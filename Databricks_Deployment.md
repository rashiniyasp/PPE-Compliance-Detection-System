# Databricks Deployment Guide: PPE Compliance Detection System

This guide outlines the steps to package and deploy the PPE Compliance Detection System to a Databricks environment.

## 1. Project Packaging
Ensure your project folder is structured as follows before uploading to Databricks File System (DBFS):

```text
ppe_project/
│── models/
│   ├── yolo11_best.pt        # Your trained YOLO detection model
│   └── yolo11n-pose.pt       # Pretrained YOLO pose model
│── src/
│   ├── tracker.py            # Tracking logic
│   ├── pipeline.py           # YOLO-Pose + Detection association logic
│   └── app.py                # Streamlit dashboard
│── requirements.txt          # Dependencies
└── README.md
```

**`requirements.txt` should include:**
```text
ultralytics
streamlit
opencv-python-headless
pandas
numpy
scipy
```

Zip the `ppe_project` directory and upload it to DBFS (e.g., `dbfs:/FileStore/ppe_project.zip`).

## 2. Databricks Cluster Setup
1.  **Create a Cluster**: Go to the **Compute** tab and create a new cluster.
2.  **Runtime**: Select a Databricks Runtime for Machine Learning (e.g., `14.3 LTS ML`) as it comes with many data science libraries pre-installed.
    *   *Note*: If you are doing heavy video inference on the cluster, select a GPU instance type (e.g., `g4dn.xlarge` on AWS or `Standard_NC4as_T4_v3` on Azure) and ensure the runtime is "GPU-supported".
3.  **Init Scripts / Libraries**: 
    *   Once the cluster is created, go to the **Libraries** tab.
    *   Click **Install New** -> **PyPI** and paste the contents of `requirements.txt` or upload it.

## 3. Extract and Run Inference Pipeline (Databricks Notebook)
Create a new Notebook attached to your cluster to run the inference pipeline.

```python
# Unzip the project
%sh
unzip /dbfs/FileStore/ppe_project.zip -d /tmp/ppe_project
```

```python
import sys
sys.path.append("/tmp/ppe_project/src")

from pipeline import PPEPipeline
import cv2
import pandas as pd

# Initialize pipeline
pipeline = PPEPipeline(
    detection_model_path="/tmp/ppe_project/models/yolo11_best.pt",
    pose_model_path="/tmp/ppe_project/models/yolo11n-pose.pt"
)

video_path = "/dbfs/FileStore/test_video.mp4" # Upload your test video to DBFS
cap = cv2.VideoCapture(video_path)

all_results = []
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    # Process frame
    persons_data = pipeline.process_frame(frame)
    compliance_data = pipeline.check_compliance(persons_data)
    
    for c in compliance_data:
        c['frame'] = frame_count
        all_results.append(c)
        
    frame_count += 1

cap.release()

# Save results to DBFS for the Streamlit UI or analytics
df = pd.DataFrame(all_results)
df.to_csv("/dbfs/FileStore/ppe_compliance_results.csv", index=False)
display(df)
```

## 4. Hosting Streamlit UI
Databricks now supports running Streamlit directly via **Databricks Apps** (in public preview/GA depending on your workspace).

1.  Navigate to **Compute** -> **Apps** (or use the Databricks CLI).
2.  Create a new App, select the `app.py` as the entry point.
3.  Point the app to read the generated `ppe_compliance_results.csv` from DBFS or Unity Catalog.

Alternatively, you can run Streamlit via a driver node proxy using the `dbtunnel` or `databricks-streamlit` package if native Databricks Apps are not available in your workspace version.

## 5. Saving Outputs
Make sure the outputs (alert frames, ZIP files, CSV tables) are written directly to DBFS paths (e.g., `/dbfs/FileStore/ppe_alerts/`) so they persist after the cluster terminates.
