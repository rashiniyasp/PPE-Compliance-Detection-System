import streamlit as st
import pandas as pd
import zipfile
import os
import io

st.set_page_config(
    page_title="PPE Compliance Detection",
    page_icon="👷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for aesthetics
st.markdown("""
<style>
    :root {
        --primary: #0ea5e9;
        --bg-dark: #0f172a;
        --card-dark: #1e293b;
    }
    .stApp {
        background-color: var(--bg-dark);
        color: #f8fafc;
    }
    .metric-card {
        background: var(--card-dark);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        text-align: center;
        border: 1px solid #334155;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: var(--primary);
    }
    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
    }
    .alert-text {
        color: #ef4444;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.title("👷‍♂️ PPE Compliance Detection Dashboard")
st.markdown("Monitor real-time compliance for helmets, vests, gloves, goggles, and boots across your site.")

# Sidebar
st.sidebar.header("Controls")
video_file = st.sidebar.file_uploader("Upload Test Video", type=['mp4', 'avi'])
run_inference = st.sidebar.button("Run Pipeline")

# Dummy data for UI structure demonstration until pipeline is hooked up
if 'results_df' not in st.session_state:
    st.session_state.results_df = pd.DataFrame({
        'Frame No.': ['00312', '00460', '00891'],
        'Time (s)': [10.4, 15.3, 29.7],
        'Person ID': ['Person A', 'Person B', 'Person C'],
        'Track ID': [1, 2, 3],
        'Detected PPE': ['vest, boots', 'helmet, vest, gloves, boots', 'helmet, boots'],
        'Missing PPE': ['helmet, gloves', '—', 'vest, gloves, goggles'],
        'Alert': ['⚠️ Yes', 'No', '⚠️ Yes']
    })

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="metric-card"><div class="metric-value">94.2%</div><div class="metric-label">mAP50</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric-card"><div class="metric-value">88.5%</div><div class="metric-label">Precision</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="metric-card"><div class="metric-value">91.0%</div><div class="metric-label">Recall</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div class="metric-card"><div class="metric-value">32</div><div class="metric-label">Inference FPS</div></div>', unsafe_allow_html=True)

st.write("---")

col_video, col_table = st.columns([1, 1])

with col_video:
    st.subheader("Live Feed / Replay")
    if video_file:
        st.video(video_file)
    else:
        st.info("Upload a video in the sidebar to view playback.")

with col_table:
    st.subheader("Compliance Table")
    st.dataframe(
        st.session_state.results_df,
        use_container_width=True,
        hide_index=True
    )
    
    st.subheader("Export Alerts")
    
    # Create dummy zip for download
    dummy_zip = io.BytesIO()
    with zipfile.ZipFile(dummy_zip, 'w') as zf:
        zf.writestr("alert_frame_00312.txt", "Bounding box data for frame 312")
    
    st.download_button(
        label="📥 Download Alert Frames (ZIP)",
        data=dummy_zip.getvalue(),
        file_name="ppe_alerts.zip",
        mime="application/zip",
        use_container_width=True
    )

st.write("---")
st.subheader("Detailed Evaluation Metrics")
metrics_data = {
    'Metric': ['mAP50', 'mAP50-95', 'Precision (Overall)', 'Recall (Overall)', 'F1 Score (Overall)', 'Macro F1', 'ID Switches', 'Alert Latency (frames)'],
    'Value': ['94.2%', '72.1%', '88.5%', '91.0%', '89.7%', '87.4%', '4', '1.2']
}
st.table(pd.DataFrame(metrics_data))
