import streamlit as st
import pandas as pd
import zipfile
import os
import io
import tempfile
import cv2
import time
import numpy as np
import datetime
import shutil

from pipeline import PPEPipeline

st.set_page_config(
    page_title="PPE Compliance Hub",
    page_icon="👷",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    :root {
        --primary: #0ea5e9;
        --bg-dark: #0f172a;
        --card-dark: #1e293b;
    }
    .stApp {
        background-color: var(--bg-dark);
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    .metric-card {
        background: var(--card-dark);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        text-align: center;
        border: 1px solid #334155;
        margin-bottom: 20px;
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
    .alert-text {
        color: #ef4444;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.sidebar.header("Navigation")
app_mode = st.sidebar.radio("Select Module", ["Live Monitoring Dashboard", "Tracking Method Comparison"])

if app_mode == "Live Monitoring Dashboard":
    st.title("👷‍♂️ Live Monitoring Dashboard")
    st.markdown("Monitor real-time compliance for helmets, vests, gloves, goggles, and boots.")
    
    st.sidebar.header("Controls")
    video_file = st.sidebar.file_uploader("Upload Test Video", type=['mp4', 'avi'])
    use_default = st.sidebar.checkbox("Use Default Test Video", value=True, help="from_lab/PPE_systech_Test file.mp4")
    run_inference = st.sidebar.button("Run Pipeline")
    
    if 'results_df' not in st.session_state:
        st.session_state.results_df = pd.DataFrame(columns=['Frame No.', 'Time (s)', 'Person ID', 'Detected PPE', 'Confidence', 'Missing PPE', 'Alert'])
    if 'alert_zip_bytes' not in st.session_state:
        st.session_state.alert_zip_bytes = None
    if 'output_video_path' not in st.session_state:
        st.session_state.output_video_path = None
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><div class="metric-value">83.9%</div><div class="metric-label">mAP50 (Positive)</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><div class="metric-value">80.2%</div><div class="metric-label">Precision (Positive)</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><div class="metric-value">79.9%</div><div class="metric-label">Recall (Positive)</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><div class="metric-value">80.0%</div><div class="metric-label">F1 Score (Positive)</div></div>', unsafe_allow_html=True)
    
    st.write("---")
    
    # PPE Colors for drawing individual PPE bounding boxes
    PPE_COLORS = {
        'helmet':  (255, 165, 0),   # Orange
        'vest':    (0, 255, 127),   # Spring Green
        'gloves':  (255, 0, 255),   # Magenta
        'boots':   (0, 191, 255),   # Deep Sky Blue
        'goggles': (255, 255, 0),   # Yellow
    }
    
    col_video, col_table = st.columns([1, 1])
    
    with col_video:
        st.subheader("Live Feed / Replay")
        video_placeholder = st.empty()
        fps_placeholder = st.empty()
        
        if run_inference:
            with st.spinner("Loading models and starting inference..."):
                
                video_path = None
                if video_file is not None:
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                    tfile.write(video_file.read())
                    video_path = tfile.name
                elif use_default:
                    video_path = "from_lab/PPE_systech_Test file.mp4"
                    
                if video_path and os.path.exists(video_path):
                    cap = cv2.VideoCapture(video_path)
                    fps_video = cap.get(cv2.CAP_PROP_FPS)
                    if not fps_video or fps_video == 0:
                        fps_video = 30.0
                    
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    
                    pipeline = PPEPipeline(fps=fps_video, frame_width=width, frame_height=height)
                    
                    # Output video
                    os.makedirs("outputs", exist_ok=True)
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_path = f"outputs/annotated_video_{timestamp}.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(output_path, fourcc, fps_video, (width, height))
                    
                    # Alert frames temp directory
                    alert_dir = tempfile.mkdtemp(prefix="ppe_alerts_")
                        
                    frame_idx = 0
                    results_list = []
                    alert_frame_count = 0
                    
                    while cap.isOpened():
                        start_time = time.time()
                        ret, frame = cap.read()
                        if not ret:
                            break
                            
                        frame_idx += 1
                        
                        # Process frame
                        persons = pipeline.process_frame(frame)
                        compliance_results = pipeline.check_compliance(persons)
                        
                        has_violation = False
                        
                        # Render Annotations
                        for person in persons:
                            box = person['bbox']
                            x1, y1, x2, y2 = map(int, box)
                            
                            # Box color based on compliance
                            is_compliant = len(set(['helmet', 'vest', 'gloves', 'boots']) - set(person['assigned_ppe'])) == 0
                            person_color = (0, 255, 0) if is_compliant else (0, 0, 255)
                            
                            if not is_compliant:
                                has_violation = True
                            
                            # Draw person bounding box
                            cv2.rectangle(frame, (x1, y1), (x2, y2), person_color, 2)
                            
                            # Person ID label at top-left above box
                            label = f"Person {person['id']}"
                            lbl_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                            cv2.rectangle(frame, (x1, y1 - lbl_size[1] - 10), (x1 + lbl_size[0], y1), person_color, -1)
                            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                            
                            # Draw PPE-specific bounding boxes with class + confidence
                            for ppe_detail in person.get('ppe_details', []):
                                px1, py1, px2, py2 = map(int, ppe_detail['bbox'])
                                ppe_color = PPE_COLORS.get(ppe_detail['class'], (200, 200, 200))
                                cv2.rectangle(frame, (px1, py1), (px2, py2), ppe_color, 2)
                                
                                ppe_label = f"{ppe_detail['class']} {ppe_detail['conf']:.2f}"
                                plbl_size, _ = cv2.getTextSize(ppe_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                                cv2.rectangle(frame, (px1, py1 - plbl_size[1] - 6), (px1 + plbl_size[0], py1), ppe_color, -1)
                                cv2.putText(frame, ppe_label, (px1, py1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                            
                            # Missing PPE text inside top-right of person box
                            missing_items = set(['helmet', 'vest', 'gloves', 'boots']) - set(person['assigned_ppe'])
                            if missing_items:
                                miss_text = f"MISSING: {', '.join(missing_items)}"
                                mt_size, _ = cv2.getTextSize(miss_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                                mt_x = max(x1 + 5, x2 - mt_size[0] - 5)
                                mt_y = y1 + mt_size[1] + 8
                                cv2.rectangle(frame, (mt_x - 2, mt_y - mt_size[1] - 4), (mt_x + mt_size[0] + 2, mt_y + 4), (0, 0, 180), -1)
                                cv2.putText(frame, miss_text, (mt_x, mt_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                            
                            # Draw skeleton keypoints
                            if person.get('keypoints') is not None:
                                for kp in person['keypoints']:
                                    if kp[2] > 0.5:
                                        cv2.circle(frame, (int(kp[0]), int(kp[1])), 3, (0, 255, 255), -1)
    
                        # Update UI Image
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        video_placeholder.image(frame_rgb, channels="RGB")
                        
                        # Write to video file
                        out.write(frame)
                        
                        # Save alert frame if violation detected
                        if has_violation:
                            alert_frame_count += 1
                            alert_path = os.path.join(alert_dir, f"alert_frame_{frame_idx:05d}.jpg")
                            cv2.imwrite(alert_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                        
                        # Update compliance table
                        time_s = round(frame_idx / fps_video, 2)
                        for res in compliance_results:
                            results_list.append({
                                'Frame No.': f"{frame_idx:05d}",
                                'Time (s)': time_s,
                                'Person ID': f"Person {res['person_id']}",
                                'Detected PPE': ", ".join(res['detected_ppe']) if res['detected_ppe'] else "â€”",
                                'Confidence': res['conf_str'],
                                'Missing PPE': ", ".join(res['missing_ppe']) if res['missing_ppe'] else "â€”",
                                'Alert': 'âš ï¸ Yes' if res['alert'] else 'No'
                            })
                            
                        end_time = time.time()
                        inf_fps = round(1.0 / (max(end_time - start_time, 0.001)), 1)
                        fps_placeholder.info(f"Live Inference FPS: {inf_fps}")
                        
                    cap.release()
                    out.release()
                    st.session_state.results_df = pd.DataFrame(results_list)
                    st.session_state.output_video_path = output_path
                    
                    # Bundle alert frames into ZIP
                    if alert_frame_count > 0:
                        zip_buffer = io.BytesIO()
                        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                            for fname in sorted(os.listdir(alert_dir)):
                                fpath = os.path.join(alert_dir, fname)
                                zf.write(fpath, fname)
                        zip_buffer.seek(0)
                        st.session_state.alert_zip_bytes = zip_buffer.getvalue()
                        shutil.rmtree(alert_dir, ignore_errors=True)
                        st.success(f"Inference Complete! Video saved to `{output_path}`. {alert_frame_count} alert frames captured.")
                    else:
                        st.session_state.alert_zip_bytes = None
                        shutil.rmtree(alert_dir, ignore_errors=True)
                        st.success(f"Inference Complete! Video saved to `{output_path}`. No violations detected.")
                else:
                    st.error(f"Video file not found at {video_path}. Please upload or check default path.")
    
    with col_table:
        st.subheader("Compliance Table")
        # Show last 50 entries to avoid massive tables lagging UI
        display_df = st.session_state.results_df.tail(50) if not st.session_state.results_df.empty else st.session_state.results_df
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
        
        st.subheader("Export & Downloads")
        
        if not st.session_state.results_df.empty:
            dl_col1, dl_col2 = st.columns(2)
            
            with dl_col1:
                # CSV Download
                csv = st.session_state.results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="ðŸ“¥ Download Compliance Logs (CSV)",
                    data=csv,
                    file_name="ppe_compliance_logs.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with dl_col2:
                # ZIP Download
                if st.session_state.alert_zip_bytes is not None:
                    st.download_button(
                        label="ðŸ“¦ Download Alert Frames (ZIP)",
                        data=st.session_state.alert_zip_bytes,
                        file_name="ppe_alert_frames.zip",
                        mime="application/zip",
                        use_container_width=True
                    )
                else:
                    st.info("No alert frames to download.")
        else:
            st.info("Run the pipeline to generate exportable data.")
    
    st.write("---")
    st.subheader("Detailed Evaluation Metrics (From Validation Set)")
    
    col_class, col_metrics = st.columns(2)
    
    with col_class:
        st.markdown("#### Per-Class Performance (Positive Classes)")
        class_data = {
            'Class': ['Person', 'vest', 'boots', 'goggles', 'helmet', 'gloves'],
            'Instances': ['239', '171', '151', '47', '201', '136'],
            'mAP50': ['0.883 (88.3%)', '0.867 (86.7%)', '0.838 (83.8%)', '0.824 (82.4%)', '0.821 (82.1%)', '0.800 (80.0%)']
        }
        st.table(pd.DataFrame(class_data))
    
    with col_metrics:
        st.markdown("#### Overall Metrics")
        metrics_data = {
            'Metric': ['mAP50 (Overall)', 'mAP50-95', 'Precision (Overall)', 'Recall (Overall)', 'F1 Score (Overall)'],
            'Value': ['56.8%', '27.9%', '60.1%', '53.3%', '56.5%']
        }
        st.table(pd.DataFrame(metrics_data))
    
    st.info("Note: The overall metrics include the 'negative' classes (no_helmet, no_boots, etc.) which had severe class imbalance and performed poorly. Our system correctly ignores these classes and relies on the highly accurate positive detections shown above.")
    

elif app_mode == "Tracking Method Comparison":
    st.title("⚖️ Tracking Method Comparison")
    st.markdown("Evaluate performance of SkeletonStat-Track against Baseline Trackers.")
    
    st.sidebar.header("Controls")
    video_file = st.sidebar.file_uploader("Upload Test Video", type=['mp4', 'avi'])
    use_default = st.sidebar.checkbox("Use Default Test Video", value=True, help="from_lab/PPE_systech_Test file.mp4")
    run_inference = st.sidebar.button("Run Evaluation Pipeline")
    
    PPE_COLORS = {
        'helmet':  (255, 165, 0),
        'vest':    (0, 255, 127),
        'gloves':  (255, 0, 255),
        'boots':   (0, 191, 255),
        'goggles': (255, 255, 0),
    }
    
    METHODS = [
        {'id': 'skeleton_yolo', 'name': '1. YOLO11x + SkeletonStat-Track (Proposed)'},
        {'id': 'traditional_iou', 'name': '2. Traditional BoT-SORT + IoU (Baseline 1)'},
        {'id': 'skeleton_mediapipe', 'name': '3. MediaPipe Pose + SkeletonTracker (Baseline 2)'}
    ]
    
    if run_inference:
        video_path = None
        if video_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(video_file.read())
            video_path = tfile.name
        elif use_default:
            video_path = "from_lab/PPE_systech_Test file.mp4"
            
        if video_path and os.path.exists(video_path):
            cap = cv2.VideoCapture(video_path)
            fps_video = cap.get(cv2.CAP_PROP_FPS)
            if not fps_video or fps_video == 0:
                fps_video = 30.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
    
            st.session_state['eval_results'] = {}
    
            for method in METHODS:
                st.write(f"---")
                st.header(f"Evaluating: {method['name']}")
                
                col_video, col_metrics = st.columns([2, 1])
                with col_video:
                    video_placeholder = st.empty()
                with col_metrics:
                    fps_placeholder = st.empty()
                    progress_bar = st.progress(0)
                    
                pipeline = PPEPipeline(tracking_method=method['id'], fps=fps_video, frame_width=width, frame_height=height)
                
                os.makedirs("outputs", exist_ok=True)
                output_path = f"outputs/annotated_{method['id']}.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps_video, (width, height))
                
                cap = cv2.VideoCapture(video_path)
                frame_idx = 0
                results_list = []
                
                total_fps = 0
                
                while cap.isOpened():
                    start_time = time.time()
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    frame_idx += 1
                    persons = pipeline.process_frame(frame)
                    compliance_results = pipeline.check_compliance(persons)
                    
                    # Render Annotations
                    for person in persons:
                        box = person['bbox']
                        x1, y1, x2, y2 = map(int, box)
                        
                        is_compliant = len(set(['helmet', 'vest', 'gloves', 'boots']) - set(person['assigned_ppe'])) == 0
                        person_color = (0, 255, 0) if is_compliant else (0, 0, 255)
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), person_color, 2)
                        label = f"ID {person['id']}"
                        lbl_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                        cv2.rectangle(frame, (x1, y1 - lbl_size[1] - 10), (x1 + lbl_size[0], y1), person_color, -1)
                        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        for ppe_detail in person.get('ppe_details', []):
                            px1, py1, px2, py2 = map(int, ppe_detail['bbox'])
                            ppe_color = PPE_COLORS.get(ppe_detail['class'], (200, 200, 200))
                            cv2.rectangle(frame, (px1, py1), (px2, py2), ppe_color, 2)
                        
                        missing_items = set(['helmet', 'vest', 'gloves', 'boots']) - set(person['assigned_ppe'])
                        if missing_items:
                            miss_text = f"MISSING: {', '.join(missing_items)}"
                            mt_size, _ = cv2.getTextSize(miss_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                            mt_x = max(x1 + 5, x2 - mt_size[0] - 5)
                            mt_y = y1 + mt_size[1] + 8
                            cv2.rectangle(frame, (mt_x - 2, mt_y - mt_size[1] - 4), (mt_x + mt_size[0] + 2, mt_y + 4), (0, 0, 180), -1)
                            cv2.putText(frame, miss_text, (mt_x, mt_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        
                        if person.get('keypoints') is not None:
                            for kp in person['keypoints']:
                                if kp[2] > 0.5:
                                    cv2.circle(frame, (int(kp[0]), int(kp[1])), 3, (0, 255, 255), -1)
    
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(frame_rgb, channels="RGB")
                    out.write(frame)
                    
                    for res in compliance_results:
                        results_list.append({
                            'Frame': frame_idx,
                            'Person ID': res['person_id'],
                            'Missing': ", ".join(res['missing_ppe']) if res['missing_ppe'] else "—",
                            'Alert': 1 if res['alert'] else 0
                        })
                        
                    end_time = time.time()
                    inf_fps = round(1.0 / (max(end_time - start_time, 0.001)), 1)
                    total_fps += inf_fps
                    fps_placeholder.info(f"Live Inference FPS: {inf_fps}")
                    if total_frames > 0:
                        progress_bar.progress(min(frame_idx / total_frames, 1.0))
                    
                cap.release()
                out.release()
                
                avg_fps = total_fps / frame_idx if frame_idx > 0 else 0
                df = pd.DataFrame(results_list)
                total_violations = df['Alert'].sum() if not df.empty else 0
                unique_ids = df['Person ID'].nunique() if not df.empty else 0
                
                st.session_state['eval_results'][method['id']] = {
                    'name': method['name'],
                    'video_path': output_path,
                    'df': df,
                    'avg_fps': avg_fps,
                    'total_violations': total_violations,
                    'unique_ids': unique_ids
                }
                
                st.success(f"Completed {method['name']}!")
    
    # Display Final Comparison if available
    if 'eval_results' in st.session_state and len(st.session_state['eval_results']) > 0:
        st.write("---")
        st.title("📊 Evaluation Results Comparison")
        
        cols = st.columns(len(st.session_state['eval_results']))
        
        for idx, method_id in enumerate(st.session_state['eval_results']):
            res = st.session_state['eval_results'][method_id]
            with cols[idx]:
                st.subheader(res['name'])
                st.markdown(f'<div class="metric-card"><div class="metric-value">{res["avg_fps"]:.1f}</div><div class="metric-label">Average FPS</div></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-card"><div class="metric-value">{res["total_violations"]}</div><div class="metric-label">Total Violations Detected</div></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-card"><div class="metric-value">{res["unique_ids"]}</div><div class="metric-label">Unique IDs Tracked (Lower is better if few people)</div></div>', unsafe_allow_html=True)
                
                with open(res['video_path'], 'rb') as f:
                    st.download_button(
                        label=f"⬇️ Download {method_id} Video",
                        data=f,
                        file_name=f"output_{method_id}.mp4",
                        mime="video/mp4",
                        key=f"dl_{method_id}"
                    )
                
                st.dataframe(res['df'].head(50))
    
