import os
import cv2
import argparse
import glob
import numpy as np
import pandas as pd
from ultralytics import YOLO

from config import Config
from skeleton_features import SkeletonFeatureExtractor
from tracker import SkeletonTracker

SKELETON_PAIRS = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
    (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9),
    (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4),
    (3, 5), (4, 6)
]

# Generate distinct colors for tracks
def get_color(track_id):
    np.random.seed(track_id)
    return tuple(int(x) for x in np.random.randint(0, 255, 3))

def process_video(video_path, args, model, feat_extractor):
    video_name = os.path.basename(video_path)
    print(f"Processing {video_name} (Ablation Mode {args.ablation_mode})...")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if fps <= 0 or np.isnan(fps):
        print(f"Warning: Invalid FPS {fps} detected for {video_name}. Falling back to 30 FPS for velocity calc.")
        fps = 30.0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = os.path.join(args.output_dir, f"tracked_mode{args.ablation_mode}_{video_name}")
    out = cv2.VideoWriter(out_path, fourcc, fps, (frame_width, frame_height))

    tracker = SkeletonTracker(Config, args.ablation_mode, fps, frame_width, frame_height)
    
    log_data = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        timestamp_sec = frame_idx / fps

        # YOLO inference
        results = model(frame, verbose=False, conf=Config.conf_threshold, classes=[0]) # class 0 is person
        
        detections = []
        if len(results) > 0:
            result = results[0]
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            
            if result.keypoints is not None:
                keypoints = result.keypoints.data.cpu().numpy() # [N, 17, 3]
                
                for i in range(len(boxes)):
                    box = boxes[i]
                    kpts = keypoints[i]
                    conf = confs[i]
                    
                    bbox_height = max(box[3] - box[1], 1e-6)
                    features, mask = feat_extractor.extract(kpts, bbox_height)
                    
                    detections.append({
                        'bbox': box,
                        'keypoints': kpts,
                        'features': features,
                        'mask': mask,
                        'conf': conf
                    })

        # Update tracker
        tracks = tracker.update(detections, frame_idx)

        # Draw and log
        for trk in tracks:
            # Only draw confirmed tracks
            if trk.hits >= Config.min_hits and trk.is_matched:
                color = get_color(trk.track_id)
                x1, y1, x2, y2 = map(int, trk.bbox)
                
                # Draw bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw Track ID
                cv2.putText(frame, f"ID: {trk.track_id}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Draw Skeleton
                kpts = trk.keypoints
                for p1_idx, p2_idx in SKELETON_PAIRS:
                    if p1_idx < len(kpts) and p2_idx < len(kpts):
                        p1 = kpts[p1_idx]
                        p2 = kpts[p2_idx]
                        if p1[2] > Config.keypoint_conf_threshold and p2[2] > Config.keypoint_conf_threshold:
                            pt1 = (int(p1[0]), int(p1[1]))
                            pt2 = (int(p2[0]), int(p2[1]))
                            cv2.line(frame, pt1, pt2, color, 2)
                            
            # Log data
            log_data.append({
                'video_name': video_name,
                'frame_idx': frame_idx,
                'timestamp_sec': timestamp_sec,
                'track_id': trk.track_id,
                'bbox_x1': trk.bbox[0],
                'bbox_y1': trk.bbox[1],
                'bbox_x2': trk.bbox[2],
                'bbox_y2': trk.bbox[3],
                'det_conf': trk.conf,
                'assigned_cost': trk.assigned_cost,
                'iou_cost': trk.last_iou_cost,
                'center_cost': trk.last_center_cost,
                'feature_cost': trk.last_feature_cost,
                'matched_flag': trk.is_matched,
                'missed_count': trk.missed,
                'ablation_mode': args.ablation_mode
            })

        out.write(frame)
        frame_idx += 1
        
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx} frames...")

    cap.release()
    out.release()
    
    # Save CSV
    df = pd.DataFrame(log_data)
    csv_path = os.path.join(args.output_dir, "tracking_logs", f"mode{args.ablation_mode}_{video_name}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved output to {out_path} and {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Skeleton-based Multi-Person Tracker")
    parser.add_argument("--input_dir", type=str, default=Config.input_dir, help="Directory containing input videos")
    parser.add_argument("--output_dir", type=str, default=Config.output_dir, help="Directory to save outputs")
    parser.add_argument("--ablation_mode", type=int, default=3, choices=[1, 2, 3], 
                        help="1: IoU only, 2: IoU + Center, 3: Full Features")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "tracking_logs"), exist_ok=True)

    print(f"Loading YOLOv8 pose model ({Config.model_path})...")
    model = YOLO(Config.model_path)
    feat_extractor = SkeletonFeatureExtractor(conf_threshold=Config.keypoint_conf_threshold)

    search_path = os.path.join(args.input_dir, "*.mp4")
    videos = glob.glob(search_path)
    
    if not videos:
        print(f"No .mp4 videos found in {args.input_dir}")
        return

    for video_path in videos:
        process_video(video_path, args, model, feat_extractor)

if __name__ == "__main__":
    main()
