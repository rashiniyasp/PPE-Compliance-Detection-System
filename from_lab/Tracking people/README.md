# Baseline Multi-Person Skeleton Tracker

This is a baseline project for multi-person tracking in videos utilizing YOLOv8-pose keypoints and skeleton kinematics, without relying on DeepSORT or custom training.

## Installation

1. Make sure you have Python 3.8+ installed.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

By default, the script reads `.mp4` videos from the current directory and saves output videos and tracking CSV logs to `./outputs`.

```bash
# Run with default configuration (ablation_mode 3: Full skeleton + center + IoU tracking)
python main.py

# Run ablation mode 1: IoU tracking only
python main.py --ablation_mode 1

# Run ablation mode 2: IoU + Center distance tracking
python main.py --ablation_mode 2

# Provide custom paths
python main.py --input_dir ./my_videos --output_dir ./my_outputs
```

## Tuning Parameters for ID Switches

If you experience frequent ID switches, check `config.py` and adjust the following parameters:
- `velocity_window`: Increase this to smooth short-term velocity calculations.
- `iou_weight`, `feature_weight`, `center_weight`: Adjust the relative importance of these cost components in `config.py`.
- `max_cost`: Lower this threshold to prevent bad matches, or increase it if valid tracks are being dropped.
- `max_age`: Increase this value so the tracker waits longer before deleting an unmatched track.

## Outputs
- **Annotated Videos**: Saved as `outputs/tracked_<video_name>.mp4`. They include bounding boxes, colored skeletons, and track IDs.
- **CSV Logs**: Saved as `outputs/tracking_logs/<video_name>.csv`. They contain detailed frame-by-frame stats such as cost components and detection confidences.
