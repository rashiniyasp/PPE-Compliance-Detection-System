import os

class Config:
    # Model
    model_path = "yolo11x-pose.pt" # Upgraded to big YOLO11 pose model
    
    # Paths
    input_dir = "."  # Default to current directory
    output_dir = "./outputs"
    
    # Detection thresholds
    conf_threshold = 0.3            # Min confidence to keep a person detection (lowered to 0.3)
    keypoint_conf_threshold = 0.3  # Min confidence to consider a keypoint valid
    
    # Tracking Weights
    # These base weights are used for the full tracking (ablation mode 3)
    iou_weight = 0.4
    feature_weight = 0.4
    center_weight = 0.2
    
    # Tracker parameters
    max_cost = 1.0            # Assignments above this cost are rejected
    max_age_sec = 4.0         # Delete track if missed for this many seconds
    max_age = 120              # Fallback if fps is not available
    min_hits = 1              # How many hits required to confirm a track. Keeping it 1 for baseline.
    velocity_window = 5       # Frames to average over for motion stats
    
    # Helper to get the correct weights based on ablation mode
    @classmethod
    def get_weights(cls, ablation_mode: int):
        """
        Returns a tuple of (iou_weight, feature_weight, center_weight) based on the ablation mode.
        Mode 1: IoU only
        Mode 2: IoU + Center
        Mode 3: IoU + Center + Features
        """
        if ablation_mode == 1:
            return (1.0, 0.0, 0.0)
        elif ablation_mode == 2:
            return (0.7, 0.0, 0.3)
        elif ablation_mode == 3:
            return (cls.iou_weight, cls.feature_weight, cls.center_weight)
        else:
            raise ValueError(f"Unknown ablation mode: {ablation_mode}")
