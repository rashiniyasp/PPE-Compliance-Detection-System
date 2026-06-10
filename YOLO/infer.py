import cv2
import argparse
import os
import torch
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLO11 Inference on a video")
    parser.add_argument("--weights", type=str, default="runs/train/ppe_detection/weights/best.pt", help="Path to trained YOLO weights")
    parser.add_argument("--source", type=str, required=True, help="Path to input video file")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument("--save-dir", type=str, default="runs/detect", help="Directory to save the output video")
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not os.path.exists(args.weights):
        print(f"Error: Model weights not found at {args.weights}")
        print("Please train the model first or provide the correct path.")
        return

    print(f"Loading model: {args.weights}")
    model = YOLO(args.weights)
    
    print(f"Running inference on: {args.source}")
    print("This will save the annotated video and the raw text labels for each frame.")
    
    # Run prediction
    results = model.predict(
        source=args.source,
        conf=args.conf,
        save=True,          # Save the annotated video/images
        save_txt=True,      # Save the raw detection boxes to text files
        save_conf=True,     # Save confidences in the text files
        project=args.save_dir,
        name="ppe_test",
        exist_ok=True,
        stream=True         # Stream results for large videos
    )
    
    # Iterate through the generator to execute inference
    for frame_idx, r in enumerate(results):
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx} frames...")
            
    print(f"\nInference complete!")
    print(f"Annotated video and raw detection labels are saved in: {args.save_dir}/ppe_test")

if __name__ == "__main__":
    main()
