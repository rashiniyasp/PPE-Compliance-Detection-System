import torch
import sys
import os
from ultralytics import YOLO

def check_gpu():
    print("--- PyTorch GPU Check ---")
    print(f"PyTorch Version: {torch.__version__}")
    if not torch.cuda.is_available():
        print("ERROR: CUDA is NOT available. PyTorch cannot see the GPU.")
        print("Please check your PyTorch installation and ensure you installed the CUDA-enabled version.")
        print("You can install it via: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 (or cu121)")
        sys.exit(1)
    
    print(f"CUDA Version: {torch.version.cuda}")
    try:
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
    except:
        pass
        
    gpu_count = torch.cuda.device_count()
    print(f"Available GPUs: {gpu_count}")
    
    for i in range(gpu_count):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Compute Capability: {torch.cuda.get_device_capability(i)}")
        
    print("GPU Check Passed!\n")

def main():
    # 1. Check GPU compatibility to avoid Blackwell/overhead issues
    check_gpu()
    
    # 2. Load the YOLO model (using YOLO11x for detection since dataset has bounding boxes)
    print("Loading YOLO11x model...")
    model = YOLO('yolo11x.pt')  # load a pretrained model
    
    # 3. Start training
    print("Starting training...")
    # Get the absolute path to data.yaml
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data.yaml')
    
    results = model.train(
        data=data_path,
        epochs=100,      # Number of epochs
        imgsz=640,       # Image size
        batch=-1,        # AutoBatch for optimal GPU memory utilization
        device=0,        # Run on GPU 0
        project='runs/train',
        name='ppe_detection',
        exist_ok=True,
        # amp=True, # Automatic Mixed Precision (enabled by default in YOLO)
    )
    
    print("\nTraining complete! Model saved to: runs/train/ppe_detection/weights/best.pt")
    
    # 4. Validate the model to get mAP50 and mAP50-95
    print("\nRunning validation...")
    metrics = model.val()
    print("Validation complete!")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50:    {metrics.box.map50:.4f}")

if __name__ == '__main__':
    main()
