#!/usr/bin/env python3
"""
GPU Training Script for Pyronear 2025 Wildfire Detection Dataset.
This script handles the complete training workflow: dataset conversion, GPU training, and evaluation.
"""

import os
import sys
import subprocess
import argparse
import yaml
from pathlib import Path
from loguru import logger


def run_command_with_progress(command: str, description: str) -> bool:
    """Run a command with real-time progress updates."""
    print(f"\n🔄 {description}")
    print(f"Command: {command}")
    print("=" * 60)
    
    try:
        process = subprocess.Popen(
            command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            universal_newlines=True, bufsize=1
        )
        
        # Print output in real-time (filtered for important updates)
        for line in process.stdout:
            line = line.rstrip()
            # Only show important progress updates
            if any(keyword in line.lower() for keyword in [
                'epoch', 'loss', 'map', 'fps', 'memory', 'temperature',
                'completed', 'failed', 'error', 'warning', 'info'
            ]):
                print(line)
        
        process.wait()
        
        if process.returncode == 0:
            print(f"\n✅ {description} completed successfully")
            return True
        else:
            print(f"\n❌ {description} failed with exit code: {process.returncode}")
            return False
    except Exception as e:
        print(f"\n❌ {description} failed: {e}")
        return False


def check_gpu_availability() -> bool:
    """Check if GPU is available for training."""
    import torch
    
    has_gpu = torch.cuda.is_available()
    if has_gpu:
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✅ GPU detected: {gpu_name}")
    else:
        print("❌ No GPU detected. Training requires GPU for optimal performance.")
        print("💡 Please run training on a machine with NVIDIA GPU and CUDA support.")
    
    return has_gpu


def create_gpu_training_config(output_dir: Path, epochs: int) -> str:
    """Create GPU-optimized training configuration."""
    config = {
        'model': {
            'model_size': 'nano',
            'input_size': 640,  # Full resolution for better accuracy
            'batch_size': 16,   # Large batch size for GPU training
            'confidence_threshold': 0.25,
            'iou_threshold': 0.45,
            'max_detections': 100,
            'use_onnx': False,      # Don't use ONNX during training
            'use_quantization': False,  # Don't quantize during training
            'use_fp16': True,       # Use FP16 for faster GPU training
            'enable_tensorrt': False
        },
        'training': {
            'epochs': epochs,
            'learning_rate': 0.01,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            'augmentations': {
                'hsv_h': 0.015,
                'hsv_s': 0.7,
                'hsv_v': 0.4,
                'degrees': 10.0,    # More rotation for training
                'translate': 0.2,   # More translation
                'scale': 0.9,       # More scaling
                'shear': 2.0,       # Shear augmentation
                'perspective': 0.001, # Perspective transform
                'flipud': 0.0,      # No vertical flip for wildfires
                'fliplr': 0.5,      # Horizontal flip
                'mosaic': 1.0,      # Full mosaic augmentation
                'mixup': 0.1,       # Mixup augmentation
                'copy_paste': 0.0,  # Not used for wildfire data
                'auto_augment': 'randaugment', # Auto augmentation
                'erasing': 0.4      # Random erasing
            },
            'pin_memory': True,     # Enable for GPU
            'num_workers': 4,       # Multiple workers for GPU
            'prefetch_factor': 2
        },
        'inference': {
            'inference_batch_size': 2,  # Will be used for Pi inference
            'enable_streaming': True,
            'max_queue_size': 15,
            'target_fps': 8.0,
            'skip_frames': 1,
            'max_memory_usage': 0.85,
            'enable_garbage_collection': True,
            'save_predictions': False,
            'save_annotated_images': False,
            'save_video': False
        },
        'storage': {
            'model_cache_dir': './models',
            'prediction_cache_dir': './predictions',
            'log_dir': './logs',
            'log_level': 'INFO'
        },
        'dataset': {
            'train_path': str(output_dir / 'images' / 'train'),
            'val_path': str(output_dir / 'images' / 'val'),
            'test_path': str(output_dir / 'images' / 'test'),
            'classes': ['fire', 'smoke'],
            'num_classes': 2,
            'validate_images': True,
            'remove_corrupted': True,
            'min_image_size': 100,
            'enable_auto_augment': True,  # Enable for training
            'enable_mixup': True
        }
    }
    
    config_path = output_dir / "gpu_training_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"✅ Created GPU training config: {config_path}")
    return str(config_path)


def run_yolo_training(config_path: str, output_dir: Path, model_weights: str, project_name: str) -> bool:
    """Run YOLO training with the given configuration."""
    try:
        from ultralytics import YOLO
        import yaml
        
        # Load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Create dataset config
        dataset_config = {
            'path': str(output_dir),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': config['dataset']['num_classes'],
            'names': config['dataset']['classes']
        }
        
        dataset_yaml_path = output_dir / "dataset.yaml"
        with open(dataset_yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        print(f"📄 Dataset config created: {dataset_yaml_path}")
        
        # Load model
        print(f"🤖 Loading model: {model_weights}")
        model = YOLO(model_weights)
        
        # Training arguments
        train_args = {
            'data': str(dataset_yaml_path),
            'epochs': config['training']['epochs'],
            'imgsz': config['model']['input_size'],
            'batch': config['model']['batch_size'],
            'project': f"runs/{project_name}",
            'name': 'training',
            'patience': 50,  # Early stopping
            'save_period': 10,  # Save every 10 epochs
            'cache': False,
            'workers': config['training']['num_workers'],
            'lr0': config['training']['learning_rate'],
            'weight_decay': config['training']['weight_decay'],
            'warmup_epochs': config['training']['warmup_epochs'],
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            'hsv_h': config['training']['augmentations']['hsv_h'],
            'hsv_s': config['training']['augmentations']['hsv_s'],
            'hsv_v': config['training']['augmentations']['hsv_v'],
            'degrees': config['training']['augmentations']['degrees'],
            'translate': config['training']['augmentations']['translate'],
            'scale': config['training']['augmentations']['scale'],
            'shear': config['training']['augmentations']['shear'],
            'perspective': config['training']['augmentations']['perspective'],
            'flipud': config['training']['augmentations']['flipud'],
            'fliplr': config['training']['augmentations']['fliplr'],
            'mosaic': config['training']['augmentations']['mosaic'],
            'mixup': config['training']['augmentations']['mixup'],
            'copy_paste': config['training']['augmentations']['copy_paste'],
            'auto_augment': config['training']['augmentations']['auto_augment'],
            'erasing': config['training']['augmentations']['erasing']
        }
        
        # Add FP16 if enabled
        if config['model'].get('use_fp16', False):
            train_args['amp'] = True
        
        print("🎯 Starting training with GPU optimizations...")
        print(f"   • Epochs: {config['training']['epochs']}")
        print(f"   • Input size: {config['model']['input_size']}")
        print(f"   • Batch size: {config['model']['batch_size']}")
        print(f"   • Learning rate: {config['training']['learning_rate']}")
        
        # Start training
        results = model.train(**train_args)
        
        print("✅ Training completed successfully!")
        
        # Show final metrics
        if results:
            print("\n📊 Final Training Metrics:")
            print("=" * 40)
            metrics = results.results_dict
            print(f"🎯 mAP50: {metrics.get('metrics/mAP50(B)', 'N/A')}")
            print(f"🎯 mAP50-95: {metrics.get('metrics/mAP50-95(B)', 'N/A')}")
            print(f"🎯 Precision: {metrics.get('metrics/precision(B)', 'N/A')}")
            print(f"🎯 Recall: {metrics.get('metrics/recall(B)', 'N/A')}")
            print("=" * 40)
        
        return True
        
    except Exception as e:
        print(f"❌ Training error: {e}")
        return False


def main():
    """Main training function for GPU training."""
    parser = argparse.ArgumentParser(description='GPU training for Pyronear 2025 wildfire detection')
    parser.add_argument('--pyronear_path', type=str, required=True,
                       help='Path to Pyronear 2025 dataset (containing train/val/test folders)')
    parser.add_argument('--output_dir', type=str, default='./pyronear_dataset',
                       help='Output directory for converted dataset')
    parser.add_argument('--model_weights', type=str, default='yolov8n.pt',
                       help='Pre-trained model weights to use')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--skip_conversion', action='store_true',
                       help='Skip dataset conversion (use existing converted dataset)')
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip training (only convert dataset)')
    parser.add_argument('--skip_evaluation', action='store_true',
                       help='Skip evaluation after training')
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.pyronear_path):
        print(f"❌ Pyronear dataset path does not exist: {args.pyronear_path}")
        sys.exit(1)
    
    # Check GPU availability
    if not check_gpu_availability():
        print("❌ GPU training requires CUDA-capable GPU. Exiting.")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🚀 GPU Training for Pyronear 2025 Wildfire Detection")
    print("=" * 70)
    print("🎯 GPU Training Configuration:")
    print("   • GPU-accelerated training")
    print("   • Full resolution (640x640)")
    print("   • Large batch size (16)")
    print("   • Complete data augmentation")
    print("   • FP16 mixed precision")
    print("   • Multiple workers for data loading")
    print("=" * 70)
    print("💡 Note: Train on GPU, deploy optimized model to Pi for inference")
    print("=" * 70)
    print(f"📁 Pyronear dataset: {args.pyronear_path}")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"🤖 Model weights: {args.model_weights}")
    print(f"🔄 Training epochs: {args.epochs}")
    
    # Step 1: Convert dataset
    if not args.skip_conversion:
        print("\n" + "=" * 70)
        print("STEP 1: Converting Pyronear dataset to YOLO format")
        print("=" * 70)
        
        conversion_cmd = f"python convert_pyronear_dataset.py --input_path {args.pyronear_path} --output_dir {args.output_dir}"
        if not run_command_with_progress(conversion_cmd, "Converting Pyronear dataset"):
            print("❌ Dataset conversion failed")
            sys.exit(1)
    else:
        print("\n⏭️  Skipping dataset conversion")
    
    # Step 2: Create GPU training config
    print("\n" + "=" * 70)
    print("STEP 2: Creating GPU training configuration")
    print("=" * 70)
    
    config_path = create_gpu_training_config(output_dir, args.epochs)
    
    # Step 3: Train model
    if not args.skip_training:
        print("\n" + "=" * 70)
        print("STEP 3: Training YOLO model (GPU optimized)")
        print("=" * 70)
        
        # Run training directly
        print("🚀 Starting YOLO training...")
        if not run_yolo_training(config_path, output_dir, args.model_weights, "gpu-wildfire-detection"):
            print("❌ Training failed")
            sys.exit(1)
    else:
        print("\n⏭️  Skipping training")
    
    # Step 4: Evaluate model
    if not args.skip_evaluation and not args.skip_training:
        print("\n" + "=" * 70)
        print("STEP 4: Evaluating trained model")
        print("=" * 70)
        
        # Find the best model
        best_model_path = f"runs/gpu-wildfire-detection/weights/best.pt"
        if not os.path.exists(best_model_path):
            # Try to find the model in the latest run
            runs_dir = Path("runs/gpu-wildfire-detection")
            if runs_dir.exists():
                latest_run = max(runs_dir.iterdir(), key=lambda x: x.stat().st_mtime if x.is_dir() else 0)
                best_model_path = latest_run / "weights" / "best.pt"
        
        if os.path.exists(best_model_path):
            eval_cmd = f"python evaluate.py --model_path {best_model_path} --test_path {output_dir}/images/test --config {config_path}"
            if not run_command_with_progress(eval_cmd, "Evaluating model"):
                print("❌ Evaluation failed")
                sys.exit(1)
        else:
            print(f"⚠️  Best model not found at {best_model_path}")
    else:
        print("\n⏭️  Skipping evaluation")
    
    # Step 5: Export for Pi deployment
    print("\n" + "=" * 70)
    print("STEP 5: Preparing model for Pi deployment")
    print("=" * 70)
    
    best_model_path = f"runs/gpu-wildfire-detection/weights/best.pt"
    if not os.path.exists(best_model_path):
        runs_dir = Path("runs/gpu-wildfire-detection")
        if runs_dir.exists():
            latest_run = max(runs_dir.iterdir(), key=lambda x: x.stat().st_mtime if x.is_dir() else 0)
            best_model_path = latest_run / "weights" / "best.pt"
    
    if os.path.exists(best_model_path):
        print(f"✅ Best model found: {best_model_path}")
        print("📋 Next steps for Pi deployment:")
        print("   1. Copy the trained model to your Pi:")
        print(f"      scp {best_model_path} pi@your-pi-ip:~/wildfire_detection/")
        print("   2. On Pi, run optimized inference:")
        print("      python3 inference.py --model_path best.pt --source 0 --stream")
        print(f"   3. Model path: {best_model_path}")
    else:
        print("⚠️  No trained model found")
    
    print("\n" + "=" * 70)
    print("🎉 GPU Training completed!")
    print("=" * 70)
    print("📊 Training Summary:")
    print(f"   • Dataset: {args.pyronear_path}")
    print(f"   • Output: {args.output_dir}")
    print(f"   • Epochs: {args.epochs}")
    print(f"   • Config: {config_path}")
    if os.path.exists(best_model_path):
        print(f"   • Model: {best_model_path}")
    print("=" * 70)


if __name__ == "__main__":
    main() 