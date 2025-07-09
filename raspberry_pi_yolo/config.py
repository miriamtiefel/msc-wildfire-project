"""
Configuration management for Raspberry Pi YOLO wildfire detection.
Enhanced with edge-specific optimizations and deployment settings.
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path


@dataclass
class ModelConfig:
    """Model-specific configuration for Raspberry Pi optimization."""
    model_size: str = "nano"  # nano, small, medium, large
    input_size: int = 320  # Reduced from 640 for Pi optimization
    batch_size: int = 1  # Single batch for Pi memory constraints
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    max_detections: int = 100
    
    # Optimization flags
    use_onnx: bool = True
    use_quantization: bool = True
    use_fp16: bool = False  # Pi may not support FP16
    enable_tensorrt: bool = False  # Future optimization


@dataclass
class TrainingConfig:
    """Enhanced training configuration with Pi-specific optimizations."""
    epochs: int = 100
    learning_rate: float = 0.01
    weight_decay: float = 0.0005
    warmup_epochs: int = 3
    
    # Data augmentation (reduced for Pi training)
    augmentations: Dict[str, Any] = field(default_factory=lambda: {
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 0.5,  # Reduced from 1.0
        "mixup": 0.0,
        "copy_paste": 0.0,
        "auto_augment": None,  # Disabled for Pi
        "erasing": 0.0
    })
    
    # Memory optimization
    pin_memory: bool = False  # Disabled for Pi
    num_workers: int = 0  # Single process for Pi
    prefetch_factor: int = 2


@dataclass
class InferenceConfig:
    """Inference configuration optimized for real-time processing."""
    # Performance settings
    inference_batch_size: int = 1
    enable_streaming: bool = True
    max_queue_size: int = 10
    
    # Real-time processing
    target_fps: float = 5.0  # Reduced for Pi capabilities
    skip_frames: int = 2  # Process every Nth frame
    
    # Memory management
    max_memory_usage: float = 0.8  # 80% of available RAM
    enable_garbage_collection: bool = True
    
    # Output settings
    save_predictions: bool = False
    save_annotated_images: bool = False
    save_video: bool = False


@dataclass
class StorageConfig:
    """Storage configuration for local file management."""
    # Storage paths
    model_cache_dir: str = "./models"
    prediction_cache_dir: str = "./predictions"
    log_dir: str = "./logs"
    log_level: str = "INFO"


@dataclass
class DatasetConfig:
    """Enhanced dataset configuration with better management."""
    train_path: str = ""
    val_path: str = ""
    test_path: str = ""
    
    # Class configuration
    classes: List[str] = field(default_factory=lambda: ["fire", "smoke"])
    num_classes: int = 2
    
    # Data validation
    validate_images: bool = True
    remove_corrupted: bool = True
    min_image_size: int = 100
    
    # Augmentation settings
    enable_auto_augment: bool = False  # Disabled for Pi
    enable_mixup: bool = False  # Disabled for Pi


@dataclass
class PiOptimizationConfig:
    """Raspberry Pi specific optimizations."""
    # Hardware optimization
    use_cpu_only: bool = True
    enable_arm_neon: bool = True
    optimize_memory_layout: bool = True
    
    # Thermal management
    enable_thermal_throttling: bool = True
    max_cpu_temp: float = 70.0  # Celsius
    thermal_check_interval: int = 30  # seconds
    
    # Power management
    enable_power_saving: bool = True
    cpu_frequency_limit: Optional[int] = None  # MHz
    
    # Storage optimization
    use_ramdisk_for_cache: bool = False
    cache_size_limit: int = 100  # MB


class Config:
    """Main configuration class that combines all configs."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.inference = InferenceConfig()
        self.storage = StorageConfig()
        self.dataset = DatasetConfig()
        self.pi_optimization = PiOptimizationConfig()
        
        if config_path and os.path.exists(config_path):
            self.load_from_file(config_path)
    
    def load_from_file(self, config_path: str):
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Update each config section
        for section_name, section_data in config_data.items():
            if hasattr(self, section_name):
                section = getattr(self, section_name)
                for key, value in section_data.items():
                    if hasattr(section, key):
                        setattr(section, key, value)
    
    def save_to_file(self, config_path: str):
        """Save current configuration to YAML file."""
        config_data = {}
        
        for section_name in ['model', 'training', 'inference', 'storage', 'dataset', 'pi_optimization']:
            section = getattr(self, section_name)
            config_data[section_name] = {
                key: getattr(section, key) 
                for key in section.__dataclass_fields__.keys()
            }
        
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, indent=2)
    
    def get_model_name(self) -> str:
        """Generate model name based on configuration."""
        return f"yolov8{self.model.model_size}-pi-{self.model.input_size}"
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of warnings/errors."""
        warnings = []
        
        # Check input size compatibility
        if self.model.input_size % 32 != 0:
            warnings.append("Input size should be divisible by 32 for optimal performance")
        
        # Check memory constraints
        if self.model.batch_size > 4:
            warnings.append("Large batch size may cause memory issues on Raspberry Pi")
        
        # Check thermal management
        if not self.pi_optimization.enable_thermal_throttling:
            warnings.append("Thermal throttling disabled - monitor CPU temperature")
        
        return warnings


# Default configuration instance
default_config = Config() 