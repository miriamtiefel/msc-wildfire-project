"""
Convert Pyronear 2025 dataset to standard YOLO format for Raspberry Pi training.
This script handles the nested folder structure and converts it to the expected format.
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import yaml
from loguru import logger


class PyronearDatasetConverter:
    """Convert Pyronear 2025 dataset to standard YOLO format."""
    
    def __init__(self, source_dir: str, output_dir: str):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        
        # Create output directory structure
        self._create_directory_structure()
        
        # Class mapping for Pyronear dataset
        # Based on the label format, it appears class 1 is fire/smoke
        self.class_mapping = {
            0: "background",  # If any background class exists
            1: "fire"         # Main fire/smoke class
        }
        
        logger.info(f"Initialized converter: {source_dir} -> {output_dir}")
    
    def _create_directory_structure(self):
        """Create standard YOLO directory structure."""
        splits = ['train', 'val', 'test']
        
        for split in splits:
            # Create images directory
            (self.output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
            # Create labels directory
            (self.output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Created directory structure in {self.output_dir}")
    
    def _find_all_images_and_labels(self, split_dir: Path) -> List[Tuple[Path, Path]]:
        """Find all image-label pairs in a split directory."""
        pairs = []
        
        # Supported image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # Walk through all subdirectories
        for location_dir in split_dir.iterdir():
            if not location_dir.is_dir():
                continue
            
            logger.info(f"Processing location: {location_dir.name}")
            
            # Find images in the location directory
            for image_file in location_dir.glob('*'):
                if image_file.suffix.lower() in image_extensions:
                    # Look for corresponding label file
                    label_file = location_dir / 'labels' / f"{image_file.stem}.txt"
                    
                    if label_file.exists():
                        pairs.append((image_file, label_file))
                    else:
                        logger.warning(f"No label file found for {image_file}")
        
        logger.info(f"Found {len(pairs)} image-label pairs in {split_dir.name}")
        return pairs
    
    def _validate_label_file(self, label_path: Path) -> bool:
        """Validate YOLO format label file."""
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            for line_num, line in enumerate(lines, 1):
                parts = line.strip().split()
                
                # Check format: class_id x_center y_center width height
                if len(parts) != 5:
                    logger.warning(f"Invalid format in {label_path}:{line_num}: {line.strip()}")
                    return False
                
                # Check if all values are valid floats
                try:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                except ValueError:
                    logger.warning(f"Invalid values in {label_path}:{line_num}: {line.strip()}")
                    return False
                
                # Check if coordinates are normalized (0-1)
                if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1):
                    logger.warning(f"Coordinates not normalized in {label_path}:{line_num}: {line.strip()}")
                    return False
                
                # Check if class_id is valid
                if class_id not in self.class_mapping:
                    logger.warning(f"Unknown class_id {class_id} in {label_path}:{line_num}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error reading {label_path}: {e}")
            return False
    
    def _get_class_statistics(self, pairs: List[Tuple[Path, Path]]) -> Dict[int, int]:
        """Get statistics about class distribution."""
        class_counts = {}
        
        for image_path, label_path in pairs:
            try:
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 1:
                            class_id = int(parts[0])
                            class_counts[class_id] = class_counts.get(class_id, 0) + 1
            except Exception as e:
                logger.warning(f"Error reading {label_path}: {e}")
        
        return class_counts
    
    def convert_dataset(self, copy_files: bool = True) -> Dict[str, int]:
        """Convert the entire dataset."""
        splits = ['train', 'val', 'test']
        total_stats = {}
        
        for split in splits:
            split_dir = self.source_dir / split
            
            if not split_dir.exists():
                logger.warning(f"Split directory not found: {split_dir}")
                continue
            
            logger.info(f"Processing {split} split...")
            
            # Find all image-label pairs
            pairs = self._find_all_images_and_labels(split_dir)
            
            if not pairs:
                logger.warning(f"No pairs found in {split} split")
                continue
            
            # Validate pairs
            valid_pairs = []
            for image_path, label_path in pairs:
                if self._validate_label_file(label_path):
                    valid_pairs.append((image_path, label_path))
                else:
                    logger.warning(f"Skipping invalid pair: {image_path}")
            
            logger.info(f"Valid pairs in {split}: {len(valid_pairs)}/{len(pairs)}")
            
            # Get class statistics
            class_stats = self._get_class_statistics(valid_pairs)
            logger.info(f"Class distribution in {split}: {class_stats}")
            
            if copy_files:
                # Copy files to output directory
                for image_path, label_path in valid_pairs:
                    # Copy image
                    dest_image = self.output_dir / 'images' / split / image_path.name
                    shutil.copy2(image_path, dest_image)
                    
                    # Copy label
                    dest_label = self.output_dir / 'labels' / split / label_path.name
                    shutil.copy2(label_path, dest_label)
                
                logger.info(f"Copied {len(valid_pairs)} pairs to {split} split")
            
            total_stats[split] = len(valid_pairs)
        
        # Create dataset configuration
        if copy_files:
            self._create_dataset_config()
            self._create_training_config()
        
        return total_stats
    
    def _create_dataset_config(self):
        """Create YAML dataset configuration file."""
        config = {
            'path': str(self.output_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': len(self.class_mapping),
            'names': [self.class_mapping[i] for i in range(len(self.class_mapping))]
        }
        
        config_path = self.output_dir / 'dataset.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        logger.info(f"Created dataset config: {config_path}")
    
    def _create_training_config(self):
        """Create training configuration optimized for Raspberry Pi 4 B 8GB."""
        config = {
            'model': {
                'model_size': 'nano',
                'input_size': 416,  # Increased for Pi 4 B
                'batch_size': 2,    # Increased for 8GB RAM
                'confidence_threshold': 0.25,
                'use_onnx': True,
                'use_quantization': True
            },
            'training': {
                'epochs': 100,
                'learning_rate': 0.01,
                'batch_size': 2,    # Increased for 8GB RAM
                'num_workers': 2,   # Increased for Pi 4 B
                'augmentations': {
                    'hsv_h': 0.015,
                    'hsv_s': 0.7,
                    'hsv_v': 0.4,
                    'degrees': 5.0,  # Slight rotation for better generalization
                    'translate': 0.1,
                    'scale': 0.5,
                    'shear': 0.0,
                    'perspective': 0.0,
                    'flipud': 0.0,
                    'fliplr': 0.5,
                    'mosaic': 0.7,  # Increased for Pi 4 B
                    'mixup': 0.1,   # Enabled for Pi 4 B
                    'copy_paste': 0.0,
                    'auto_augment': None,
                    'erasing': 0.0
                }
            },
            'dataset': {
                'train_path': str(self.output_dir / 'images' / 'train'),
                'val_path': str(self.output_dir / 'images' / 'val'),
                'test_path': str(self.output_dir / 'images' / 'test'),
                'classes': [self.class_mapping[i] for i in range(len(self.class_mapping))],
                'num_classes': len(self.class_mapping)
            },
            'inference': {
                'target_fps': 8.0,      # Increased for Pi 4 B
                'skip_frames': 1,       # Process every frame
                'max_memory_usage': 0.85, # Increased for 8GB RAM
                'enable_streaming': True
            },
            'pi_optimization': {
                'enable_thermal_throttling': True,
                'max_cpu_temp': 75.0,   # Increased for Pi 4 B
                'use_cpu_only': True
            }
        }
        
        config_path = self.output_dir / 'pi_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        logger.info(f"Created Pi-optimized config: {config_path}")


def main():
    """Main function for dataset conversion."""
    parser = argparse.ArgumentParser(description='Convert Pyronear 2025 dataset to YOLO format')
    parser.add_argument('--source_dir', type=str, required=True, 
                       help='Path to Pyronear dataset (containing train/val/test folders)')
    parser.add_argument('--output_dir', type=str, required=True, 
                       help='Output directory for converted dataset')
    parser.add_argument('--validate_only', action='store_true', 
                       help='Only validate dataset, do not copy files')
    parser.add_argument('--create_configs', action='store_true', 
                       help='Create configuration files')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.source_dir):
        logger.error(f"Source directory does not exist: {args.source_dir}")
        sys.exit(1)
    
    try:
        # Create converter
        converter = PyronearDatasetConverter(args.source_dir, args.output_dir)
        
        # Convert dataset
        stats = converter.convert_dataset(copy_files=not args.validate_only)
        
        # Print results
        print(f"\nDataset conversion completed:")
        for split, count in stats.items():
            print(f"  {split.capitalize()}: {count} images")
        
        total_images = sum(stats.values())
        print(f"  Total: {total_images} images")
        
        if args.create_configs and not args.validate_only:
            print(f"\nConfiguration files created:")
            print(f"  Dataset config: {args.output_dir}/dataset.yaml")
            print(f"  Pi config: {args.output_dir}/pi_config.yaml")
        
        print(f"\nDataset ready for training!")
        print(f"Use: python train.py --config {args.output_dir}/pi_config.yaml")
        
    except Exception as e:
        logger.error(f"Dataset conversion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 