#!/usr/bin/env python3
"""
Raspberry Pi 4 B 8GB Optimized Inference Script
Designed for real-time wildfire detection on Pi 4 B 8GB
"""

import os
import sys
import time
import cv2
import numpy as np
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
import threading
from queue import Queue, Empty
import json
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from utils.optimization import PerformanceOptimizer, InferenceOptimizer, SystemMetrics
from loguru import logger


@dataclass
class Detection:
    """Detection result with enhanced metadata."""
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str
    timestamp: float
    frame_id: int


@dataclass
class InferenceResult:
    """Complete inference result with metadata."""
    detections: List[Detection]
    processing_time: float
    frame_shape: Tuple[int, int, int]
    timestamp: float
    frame_id: int


class PiYOLOInference:
    """Enhanced YOLO inference optimized for Raspberry Pi."""
    
    def __init__(self, config: Config, model_path: str):
        self.config = config
        self.model_path = model_path
        self.performance_optimizer = PerformanceOptimizer(config)
        self.inference_optimizer = InferenceOptimizer(config)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize model
        self.model = self._load_model()
        
        # Performance tracking
        self.frame_count = 0
        self.total_processing_time = 0.0
        self.detection_history = []
        
        # Streaming setup
        self.frame_queue = Queue(maxsize=config.inference.max_queue_size)
        self.result_queue = Queue(maxsize=config.inference.max_queue_size)
        self.is_streaming = False
        self.stream_thread = None
        
        # Start performance optimization
        self.performance_optimizer.optimize_for_inference()
    
    def _setup_logging(self):
        """Setup enhanced logging for inference."""
        log_dir = Path(self.config.storage.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure loguru
        logger.remove()  # Remove default handler
        logger.add(
            log_dir / "inference.log",
            rotation="10 MB",
            retention="7 days",
            level=self.config.storage.log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
        )
        logger.add(sys.stderr, level="INFO")
    
    def _load_model(self):
        """Load and optimize model for inference."""
        logger.info(f"Loading model: {self.model_path}")
        
        from ultralytics import YOLO
        model = YOLO(self.model_path)
        
        # Apply Pi optimizations
        if hasattr(self.config.model, 'use_quantization') and self.config.model.use_quantization:
            from utils.optimization import ModelOptimizer
            model = ModelOptimizer.quantize_model(model)
        
        return model
    
    def predict_single(self, image: np.ndarray) -> InferenceResult:
        """Run inference on a single image."""
        start_time = time.time()
        
        # Run inference
        results = self.model.predict(
            source=image,
            conf=self.config.model.confidence_threshold,
            iou=self.config.model.iou_threshold,
            max_det=self.config.model.max_detections,
            verbose=False
        )
        
        processing_time = time.time() - start_time
        
        # Process results
        detections = []
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy()
                
                for box, conf, class_id in zip(boxes, confidences, class_ids):
                    detection = Detection(
                        bbox=box.tolist(),
                        confidence=float(conf),
                        class_id=int(class_id),
                        class_name=self.config.dataset.classes[int(class_id)] if int(class_id) < len(self.config.dataset.classes) else f"class_{int(class_id)}",
                        timestamp=time.time(),
                        frame_id=self.frame_count
                    )
                    detections.append(detection)
        
        # Update tracking
        self.frame_count += 1
        self.total_processing_time += processing_time
        self.detection_history.extend(detections)
        
        return InferenceResult(
            detections=detections,
            processing_time=processing_time,
            frame_shape=image.shape,
            timestamp=time.time(),
            frame_id=self.frame_count
        )
    
    def start_streaming(self, source: str, save_output: bool = False, output_dir: Optional[str] = None):
        """Start real-time streaming inference."""
        self.is_streaming = True
        self.stream_thread = threading.Thread(target=self._streaming_loop, args=(source, save_output, output_dir))
        self.stream_thread.start()
    
    def stop_streaming(self):
        """Stop streaming inference."""
        self.is_streaming = False
        if self.stream_thread:
            self.stream_thread.join()
    
    def _streaming_loop(self, source: str, save_output: bool, output_dir: Optional[str] = None):
        """Main streaming loop."""
        cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
        
        if not cap.isOpened():
            logger.error(f"Failed to open source: {source}")
            return
        
        # Setup video writer if saving
        writer = None
        if save_output and output_dir:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_path = os.path.join(output_dir, f"inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
            writer = cv2.VideoWriter(output_path, fourcc, 20.0, (640, 480))
        
        try:
            while self.is_streaming:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Run inference
                result = self.predict_single(frame)
                
                # Draw detections
                annotated_frame = self._draw_detections(frame, result.detections)
                
                # Add performance info
                fps = 1.0 / result.processing_time if result.processing_time > 0 else 0
                cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Detections: {len(result.detections)}", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Display frame
                cv2.imshow('Wildfire Detection', annotated_frame)
                
                # Save frame if requested
                if writer:
                    writer.write(annotated_frame)
                
                # Check for key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s') and save_output and output_dir:
                    # Save current frame
                    frame_path = os.path.join(output_dir, f"frame_{self.frame_count}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                    cv2.imwrite(frame_path, annotated_frame)
                    print(f"Saved frame: {frame_path}")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
    
    def _draw_detections(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Draw detection boxes on frame."""
        annotated_frame = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection.bbox)
            conf = detection.confidence
            class_name = detection.class_name
            
            # Choose color based on class
            color = (0, 0, 255) if class_name == 'fire' else (255, 0, 0)  # Red for fire, Blue for smoke
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(annotated_frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return annotated_frame
    
    def get_performance_metrics(self) -> SystemMetrics:
        """Get current performance metrics."""
        return self.performance_optimizer.get_system_metrics()
    
    def get_detection_stats(self) -> Dict[str, Any]:
        """Get detection statistics."""
        if not self.detection_history:
            return {}
        
        class_counts = {}
        for detection in self.detection_history:
            class_name = detection.class_name
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        return {
            'total_detections': len(self.detection_history),
            'class_counts': class_counts,
            'avg_confidence': sum(d.confidence for d in self.detection_history) / len(self.detection_history),
            'frames_processed': self.frame_count,
            'avg_processing_time': self.total_processing_time / self.frame_count if self.frame_count > 0 else 0
        }
    
    def cleanup(self):
        """Cleanup resources."""
        self.stop_streaming()
        self.performance_optimizer.cleanup()


def main():
    """Main inference function optimized for Pi 4 B 8GB."""
    parser = argparse.ArgumentParser(description='Pi 4 B 8GB Optimized Wildfire Detection')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--source', type=str, default='0', help='Camera source (0 for webcam) or video file')
    parser.add_argument('--stream', action='store_true', help='Enable real-time streaming')
    parser.add_argument('--save_output', action='store_true', help='Save output video/images')
    parser.add_argument('--output_dir', type=str, default='./inference_output', help='Output directory')
    
    args = parser.parse_args()
    
    # Create default Pi 4 B 8GB config
    config = Config()
    
    print("🔥 Raspberry Pi 4 B 8GB Wildfire Detection")
    print("=" * 50)
    print(f"🤖 Model: {args.model_path}")
    print(f"📹 Source: {args.source}")
    print(f"🎯 Stream: {args.stream}")
    print("=" * 50)
    
    # Create output directory
    if args.save_output:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize inference engine
        print("🚀 Initializing Pi 4 B 8GB inference engine...")
        inference_engine = PiYOLOInference(config, args.model_path)
        
        # Pi 4 B 8GB specific optimizations
        print("🔧 Applying Pi 4 B 8GB optimizations...")
        print(f"   • Input size: {config.model.input_size}x{config.model.input_size}")
        print(f"   • Batch size: {config.inference.inference_batch_size}")
        print(f"   • Target FPS: {config.inference.target_fps}")
        print(f"   • Memory limit: {config.inference.max_memory_usage*100:.0f}%")
        
        # Start inference
        if args.stream:
            print("\n📺 Starting real-time wildfire detection...")
            print("Press 'q' to quit, 's' to save frame")
            
            inference_engine.start_streaming(
                source=args.source,
                save_output=args.save_output,
                output_dir=str(output_dir) if args.save_output else None
            )
        else:
            print("\n🖼️  Processing single image/video...")
            # For single image/video, use streaming with one frame
            inference_engine.start_streaming(
                source=args.source,
                save_output=args.save_output,
                output_dir=str(output_dir) if args.save_output else None
            )
        
        # Show final performance metrics
        print("\n📊 Final Performance Metrics:")
        print("=" * 30)
        metrics = inference_engine.get_performance_metrics()
        print(f"🎯 Average FPS: {metrics.fps:.1f}")
        print(f"⏱️  Avg Inference Time: {metrics.inference_time*1000:.1f}ms")
        print(f"💾 Memory Usage: {metrics.memory_percent:.1f}%")
        print(f"🔥 CPU Usage: {metrics.cpu_percent:.1f}%")
        if metrics.temperature:
            print(f"🌡️  Temperature: {metrics.temperature:.1f}°C")
        
        # Show detection stats
        stats = inference_engine.get_detection_stats()
        if stats:
            print(f"🎯 Total Detections: {stats.get('total_detections', 0)}")
            print(f"🎯 Class Counts: {stats.get('class_counts', {})}")
            print(f"🎯 Avg Confidence: {stats.get('avg_confidence', 0):.3f}")
        print("=" * 30)
        
    except KeyboardInterrupt:
        print("\n⏹️  Inference stopped by user")
    except Exception as e:
        print(f"\n❌ Inference failed: {e}")
        logger.error(f"Inference error: {e}")
    finally:
        # Cleanup
        if 'inference_engine' in locals():
            inference_engine.cleanup()
        print("🧹 Cleanup completed")


if __name__ == "__main__":
    main() 