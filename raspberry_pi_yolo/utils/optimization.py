"""
Optimization utilities for Raspberry Pi YOLO deployment.
Enhanced with memory management, thermal monitoring, and performance optimization.
"""

import os
import gc
import time
import psutil
import subprocess
import threading
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
import numpy as np
import torch
from loguru import logger


@dataclass
class SystemMetrics:
    """System performance metrics for monitoring."""
    cpu_percent: float
    memory_percent: float
    memory_available: int
    temperature: Optional[float]
    gpu_memory_used: Optional[int]
    inference_time: float
    fps: float


class ThermalManager:
    """Thermal management for Raspberry Pi to prevent throttling."""
    
    def __init__(self, max_temp: float = 70.0, check_interval: int = 30):
        self.max_temp = max_temp
        self.check_interval = check_interval
        self.current_temp = 0.0
        self.is_monitoring = False
        self.monitor_thread = None
        
    def get_cpu_temperature(self) -> Optional[float]:
        """Get CPU temperature from system."""
        try:
            # Try different temperature sources
            temp_sources = [
                "/sys/class/thermal/thermal_zone0/temp",
                "/sys/class/hwmon/hwmon0/temp1_input",
                "/proc/acpi/thermal_zone/THM0/temperature"
            ]
            
            for source in temp_sources:
                if os.path.exists(source):
                    with open(source, 'r') as f:
                        temp_raw = f.read().strip()
                        # Convert to Celsius (most sources provide millidegrees)
                        if len(temp_raw) > 3:
                            temp = float(temp_raw) / 1000.0
                        else:
                            temp = float(temp_raw)
                        return temp
            
            # Fallback: try vcgencmd
            result = subprocess.run(['vcgencmd', 'measure_temp'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                temp_str = result.stdout.strip()
                temp = float(temp_str.replace('temp=', '').replace("'C", ''))
                return temp
                
        except Exception as e:
            logger.warning(f"Could not read temperature: {e}")
        
        return None
    
    def start_monitoring(self):
        """Start thermal monitoring in background thread."""
        if self.is_monitoring:
            return
            
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Thermal monitoring started")
    
    def stop_monitoring(self):
        """Stop thermal monitoring."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logger.info("Thermal monitoring stopped")
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.is_monitoring:
            self.current_temp = self.get_cpu_temperature() or 0.0
            
            if self.current_temp > self.max_temp:
                logger.warning(f"High temperature detected: {self.current_temp:.1f}°C")
                self._apply_thermal_throttling()
            
            time.sleep(self.check_interval)
    
    def _apply_thermal_throttling(self):
        """Apply thermal throttling measures."""
        # Note: Disabled CPU frequency reduction (requires sudo)
        logger.info("High temperature detected: monitoring only (thermal throttling requires sudo)")


class MemoryManager:
    """Memory management for Raspberry Pi optimization."""
    
    def __init__(self, max_memory_usage: float = 0.8):
        self.max_memory_usage = max_memory_usage
        self.memory_threshold = self._get_memory_threshold()
        
    def _get_memory_threshold(self) -> int:
        """Get memory threshold in bytes."""
        total_memory = psutil.virtual_memory().total
        return int(total_memory * self.max_memory_usage)
    
    def check_memory_usage(self) -> Dict[str, Any]:
        """Check current memory usage."""
        memory = psutil.virtual_memory()
        return {
            'total': memory.total,
            'available': memory.available,
            'used': memory.used,
            'percent': memory.percent,
            'threshold': self.memory_threshold
        }
    
    def is_memory_critical(self) -> bool:
        """Check if memory usage is critical."""
        memory_info = self.check_memory_usage()
        return memory_info['used'] > self.memory_threshold
    
    def optimize_memory(self):
        """Apply memory optimization techniques."""
        # Force garbage collection
        gc.collect()
        
        # Clear PyTorch cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Note: Disabled system cache clearing (requires sudo)
        if self.is_memory_critical():
            logger.warning("High memory usage detected: system cache clearing disabled (requires sudo)")


class PerformanceOptimizer:
    """Performance optimization for Raspberry Pi inference."""
    
    def __init__(self, config: Any):
        self.config = config
        self.thermal_manager = ThermalManager(
            max_temp=config.pi_optimization.max_cpu_temp,
            check_interval=config.pi_optimization.thermal_check_interval
        )
        self.memory_manager = MemoryManager(
            max_memory_usage=config.inference.max_memory_usage
        )
        self.inference_times = []
        self.fps_history = []
        
    def optimize_for_inference(self):
        """Apply inference optimizations."""
        # Note: Disabled system-level optimizations to avoid sudo requirements
        logger.info("Skipping system-level optimizations (requires sudo)")
        
        # Start thermal monitoring (doesn't require sudo)
        if self.config.pi_optimization.enable_thermal_throttling:
            self.thermal_manager.start_monitoring()
    
    def _disable_unnecessary_services(self):
        """Disable unnecessary system services for better performance."""
        # Note: Disabled service management (requires sudo)
        logger.info("Skipping service management (requires sudo)")
    
    def measure_inference_time(self, inference_func: Callable, *args, **kwargs) -> float:
        """Measure inference time with optimization."""
        start_time = time.time()
        
        # Check memory before inference
        if self.memory_manager.is_memory_critical():
            self.memory_manager.optimize_memory()
        
        # Run inference
        result = inference_func(*args, **kwargs)
        
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        # Keep only recent measurements
        if len(self.inference_times) > 100:
            self.inference_times = self.inference_times[-100:]
        
        return inference_time
    
    def get_performance_metrics(self) -> SystemMetrics:
        """Get comprehensive performance metrics."""
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_info = self.memory_manager.check_memory_usage()
        temperature = self.thermal_manager.current_temp
        
        # GPU metrics (if available)
        gpu_memory_used = None
        if torch.cuda.is_available():
            gpu_memory_used = torch.cuda.memory_allocated()
        
        # Inference metrics
        avg_inference_time = np.mean(self.inference_times) if self.inference_times else 0
        fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
        
        return SystemMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory_info['percent'],
            memory_available=memory_info['available'],
            temperature=temperature,
            gpu_memory_used=gpu_memory_used,
            inference_time=avg_inference_time,
            fps=fps
        )
    
    def cleanup(self):
        """Cleanup optimization resources."""
        self.thermal_manager.stop_monitoring()
        
        # Note: Skipping CPU governor restoration (requires sudo)
        logger.info("Skipping system-level cleanup (requires sudo)")


class ModelOptimizer:
    """Model optimization utilities for Raspberry Pi."""
    
    @staticmethod
    def quantize_model(model, quantization_type: str = "int8"):
        """Quantize model for better performance on Pi."""
        if quantization_type == "int8":
            # Dynamic quantization for PyTorch models
            if hasattr(model, 'model'):
                model.model = torch.quantization.quantize_dynamic(
                    model.model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
                )
                logger.info("Applied dynamic quantization to model")
        
        return model
    
    @staticmethod
    def export_to_onnx(model, output_path: str, input_shape: tuple = (1, 3, 320, 320)):
        """Export model to ONNX format for optimized inference."""
        try:
            # Create dummy input
            dummy_input = torch.randn(input_shape)
            
            # Export to ONNX
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            logger.info(f"Model exported to ONNX: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export model to ONNX: {e}")
            return False
    
    @staticmethod
    def optimize_onnx_model(onnx_path: str, optimized_path: str):
        """Optimize ONNX model for better performance."""
        try:
            import onnx
            from onnxruntime.quantization import quantize_dynamic
            
            # Load and optimize ONNX model
            quantize_dynamic(
                model_input=onnx_path,
                model_output=optimized_path,
                weight_type=onnx.quantization.QuantType.QUInt8
            )
            logger.info(f"ONNX model optimized: {optimized_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to optimize ONNX model: {e}")
            return False


class InferenceOptimizer:
    """Inference-specific optimizations."""
    
    def __init__(self, config: Any):
        self.config = config
        self.frame_skip_counter = 0
        
    def should_process_frame(self) -> bool:
        """Determine if current frame should be processed based on skip_frames setting."""
        if self.config.inference.skip_frames <= 1:
            return True
        
        self.frame_skip_counter += 1
        if self.frame_skip_counter >= self.config.inference.skip_frames:
            self.frame_skip_counter = 0
            return True
        
        return False
    
    def optimize_batch_size(self, available_memory: int) -> int:
        """Dynamically adjust batch size based on available memory."""
        base_batch_size = self.config.inference.inference_batch_size
        
        # Estimate memory per sample (rough approximation)
        memory_per_sample = 100 * 1024 * 1024  # 100MB per sample
        
        max_batch_size = max(1, available_memory // memory_per_sample)
        optimal_batch_size = min(base_batch_size, max_batch_size)
        
        return optimal_batch_size 