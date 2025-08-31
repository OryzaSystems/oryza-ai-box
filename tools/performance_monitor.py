#!/usr/bin/env python3
"""
Performance Monitor for AI Box Real-World Testing

This tool provides comprehensive performance monitoring for:
- FPS measurement
- Memory usage tracking
- CPU/GPU utilization
- Temperature monitoring
- Power consumption (edge devices)
"""

import time
import psutil
import threading
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass, asdict
import numpy as np

# Try to import GPU monitoring
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("⚠️ GPUtil not available - GPU monitoring disabled")

# Try to import temperature monitoring
try:
    import psutil
    TEMP_AVAILABLE = True
except ImportError:
    TEMP_AVAILABLE = False
    print("⚠️ Temperature monitoring not available")

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    timestamp: float
    fps: float
    memory_usage_mb: float
    cpu_percent: float
    gpu_percent: Optional[float] = None
    gpu_memory_mb: Optional[float] = None
    temperature_c: Optional[float] = None
    power_watts: Optional[float] = None
    model_name: str = ""
    platform: str = ""
    resolution: str = ""

class PerformanceMonitor:
    """Real-time performance monitoring for AI models."""
    
    def __init__(self, output_file: str = "performance_log.json"):
        self.output_file = output_file
        self.metrics: List[PerformanceMetrics] = []
        self.is_monitoring = False
        self.monitor_thread = None
        self.fps_counter = FPSCounter()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.start_time = None
        self.frame_count = 0
        
    def start_monitoring(self, model_name: str = "", platform: str = "", resolution: str = ""):
        """Start performance monitoring."""
        if self.is_monitoring:
            self.logger.warning("Monitoring already started")
            return
            
        self.is_monitoring = True
        self.start_time = time.time()
        self.frame_count = 0
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(model_name, platform, resolution),
            daemon=True
        )
        self.monitor_thread.start()
        
        self.logger.info(f"🚀 Performance monitoring started for {model_name}")
        
    def stop_monitoring(self):
        """Stop performance monitoring."""
        if not self.is_monitoring:
            return
            
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
            
        self.logger.info("🛑 Performance monitoring stopped")
        
    def record_frame(self):
        """Record a processed frame for FPS calculation."""
        self.frame_count += 1
        self.fps_counter.record_frame()
        
    def _monitor_loop(self, model_name: str, platform: str, resolution: str):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Collect metrics
                metrics = self._collect_metrics(model_name, platform, resolution)
                self.metrics.append(metrics)
                
                # Log current status
                self._log_status(metrics)
                
                # Save to file periodically
                if len(self.metrics) % 10 == 0:
                    self._save_metrics()
                    
                time.sleep(1)  # Update every second
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(1)
                
    def _collect_metrics(self, model_name: str, platform: str, resolution: str) -> PerformanceMetrics:
        """Collect current performance metrics."""
        # Get FPS
        fps = self.fps_counter.get_fps()
        
        # Get memory usage
        memory = psutil.virtual_memory()
        memory_mb = memory.used / (1024 * 1024)
        
        # Get CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Get GPU metrics
        gpu_percent = None
        gpu_memory_mb = None
        if GPU_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Use first GPU
                    gpu_percent = gpu.load * 100
                    gpu_memory_mb = gpu.memoryUsed
            except:
                pass
                
        # Get temperature
        temperature_c = None
        if TEMP_AVAILABLE:
            try:
                # Try to get CPU temperature
                temps = psutil.sensors_temperatures()
                if temps:
                    for name, entries in temps.items():
                        if entries:
                            temperature_c = entries[0].current
                            break
            except:
                pass
                
        # Get power consumption (platform-specific)
        power_watts = self._get_power_consumption(platform)
        
        return PerformanceMetrics(
            timestamp=time.time(),
            fps=fps,
            memory_usage_mb=memory_mb,
            cpu_percent=cpu_percent,
            gpu_percent=gpu_percent,
            gpu_memory_mb=gpu_memory_mb,
            temperature_c=temperature_c,
            power_watts=power_watts,
            model_name=model_name,
            platform=platform,
            resolution=resolution
        )
        
    def _get_power_consumption(self, platform: str) -> Optional[float]:
        """Get power consumption for edge devices."""
        try:
            if platform.lower() in ['raspberry_pi', 'pi5', 'pi']:
                # Raspberry Pi power monitoring
                return self._get_pi_power()
            elif platform.lower() in ['radxa', 'rock5']:
                # Radxa Rock power monitoring
                return self._get_radxa_power()
            elif platform.lower() in ['jetson', 'nano']:
                # Jetson power monitoring
                return self._get_jetson_power()
            else:
                return None
        except:
            return None
            
    def _get_pi_power(self) -> Optional[float]:
        """Get Raspberry Pi power consumption."""
        try:
            # Read from power monitoring if available
            with open('/sys/class/hwmon/hwmon0/power1_input', 'r') as f:
                power_mw = float(f.read().strip())
                return power_mw / 1000.0  # Convert to watts
        except:
            return None
            
    def _get_radxa_power(self) -> Optional[float]:
        """Get Radxa Rock power consumption."""
        try:
            # Similar to Pi but different path
            with open('/sys/class/hwmon/hwmon1/power1_input', 'r') as f:
                power_mw = float(f.read().strip())
                return power_mw / 1000.0
        except:
            return None
            
    def _get_jetson_power(self) -> Optional[float]:
        """Get Jetson power consumption."""
        try:
            # Jetson power monitoring
            with open('/sys/devices/platform/tegra-ahci.0/regulator/regulator.0/device/power', 'r') as f:
                power_mw = float(f.read().strip())
                return power_mw / 1000.0
        except:
            return None
            
    def _log_status(self, metrics: PerformanceMetrics):
        """Log current performance status."""
        status = f"📊 {metrics.model_name} | FPS: {metrics.fps:.1f} | "
        status += f"Memory: {metrics.memory_usage_mb:.0f}MB | "
        status += f"CPU: {metrics.cpu_percent:.1f}%"
        
        if metrics.gpu_percent:
            status += f" | GPU: {metrics.gpu_percent:.1f}%"
        if metrics.temperature_c:
            status += f" | Temp: {metrics.temperature_c:.1f}°C"
        if metrics.power_watts:
            status += f" | Power: {metrics.power_watts:.1f}W"
            
        self.logger.info(status)
        
    def _save_metrics(self):
        """Save metrics to JSON file."""
        try:
            data = {
                'timestamp': datetime.now().isoformat(),
                'metrics': [asdict(m) for m in self.metrics]
            }
            
            with open(self.output_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving metrics: {e}")
            
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.metrics:
            return {}
            
        fps_values = [m.fps for m in self.metrics if m.fps > 0]
        memory_values = [m.memory_usage_mb for m in self.metrics]
        cpu_values = [m.cpu_percent for m in self.metrics]
        
        summary = {
            'total_frames': self.frame_count,
            'monitoring_duration': time.time() - self.start_time if self.start_time else 0,
            'avg_fps': np.mean(fps_values) if fps_values else 0,
            'max_fps': np.max(fps_values) if fps_values else 0,
            'min_fps': np.min(fps_values) if fps_values else 0,
            'avg_memory_mb': np.mean(memory_values),
            'max_memory_mb': np.max(memory_values),
            'avg_cpu_percent': np.mean(cpu_values),
            'max_cpu_percent': np.max(cpu_values),
        }
        
        # Add GPU metrics if available
        gpu_values = [m.gpu_percent for m in self.metrics if m.gpu_percent is not None]
        if gpu_values:
            summary.update({
                'avg_gpu_percent': np.mean(gpu_values),
                'max_gpu_percent': np.max(gpu_values)
            })
            
        # Add temperature metrics if available
        temp_values = [m.temperature_c for m in self.metrics if m.temperature_c is not None]
        if temp_values:
            summary.update({
                'avg_temperature_c': np.mean(temp_values),
                'max_temperature_c': np.max(temp_values)
            })
            
        # Add power metrics if available
        power_values = [m.power_watts for m in self.metrics if m.power_watts is not None]
        if power_values:
            summary.update({
                'avg_power_watts': np.mean(power_values),
                'max_power_watts': np.max(power_values)
            })
            
        return summary
        
    def print_summary(self):
        """Print performance summary."""
        summary = self.get_summary()
        if not summary:
            print("❌ No performance data available")
            return
            
        print("\n" + "="*60)
        print("📊 PERFORMANCE SUMMARY")
        print("="*60)
        print(f"🎯 Model: {self.metrics[0].model_name if self.metrics else 'Unknown'}")
        print(f"🖥️ Platform: {self.metrics[0].platform if self.metrics else 'Unknown'}")
        print(f"📐 Resolution: {self.metrics[0].resolution if self.metrics else 'Unknown'}")
        print(f"⏱️ Duration: {summary['monitoring_duration']:.1f}s")
        print(f"📊 Total Frames: {summary['total_frames']}")
        print()
        print("🚀 PERFORMANCE METRICS:")
        print(f"   FPS: {summary['avg_fps']:.1f} avg, {summary['max_fps']:.1f} max")
        print(f"   Memory: {summary['avg_memory_mb']:.0f}MB avg, {summary['max_memory_mb']:.0f}MB max")
        print(f"   CPU: {summary['avg_cpu_percent']:.1f}% avg, {summary['max_cpu_percent']:.1f}% max")
        
        if 'avg_gpu_percent' in summary:
            print(f"   GPU: {summary['avg_gpu_percent']:.1f}% avg, {summary['max_gpu_percent']:.1f}% max")
        if 'avg_temperature_c' in summary:
            print(f"   Temperature: {summary['avg_temperature_c']:.1f}°C avg, {summary['max_temperature_c']:.1f}°C max")
        if 'avg_power_watts' in summary:
            print(f"   Power: {summary['avg_power_watts']:.1f}W avg, {summary['max_power_watts']:.1f}W max")
            
        print("="*60)

class FPSCounter:
    """Simple FPS counter."""
    
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.frame_times = []
        
    def record_frame(self):
        """Record a frame timestamp."""
        current_time = time.time()
        self.frame_times.append(current_time)
        
        # Keep only recent frames
        if len(self.frame_times) > self.window_size:
            self.frame_times.pop(0)
            
    def get_fps(self) -> float:
        """Calculate current FPS."""
        if len(self.frame_times) < 2:
            return 0.0
            
        time_span = self.frame_times[-1] - self.frame_times[0]
        if time_span <= 0:
            return 0.0
            
        return (len(self.frame_times) - 1) / time_span

def demo_performance_monitor():
    """Demo performance monitoring functionality."""
    print("🧪 Performance Monitor Demo")
    print("="*50)
    
    # Create monitor
    monitor = PerformanceMonitor("demo_performance.json")
    
    # Start monitoring
    monitor.start_monitoring("Demo Model", "myai", "1920x1080")
    
    # Simulate processing
    print("🚀 Simulating AI processing...")
    for i in range(30):
        monitor.record_frame()
        time.sleep(0.1)  # Simulate processing time
        
    # Stop monitoring
    monitor.stop_monitoring()
    
    # Print summary
    monitor.print_summary()
    
    print("\n✅ Performance monitoring demo completed!")

if __name__ == "__main__":
    demo_performance_monitor()
