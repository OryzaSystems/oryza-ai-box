#!/usr/bin/env python3
"""
Camera Test Suite for AI Box Real-World Testing

This tool provides comprehensive camera testing for:
- Webcam integration testing
- IP camera connection testing
- Video stream processing
- Frame capture và analysis
"""

import cv2
import time
import threading
import numpy as np
from typing import Optional, Dict, List, Tuple, Any
import logging
from dataclasses import dataclass
import json
import os
from datetime import datetime

@dataclass
class CameraConfig:
    """Camera configuration."""
    name: str
    source: str  # URL, device index, or file path
    camera_type: str  # 'webcam', 'ip_camera', 'file'
    resolution: Tuple[int, int] = (640, 480)
    fps: int = 30
    timeout: int = 10
    retry_count: int = 3

@dataclass
class CameraTestResult:
    """Camera test result."""
    camera_name: str
    success: bool
    fps_actual: float
    resolution_actual: Tuple[int, int]
    frame_count: int
    error_message: str = ""
    test_duration: float = 0.0
    connection_time: float = 0.0

class CameraTestSuite:
    """Comprehensive camera testing suite."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Test results
        self.test_results: List[CameraTestResult] = []
        
        # Camera configurations
        self.camera_configs = self._get_default_configs()
        
    def _get_default_configs(self) -> List[CameraConfig]:
        """Get default camera configurations."""
        configs = [
            # Webcam configurations
            CameraConfig("Webcam 0", "0", "webcam", (640, 480), 30),
            CameraConfig("Webcam 1", "1", "webcam", (1280, 720), 30),
            
            # IP Camera configurations (examples)
            CameraConfig("IP Camera 1", "http://admin:password@192.168.1.100:8080/video", "ip_camera", (640, 480), 30),
            CameraConfig("IP Camera 2", "rtsp://admin:password@192.168.1.101:554/stream", "ip_camera", (1280, 720), 30),
            
            # File-based testing
            CameraConfig("Test Video", "test_data/sample_video.mp4", "file", (640, 480), 30),
        ]
        return configs
        
    def add_camera_config(self, config: CameraConfig):
        """Add custom camera configuration."""
        self.camera_configs.append(config)
        
    def test_webcam(self, device_index: int = 0, duration: int = 10) -> CameraTestResult:
        """Test webcam functionality."""
        config = CameraConfig(f"Webcam {device_index}", str(device_index), "webcam")
        return self._test_camera(config, duration)
        
    def test_ip_camera(self, url: str, duration: int = 10) -> CameraTestResult:
        """Test IP camera functionality."""
        config = CameraConfig("IP Camera", url, "ip_camera")
        return self._test_camera(config, duration)
        
    def test_video_file(self, file_path: str, duration: int = 10) -> CameraTestResult:
        """Test video file processing."""
        config = CameraConfig("Video File", file_path, "file")
        return self._test_camera(config, duration)
        
    def _test_camera(self, config: CameraConfig, duration: int = 10) -> CameraTestResult:
        """Test camera with given configuration."""
        self.logger.info(f"🧪 Testing camera: {config.name}")
        
        start_time = time.time()
        cap = None
        frame_count = 0
        fps_start_time = time.time()
        
        try:
            # Try to open camera
            connection_start = time.time()
            
            if config.camera_type == "webcam":
                cap = cv2.VideoCapture(int(config.source))
            elif config.camera_type == "ip_camera":
                cap = cv2.VideoCapture(config.source)
            elif config.camera_type == "file":
                cap = cv2.VideoCapture(config.source)
            else:
                raise ValueError(f"Unknown camera type: {config.camera_type}")
                
            # Check if camera opened successfully
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open camera: {config.source}")
                
            connection_time = time.time() - connection_start
            self.logger.info(f"✅ Camera connected in {connection_time:.2f}s")
            
            # Set camera properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.resolution[1])
            cap.set(cv2.CAP_PROP_FPS, config.fps)
            
            # Get actual properties
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            
            self.logger.info(f"📐 Resolution: {actual_width}x{actual_height}")
            self.logger.info(f"🎬 FPS: {actual_fps}")
            
            # Capture frames for specified duration
            fps_start_time = time.time()
            frame_count = 0
            
            while time.time() - start_time < duration:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_count += 1
                
                # Display frame (optional)
                cv2.imshow(f"Camera Test - {config.name}", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            # Calculate actual FPS
            test_duration = time.time() - start_time
            fps_actual = frame_count / test_duration if test_duration > 0 else 0
            
            self.logger.info(f"✅ Test completed: {frame_count} frames in {test_duration:.1f}s")
            self.logger.info(f"📊 Actual FPS: {fps_actual:.1f}")
            
            return CameraTestResult(
                camera_name=config.name,
                success=True,
                fps_actual=fps_actual,
                resolution_actual=(actual_width, actual_height),
                frame_count=frame_count,
                test_duration=test_duration,
                connection_time=connection_time
            )
            
        except Exception as e:
            error_msg = f"Camera test failed: {str(e)}"
            self.logger.error(error_msg)
            
            return CameraTestResult(
                camera_name=config.name,
                success=False,
                fps_actual=0.0,
                resolution_actual=(0, 0),
                frame_count=0,
                error_message=error_msg,
                test_duration=time.time() - start_time
            )
            
        finally:
            if cap:
                cap.release()
            cv2.destroyAllWindows()
            
    def test_all_cameras(self, duration: int = 5) -> List[CameraTestResult]:
        """Test all configured cameras."""
        self.logger.info(f"🧪 Testing {len(self.camera_configs)} cameras...")
        
        results = []
        for config in self.camera_configs:
            result = self._test_camera(config, duration)
            results.append(result)
            self.test_results.append(result)
            
            # Brief pause between tests
            time.sleep(1)
            
        return results
        
    def generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        if not self.test_results:
            return {"error": "No test results available"}
            
        successful_tests = [r for r in self.test_results if r.success]
        failed_tests = [r for r in self.test_results if not r.success]
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_cameras": len(self.test_results),
                "successful": len(successful_tests),
                "failed": len(failed_tests),
                "success_rate": len(successful_tests) / len(self.test_results) * 100
            },
            "performance": {
                "avg_fps": np.mean([r.fps_actual for r in successful_tests]) if successful_tests else 0,
                "max_fps": np.max([r.fps_actual for r in successful_tests]) if successful_tests else 0,
                "avg_connection_time": np.mean([r.connection_time for r in successful_tests]) if successful_tests else 0,
                "avg_frame_count": np.mean([r.frame_count for r in successful_tests]) if successful_tests else 0
            },
            "successful_cameras": [
                {
                    "name": r.camera_name,
                    "fps": r.fps_actual,
                    "resolution": r.resolution_actual,
                    "frame_count": r.frame_count,
                    "connection_time": r.connection_time
                }
                for r in successful_tests
            ],
            "failed_cameras": [
                {
                    "name": r.camera_name,
                    "error": r.error_message
                }
                for r in failed_tests
            ]
        }
        
        return report
        
    def save_test_report(self, filename: str = "camera_test_report.json"):
        """Save test report to file."""
        report = self.generate_test_report()
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
            
        self.logger.info(f"📄 Test report saved to {filename}")
        
    def print_test_report(self):
        """Print test report to console."""
        report = self.generate_test_report()
        
        if "error" in report:
            print("❌ No test results available")
            return
            
        print("\n" + "="*60)
        print("📹 CAMERA TEST REPORT")
        print("="*60)
        
        # Summary
        summary = report["summary"]
        print(f"📊 SUMMARY:")
        print(f"   Total Cameras: {summary['total_cameras']}")
        print(f"   Successful: {summary['successful']}")
        print(f"   Failed: {summary['failed']}")
        print(f"   Success Rate: {summary['success_rate']:.1f}%")
        
        # Performance
        perf = report["performance"]
        print(f"\n🚀 PERFORMANCE:")
        print(f"   Average FPS: {perf['avg_fps']:.1f}")
        print(f"   Max FPS: {perf['max_fps']:.1f}")
        print(f"   Avg Connection Time: {perf['avg_connection_time']:.2f}s")
        print(f"   Avg Frame Count: {perf['avg_frame_count']:.0f}")
        
        # Successful cameras
        if report["successful_cameras"]:
            print(f"\n✅ SUCCESSFUL CAMERAS:")
            for cam in report["successful_cameras"]:
                print(f"   📹 {cam['name']}: {cam['fps']:.1f} FPS, {cam['resolution'][0]}x{cam['resolution'][1]}")
                
        # Failed cameras
        if report["failed_cameras"]:
            print(f"\n❌ FAILED CAMERAS:")
            for cam in report["failed_cameras"]:
                print(f"   📹 {cam['name']}: {cam['error']}")
                
        print("="*60)

class RealTimeCameraProcessor:
    """Real-time camera processing for AI model testing."""
    
    def __init__(self, camera_source: str, model_processor=None):
        self.camera_source = camera_source
        self.model_processor = model_processor
        self.is_running = False
        self.cap = None
        self.logger = logging.getLogger(__name__)
        
    def start_processing(self, duration: int = 30):
        """Start real-time processing."""
        self.logger.info(f"🚀 Starting real-time processing for {duration}s")
        
        try:
            self.cap = cv2.VideoCapture(self.camera_source)
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open camera: {self.camera_source}")
                
            self.is_running = True
            start_time = time.time()
            frame_count = 0
            
            while self.is_running and (time.time() - start_time) < duration:
                ret, frame = self.cap.read()
                if not ret:
                    break
                    
                frame_count += 1
                
                # Process frame with AI model if available
                if self.model_processor:
                    processed_frame = self.model_processor(frame)
                else:
                    processed_frame = frame
                    
                # Display frame
                cv2.imshow("Real-time Processing", processed_frame)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            # Calculate FPS
            duration_actual = time.time() - start_time
            fps = frame_count / duration_actual if duration_actual > 0 else 0
            
            self.logger.info(f"✅ Processing completed: {frame_count} frames, {fps:.1f} FPS")
            
        except Exception as e:
            self.logger.error(f"Processing error: {e}")
            
        finally:
            self.stop_processing()
            
    def stop_processing(self):
        """Stop real-time processing."""
        self.is_running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.logger.info("🛑 Processing stopped")

def demo_camera_test_suite():
    """Demo camera testing functionality."""
    print("🧪 Camera Test Suite Demo")
    print("="*50)
    
    # Create test suite
    suite = CameraTestSuite()
    
    # Test webcam
    print("\n📹 Testing webcam...")
    result = suite.test_webcam(device_index=0, duration=5)
    
    if result.success:
        print(f"✅ Webcam test successful: {result.fps_actual:.1f} FPS")
    else:
        print(f"❌ Webcam test failed: {result.error_message}")
        
    # Generate report
    suite.print_test_report()
    
    print("\n✅ Camera testing demo completed!")

def demo_real_time_processing():
    """Demo real-time processing."""
    print("🧪 Real-time Processing Demo")
    print("="*50)
    
    # Create processor
    processor = RealTimeCameraProcessor("0")  # Use webcam 0
    
    # Start processing
    processor.start_processing(duration=10)
    
    print("✅ Real-time processing demo completed!")

if __name__ == "__main__":
    # Run demos
    demo_camera_test_suite()
    # demo_real_time_processing()  # Uncomment to test real-time processing
