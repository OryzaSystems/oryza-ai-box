#!/usr/bin/env python3
"""
Simple Camera Test - No GUI Required

This tool provides basic camera testing without GUI display:
- Webcam connection testing
- Frame capture testing
- Performance measurement
"""

import cv2
import time
import numpy as np
from typing import Optional, Dict, Any
import logging
from dataclasses import dataclass

@dataclass
class SimpleCameraResult:
    """Simple camera test result."""
    success: bool
    fps: float
    resolution: tuple
    frame_count: int
    error_message: str = ""
    test_duration: float = 0.0

class SimpleCameraTester:
    """Simple camera tester without GUI."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
    def test_webcam(self, device_index: int = 0, duration: int = 10) -> SimpleCameraResult:
        """Test webcam without GUI display."""
        self.logger.info(f"🧪 Testing webcam {device_index}...")
        
        start_time = time.time()
        cap = None
        
        try:
            # Try to open camera
            cap = cv2.VideoCapture(device_index)
            
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open camera {device_index}")
                
            self.logger.info(f"✅ Camera {device_index} opened successfully")
            
            # Get camera properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            self.logger.info(f"📐 Resolution: {width}x{height}")
            self.logger.info(f"🎬 FPS: {fps}")
            
            # Capture frames for specified duration
            frame_count = 0
            frame_start_time = time.time()
            
            while time.time() - start_time < duration:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_count += 1
                
                # Process frame (just to simulate work)
                if frame is not None:
                    # Simple processing - convert to grayscale
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
            # Calculate actual FPS
            test_duration = time.time() - start_time
            actual_fps = frame_count / test_duration if test_duration > 0 else 0
            
            self.logger.info(f"✅ Test completed: {frame_count} frames in {test_duration:.1f}s")
            self.logger.info(f"📊 Actual FPS: {actual_fps:.1f}")
            
            return SimpleCameraResult(
                success=True,
                fps=actual_fps,
                resolution=(width, height),
                frame_count=frame_count,
                test_duration=test_duration
            )
            
        except Exception as e:
            error_msg = f"Camera test failed: {str(e)}"
            self.logger.error(error_msg)
            
            return SimpleCameraResult(
                success=False,
                fps=0.0,
                resolution=(0, 0),
                frame_count=0,
                error_message=error_msg,
                test_duration=time.time() - start_time
            )
            
        finally:
            if cap:
                cap.release()
                
    def test_camera_availability(self) -> Dict[str, Any]:
        """Test which cameras are available."""
        self.logger.info("🔍 Testing camera availability...")
        
        available_cameras = []
        max_cameras = 5  # Test first 5 camera indices
        
        for i in range(max_cameras):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    # Get basic info
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    
                    available_cameras.append({
                        'index': i,
                        'resolution': (width, height),
                        'available': True
                    })
                    
                    self.logger.info(f"✅ Camera {i}: {width}x{height}")
                    cap.release()
                else:
                    self.logger.info(f"❌ Camera {i}: Not available")
                    
            except Exception as e:
                self.logger.info(f"❌ Camera {i}: Error - {e}")
                
        return {
            'total_cameras': len(available_cameras),
            'available_cameras': available_cameras
        }
        
    def test_video_file(self, file_path: str, duration: int = 10) -> SimpleCameraResult:
        """Test video file processing."""
        self.logger.info(f"🎬 Testing video file: {file_path}")
        
        start_time = time.time()
        cap = None
        
        try:
            cap = cv2.VideoCapture(file_path)
            
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video file: {file_path}")
                
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            self.logger.info(f"📐 Video: {width}x{height}, {fps} FPS, {total_frames} frames")
            
            # Process frames
            frame_count = 0
            while time.time() - start_time < duration:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_count += 1
                
                # Simple processing
                if frame is not None:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
            # Calculate actual FPS
            test_duration = time.time() - start_time
            actual_fps = frame_count / test_duration if test_duration > 0 else 0
            
            self.logger.info(f"✅ Video test completed: {frame_count} frames, {actual_fps:.1f} FPS")
            
            return SimpleCameraResult(
                success=True,
                fps=actual_fps,
                resolution=(width, height),
                frame_count=frame_count,
                test_duration=test_duration
            )
            
        except Exception as e:
            error_msg = f"Video test failed: {str(e)}"
            self.logger.error(error_msg)
            
            return SimpleCameraResult(
                success=False,
                fps=0.0,
                resolution=(0, 0),
                frame_count=0,
                error_message=error_msg,
                test_duration=time.time() - start_time
            )
            
        finally:
            if cap:
                cap.release()

def demo_simple_camera_test():
    """Demo simple camera testing."""
    print("🧪 Simple Camera Test Demo")
    print("="*50)
    
    tester = SimpleCameraTester()
    
    # Test camera availability
    print("\n🔍 Testing camera availability...")
    availability = tester.test_camera_availability()
    
    print(f"📊 Found {availability['total_cameras']} available cameras")
    
    # Test first available camera
    if availability['available_cameras']:
        first_camera = availability['available_cameras'][0]
        print(f"\n📹 Testing camera {first_camera['index']}...")
        
        result = tester.test_webcam(
            device_index=first_camera['index'],
            duration=5
        )
        
        if result.success:
            print(f"✅ Camera test successful: {result.fps:.1f} FPS")
        else:
            print(f"❌ Camera test failed: {result.error_message}")
    else:
        print("❌ No cameras available for testing")
        
    print("\n✅ Simple camera testing demo completed!")

if __name__ == "__main__":
    demo_simple_camera_test()
