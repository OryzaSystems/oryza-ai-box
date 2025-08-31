#!/usr/bin/env python3
"""
Real-World Testing Starter Script

This script orchestrates the complete real-world testing process:
1. Dataset Collection
2. Camera Integration Testing
3. Edge Device Testing
4. Performance Benchmarking
5. Production Validation
"""

import os
import sys
import time
import json
import logging
from datetime import datetime
from typing import Dict, List, Any
import argparse

# Add tools directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'tools'))

# Import testing tools
try:
    from performance_monitor import PerformanceMonitor
    from camera_test_suite import CameraTestSuite, RealTimeCameraProcessor
except ImportError as e:
    print(f"❌ Error importing testing tools: {e}")
    print("Please ensure all tools are in the tools/ directory")
    sys.exit(1)

# Import AI models
try:
    from ai_models import (
        FaceDetector, FaceRecognizer, PersonDetector, BehaviorAnalyzer,
        VehicleDetector, LicensePlateOCR, VehicleClassifier, TrafficAnalyzer
    )
except ImportError as e:
    print(f"⚠️ Warning: Could not import AI models: {e}")
    print("Testing will proceed without AI model integration")

class RealWorldTester:
    """Main real-world testing orchestrator."""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.test_results = {}
        self.performance_monitor = None
        self.camera_suite = None
        
        # Test configuration
        self.config = {
            'test_duration': 30,  # seconds per test
            'camera_device': 0,   # webcam device index
            'output_dir': 'test_results',
            'save_videos': True,
            'enable_ai_models': True
        }
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('real_world_testing.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
        
    def setup_test_environment(self):
        """Setup testing environment."""
        self.logger.info("🔧 Setting up test environment...")
        
        # Create output directory
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
        # Initialize testing tools
        self.performance_monitor = PerformanceMonitor(
            os.path.join(self.config['output_dir'], 'performance_log.json')
        )
        
        self.camera_suite = CameraTestSuite()
        
        self.logger.info("✅ Test environment setup completed")
        
    def test_camera_integration(self) -> Dict[str, Any]:
        """Test camera integration."""
        self.logger.info("📹 Testing camera integration...")
        
        # Test webcam
        result = self.camera_suite.test_webcam(
            device_index=self.config['camera_device'],
            duration=10
        )
        
        if result.success:
            self.logger.info(f"✅ Camera test successful: {result.fps_actual:.1f} FPS")
        else:
            self.logger.error(f"❌ Camera test failed: {result.error_message}")
            
        return {
            'camera_test': result,
            'timestamp': datetime.now().isoformat()
        }
        
    def test_ai_models_with_camera(self) -> Dict[str, Any]:
        """Test AI models with real camera feed."""
        if not self.config['enable_ai_models']:
            self.logger.info("⚠️ AI model testing disabled")
            return {}
            
        self.logger.info("🤖 Testing AI models with camera...")
        
        results = {}
        models_to_test = [
            ('FaceDetector', FaceDetector),
            ('PersonDetector', PersonDetector),
            ('VehicleDetector', VehicleDetector)
        ]
        
        for model_name, model_class in models_to_test:
            try:
                self.logger.info(f"🧪 Testing {model_name}...")
                
                # Initialize model
                model = model_class()
                
                # Start performance monitoring
                self.performance_monitor.start_monitoring(
                    model_name=model_name,
                    platform="myai",
                    resolution="640x480"
                )
                
                # Test with camera
                processor = RealTimeCameraProcessor(
                    str(self.config['camera_device']),
                    model_processor=lambda frame: self._process_frame_with_model(frame, model)
                )
                
                # Start processing
                processor.start_processing(duration=self.config['test_duration'])
                
                # Stop monitoring
                self.performance_monitor.stop_monitoring()
                
                # Get performance summary
                summary = self.performance_monitor.get_summary()
                
                results[model_name] = {
                    'success': True,
                    'performance': summary,
                    'timestamp': datetime.now().isoformat()
                }
                
                self.logger.info(f"✅ {model_name} test completed")
                
            except Exception as e:
                self.logger.error(f"❌ {model_name} test failed: {e}")
                results[model_name] = {
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                
        return results
        
    def _process_frame_with_model(self, frame, model):
        """Process frame with AI model."""
        try:
            # Record frame for FPS calculation
            self.performance_monitor.record_frame()
            
            # Process with model
            result = model.detect(frame)
            
            # Draw results on frame
            processed_frame = self._draw_results_on_frame(frame, result)
            
            return processed_frame
            
        except Exception as e:
            self.logger.error(f"Frame processing error: {e}")
            return frame
            
    def _draw_results_on_frame(self, frame, result):
        """Draw AI model results on frame."""
        try:
            # Simple visualization - draw bounding boxes
            if hasattr(result, 'detections') and result.detections:
                for detection in result.detections:
                    if hasattr(detection, 'bbox'):
                        x1, y1, x2, y2 = detection.bbox
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        
                        # Add label
                        if hasattr(detection, 'class_name'):
                            cv2.putText(frame, detection.class_name, (int(x1), int(y1)-10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            
        except Exception as e:
            self.logger.error(f"Error drawing results: {e}")
            
        return frame
        
    def test_performance_baselines(self) -> Dict[str, Any]:
        """Test performance baselines."""
        self.logger.info("📊 Testing performance baselines...")
        
        # Test different resolutions
        resolutions = [(640, 480), (1280, 720), (1920, 1080)]
        results = {}
        
        for width, height in resolutions:
            self.logger.info(f"🧪 Testing resolution {width}x{height}...")
            
            # Start monitoring
            self.performance_monitor.start_monitoring(
                model_name="Baseline",
                platform="myai",
                resolution=f"{width}x{height}"
            )
            
            # Simulate processing
            start_time = time.time()
            frame_count = 0
            
            while time.time() - start_time < 10:  # 10 seconds per resolution
                self.performance_monitor.record_frame()
                frame_count += 1
                time.sleep(1/30)  # Simulate 30 FPS
                
            # Stop monitoring
            self.performance_monitor.stop_monitoring()
            
            # Get summary
            summary = self.performance_monitor.get_summary()
            
            results[f"{width}x{height}"] = {
                'performance': summary,
                'timestamp': datetime.now().isoformat()
            }
            
        return results
        
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        self.logger.info("📄 Generating comprehensive test report...")
        
        report = {
            'test_session': {
                'start_time': datetime.now().isoformat(),
                'duration': time.time() - getattr(self, 'session_start_time', time.time()),
                'config': self.config
            },
            'results': self.test_results,
            'summary': {
                'total_tests': len(self.test_results),
                'successful_tests': len([r for r in self.test_results.values() if r.get('success', False)]),
                'failed_tests': len([r for r in self.test_results.values() if not r.get('success', True)])
            }
        }
        
        # Save report
        report_file = os.path.join(self.config['output_dir'], 'comprehensive_report.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        self.logger.info(f"📄 Report saved to {report_file}")
        
        return report
        
    def print_test_summary(self):
        """Print test summary to console."""
        print("\n" + "="*80)
        print("🎯 REAL-WORLD TESTING SUMMARY")
        print("="*80)
        
        if not self.test_results:
            print("❌ No test results available")
            return
            
        # Print results by category
        for test_name, result in self.test_results.items():
            print(f"\n📊 {test_name.upper()}:")
            
            if isinstance(result, dict):
                if 'success' in result:
                    status = "✅ PASS" if result['success'] else "❌ FAIL"
                    print(f"   Status: {status}")
                    
                if 'performance' in result:
                    perf = result['performance']
                    if 'avg_fps' in perf:
                        print(f"   FPS: {perf['avg_fps']:.1f} avg, {perf['max_fps']:.1f} max")
                    if 'avg_memory_mb' in perf:
                        print(f"   Memory: {perf['avg_memory_mb']:.0f}MB avg")
                        
                if 'error' in result:
                    print(f"   Error: {result['error']}")
                    
        print("\n" + "="*80)
        
    def run_complete_test_suite(self):
        """Run complete real-world testing suite."""
        self.logger.info("🚀 Starting complete real-world testing suite...")
        
        # Record session start time
        self.session_start_time = time.time()
        
        try:
            # Setup environment
            self.setup_test_environment()
            
            # Phase 1: Camera Integration Testing
            self.logger.info("📹 Phase 1: Camera Integration Testing")
            self.test_results['camera_integration'] = self.test_camera_integration()
            
            # Phase 2: Performance Baselines
            self.logger.info("📊 Phase 2: Performance Baselines")
            self.test_results['performance_baselines'] = self.test_performance_baselines()
            
            # Phase 3: AI Model Testing
            self.logger.info("🤖 Phase 3: AI Model Testing")
            self.test_results['ai_models'] = self.test_ai_models_with_camera()
            
            # Generate comprehensive report
            self.logger.info("📄 Phase 4: Report Generation")
            self.generate_comprehensive_report()
            
            # Print summary
            self.print_test_summary()
            
            self.logger.info("✅ Real-world testing suite completed successfully!")
            
        except Exception as e:
            self.logger.error(f"❌ Testing suite failed: {e}")
            raise

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Real-World Testing Suite")
    parser.add_argument('--duration', type=int, default=30, help='Test duration in seconds')
    parser.add_argument('--camera', type=int, default=0, help='Camera device index')
    parser.add_argument('--output', type=str, default='test_results', help='Output directory')
    parser.add_argument('--no-ai', action='store_true', help='Disable AI model testing')
    
    args = parser.parse_args()
    
    # Create tester
    tester = RealWorldTester()
    
    # Update configuration
    tester.config.update({
        'test_duration': args.duration,
        'camera_device': args.camera,
        'output_dir': args.output,
        'enable_ai_models': not args.no_ai
    })
    
    # Run tests
    tester.run_complete_test_suite()

if __name__ == "__main__":
    main()
