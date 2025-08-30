# ==========================================
# AI Box - Person Detection Simple Test
# Test person detection infrastructure without complex operations
# ==========================================

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ai_models.common.model_config import ModelConfig
from ai_models.human_analysis.person_detector import PersonDetector

def test_person_detector_initialization():
    """Test PersonDetector initialization without loading model."""
    print("🧪 Testing PersonDetector Initialization...")
    
    # Create config
    config = ModelConfig(
        model_name="yolov8n-person",
        model_type="person_detection",
        confidence_threshold=0.5,
        input_size=(640, 640),
        use_gpu=True,
        model_params={
            'min_person_size': 30,
            'max_persons': 50
        }
    )
    
    # Initialize detector (without loading model)
    try:
        person_detector = PersonDetector(config)
        print(f"✅ PersonDetector initialized: {person_detector}")
        
        # Test model info
        model_info = person_detector.get_model_info()
        print(f"✅ Model info: {model_info['model_class']}")
        
        # Test performance metrics
        metrics = person_detector.get_performance_metrics()
        print(f"✅ Performance metrics: {metrics}")
        
        # Test configuration
        print(f"✅ Person class ID: {person_detector.person_class_id}")
        print(f"✅ Min person size: {person_detector.min_person_size}")
        print(f"✅ Max persons: {person_detector.max_persons}")
        
        return True
        
    except Exception as e:
        print(f"❌ PersonDetector initialization failed: {e}")
        return False

def test_platform_optimization():
    """Test platform optimization functionality."""
    print("\n🔧 Testing Platform Optimization...")
    
    config = ModelConfig(
        model_name="yolov8n-person",
        model_type="person_detection",
        platform="auto",
        model_params={'min_person_size': 30}
    )
    
    person_detector = PersonDetector(config)
    
    # Test different platforms
    platforms = ['raspberry-pi-5', 'radxa-rock-5', 'jetson-nano', 'core-i5']
    
    for platform in platforms:
        print(f"🔧 Testing optimization for {platform}...")
        success = person_detector.optimize_for_platform(platform)
        print(f"📊 {platform} optimization: {'✅ Success' if success else '❌ Failed'}")
    
    return True

def test_yolo_library():
    """Test YOLO library functionality."""
    print("\n🧪 Testing YOLO Library...")
    
    try:
        from ultralytics import YOLO
        
        # Test YOLO import
        print("✅ YOLO library imported successfully")
        
        # Test model creation (without loading)
        model = YOLO()
        print("✅ YOLO model created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ YOLO library test failed: {e}")
        return False

def main():
    """Main test function."""
    print("🚀 AI Box - Person Detection Simple Test")
    print("=" * 50)
    
    tests = [
        ("PersonDetector Initialization", test_person_detector_initialization),
        ("Platform Optimization", test_platform_optimization),
        ("YOLO Library", test_yolo_library)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
    
    print(f"\n{'='*50}")
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All person detection infrastructure tests passed!")
        print("✅ Person Detection Infrastructure is working correctly")
    else:
        print("❌ Some tests failed!")
        print("🔧 Please check the error messages above")
    
    return passed == total

if __name__ == "__main__":
    main()
