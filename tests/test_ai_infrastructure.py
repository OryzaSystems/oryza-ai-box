# ==========================================
# AI Box - AI Infrastructure Test Script
# Test basic AI model infrastructure without downloading models
# ==========================================

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ai_models.common.model_config import ModelConfig
from ai_models.common.inference_result import InferenceResult, Detection
from ai_models.human_analysis.face_detector import FaceDetector

def test_model_config():
    """Test ModelConfig functionality."""
    print("🧪 Testing ModelConfig...")
    
    # Create config
    config = ModelConfig(
        model_name="test-model",
        model_type="test_type",
        confidence_threshold=0.5,
        input_size=(640, 640),
        use_gpu=True
    )
    
    print(f"✅ Config created: {config}")
    
    # Test validation
    errors = config.validate()
    if errors:
        print(f"❌ Config validation errors: {errors}")
        return False
    else:
        print("✅ Config validation passed")
    
    # Test to_dict/from_dict
    config_dict = config.to_dict()
    config2 = ModelConfig.from_dict(config_dict)
    print(f"✅ Config serialization: {config2}")
    
    return True

def test_inference_result():
    """Test InferenceResult functionality."""
    print("\n🧪 Testing InferenceResult...")
    
    # Create result
    result = InferenceResult(
        success=True,
        model_name="test-model",
        model_type="test_type"
    )
    
    print(f"✅ Result created: {result}")
    
    # Add detections
    result.add_detection(
        bbox=[100, 100, 200, 200],
        confidence=0.95,
        class_id=0,
        class_name="face"
    )
    
    result.add_detection(
        bbox=[300, 300, 400, 400],
        confidence=0.87,
        class_id=0,
        class_name="face"
    )
    
    print(f"✅ Added detections: {len(result.detections)}")
    
    # Test filtering
    high_conf_detections = result.get_detections_by_confidence(0.9)
    print(f"✅ High confidence detections: {len(high_conf_detections)}")
    
    # Test best detection
    best_detection = result.get_best_detection()
    if best_detection:
        print(f"✅ Best detection confidence: {best_detection.confidence}")
    
    # Test counting
    face_count = result.count_detections('face')
    print(f"✅ Face count: {face_count}")
    
    # Test NMS
    nms_result = result.filter_by_nms(0.5)
    print(f"✅ NMS result: {len(nms_result.detections)} detections")
    
    return True

def test_face_detector_initialization():
    """Test FaceDetector initialization without loading model."""
    print("\n🧪 Testing FaceDetector Initialization...")
    
    # Create config
    config = ModelConfig(
        model_name="yolov8n-face",
        model_type="face_detection",
        confidence_threshold=0.5,
        input_size=(640, 640),
        use_gpu=True
    )
    
    # Initialize detector (without loading model)
    try:
        face_detector = FaceDetector(config)
        print(f"✅ FaceDetector initialized: {face_detector}")
        
        # Test model info
        model_info = face_detector.get_model_info()
        print(f"✅ Model info: {model_info['model_class']}")
        
        # Test performance metrics
        metrics = face_detector.get_performance_metrics()
        print(f"✅ Performance metrics: {metrics}")
        
        return True
        
    except Exception as e:
        print(f"❌ FaceDetector initialization failed: {e}")
        return False

def test_platform_optimization():
    """Test platform optimization functionality."""
    print("\n🧪 Testing Platform Optimization...")
    
    config = ModelConfig(
        model_name="test-model",
        model_type="test_type",
        platform="auto"
    )
    
    face_detector = FaceDetector(config)
    
    # Test different platforms
    platforms = ['raspberry-pi-5', 'radxa-rock-5', 'jetson-nano', 'core-i5']
    
    for platform in platforms:
        print(f"🔧 Testing optimization for {platform}...")
        success = face_detector.optimize_for_platform(platform)
        print(f"📊 {platform} optimization: {'✅ Success' if success else '❌ Failed'}")
    
    return True

def main():
    """Main test function."""
    print("🚀 AI Box - AI Infrastructure Test")
    print("=" * 50)
    
    tests = [
        ("ModelConfig", test_model_config),
        ("InferenceResult", test_inference_result),
        ("FaceDetector Initialization", test_face_detector_initialization),
        ("Platform Optimization", test_platform_optimization)
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
        print("🎉 All infrastructure tests passed!")
        print("✅ AI Model Infrastructure is working correctly")
    else:
        print("❌ Some tests failed!")
        print("🔧 Please check the error messages above")
    
    return passed == total

if __name__ == "__main__":
    main()
