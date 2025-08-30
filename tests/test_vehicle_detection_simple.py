# ==========================================
# AI Box - Vehicle Detection Simple Test
# Test vehicle detection infrastructure
# ==========================================

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ai_models.common.model_config import ModelConfig
from ai_models.vehicle_analysis.vehicle_detector import VehicleDetector

def test_vehicle_detector_initialization():
    """Test VehicleDetector initialization without loading model."""
    print("🧪 Testing VehicleDetector Initialization...")
    
    # Create config
    config = ModelConfig(
        model_name="vehicle-detector",
        model_type="vehicle_detection",
        confidence_threshold=0.5,
        input_size=(640, 640),
        use_gpu=True,
        model_params={
            'min_vehicle_size': 100,
            'vehicle_confidence': 0.3
        }
    )
    
    # Initialize detector (without loading model)
    try:
        vehicle_detector = VehicleDetector(config)
        print(f"✅ VehicleDetector initialized: {vehicle_detector}")
        
        # Test model info
        model_info = vehicle_detector.get_model_info()
        print(f"✅ Model info: {model_info['model_class']}")
        
        # Test performance metrics
        metrics = vehicle_detector.get_performance_metrics()
        print(f"✅ Performance metrics: {metrics}")
        
        # Test configuration
        print(f"✅ Vehicle classes: {vehicle_detector.vehicle_classes}")
        print(f"✅ Vehicle class IDs: {vehicle_detector.vehicle_class_ids}")
        print(f"✅ Min vehicle size: {vehicle_detector.min_vehicle_size}")
        print(f"✅ Vehicle confidence threshold: {vehicle_detector.vehicle_confidence_threshold}")
        print(f"✅ Supported vehicles: {vehicle_detector.metadata['supported_vehicles']}")
        
        return True
        
    except Exception as e:
        print(f"❌ VehicleDetector initialization failed: {e}")
        return False

def test_platform_optimization():
    """Test platform optimization functionality."""
    print("\n🔧 Testing Platform Optimization...")
    
    config = ModelConfig(
        model_name="vehicle-detector",
        model_type="vehicle_detection",
        platform="auto",
        model_params={'min_vehicle_size': 100}
    )
    
    vehicle_detector = VehicleDetector(config)
    
    # Test different platforms
    platforms = ['raspberry-pi-5', 'radxa-rock-5', 'jetson-nano', 'core-i5']
    
    for platform in platforms:
        print(f"🔧 Testing optimization for {platform}...")
        success = vehicle_detector.optimize_for_platform(platform)
        print(f"📊 {platform} optimization: {'✅ Success' if success else '❌ Failed'}")
    
    return True

def test_vehicle_classes():
    """Test vehicle classification setup."""
    print("\n🧪 Testing Vehicle Classes...")
    
    config = ModelConfig(
        model_name="vehicle-detector",
        model_type="vehicle_detection"
    )
    
    vehicle_detector = VehicleDetector(config)
    
    # Test vehicle classes
    expected_vehicles = ['bicycle', 'car', 'motorcycle', 'bus', 'truck']
    actual_vehicles = list(vehicle_detector.vehicle_classes.values())
    
    print(f"📊 Expected vehicles: {expected_vehicles}")
    print(f"📊 Actual vehicles: {actual_vehicles}")
    
    # Check if all expected vehicles are present
    missing_vehicles = set(expected_vehicles) - set(actual_vehicles)
    if not missing_vehicles:
        print(f"✅ All vehicle types present")
    else:
        print(f"❌ Missing vehicles: {missing_vehicles}")
        return False
    
    # Test COCO class IDs
    expected_class_ids = [1, 2, 3, 5, 7]  # COCO dataset IDs
    actual_class_ids = vehicle_detector.vehicle_class_ids
    
    print(f"📊 Vehicle class IDs: {actual_class_ids}")
    
    if set(expected_class_ids) == set(actual_class_ids):
        print(f"✅ Vehicle class IDs correct")
    else:
        print(f"❌ Vehicle class IDs mismatch")
        return False
    
    return True

def main():
    """Main test function."""
    print("🚀 AI Box - Vehicle Detection Simple Test")
    print("=" * 50)
    
    tests = [
        ("VehicleDetector Initialization", test_vehicle_detector_initialization),
        ("Platform Optimization", test_platform_optimization),
        ("Vehicle Classes", test_vehicle_classes)
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
        print("🎉 All vehicle detection infrastructure tests passed!")
        print("✅ Vehicle Detection Infrastructure is working correctly")
    else:
        print("❌ Some tests failed!")
        print("🔧 Please check the error messages above")
    
    return passed == total

if __name__ == "__main__":
    main()
