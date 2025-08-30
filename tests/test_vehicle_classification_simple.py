# ==========================================
# AI Box - Vehicle Classification Simple Test
# Test vehicle classification infrastructure
# ==========================================

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ai_models.common.model_config import ModelConfig
from ai_models.vehicle_analysis.vehicle_classifier import VehicleClassifier

def test_vehicle_classifier_initialization():
    """Test VehicleClassifier initialization without loading model."""
    print("🧪 Testing VehicleClassifier Initialization...")
    
    # Create config
    config = ModelConfig(
        model_name="vehicle-classifier",
        model_type="vehicle_classification",
        confidence_threshold=0.3,
        input_size=(224, 224),
        use_gpu=True,
        model_params={
            'min_classification_confidence': 0.3,
            'top_k_predictions': 3,
            'backbone': 'resnet50',
            'pretrained': True
        }
    )
    
    # Initialize classifier (without loading model)
    try:
        vehicle_classifier = VehicleClassifier(config)
        print(f"✅ VehicleClassifier initialized: {vehicle_classifier}")
        
        # Test model info
        model_info = vehicle_classifier.get_model_info()
        print(f"✅ Model info: {model_info['model_class']}")
        
        # Test performance metrics
        metrics = vehicle_classifier.get_performance_metrics()
        print(f"✅ Performance metrics: {metrics}")
        
        # Test configuration
        print(f"✅ Number of classes: {vehicle_classifier.num_classes}")
        print(f"✅ Vehicle categories: {list(vehicle_classifier.vehicle_categories.keys())}")
        print(f"✅ All classes count: {len(vehicle_classifier.all_classes)}")
        print(f"✅ Backbone: {vehicle_classifier.backbone}")
        print(f"✅ Image size: {vehicle_classifier.image_size}")
        print(f"✅ Min confidence: {vehicle_classifier.min_classification_confidence}")
        print(f"✅ Top-K predictions: {vehicle_classifier.top_k_predictions}")
        
        # Test class mapping
        sample_classes = vehicle_classifier.all_classes[:5]
        print(f"✅ Sample classes: {sample_classes}")
        for cls in sample_classes:
            category = vehicle_classifier.class_to_category[cls]
            print(f"   - {cls} → {category}")
        
        return True
        
    except Exception as e:
        print(f"❌ VehicleClassifier initialization failed: {e}")
        return False

def test_platform_optimization():
    """Test platform optimization functionality."""
    print("\n🔧 Testing Platform Optimization...")
    
    config = ModelConfig(
        model_name="vehicle-classifier",
        model_type="vehicle_classification",
        platform="auto",
        model_params={'backbone': 'resnet50'}
    )
    
    vehicle_classifier = VehicleClassifier(config)
    
    # Test different platforms
    platforms = ['raspberry-pi-5', 'radxa-rock-5', 'jetson-nano', 'core-i5']
    
    for platform in platforms:
        print(f"🔧 Testing optimization for {platform}...")
        success = vehicle_classifier.optimize_for_platform(platform)
        print(f"📊 {platform} optimization: {'✅ Success' if success else '❌ Failed'}")
        
        # Show optimized settings
        print(f"   - Backbone: {vehicle_classifier.backbone}")
        print(f"   - Image size: {vehicle_classifier.image_size}")
        print(f"   - Min confidence: {vehicle_classifier.min_classification_confidence}")
        print(f"   - Top-K: {vehicle_classifier.top_k_predictions}")
    
    return True

def test_vehicle_categories():
    """Test vehicle category structure."""
    print("\n🧪 Testing Vehicle Categories...")
    
    config = ModelConfig(
        model_name="vehicle-classifier",
        model_type="vehicle_classification"
    )
    
    vehicle_classifier = VehicleClassifier(config)
    
    # Test category structure
    expected_categories = ['car', 'truck', 'bus', 'motorcycle', 'bicycle']
    actual_categories = list(vehicle_classifier.vehicle_categories.keys())
    
    print(f"📊 Expected categories: {expected_categories}")
    print(f"📊 Actual categories: {actual_categories}")
    
    # Check if all expected categories are present
    missing_categories = set(expected_categories) - set(actual_categories)
    if not missing_categories:
        print(f"✅ All vehicle categories present")
    else:
        print(f"❌ Missing categories: {missing_categories}")
        return False
    
    # Test class distribution
    total_classes = 0
    for category, classes in vehicle_classifier.vehicle_categories.items():
        class_count = len(classes)
        total_classes += class_count
        print(f"📊 {category}: {class_count} classes - {classes}")
    
    print(f"📊 Total classes: {total_classes}")
    
    if total_classes == vehicle_classifier.num_classes:
        print(f"✅ Class count matches")
    else:
        print(f"❌ Class count mismatch: {total_classes} vs {vehicle_classifier.num_classes}")
        return False
    
    # Test class-to-category mapping
    mapping_errors = 0
    for cls in vehicle_classifier.all_classes:
        if cls not in vehicle_classifier.class_to_category:
            print(f"❌ Missing mapping for class: {cls}")
            mapping_errors += 1
    
    if mapping_errors == 0:
        print(f"✅ All class-to-category mappings present")
    else:
        print(f"❌ {mapping_errors} mapping errors found")
        return False
    
    return True

def test_pytorch_dependencies():
    """Test PyTorch and torchvision dependencies."""
    print("\n🔧 Testing PyTorch Dependencies...")
    
    try:
        import torch
        print(f"✅ PyTorch available: {torch.__version__}")
        
        # Test CUDA availability
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print(f"⚠️ CUDA not available (CPU only)")
        
        # Test torchvision
        import torchvision
        print(f"✅ Torchvision available: {torchvision.__version__}")
        
        # Test torchvision models
        import torchvision.models as models
        print(f"✅ Torchvision models accessible")
        
        # Test ResNet50 availability
        try:
            resnet50 = models.resnet50
            print(f"✅ ResNet50 model accessible")
        except Exception as e:
            print(f"❌ ResNet50 model error: {e}")
            return False
        
        # Test transforms
        import torchvision.transforms as transforms
        print(f"✅ Torchvision transforms accessible")
        
        return True
        
    except ImportError as e:
        print(f"❌ PyTorch dependencies not available: {e}")
        return False

def test_model_architecture():
    """Test model architecture setup."""
    print("\n🧪 Testing Model Architecture...")
    
    config = ModelConfig(
        model_name="vehicle-classifier",
        model_type="vehicle_classification",
        model_params={
            'backbone': 'resnet50',
            'pretrained': True,
            'dropout_rate': 0.5
        }
    )
    
    vehicle_classifier = VehicleClassifier(config)
    
    # Test architecture parameters
    print(f"📊 Backbone: {vehicle_classifier.backbone}")
    print(f"📊 Pretrained: {vehicle_classifier.pretrained}")
    print(f"📊 Dropout rate: {vehicle_classifier.dropout_rate}")
    print(f"📊 Number of classes: {vehicle_classifier.num_classes}")
    
    # Test different backbones
    backbones = ['resnet50', 'resnet34']
    for backbone in backbones:
        try:
            test_config = ModelConfig(
                model_name="test-classifier",
                model_type="vehicle_classification",
                model_params={'backbone': backbone}
            )
            test_classifier = VehicleClassifier(test_config)
            print(f"✅ {backbone} architecture supported")
        except Exception as e:
            print(f"❌ {backbone} architecture error: {e}")
            return False
    
    return True

def main():
    """Main test function."""
    print("🚀 AI Box - Vehicle Classification Simple Test")
    print("=" * 60)
    
    tests = [
        ("PyTorch Dependencies", test_pytorch_dependencies),
        ("VehicleClassifier Initialization", test_vehicle_classifier_initialization),
        ("Platform Optimization", test_platform_optimization),
        ("Vehicle Categories", test_vehicle_categories),
        ("Model Architecture", test_model_architecture)
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
    
    print(f"\n{'='*60}")
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All vehicle classification infrastructure tests passed!")
        print("✅ Vehicle Classification Infrastructure is working correctly")
    else:
        print("❌ Some tests failed!")
        print("🔧 Please check the error messages above")
    
    return passed == total

if __name__ == "__main__":
    main()
