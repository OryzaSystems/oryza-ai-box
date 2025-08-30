# ==========================================
# AI Box - Behavior Analysis Simple Test
# Test behavior analysis infrastructure without complex operations
# ==========================================

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ai_models.common.model_config import ModelConfig
from ai_models.human_analysis.behavior_analyzer import BehaviorAnalyzer

def test_behavior_analyzer_initialization():
    """Test BehaviorAnalyzer initialization without loading model."""
    print("🧪 Testing BehaviorAnalyzer Initialization...")
    
    # Create config
    config = ModelConfig(
        model_name="behavior-analyzer",
        model_type="behavior_analysis",
        confidence_threshold=0.5,
        input_size=(224, 224),
        use_gpu=True,
        model_params={
            'temporal_window': 5,
            'smoothing_factor': 0.7,
            'dropout_rate': 0.5
        }
    )
    
    # Initialize analyzer (without loading model)
    try:
        behavior_analyzer = BehaviorAnalyzer(config)
        print(f"✅ BehaviorAnalyzer initialized: {behavior_analyzer}")
        
        # Test model info
        model_info = behavior_analyzer.get_model_info()
        print(f"✅ Model info: {model_info['model_class']}")
        
        # Test performance metrics
        metrics = behavior_analyzer.get_performance_metrics()
        print(f"✅ Performance metrics: {metrics}")
        
        # Test configuration
        print(f"✅ Behavior classes: {behavior_analyzer.behavior_classes}")
        print(f"✅ Number of classes: {behavior_analyzer.num_classes}")
        print(f"✅ Temporal window: {behavior_analyzer.temporal_window}")
        print(f"✅ Smoothing factor: {behavior_analyzer.smoothing_factor}")
        print(f"✅ Input size: {behavior_analyzer.input_size}")
        print(f"✅ Dropout rate: {behavior_analyzer.dropout_rate}")
        
        return True
        
    except Exception as e:
        print(f"❌ BehaviorAnalyzer initialization failed: {e}")
        return False

def test_platform_optimization():
    """Test platform optimization functionality."""
    print("\n🔧 Testing Platform Optimization...")
    
    config = ModelConfig(
        model_name="behavior-analyzer",
        model_type="behavior_analysis",
        platform="auto",
        model_params={'temporal_window': 5}
    )
    
    behavior_analyzer = BehaviorAnalyzer(config)
    
    # Test different platforms
    platforms = ['raspberry-pi-5', 'radxa-rock-5', 'jetson-nano', 'core-i5']
    
    for platform in platforms:
        print(f"🔧 Testing optimization for {platform}...")
        success = behavior_analyzer.optimize_for_platform(platform)
        print(f"📊 {platform} optimization: {'✅ Success' if success else '❌ Failed'}")
    
    return True

def test_pytorch_cnn():
    """Test PyTorch CNN functionality."""
    print("\n🧪 Testing PyTorch CNN...")
    
    try:
        import torch
        import torch.nn as nn
        
        from ai_models.human_analysis.behavior_analyzer import BehaviorCNN
        
        # Test CNN creation
        model = BehaviorCNN(num_classes=4, dropout_rate=0.5)
        print("✅ BehaviorCNN created successfully")
        
        # Test forward pass with dummy input
        dummy_input = torch.randn(1, 3, 224, 224)
        output = model(dummy_input)
        print(f"✅ Forward pass successful: output shape {output.shape}")
        
        # Test parameter count
        param_count = sum(p.numel() for p in model.parameters())
        print(f"✅ Model parameters: {param_count:,}")
        
        return True
        
    except Exception as e:
        print(f"❌ PyTorch CNN test failed: {e}")
        return False

def test_behavior_classes():
    """Test behavior classification setup."""
    print("\n🧪 Testing Behavior Classes...")
    
    config = ModelConfig(
        model_name="behavior-analyzer",
        model_type="behavior_analysis"
    )
    
    behavior_analyzer = BehaviorAnalyzer(config)
    
    # Test behavior classes
    expected_classes = ['standing', 'walking', 'running', 'sitting']
    actual_classes = behavior_analyzer.behavior_classes
    
    if actual_classes == expected_classes:
        print(f"✅ Behavior classes correct: {actual_classes}")
    else:
        print(f"❌ Behavior classes mismatch: expected {expected_classes}, got {actual_classes}")
        return False
    
    # Test number of classes
    if behavior_analyzer.num_classes == 4:
        print(f"✅ Number of classes correct: {behavior_analyzer.num_classes}")
    else:
        print(f"❌ Number of classes incorrect: {behavior_analyzer.num_classes}")
        return False
    
    return True

def main():
    """Main test function."""
    print("🚀 AI Box - Behavior Analysis Simple Test")
    print("=" * 50)
    
    tests = [
        ("BehaviorAnalyzer Initialization", test_behavior_analyzer_initialization),
        ("Platform Optimization", test_platform_optimization),
        ("PyTorch CNN", test_pytorch_cnn),
        ("Behavior Classes", test_behavior_classes)
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
        print("🎉 All behavior analysis infrastructure tests passed!")
        print("✅ Behavior Analysis Infrastructure is working correctly")
    else:
        print("❌ Some tests failed!")
        print("🔧 Please check the error messages above")
    
    return passed == total

if __name__ == "__main__":
    main()
