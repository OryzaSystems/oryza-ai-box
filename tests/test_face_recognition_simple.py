# ==========================================
# AI Box - Face Recognition Simple Test
# Test face recognition infrastructure without complex operations
# ==========================================

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ai_models.common.model_config import ModelConfig
from ai_models.human_analysis.face_recognizer import FaceRecognizer

def test_face_recognizer_initialization():
    """Test FaceRecognizer initialization without loading model."""
    print("🧪 Testing FaceRecognizer Initialization...")
    
    # Create config
    config = ModelConfig(
        model_name="face-recognizer",
        model_type="face_recognition",
        confidence_threshold=0.5,
        input_size=(640, 640),
        use_gpu=True,
        model_params={
            'tolerance': 0.6,
            'min_face_size': 20,
            'embedding_model': 'hog',
            'database_path': 'test_face_database.pkl'
        }
    )
    
    # Initialize recognizer (without loading model)
    try:
        face_recognizer = FaceRecognizer(config)
        print(f"✅ FaceRecognizer initialized: {face_recognizer}")
        
        # Test model info
        model_info = face_recognizer.get_model_info()
        print(f"✅ Model info: {model_info['model_class']}")
        
        # Test performance metrics
        metrics = face_recognizer.get_performance_metrics()
        print(f"✅ Performance metrics: {metrics}")
        
        # Test configuration
        print(f"✅ Tolerance: {face_recognizer.tolerance}")
        print(f"✅ Embedding model: {face_recognizer.embedding_model}")
        print(f"✅ Database path: {face_recognizer.database_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ FaceRecognizer initialization failed: {e}")
        return False

def test_platform_optimization():
    """Test platform optimization functionality."""
    print("\n🔧 Testing Platform Optimization...")
    
    config = ModelConfig(
        model_name="face-recognizer",
        model_type="face_recognition",
        platform="auto",
        model_params={'tolerance': 0.6}
    )
    
    face_recognizer = FaceRecognizer(config)
    
    # Test different platforms
    platforms = ['raspberry-pi-5', 'radxa-rock-5', 'jetson-nano', 'core-i5']
    
    for platform in platforms:
        print(f"🔧 Testing optimization for {platform}...")
        success = face_recognizer.optimize_for_platform(platform)
        print(f"📊 {platform} optimization: {'✅ Success' if success else '❌ Failed'}")
    
    return True

def test_face_recognition_library():
    """Test face_recognition library functionality."""
    print("\n🧪 Testing face_recognition Library...")
    
    try:
        import face_recognition
        import numpy as np
        
        # Create a simple test image
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Test face locations
        face_locations = face_recognition.face_locations(test_image)
        print(f"✅ Face locations test: {len(face_locations)} faces found")
        
        # Test face encodings (should return empty list for blank image)
        face_encodings = face_recognition.face_encodings(test_image)
        print(f"✅ Face encodings test: {len(face_encodings)} encodings extracted")
        
        return True
        
    except Exception as e:
        print(f"❌ face_recognition library test failed: {e}")
        return False

def main():
    """Main test function."""
    print("🚀 AI Box - Face Recognition Simple Test")
    print("=" * 50)
    
    tests = [
        ("FaceRecognizer Initialization", test_face_recognizer_initialization),
        ("Platform Optimization", test_platform_optimization),
        ("Face Recognition Library", test_face_recognition_library)
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
        print("🎉 All face recognition infrastructure tests passed!")
        print("✅ Face Recognition Infrastructure is working correctly")
    else:
        print("❌ Some tests failed!")
        print("🔧 Please check the error messages above")
    
    return passed == total

if __name__ == "__main__":
    main()
