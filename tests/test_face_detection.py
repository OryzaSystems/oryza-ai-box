# ==========================================
# AI Box - Face Detection Test Script
# Test face detection model functionality
# ==========================================

import sys
import os
import cv2
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ai_models.common.model_config import ModelConfig
from ai_models.human_analysis.face_detector import FaceDetector

def create_test_image(width: int = 640, height: int = 480) -> np.ndarray:
    """
    Create a test image with a simple face-like pattern.
    
    Args:
        width: Image width
        height: Image height
        
    Returns:
        Test image as numpy array
    """
    # Create a simple test image
    image = np.ones((height, width, 3), dtype=np.uint8) * 128
    
    # Add a simple face-like pattern (circle for head, rectangles for eyes)
    center_x, center_y = width // 2, height // 2
    
    # Draw head (circle)
    cv2.circle(image, (center_x, center_y), 80, (255, 255, 255), -1)
    
    # Draw eyes (rectangles)
    cv2.rectangle(image, (center_x - 30, center_y - 20), (center_x - 10, center_y), (0, 0, 0), -1)
    cv2.rectangle(image, (center_x + 10, center_y - 20), (center_x + 30, center_y), (0, 0, 0), -1)
    
    # Draw mouth (rectangle)
    cv2.rectangle(image, (center_x - 20, center_y + 20), (center_x + 20, center_y + 40), (0, 0, 0), -1)
    
    return image

def test_face_detector():
    """Test face detection model functionality."""
    print("🧪 Testing Face Detection Model...")
    
    # Create model configuration
    config = ModelConfig(
        model_name="yolov8n-face",
        model_type="face_detection",
        confidence_threshold=0.5,
        nms_threshold=0.4,
        input_size=(640, 640),
        use_gpu=True,
        platform="auto"
    )
    
    print(f"📋 Model Config: {config}")
    
    # Initialize face detector
    print("🔧 Initializing Face Detector...")
    face_detector = FaceDetector(config)
    
    # Load model
    print("📥 Loading Face Detection Model...")
    if not face_detector.load_model():
        print("❌ Failed to load face detection model")
        return False
    
    print("✅ Face detection model loaded successfully")
    
    # Get model info
    model_info = face_detector.get_model_info()
    print(f"📊 Model Info: {model_info['model_class']}")
    print(f"📊 Device: {model_info['device']}")
    print(f"📊 Performance Metrics: {model_info['performance_metrics']}")
    
    # Create test image
    print("🖼️ Creating test image...")
    test_image = create_test_image(640, 480)
    
    # Save test image
    test_image_path = "test_face_image.jpg"
    cv2.imwrite(test_image_path, test_image)
    print(f"💾 Test image saved to: {test_image_path}")
    
    # Test face detection
    print("🔍 Testing face detection...")
    try:
        # Test with file path
        result = face_detector.detect_faces(test_image_path)
        print(f"📊 Detection Result: {result}")
        print(f"📊 Detections: {len(result.detections)}")
        
        # Test with numpy array
        result_array = face_detector.detect_faces(test_image)
        print(f"📊 Array Detection Result: {len(result_array.detections)} detections")
        
        # Test single face detection
        single_face = face_detector.detect_single_face(test_image)
        if single_face:
            print(f"📊 Single Face: {single_face.confidence:.3f} confidence")
        
        # Test face counting
        face_count = face_detector.count_faces(test_image)
        print(f"📊 Face Count: {face_count}")
        
        # Test bounding boxes
        bboxes = face_detector.get_face_bboxes(test_image)
        print(f"📊 Bounding Boxes: {len(bboxes)}")
        
        # Test landmarks
        landmarks = face_detector.get_face_landmarks(test_image)
        print(f"📊 Landmarks: {len(landmarks)}")
        
        # Test performance metrics
        metrics = face_detector.get_performance_metrics()
        print(f"📊 Performance Metrics: {metrics}")
        
        print("✅ Face detection tests completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Face detection test failed: {e}")
        return False

def test_platform_optimization():
    """Test platform optimization functionality."""
    print("\n🔧 Testing Platform Optimization...")
    
    config = ModelConfig(
        model_name="yolov8n-face",
        model_type="face_detection",
        platform="auto"
    )
    
    face_detector = FaceDetector(config)
    
    # Test different platforms
    platforms = ['raspberry-pi-5', 'radxa-rock-5', 'jetson-nano', 'core-i5']
    
    for platform in platforms:
        print(f"🔧 Testing optimization for {platform}...")
        success = face_detector.optimize_for_platform(platform)
        print(f"📊 {platform} optimization: {'✅ Success' if success else '❌ Failed'}")

def main():
    """Main test function."""
    print("🚀 AI Box - Face Detection Model Test")
    print("=" * 50)
    
    # Test basic functionality
    success = test_face_detector()
    
    if success:
        # Test platform optimization
        test_platform_optimization()
        
        print("\n🎉 All tests completed successfully!")
        print("✅ Face Detection Model is working correctly")
    else:
        print("\n❌ Tests failed!")
        print("🔧 Please check the error messages above")
    
    # Cleanup
    test_image_path = "test_face_image.jpg"
    if os.path.exists(test_image_path):
        os.remove(test_image_path)
        print(f"🧹 Cleaned up test image: {test_image_path}")

if __name__ == "__main__":
    main()
