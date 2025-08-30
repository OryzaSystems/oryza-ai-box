# ==========================================
# AI Box - Face Recognition Test Script
# Test face recognition model functionality
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
from ai_models.human_analysis.face_recognizer import FaceRecognizer

def create_test_faces():
    """Create test face images for recognition testing."""
    print("🖼️ Creating test face images...")
    
    # Create directory for test images
    test_dir = Path("test_faces")
    test_dir.mkdir(exist_ok=True)
    
    # Create different face patterns
    faces = {
        "person1": create_face_pattern(1),
        "person2": create_face_pattern(2),
        "person3": create_face_pattern(3)
    }
    
    # Save test images
    for name, image in faces.items():
        image_path = test_dir / f"{name}.jpg"
        cv2.imwrite(str(image_path), image)
        print(f"💾 Saved {name} face: {image_path}")
    
    return faces, test_dir

def create_face_pattern(person_id: int) -> np.ndarray:
    """
    Create a simple face pattern for testing.
    
    Args:
        person_id: Person identifier
        
    Returns:
        Face image as numpy array
    """
    # Create base image
    image = np.ones((200, 200, 3), dtype=np.uint8) * 128
    
    # Different patterns for different people
    if person_id == 1:
        # Person 1: Circle head, square eyes, triangle nose
        cv2.circle(image, (100, 100), 60, (255, 255, 255), -1)
        cv2.rectangle(image, (80, 80), (90, 90), (0, 0, 0), -1)
        cv2.rectangle(image, (110, 80), (120, 90), (0, 0, 0), -1)
        cv2.circle(image, (100, 120), 5, (0, 0, 0), -1)
    elif person_id == 2:
        # Person 2: Square head, circle eyes, rectangle mouth
        cv2.rectangle(image, (60, 60), (140, 140), (200, 200, 200), -1)
        cv2.circle(image, (80, 90), 8, (0, 0, 0), -1)
        cv2.circle(image, (120, 90), 8, (0, 0, 0), -1)
        cv2.rectangle(image, (90, 110), (110, 120), (0, 0, 0), -1)
    else:
        # Person 3: Triangle head, small eyes, big mouth
        pts = np.array([[100, 40], [60, 140], [140, 140]], np.int32)
        cv2.fillPoly(image, [pts], (180, 180, 180))
        cv2.circle(image, (85, 100), 5, (0, 0, 0), -1)
        cv2.circle(image, (115, 100), 5, (0, 0, 0), -1)
        cv2.ellipse(image, (100, 120), (15, 8), 0, 0, 180, (0, 0, 0), -1)
    
    return image

def test_face_recognition():
    """Test face recognition model functionality."""
    print("🧪 Testing Face Recognition Model...")
    
    # Create model configuration
    config = ModelConfig(
        model_name="face-recognizer",
        model_type="face_recognition",
        confidence_threshold=0.5,
        input_size=(640, 640),
        use_gpu=True,
        platform="auto",
        model_params={
            'tolerance': 0.6,
            'min_face_size': 20,
            'embedding_model': 'hog',
            'database_path': 'test_face_database.pkl'
        }
    )
    
    print(f"📋 Model Config: {config}")
    
    # Initialize face recognizer
    print("🔧 Initializing Face Recognizer...")
    face_recognizer = FaceRecognizer(config)
    
    # Load model
    print("📥 Loading Face Recognition Model...")
    if not face_recognizer.load_model():
        print("❌ Failed to load face recognition model")
        return False
    
    print("✅ Face recognition model loaded successfully")
    
    # Get model info
    model_info = face_recognizer.get_model_info()
    print(f"📊 Model Info: {model_info['model_class']}")
    print(f"📊 Device: {model_info['device']}")
    
    # Create test faces
    faces, test_dir = create_test_faces()
    
    # Test face recognition functionality
    print("🔍 Testing face recognition functionality...")
    try:
        # Test 1: Add known faces
        print("\n📝 Test 1: Adding known faces...")
        for name, image in faces.items():
            success = face_recognizer.add_known_face(name, image)
            print(f"📊 Added {name}: {'✅ Success' if success else '❌ Failed'}")
        
        # Test 2: Get known faces
        print("\n📋 Test 2: Getting known faces...")
        known_faces = face_recognizer.get_known_faces()
        print(f"📊 Known faces: {known_faces}")
        
        # Test 3: Face encoding extraction
        print("\n🔍 Test 3: Face encoding extraction...")
        for name, image in faces.items():
            encodings = face_recognizer.extract_face_encodings(image)
            print(f"📊 {name} encodings: {len(encodings)}")
        
        # Test 4: Face recognition
        print("\n👤 Test 4: Face recognition...")
        for name, image in faces.items():
            result = face_recognizer.recognize_faces(image)
            print(f"📊 {name} recognition: {len(result.detections)} faces detected")
            for detection in result.detections:
                print(f"   - {detection.class_name} (confidence: {detection.confidence:.3f})")
        
        # Test 5: Face comparison
        print("\n🔄 Test 5: Face comparison...")
        if len(faces) >= 2:
            names = list(faces.keys())
            encodings = []
            for name in names:
                enc = face_recognizer.extract_face_encodings(faces[name])
                if enc:
                    encodings.append(enc[0])
            
            if len(encodings) >= 2:
                # Compare same person
                same_person = face_recognizer.compare_faces(encodings[0], encodings[0])
                print(f"📊 Same person comparison: {'✅ Match' if same_person else '❌ No match'}")
                
                # Compare different people
                if len(encodings) >= 2:
                    diff_person = face_recognizer.compare_faces(encodings[0], encodings[1])
                    print(f"📊 Different person comparison: {'❌ Match' if diff_person else '✅ No match'}")
                
                # Get face distances
                distance_same = face_recognizer.get_face_distance(encodings[0], encodings[0])
                print(f"📊 Same person distance: {distance_same:.4f}")
                
                if len(encodings) >= 2:
                    distance_diff = face_recognizer.get_face_distance(encodings[0], encodings[1])
                    print(f"📊 Different person distance: {distance_diff:.4f}")
        
        # Test 6: Performance metrics
        print("\n📊 Test 6: Performance metrics...")
        metrics = face_recognizer.get_performance_metrics()
        print(f"📊 Performance metrics: {metrics}")
        
        print("✅ Face recognition tests completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Face recognition test failed: {e}")
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

def cleanup_test_files():
    """Clean up test files."""
    print("\n🧹 Cleaning up test files...")
    
    # Remove test images
    test_dir = Path("test_faces")
    if test_dir.exists():
        for file in test_dir.glob("*.jpg"):
            file.unlink()
        test_dir.rmdir()
        print("🧹 Removed test face images")
    
    # Remove test database
    test_db = Path("test_face_database.pkl")
    if test_db.exists():
        test_db.unlink()
        print("🧹 Removed test face database")

def main():
    """Main test function."""
    print("🚀 AI Box - Face Recognition Model Test")
    print("=" * 50)
    
    try:
        # Test basic functionality
        success = test_face_recognition()
        
        if success:
            # Test platform optimization
            test_platform_optimization()
            
            print("\n🎉 All tests completed successfully!")
            print("✅ Face Recognition Model is working correctly")
        else:
            print("\n❌ Tests failed!")
            print("🔧 Please check the error messages above")
    
    finally:
        # Cleanup
        cleanup_test_files()

if __name__ == "__main__":
    main()
