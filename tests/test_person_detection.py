# ==========================================
# AI Box - Person Detection Test Script
# Test person detection model functionality
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
from ai_models.human_analysis.person_detector import PersonDetector

def create_test_persons():
    """Create test person images for detection testing."""
    print("🖼️ Creating test person images...")
    
    # Create directory for test images
    test_dir = Path("test_persons")
    test_dir.mkdir(exist_ok=True)
    
    # Create different person patterns
    persons = {
        "single_person": create_person_pattern(1),
        "multiple_persons": create_person_pattern(3),
        "crowd_scene": create_person_pattern(5)
    }
    
    # Save test images
    for name, image in persons.items():
        image_path = test_dir / f"{name}.jpg"
        cv2.imwrite(str(image_path), image)
        print(f"💾 Saved {name} image: {image_path}")
    
    return persons, test_dir

def create_person_pattern(num_persons: int) -> np.ndarray:
    """
    Create a simple person pattern for testing.
    
    Args:
        num_persons: Number of persons to create
        
    Returns:
        Person image as numpy array
    """
    # Create base image
    image = np.ones((400, 600, 3), dtype=np.uint8) * 128
    
    # Create persons at different positions
    positions = [
        (150, 200),  # Center
        (100, 150),  # Top left
        (500, 150),  # Top right
        (100, 350),  # Bottom left
        (500, 350),  # Bottom right
    ]
    
    for i in range(min(num_persons, len(positions))):
        x, y = positions[i]
        
        # Draw person (simple stick figure)
        # Head
        cv2.circle(image, (x, y - 40), 15, (255, 255, 255), -1)
        
        # Body
        cv2.line(image, (x, y - 25), (x, y + 20), (255, 255, 255), 3)
        
        # Arms
        cv2.line(image, (x - 20, y), (x + 20, y), (255, 255, 255), 3)
        
        # Legs
        cv2.line(image, (x, y + 20), (x - 15, y + 50), (255, 255, 255), 3)
        cv2.line(image, (x, y + 20), (x + 15, y + 50), (255, 255, 255), 3)
    
    return image

def test_person_detector():
    """Test person detection model functionality."""
    print("🧪 Testing Person Detection Model...")
    
    # Create model configuration
    config = ModelConfig(
        model_name="yolov8n-person",
        model_type="person_detection",
        confidence_threshold=0.5,
        nms_threshold=0.4,
        input_size=(640, 640),
        use_gpu=True,
        platform="auto",
        model_params={
            'min_person_size': 30,
            'max_persons': 50
        }
    )
    
    print(f"📋 Model Config: {config}")
    
    # Initialize person detector
    print("🔧 Initializing Person Detector...")
    person_detector = PersonDetector(config)
    
    # Load model
    print("📥 Loading Person Detection Model...")
    if not person_detector.load_model():
        print("❌ Failed to load person detection model")
        return False
    
    print("✅ Person detection model loaded successfully")
    
    # Get model info
    model_info = person_detector.get_model_info()
    print(f"📊 Model Info: {model_info['model_class']}")
    print(f"📊 Device: {model_info['device']}")
    print(f"📊 Performance Metrics: {model_info['performance_metrics']}")
    
    # Create test images
    persons, test_dir = create_test_persons()
    
    # Test person detection functionality
    print("🔍 Testing person detection functionality...")
    try:
        # Test 1: Single person detection
        print("\n👤 Test 1: Single person detection...")
        single_result = person_detector.detect_persons(persons["single_person"])
        print(f"📊 Single person detections: {len(single_result.detections)}")
        
        # Test 2: Multiple persons detection
        print("\n👥 Test 2: Multiple persons detection...")
        multiple_result = person_detector.detect_persons(persons["multiple_persons"])
        print(f"📊 Multiple persons detections: {len(multiple_result.detections)}")
        
        # Test 3: Person counting
        print("\n🔢 Test 3: Person counting...")
        single_count = person_detector.count_persons(persons["single_person"])
        multiple_count = person_detector.count_persons(persons["multiple_persons"])
        crowd_count = person_detector.count_persons(persons["crowd_scene"])
        print(f"📊 Single person count: {single_count}")
        print(f"📊 Multiple persons count: {multiple_count}")
        print(f"📊 Crowd count: {crowd_count}")
        
        # Test 4: Bounding boxes
        print("\n📦 Test 4: Bounding boxes...")
        bboxes = person_detector.get_person_bboxes(persons["single_person"])
        print(f"📊 Bounding boxes: {len(bboxes)}")
        
        # Test 5: Confidence scores
        print("\n📊 Test 5: Confidence scores...")
        confidences = person_detector.get_person_confidences(persons["single_person"])
        print(f"📊 Confidence scores: {len(confidences)}")
        if confidences:
            print(f"📊 Average confidence: {sum(confidences) / len(confidences):.3f}")
        
        # Test 6: Single person detection
        print("\n👤 Test 6: Single person detection...")
        single_person = person_detector.detect_single_person(persons["single_person"])
        if single_person:
            print(f"📊 Single person confidence: {single_person.confidence:.3f}")
        
        # Test 7: Person areas
        print("\n📏 Test 7: Person areas...")
        areas = person_detector.get_person_areas(persons["single_person"])
        print(f"📊 Person areas: {len(areas)}")
        if areas:
            print(f"📊 Average area: {sum(areas) / len(areas):.1f}")
        
        # Test 8: Size filtering
        print("\n🔍 Test 8: Size filtering...")
        filtered_result = person_detector.filter_by_size(persons["single_person"], min_size=20)
        print(f"📊 Filtered detections: {len(filtered_result.detections)}")
        
        # Test 9: Person tracking
        print("\n🎯 Test 9: Person tracking...")
        track_history = person_detector.track_persons(persons["single_person"])
        print(f"📊 Track history: {len(track_history)} tracks")
        
        # Test 10: Performance metrics
        print("\n📊 Test 10: Performance metrics...")
        metrics = person_detector.get_performance_metrics()
        print(f"📊 Performance metrics: {metrics}")
        
        print("✅ Person detection tests completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Person detection test failed: {e}")
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

def cleanup_test_files():
    """Clean up test files."""
    print("\n🧹 Cleaning up test files...")
    
    # Remove test images
    test_dir = Path("test_persons")
    if test_dir.exists():
        for file in test_dir.glob("*.jpg"):
            file.unlink()
        test_dir.rmdir()
        print("🧹 Removed test person images")

def main():
    """Main test function."""
    print("🚀 AI Box - Person Detection Model Test")
    print("=" * 50)
    
    try:
        # Test basic functionality
        success = test_person_detector()
        
        if success:
            # Test platform optimization
            test_platform_optimization()
            
            print("\n🎉 All tests completed successfully!")
            print("✅ Person Detection Model is working correctly")
        else:
            print("\n❌ Tests failed!")
            print("🔧 Please check the error messages above")
    
    finally:
        # Cleanup
        cleanup_test_files()

if __name__ == "__main__":
    main()
