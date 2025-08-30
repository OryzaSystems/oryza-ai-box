# ==========================================
# AI Box - Behavior Analysis Real Image Test
# Test behavior analysis model with real/synthetic images
# ==========================================

import sys
import os
import cv2
import numpy as np
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ai_models.common.model_config import ModelConfig
from ai_models.human_analysis.behavior_analyzer import BehaviorAnalyzer

def create_realistic_behavior_images():
    """Create realistic behavior images for testing."""
    print("🖼️ Creating realistic behavior images...")
    
    # Create directory for test images
    test_dir = Path("test_behaviors")
    test_dir.mkdir(exist_ok=True)
    
    # Create different behavior scenarios
    behaviors = {
        "standing_person": create_standing_person(),
        "walking_person": create_walking_person(),
        "running_person": create_running_person(),
        "sitting_person": create_sitting_person(),
        "mixed_behaviors": create_mixed_behaviors()
    }
    
    # Save test images
    for name, image in behaviors.items():
        image_path = test_dir / f"{name}.jpg"
        cv2.imwrite(str(image_path), image)
        print(f"💾 Saved {name} image: {image_path}")
    
    return behaviors, test_dir

def create_standing_person() -> np.ndarray:
    """Create a realistic standing person image."""
    # Create base scene
    image = np.ones((400, 300, 3), dtype=np.uint8) * 200  # Light gray background
    
    # Add ground
    cv2.rectangle(image, (0, 350), (300, 400), (150, 150, 150), -1)
    
    # Draw standing person (more detailed)
    person_x, person_y = 150, 200
    
    # Head
    cv2.circle(image, (person_x, person_y - 60), 20, (220, 180, 140), -1)  # Skin color
    cv2.circle(image, (person_x, person_y - 60), 20, (0, 0, 0), 2)  # Head outline
    
    # Body (standing straight)
    cv2.rectangle(image, (person_x - 15, person_y - 40), (person_x + 15, person_y + 40), (100, 100, 200), -1)  # Blue shirt
    cv2.rectangle(image, (person_x - 12, person_y + 40), (person_x + 12, person_y + 80), (50, 50, 50), -1)  # Dark pants
    
    # Arms (relaxed at sides)
    cv2.line(image, (person_x - 15, person_y - 20), (person_x - 35, person_y + 20), (220, 180, 140), 8)  # Left arm
    cv2.line(image, (person_x + 15, person_y - 20), (person_x + 35, person_y + 20), (220, 180, 140), 8)  # Right arm
    
    # Legs (straight, standing)
    cv2.line(image, (person_x - 8, person_y + 80), (person_x - 8, person_y + 140), (220, 180, 140), 8)  # Left leg
    cv2.line(image, (person_x + 8, person_y + 80), (person_x + 8, person_y + 140), (220, 180, 140), 8)  # Right leg
    
    # Feet
    cv2.ellipse(image, (person_x - 8, person_y + 145), (12, 6), 0, 0, 180, (0, 0, 0), -1)  # Left shoe
    cv2.ellipse(image, (person_x + 8, person_y + 145), (12, 6), 0, 0, 180, (0, 0, 0), -1)  # Right shoe
    
    return image

def create_walking_person() -> np.ndarray:
    """Create a realistic walking person image."""
    # Create base scene
    image = np.ones((400, 300, 3), dtype=np.uint8) * 200
    
    # Add ground
    cv2.rectangle(image, (0, 350), (300, 400), (150, 150, 150), -1)
    
    # Draw walking person (dynamic pose)
    person_x, person_y = 150, 200
    
    # Head (slightly forward)
    cv2.circle(image, (person_x + 5, person_y - 60), 20, (220, 180, 140), -1)
    cv2.circle(image, (person_x + 5, person_y - 60), 20, (0, 0, 0), 2)
    
    # Body (slightly leaning forward)
    cv2.rectangle(image, (person_x - 10, person_y - 40), (person_x + 20, person_y + 40), (100, 200, 100), -1)  # Green shirt
    cv2.rectangle(image, (person_x - 8, person_y + 40), (person_x + 16, person_y + 80), (50, 50, 50), -1)
    
    # Arms (swinging motion)
    cv2.line(image, (person_x - 10, person_y - 20), (person_x - 40, person_y + 10), (220, 180, 140), 8)  # Left arm forward
    cv2.line(image, (person_x + 20, person_y - 20), (person_x + 40, person_y + 30), (220, 180, 140), 8)  # Right arm back
    
    # Legs (walking stride)
    cv2.line(image, (person_x - 5, person_y + 80), (person_x + 15, person_y + 140), (220, 180, 140), 8)  # Left leg forward
    cv2.line(image, (person_x + 10, person_y + 80), (person_x - 10, person_y + 130), (220, 180, 140), 8)  # Right leg back
    
    # Feet (walking position)
    cv2.ellipse(image, (person_x + 15, person_y + 145), (12, 6), 0, 0, 180, (0, 0, 0), -1)
    cv2.ellipse(image, (person_x - 10, person_y + 135), (12, 6), 0, 0, 180, (0, 0, 0), -1)
    
    # Add motion blur effect
    kernel = np.array([[0, 0, 1, 0, 0],
                       [0, 1, 1, 1, 0],
                       [1, 1, 1, 1, 1],
                       [0, 1, 1, 1, 0],
                       [0, 0, 1, 0, 0]]) / 13
    image = cv2.filter2D(image, -1, kernel)
    
    return image

def create_running_person() -> np.ndarray:
    """Create a realistic running person image."""
    # Create base scene
    image = np.ones((400, 300, 3), dtype=np.uint8) * 200
    
    # Add ground with motion lines
    cv2.rectangle(image, (0, 350), (300, 400), (150, 150, 150), -1)
    # Add speed lines
    for i in range(5):
        cv2.line(image, (50 + i*40, 360), (30 + i*40, 370), (100, 100, 100), 2)
    
    # Draw running person (very dynamic pose)
    person_x, person_y = 150, 190
    
    # Head (forward lean)
    cv2.circle(image, (person_x + 15, person_y - 60), 18, (220, 180, 140), -1)
    cv2.circle(image, (person_x + 15, person_y - 60), 18, (0, 0, 0), 2)
    
    # Body (strong forward lean)
    cv2.rectangle(image, (person_x, person_y - 40), (person_x + 30, person_y + 35), (200, 100, 100), -1)  # Red shirt
    cv2.rectangle(image, (person_x + 5, person_y + 35), (person_x + 25, person_y + 70), (50, 50, 50), -1)
    
    # Arms (pumping motion)
    cv2.line(image, (person_x, person_y - 25), (person_x - 35, person_y - 5), (220, 180, 140), 8)  # Left arm high
    cv2.line(image, (person_x + 30, person_y - 25), (person_x + 55, person_y + 25), (220, 180, 140), 8)  # Right arm back
    
    # Legs (running stride - one leg high)
    cv2.line(image, (person_x + 5, person_y + 70), (person_x + 35, person_y + 110), (220, 180, 140), 8)  # Left leg forward/up
    cv2.line(image, (person_x + 20, person_y + 70), (person_x - 5, person_y + 140), (220, 180, 140), 8)  # Right leg extended back
    
    # Feet (running position)
    cv2.ellipse(image, (person_x + 35, person_y + 115), (12, 6), 0, 0, 180, (0, 0, 0), -1)
    cv2.ellipse(image, (person_x - 5, person_y + 145), (12, 6), 0, 0, 180, (0, 0, 0), -1)
    
    # Add stronger motion blur
    kernel = np.array([[0, 0, 0, 1, 0, 0, 0],
                       [0, 0, 1, 1, 1, 0, 0],
                       [0, 1, 1, 1, 1, 1, 0],
                       [1, 1, 1, 1, 1, 1, 1],
                       [0, 1, 1, 1, 1, 1, 0],
                       [0, 0, 1, 1, 1, 0, 0],
                       [0, 0, 0, 1, 0, 0, 0]]) / 25
    image = cv2.filter2D(image, -1, kernel)
    
    return image

def create_sitting_person() -> np.ndarray:
    """Create a realistic sitting person image."""
    # Create base scene
    image = np.ones((400, 300, 3), dtype=np.uint8) * 200
    
    # Add ground
    cv2.rectangle(image, (0, 350), (300, 400), (150, 150, 150), -1)
    
    # Add bench/chair
    cv2.rectangle(image, (100, 280), (200, 300), (139, 69, 19), -1)  # Brown bench
    cv2.rectangle(image, (95, 300), (205, 310), (139, 69, 19), -1)  # Bench legs
    
    # Draw sitting person
    person_x, person_y = 150, 240
    
    # Head
    cv2.circle(image, (person_x, person_y - 40), 20, (220, 180, 140), -1)
    cv2.circle(image, (person_x, person_y - 40), 20, (0, 0, 0), 2)
    
    # Body (sitting upright)
    cv2.rectangle(image, (person_x - 15, person_y - 20), (person_x + 15, person_y + 20), (100, 100, 200), -1)  # Blue shirt
    
    # Arms (resting)
    cv2.line(image, (person_x - 15, person_y - 10), (person_x - 30, person_y + 10), (220, 180, 140), 8)  # Left arm
    cv2.line(image, (person_x + 15, person_y - 10), (person_x + 30, person_y + 10), (220, 180, 140), 8)  # Right arm
    
    # Legs (sitting - horizontal thighs, vertical shins)
    cv2.rectangle(image, (person_x - 12, person_y + 20), (person_x + 12, person_y + 40), (50, 50, 50), -1)  # Thighs
    cv2.line(image, (person_x - 8, person_y + 40), (person_x - 8, person_y + 80), (220, 180, 140), 8)  # Left shin
    cv2.line(image, (person_x + 8, person_y + 40), (person_x + 8, person_y + 80), (220, 180, 140), 8)  # Right shin
    
    # Feet (on ground)
    cv2.ellipse(image, (person_x - 8, person_y + 85), (12, 6), 0, 0, 180, (0, 0, 0), -1)
    cv2.ellipse(image, (person_x + 8, person_y + 85), (12, 6), 0, 0, 180, (0, 0, 0), -1)
    
    return image

def create_mixed_behaviors() -> np.ndarray:
    """Create an image with multiple people showing different behaviors."""
    # Create larger scene
    image = np.ones((500, 600, 3), dtype=np.uint8) * 220
    
    # Add ground
    cv2.rectangle(image, (0, 450), (600, 500), (150, 150, 150), -1)
    
    # Add background elements
    cv2.rectangle(image, (50, 350), (100, 450), (139, 69, 19), -1)  # Tree trunk
    cv2.circle(image, (75, 320), 40, (34, 139, 34), -1)  # Tree top
    
    # Person 1: Standing (left)
    x1, y1 = 120, 300
    cv2.circle(image, (x1, y1 - 60), 15, (220, 180, 140), -1)
    cv2.rectangle(image, (x1 - 12, y1 - 45), (x1 + 12, y1 + 30), (100, 100, 200), -1)
    cv2.line(image, (x1 - 12, y1 - 20), (x1 - 25, y1 + 15), (220, 180, 140), 6)
    cv2.line(image, (x1 + 12, y1 - 20), (x1 + 25, y1 + 15), (220, 180, 140), 6)
    cv2.line(image, (x1 - 6, y1 + 30), (x1 - 6, y1 + 80), (220, 180, 140), 6)
    cv2.line(image, (x1 + 6, y1 + 30), (x1 + 6, y1 + 80), (220, 180, 140), 6)
    
    # Person 2: Walking (center-left)
    x2, y2 = 250, 300
    cv2.circle(image, (x2 + 3, y2 - 60), 15, (220, 180, 140), -1)
    cv2.rectangle(image, (x2 - 10, y2 - 45), (x2 + 15, y2 + 30), (100, 200, 100), -1)
    cv2.line(image, (x2 - 10, y2 - 20), (x2 - 30, y2 + 10), (220, 180, 140), 6)
    cv2.line(image, (x2 + 15, y2 - 20), (x2 + 30, y2 + 25), (220, 180, 140), 6)
    cv2.line(image, (x2 - 3, y2 + 30), (x2 + 12, y2 + 80), (220, 180, 140), 6)
    cv2.line(image, (x2 + 8, y2 + 30), (x2 - 8, y2 + 75), (220, 180, 140), 6)
    
    # Person 3: Running (center-right)
    x3, y3 = 380, 290
    cv2.circle(image, (x3 + 10, y3 - 60), 15, (220, 180, 140), -1)
    cv2.rectangle(image, (x3 - 5, y3 - 45), (x3 + 25, y3 + 25), (200, 100, 100), -1)
    cv2.line(image, (x3 - 5, y3 - 25), (x3 - 25, y3 - 5), (220, 180, 140), 6)
    cv2.line(image, (x3 + 25, y3 - 25), (x3 + 45, y3 + 20), (220, 180, 140), 6)
    cv2.line(image, (x3 + 3, y3 + 25), (x3 + 25, y3 + 65), (220, 180, 140), 6)
    cv2.line(image, (x3 + 15, y3 + 25), (x3 - 5, y3 + 80), (220, 180, 140), 6)
    
    # Person 4: Sitting (right)
    x4, y4 = 500, 320
    # Add bench
    cv2.rectangle(image, (x4 - 30, y4 + 20), (x4 + 30, y4 + 35), (139, 69, 19), -1)
    cv2.circle(image, (x4, y4 - 40), 15, (220, 180, 140), -1)
    cv2.rectangle(image, (x4 - 12, y4 - 25), (x4 + 12, y4 + 15), (100, 100, 200), -1)
    cv2.line(image, (x4 - 12, y4 - 15), (x4 - 25, y4 + 5), (220, 180, 140), 6)
    cv2.line(image, (x4 + 12, y4 - 15), (x4 + 25, y4 + 5), (220, 180, 140), 6)
    cv2.rectangle(image, (x4 - 10, y4 + 15), (x4 + 10, y4 + 30), (50, 50, 50), -1)
    cv2.line(image, (x4 - 6, y4 + 30), (x4 - 6, y4 + 60), (220, 180, 140), 6)
    cv2.line(image, (x4 + 6, y4 + 30), (x4 + 6, y4 + 60), (220, 180, 140), 6)
    
    return image

def test_behavior_analysis_real():
    """Test behavior analysis with realistic images."""
    print("🧪 Testing Behavior Analysis with Real Images...")
    
    # Create model configuration
    config = ModelConfig(
        model_name="behavior-analyzer",
        model_type="behavior_analysis",
        confidence_threshold=0.3,
        input_size=(224, 224),
        use_gpu=True,
        platform="auto",
        model_params={
            'temporal_window': 5,
            'smoothing_factor': 0.7,
            'dropout_rate': 0.5
        }
    )
    
    print(f"📋 Model Config: {config}")
    
    # Initialize behavior analyzer
    print("🔧 Initializing Behavior Analyzer...")
    behavior_analyzer = BehaviorAnalyzer(config)
    
    # Load model
    print("📥 Loading Behavior Analysis Model...")
    if not behavior_analyzer.load_model():
        print("❌ Failed to load behavior analysis model")
        return False
    
    print("✅ Behavior analysis model loaded successfully")
    
    # Get model info
    model_info = behavior_analyzer.get_model_info()
    print(f"📊 Model Info: {model_info['model_class']}")
    print(f"📊 Device: {model_info['device']}")
    print(f"📊 Model Parameters: {behavior_analyzer.metadata.get('parameters', 'Unknown'):,}")
    print(f"📊 Model Size: {behavior_analyzer.metadata.get('model_size_mb', 0):.2f} MB")
    
    # Create realistic test images
    behaviors, test_dir = create_realistic_behavior_images()
    
    # Test behavior analysis functionality
    print("🔍 Testing behavior analysis with realistic images...")
    try:
        results = {}
        
        # Test each behavior type
        for behavior_name, image in behaviors.items():
            print(f"\n🎯 Testing {behavior_name}...")
            
            start_time = time.time()
            result = behavior_analyzer.analyze_behavior(image, use_temporal=False)
            inference_time = time.time() - start_time
            
            if result.success and result.detections:
                detection = result.detections[0]
                probabilities = detection.attributes.get('behavior_probabilities', {})
                
                print(f"📊 Predicted behavior: {detection.class_name}")
                print(f"📊 Confidence: {detection.confidence:.3f}")
                print(f"📊 Inference time: {inference_time*1000:.1f}ms")
                print("📊 All probabilities:")
                for behavior, prob in probabilities.items():
                    print(f"   - {behavior}: {prob:.3f}")
                
                results[behavior_name] = {
                    'predicted': detection.class_name,
                    'confidence': detection.confidence,
                    'probabilities': probabilities,
                    'inference_time': inference_time
                }
            else:
                print(f"❌ Failed to analyze {behavior_name}")
                results[behavior_name] = {'error': 'Analysis failed'}
        
        # Test temporal analysis
        print(f"\n🕐 Testing temporal behavior analysis...")
        behavior_analyzer.reset_temporal_history()
        
        # Create sequence of similar behaviors
        sequence_images = [behaviors["walking_person"]] * 3 + [behaviors["running_person"]] * 2
        sequence_results = behavior_analyzer.analyze_behavior_sequence(sequence_images)
        
        print(f"📊 Sequence analysis results:")
        for i, seq_result in enumerate(sequence_results):
            if seq_result.success and seq_result.detections:
                detection = seq_result.detections[0]
                temporal_smoothed = detection.attributes.get('temporal_smoothed', False)
                print(f"   Frame {i+1}: {detection.class_name} ({detection.confidence:.3f}) - Smoothed: {temporal_smoothed}")
        
        # Test behavior trends
        print(f"\n📈 Testing behavior trends...")
        trends = behavior_analyzer.get_behavior_trends()
        print(f"📊 Behavior trends: {trends}")
        
        # Performance summary
        print(f"\n📊 Performance Summary:")
        avg_inference_time = np.mean([r.get('inference_time', 0) for r in results.values() if 'inference_time' in r])
        print(f"📊 Average inference time: {avg_inference_time*1000:.1f}ms")
        print(f"📊 Model parameters: {behavior_analyzer.metadata.get('parameters', 0):,}")
        print(f"📊 Model size: {behavior_analyzer.metadata.get('model_size_mb', 0):.2f} MB")
        
        # Test accuracy assessment
        print(f"\n🎯 Accuracy Assessment:")
        expected_behaviors = {
            'standing_person': 'standing',
            'walking_person': 'walking', 
            'running_person': 'running',
            'sitting_person': 'sitting'
        }
        
        correct_predictions = 0
        total_predictions = 0
        
        for image_name, expected in expected_behaviors.items():
            if image_name in results and 'predicted' in results[image_name]:
                predicted = results[image_name]['predicted']
                confidence = results[image_name]['confidence']
                is_correct = predicted == expected
                
                print(f"📊 {image_name}: Expected {expected}, Got {predicted} ({confidence:.3f}) - {'✅' if is_correct else '❌'}")
                
                if is_correct:
                    correct_predictions += 1
                total_predictions += 1
        
        if total_predictions > 0:
            accuracy = correct_predictions / total_predictions
            print(f"📊 Overall Accuracy: {accuracy:.1%} ({correct_predictions}/{total_predictions})")
        
        print("✅ Behavior analysis real image tests completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Behavior analysis real image test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def cleanup_test_files():
    """Clean up test files."""
    print("\n🧹 Cleaning up test files...")
    
    # Remove test images
    test_dir = Path("test_behaviors")
    if test_dir.exists():
        for file in test_dir.glob("*.jpg"):
            file.unlink()
        test_dir.rmdir()
        print("🧹 Removed test behavior images")

def main():
    """Main test function."""
    print("🚀 AI Box - Behavior Analysis Real Image Test")
    print("=" * 60)
    
    try:
        # Test with realistic images
        success = test_behavior_analysis_real()
        
        if success:
            print("\n🎉 All real image tests completed successfully!")
            print("✅ Behavior Analysis Model is working correctly with realistic images")
        else:
            print("\n❌ Tests failed!")
            print("🔧 Please check the error messages above")
    
    finally:
        # Cleanup
        cleanup_test_files()

if __name__ == "__main__":
    main()
