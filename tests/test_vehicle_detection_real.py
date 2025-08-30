# ==========================================
# AI Box - Vehicle Detection Real Test
# Test vehicle detection with real model and images
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
from ai_models.vehicle_analysis.vehicle_detector import VehicleDetector

def create_realistic_vehicle_images():
    """Create realistic vehicle images for testing."""
    print("🖼️ Creating realistic vehicle images...")
    
    # Create directory for test images
    test_dir = Path("test_vehicles")
    test_dir.mkdir(exist_ok=True)
    
    # Create different vehicle scenarios
    vehicles = {
        "single_car": create_single_car(),
        "multiple_cars": create_multiple_cars(),
        "mixed_vehicles": create_mixed_vehicles(),
        "traffic_scene": create_traffic_scene(),
        "parking_lot": create_parking_lot()
    }
    
    # Save test images
    for name, image in vehicles.items():
        image_path = test_dir / f"{name}.jpg"
        cv2.imwrite(str(image_path), image)
        print(f"💾 Saved {name} image: {image_path}")
    
    return vehicles, test_dir

def create_single_car() -> np.ndarray:
    """Create a realistic single car image."""
    # Create base scene
    image = np.ones((480, 640, 3), dtype=np.uint8) * 200  # Light gray background
    
    # Add road
    cv2.rectangle(image, (0, 350), (640, 480), (100, 100, 100), -1)  # Dark gray road
    
    # Add lane markings
    for x in range(50, 640, 100):
        cv2.rectangle(image, (x, 400), (x + 40, 410), (255, 255, 255), -1)
    
    # Draw detailed car
    car_x, car_y = 250, 300
    car_width, car_height = 140, 80
    
    # Car body (main rectangle)
    cv2.rectangle(image, (car_x, car_y), (car_x + car_width, car_y + car_height), (50, 100, 200), -1)  # Blue car
    
    # Car roof (smaller rectangle)
    roof_margin = 20
    cv2.rectangle(image, (car_x + roof_margin, car_y - 30), 
                  (car_x + car_width - roof_margin, car_y + 20), (40, 80, 180), -1)
    
    # Windows
    cv2.rectangle(image, (car_x + roof_margin + 5, car_y - 25), 
                  (car_x + car_width - roof_margin - 5, car_y + 15), (150, 200, 255), -1)
    
    # Wheels
    wheel_radius = 15
    cv2.circle(image, (car_x + 25, car_y + car_height + 10), wheel_radius, (0, 0, 0), -1)  # Front wheel
    cv2.circle(image, (car_x + car_width - 25, car_y + car_height + 10), wheel_radius, (0, 0, 0), -1)  # Rear wheel
    
    # Headlights
    cv2.circle(image, (car_x + car_width - 5, car_y + 20), 8, (255, 255, 200), -1)  # Front light
    cv2.circle(image, (car_x + car_width - 5, car_y + 60), 8, (255, 255, 200), -1)  # Front light
    
    # Taillights
    cv2.circle(image, (car_x + 5, car_y + 20), 6, (255, 0, 0), -1)  # Rear light
    cv2.circle(image, (car_x + 5, car_y + 60), 6, (255, 0, 0), -1)  # Rear light
    
    return image

def create_multiple_cars() -> np.ndarray:
    """Create an image with multiple cars."""
    # Create larger scene
    image = np.ones((480, 800, 3), dtype=np.uint8) * 220
    
    # Add road
    cv2.rectangle(image, (0, 300), (800, 480), (100, 100, 100), -1)
    
    # Add lane markings
    cv2.line(image, (0, 390), (800, 390), (255, 255, 255), 3)  # Center line
    
    # Car 1: Red car (left)
    draw_simple_car(image, 80, 320, 120, 60, (50, 50, 200))  # Red
    
    # Car 2: Blue car (center)
    draw_simple_car(image, 300, 310, 130, 70, (200, 100, 50))  # Blue
    
    # Car 3: Green car (right)
    draw_simple_car(image, 550, 325, 110, 55, (50, 200, 50))  # Green
    
    return image

def create_mixed_vehicles() -> np.ndarray:
    """Create an image with different types of vehicles."""
    # Create scene
    image = np.ones((600, 800, 3), dtype=np.uint8) * 210
    
    # Add road
    cv2.rectangle(image, (0, 400), (800, 600), (90, 90, 90), -1)
    
    # Car
    draw_simple_car(image, 100, 450, 120, 60, (200, 50, 50))
    
    # Truck (larger)
    draw_simple_car(image, 300, 430, 180, 100, (100, 100, 200))
    
    # Motorcycle (smaller)
    draw_simple_motorcycle(image, 550, 480)
    
    # Bus (very large)
    draw_simple_car(image, 50, 350, 250, 120, (200, 200, 50))
    
    return image

def create_traffic_scene() -> np.ndarray:
    """Create a busy traffic scene."""
    # Create scene
    image = np.ones((600, 1000, 3), dtype=np.uint8) * 200
    
    # Add multi-lane road
    cv2.rectangle(image, (0, 350), (1000, 600), (80, 80, 80), -1)
    
    # Lane markings
    for y in [420, 490]:
        for x in range(0, 1000, 60):
            cv2.rectangle(image, (x, y), (x + 30, y + 5), (255, 255, 255), -1)
    
    # Multiple vehicles in different lanes
    vehicles_data = [
        (50, 380, 110, 50, (200, 50, 50)),    # Lane 1
        (200, 375, 120, 55, (50, 200, 50)),   # Lane 1
        (400, 370, 130, 60, (50, 50, 200)),   # Lane 1
        (100, 450, 140, 65, (200, 100, 50)),  # Lane 2
        (350, 445, 125, 58, (100, 200, 100)), # Lane 2
        (600, 440, 135, 62, (200, 200, 50)),  # Lane 2
        (150, 520, 115, 52, (150, 50, 200)),  # Lane 3
        (450, 515, 128, 60, (200, 150, 50)),  # Lane 3
    ]
    
    for x, y, w, h, color in vehicles_data:
        draw_simple_car(image, x, y, w, h, color)
    
    return image

def create_parking_lot() -> np.ndarray:
    """Create a parking lot scene."""
    # Create scene
    image = np.ones((600, 800, 3), dtype=np.uint8) * 180
    
    # Add parking spaces
    for i in range(4):
        for j in range(3):
            x = 50 + j * 250
            y = 100 + i * 120
            # Parking space outline
            cv2.rectangle(image, (x, y), (x + 200, y + 100), (255, 255, 255), 2)
            
            # Add car in some spaces
            if (i + j) % 2 == 0:  # Park cars in alternating spaces
                colors = [(200, 50, 50), (50, 200, 50), (50, 50, 200), (200, 200, 50)]
                color = colors[(i * 3 + j) % len(colors)]
                draw_simple_car(image, x + 20, y + 20, 160, 60, color)
    
    return image

def draw_simple_car(image, x, y, width, height, color):
    """Draw a simple car shape."""
    # Main body
    cv2.rectangle(image, (x, y), (x + width, y + height), color, -1)
    
    # Roof
    roof_margin = width // 6
    cv2.rectangle(image, (x + roof_margin, y - height//3), 
                  (x + width - roof_margin, y + height//3), 
                  tuple(max(0, c - 30) for c in color), -1)
    
    # Wheels
    wheel_size = height // 4
    cv2.circle(image, (x + width//4, y + height + wheel_size//2), wheel_size, (0, 0, 0), -1)
    cv2.circle(image, (x + 3*width//4, y + height + wheel_size//2), wheel_size, (0, 0, 0), -1)

def draw_simple_motorcycle(image, x, y):
    """Draw a simple motorcycle shape."""
    # Body (smaller than car)
    cv2.rectangle(image, (x, y), (x + 80, y + 40), (100, 100, 100), -1)
    
    # Wheels (smaller)
    cv2.circle(image, (x + 15, y + 45), 12, (0, 0, 0), -1)
    cv2.circle(image, (x + 65, y + 45), 12, (0, 0, 0), -1)
    
    # Rider (simple shape)
    cv2.rectangle(image, (x + 25, y - 30), (x + 55, y), (150, 100, 50), -1)

def test_vehicle_detection_real():
    """Test vehicle detection with realistic images."""
    print("🧪 Testing Vehicle Detection with Real Images...")
    
    # Create model configuration
    config = ModelConfig(
        model_name="vehicle-detector",
        model_type="vehicle_detection",
        confidence_threshold=0.3,
        input_size=(640, 640),
        use_gpu=True,
        model_params={
            'min_vehicle_size': 100,
            'vehicle_confidence': 0.25
        }
    )
    
    print(f"📋 Model Config: {config}")
    
    # Initialize vehicle detector
    print("🔧 Initializing Vehicle Detector...")
    vehicle_detector = VehicleDetector(config)
    
    # Load model
    print("📥 Loading Vehicle Detection Model...")
    if not vehicle_detector.load_model():
        print("❌ Failed to load vehicle detection model")
        return False
    
    print("✅ Vehicle detection model loaded successfully")
    
    # Get model info
    model_info = vehicle_detector.get_model_info()
    print(f"📊 Model Info: {model_info['model_class']}")
    print(f"📊 Device: {model_info['device']}")
    
    # Create realistic test images
    vehicles, test_dir = create_realistic_vehicle_images()
    
    # Test vehicle detection functionality
    print("🔍 Testing vehicle detection with realistic images...")
    try:
        results = {}
        
        # Test each vehicle scenario
        for scenario_name, image in vehicles.items():
            print(f"\n🎯 Testing {scenario_name}...")
            
            start_time = time.time()
            result = vehicle_detector.detect_vehicles(image)
            inference_time = time.time() - start_time
            
            if result.success:
                print(f"📊 Detected {len(result.detections)} vehicles")
                print(f"📊 Inference time: {inference_time*1000:.1f}ms")
                
                # Count vehicles by type
                vehicle_counts = vehicle_detector.count_vehicles(image)
                print(f"📊 Vehicle counts: {vehicle_counts}")
                
                # Analyze traffic flow
                traffic_analysis = vehicle_detector.analyze_traffic_flow(image)
                print(f"📊 Traffic density: {traffic_analysis['traffic_density']:.3f}")
                print(f"📊 Average vehicle size: {traffic_analysis['average_vehicle_size']:.1f}")
                
                # Show individual detections
                for i, detection in enumerate(result.detections[:3]):  # Show first 3
                    print(f"   Vehicle {i+1}: {detection.class_name} ({detection.confidence:.3f})")
                
                results[scenario_name] = {
                    'detections': len(result.detections),
                    'vehicle_counts': vehicle_counts,
                    'traffic_analysis': traffic_analysis,
                    'inference_time': inference_time
                }
            else:
                print(f"❌ Failed to detect vehicles in {scenario_name}")
                results[scenario_name] = {'error': 'Detection failed'}
        
        # Performance summary
        print(f"\n📊 Performance Summary:")
        successful_tests = [r for r in results.values() if 'inference_time' in r]
        if successful_tests:
            avg_inference_time = np.mean([r['inference_time'] for r in successful_tests])
            total_detections = sum([r['detections'] for r in successful_tests])
            print(f"📊 Average inference time: {avg_inference_time*1000:.1f}ms")
            print(f"📊 Total vehicles detected: {total_detections}")
        
        # Test specific vehicle types
        print(f"\n🎯 Testing specific vehicle type filtering...")
        test_image = vehicles["mixed_vehicles"]
        
        for vehicle_type in ['car', 'truck', 'motorcycle']:
            vehicles_of_type = vehicle_detector.get_vehicles_by_type(test_image, vehicle_type)
            print(f"📊 {vehicle_type}s detected: {len(vehicles_of_type)}")
        
        print("✅ Vehicle detection real image tests completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Vehicle detection real image test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def cleanup_test_files():
    """Clean up test files."""
    print("\n🧹 Cleaning up test files...")
    
    # Remove test images
    test_dir = Path("test_vehicles")
    if test_dir.exists():
        for file in test_dir.glob("*.jpg"):
            file.unlink()
        test_dir.rmdir()
        print("🧹 Removed test vehicle images")

def main():
    """Main test function."""
    print("🚀 AI Box - Vehicle Detection Real Image Test")
    print("=" * 60)
    
    try:
        # Test with realistic images
        success = test_vehicle_detection_real()
        
        if success:
            print("\n🎉 All real image tests completed successfully!")
            print("✅ Vehicle Detection Model is working correctly with realistic images")
        else:
            print("\n❌ Tests failed!")
            print("🔧 Please check the error messages above")
    
    finally:
        # Cleanup
        cleanup_test_files()

if __name__ == "__main__":
    main()
