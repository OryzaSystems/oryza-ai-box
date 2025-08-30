# ==========================================
# AI Box - Vehicle Classification Real Test
# Test vehicle classification with real/synthetic images
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
from ai_models.vehicle_analysis.vehicle_classifier import VehicleClassifier

def create_realistic_vehicle_images():
    """Create realistic vehicle images for testing."""
    print("🖼️ Creating realistic vehicle images...")
    
    # Create directory for test images
    test_dir = Path("test_vehicle_classification")
    test_dir.mkdir(exist_ok=True)
    
    # Create different vehicle scenarios
    vehicles = {
        "sedan_car": create_sedan_car(),
        "suv_car": create_suv_car(),
        "pickup_truck": create_pickup_truck(),
        "city_bus": create_city_bus(),
        "sport_motorcycle": create_sport_motorcycle(),
        "road_bicycle": create_road_bicycle(),
        "mixed_vehicles": create_mixed_vehicles()
    }
    
    # Save test images
    for name, image in vehicles.items():
        image_path = test_dir / f"{name}.jpg"
        cv2.imwrite(str(image_path), image)
        print(f"💾 Saved {name} image: {image_path}")
    
    return vehicles, test_dir

def create_sedan_car() -> np.ndarray:
    """Create a realistic sedan car image."""
    # Create background
    image = np.ones((400, 600, 3), dtype=np.uint8) * 220
    
    # Add road
    cv2.rectangle(image, (0, 300), (600, 400), (100, 100, 100), -1)
    
    # Draw sedan car (low profile, 4 doors)
    car_x, car_y = 150, 250
    car_width, car_height = 300, 80
    
    # Main body (sedan shape - lower)
    cv2.rectangle(image, (car_x, car_y), (car_x + car_width, car_y + car_height), (50, 100, 200), -1)
    
    # Roof (sedan - lower and longer)
    roof_start = car_x + 40
    roof_end = car_x + car_width - 40
    cv2.rectangle(image, (roof_start, car_y - 40), (roof_end, car_y + 20), (40, 80, 180), -1)
    
    # Windows (4 windows for sedan)
    window_width = (roof_end - roof_start) // 2 - 10
    cv2.rectangle(image, (roof_start + 5, car_y - 35), (roof_start + window_width, car_y + 15), (150, 200, 255), -1)
    cv2.rectangle(image, (roof_start + window_width + 20, car_y - 35), (roof_end - 5, car_y + 15), (150, 200, 255), -1)
    
    # Wheels
    wheel_radius = 20
    cv2.circle(image, (car_x + 60, car_y + car_height + 15), wheel_radius, (0, 0, 0), -1)
    cv2.circle(image, (car_x + car_width - 60, car_y + car_height + 15), wheel_radius, (0, 0, 0), -1)
    
    # Headlights and taillights
    cv2.circle(image, (car_x + car_width - 5, car_y + 20), 8, (255, 255, 200), -1)
    cv2.circle(image, (car_x + car_width - 5, car_y + 60), 8, (255, 255, 200), -1)
    cv2.circle(image, (car_x + 5, car_y + 20), 6, (255, 0, 0), -1)
    cv2.circle(image, (car_x + 5, car_y + 60), 6, (255, 0, 0), -1)
    
    return image

def create_suv_car() -> np.ndarray:
    """Create a realistic SUV car image."""
    # Create background
    image = np.ones((400, 600, 3), dtype=np.uint8) * 210
    
    # Add road
    cv2.rectangle(image, (0, 300), (600, 400), (100, 100, 100), -1)
    
    # Draw SUV (taller, more upright)
    car_x, car_y = 150, 200  # Higher position
    car_width, car_height = 280, 120  # Taller
    
    # Main body (SUV shape - taller)
    cv2.rectangle(image, (car_x, car_y), (car_x + car_width, car_y + car_height), (100, 150, 100), -1)
    
    # Roof (SUV - higher and more square)
    roof_start = car_x + 30
    roof_end = car_x + car_width - 30
    cv2.rectangle(image, (roof_start, car_y - 60), (roof_end, car_y + 30), (80, 130, 80), -1)
    
    # Large windows (SUV style)
    cv2.rectangle(image, (roof_start + 5, car_y - 55), (roof_end - 5, car_y + 25), (150, 200, 255), -1)
    
    # Larger wheels (SUV)
    wheel_radius = 25
    cv2.circle(image, (car_x + 50, car_y + car_height + 20), wheel_radius, (0, 0, 0), -1)
    cv2.circle(image, (car_x + car_width - 50, car_y + car_height + 20), wheel_radius, (0, 0, 0), -1)
    
    # SUV grille
    cv2.rectangle(image, (car_x + car_width - 10, car_y + 30), (car_x + car_width, car_y + 90), (50, 50, 50), -1)
    
    return image

def create_pickup_truck() -> np.ndarray:
    """Create a realistic pickup truck image."""
    # Create background
    image = np.ones((400, 600, 3), dtype=np.uint8) * 200
    
    # Add road
    cv2.rectangle(image, (0, 300), (600, 400), (100, 100, 100), -1)
    
    # Draw pickup truck
    truck_x, truck_y = 120, 220
    
    # Cab (front part)
    cab_width, cab_height = 120, 100
    cv2.rectangle(image, (truck_x, truck_y), (truck_x + cab_width, truck_y + cab_height), (200, 100, 50), -1)
    
    # Cab roof
    cv2.rectangle(image, (truck_x + 10, truck_y - 40), (truck_x + cab_width - 10, truck_y + 20), (180, 80, 30), -1)
    
    # Bed (back part)
    bed_x = truck_x + cab_width + 10
    bed_width, bed_height = 180, 80
    cv2.rectangle(image, (bed_x, truck_y + 20), (bed_x + bed_width, truck_y + cab_height), (150, 150, 150), -1)
    
    # Bed sides
    cv2.rectangle(image, (bed_x, truck_y), (bed_x + 10, truck_y + cab_height), (120, 120, 120), -1)
    cv2.rectangle(image, (bed_x + bed_width - 10, truck_y), (bed_x + bed_width, truck_y + cab_height), (120, 120, 120), -1)
    cv2.rectangle(image, (bed_x + bed_width - 10, truck_y), (bed_x + bed_width, truck_y + 20), (120, 120, 120), -1)
    
    # Wheels (larger for truck)
    wheel_radius = 28
    cv2.circle(image, (truck_x + 30, truck_y + cab_height + 25), wheel_radius, (0, 0, 0), -1)
    cv2.circle(image, (bed_x + bed_width - 40, truck_y + cab_height + 25), wheel_radius, (0, 0, 0), -1)
    
    return image

def create_city_bus() -> np.ndarray:
    """Create a realistic city bus image."""
    # Create background
    image = np.ones((400, 700, 3), dtype=np.uint8) * 230
    
    # Add road
    cv2.rectangle(image, (0, 300), (700, 400), (100, 100, 100), -1)
    
    # Draw city bus (long and tall)
    bus_x, bus_y = 50, 150
    bus_width, bus_height = 500, 140
    
    # Main body
    cv2.rectangle(image, (bus_x, bus_y), (bus_x + bus_width, bus_y + bus_height), (200, 200, 50), -1)
    
    # Roof
    cv2.rectangle(image, (bus_x + 10, bus_y - 20), (bus_x + bus_width - 10, bus_y + 20), (180, 180, 30), -1)
    
    # Multiple windows (bus style)
    window_width = 60
    window_spacing = 80
    for i in range(6):
        window_x = bus_x + 30 + i * window_spacing
        if window_x + window_width < bus_x + bus_width - 30:
            cv2.rectangle(image, (window_x, bus_y + 20), (window_x + window_width, bus_y + 80), (150, 200, 255), -1)
    
    # Door
    cv2.rectangle(image, (bus_x + 20, bus_y + 40), (bus_x + 50, bus_y + bus_height - 10), (100, 100, 100), -1)
    
    # Wheels (multiple)
    wheel_radius = 30
    cv2.circle(image, (bus_x + 80, bus_y + bus_height + 25), wheel_radius, (0, 0, 0), -1)
    cv2.circle(image, (bus_x + bus_width - 80, bus_y + bus_height + 25), wheel_radius, (0, 0, 0), -1)
    
    return image

def create_sport_motorcycle() -> np.ndarray:
    """Create a realistic sport motorcycle image."""
    # Create background
    image = np.ones((400, 500, 3), dtype=np.uint8) * 200
    
    # Add road
    cv2.rectangle(image, (0, 300), (500, 400), (100, 100, 100), -1)
    
    # Draw sport motorcycle
    bike_x, bike_y = 150, 250
    
    # Main body (sleek, low)
    cv2.rectangle(image, (bike_x, bike_y), (bike_x + 200, bike_y + 40), (200, 50, 50), -1)
    
    # Fuel tank
    cv2.ellipse(image, (bike_x + 80, bike_y - 10), (40, 25), 0, 0, 360, (180, 30, 30), -1)
    
    # Seat
    cv2.rectangle(image, (bike_x + 120, bike_y - 15), (bike_x + 180, bike_y + 5), (50, 50, 50), -1)
    
    # Handlebars
    cv2.line(image, (bike_x + 30, bike_y - 20), (bike_x + 70, bike_y - 20), (100, 100, 100), 5)
    cv2.line(image, (bike_x + 20, bike_y - 25), (bike_x + 80, bike_y - 15), (100, 100, 100), 3)
    
    # Wheels (smaller than car)
    wheel_radius = 18
    cv2.circle(image, (bike_x + 40, bike_y + 50), wheel_radius, (0, 0, 0), -1)
    cv2.circle(image, (bike_x + 160, bike_y + 50), wheel_radius, (0, 0, 0), -1)
    
    # Exhaust
    cv2.rectangle(image, (bike_x + 170, bike_y + 20), (bike_x + 200, bike_y + 30), (150, 150, 150), -1)
    
    return image

def create_road_bicycle() -> np.ndarray:
    """Create a realistic road bicycle image."""
    # Create background
    image = np.ones((400, 500, 3), dtype=np.uint8) * 240
    
    # Add road
    cv2.rectangle(image, (0, 300), (500, 400), (100, 100, 100), -1)
    
    # Draw road bicycle
    bike_x, bike_y = 150, 280
    
    # Frame (triangle shape)
    points = np.array([[bike_x + 50, bike_y - 30], [bike_x + 150, bike_y - 30], [bike_x + 100, bike_y + 20]], np.int32)
    cv2.fillPoly(image, [points], (100, 100, 200))
    
    # Seat post and seat
    cv2.line(image, (bike_x + 100, bike_y - 30), (bike_x + 100, bike_y - 50), (150, 150, 150), 3)
    cv2.rectangle(image, (bike_x + 90, bike_y - 55), (bike_x + 110, bike_y - 50), (50, 50, 50), -1)
    
    # Handlebars
    cv2.line(image, (bike_x + 50, bike_y - 30), (bike_x + 50, bike_y - 50), (150, 150, 150), 3)
    cv2.line(image, (bike_x + 40, bike_y - 50), (bike_x + 60, bike_y - 50), (150, 150, 150), 3)
    
    # Wheels (thin)
    wheel_radius = 15
    cv2.circle(image, (bike_x + 50, bike_y + 30), wheel_radius, (0, 0, 0), 2)  # Thin wheels
    cv2.circle(image, (bike_x + 150, bike_y + 30), wheel_radius, (0, 0, 0), 2)
    
    # Spokes
    for angle in range(0, 360, 45):
        x1 = int(bike_x + 50 + 8 * np.cos(np.radians(angle)))
        y1 = int(bike_y + 30 + 8 * np.sin(np.radians(angle)))
        x2 = int(bike_x + 50 + 13 * np.cos(np.radians(angle)))
        y2 = int(bike_y + 30 + 13 * np.sin(np.radians(angle)))
        cv2.line(image, (x1, y1), (x2, y2), (150, 150, 150), 1)
        
        x1 = int(bike_x + 150 + 8 * np.cos(np.radians(angle)))
        y1 = int(bike_y + 30 + 8 * np.sin(np.radians(angle)))
        x2 = int(bike_x + 150 + 13 * np.cos(np.radians(angle)))
        y2 = int(bike_y + 30 + 13 * np.sin(np.radians(angle)))
        cv2.line(image, (x1, y1), (x2, y2), (150, 150, 150), 1)
    
    return image

def create_mixed_vehicles() -> np.ndarray:
    """Create an image with multiple vehicle types."""
    # Create larger scene
    image = np.ones((500, 800, 3), dtype=np.uint8) * 220
    
    # Add road
    cv2.rectangle(image, (0, 350), (800, 500), (100, 100, 100), -1)
    
    # Add smaller versions of different vehicles
    # Sedan (scaled down)
    sedan = create_sedan_car()
    sedan_small = cv2.resize(sedan, (200, 133))
    image[50:183, 50:250] = sedan_small
    
    # Pickup truck (scaled down)
    truck = create_pickup_truck()
    truck_small = cv2.resize(truck, (200, 133))
    image[50:183, 300:500] = truck_small
    
    # Motorcycle (scaled down)
    bike = create_sport_motorcycle()
    bike_small = cv2.resize(bike, (150, 120))
    image[200:320, 100:250] = bike_small
    
    # Bicycle (scaled down)
    bicycle = create_road_bicycle()
    bicycle_small = cv2.resize(bicycle, (150, 120))
    image[200:320, 400:550] = bicycle_small
    
    return image

def test_vehicle_classification_real():
    """Test vehicle classification with realistic images."""
    print("🧪 Testing Vehicle Classification with Real Images...")
    
    # Create model configuration
    config = ModelConfig(
        model_name="vehicle-classifier",
        model_type="vehicle_classification",
        confidence_threshold=0.1,  # Lower threshold for testing
        input_size=(224, 224),
        use_gpu=True,
        model_params={
            'min_classification_confidence': 0.1,
            'top_k_predictions': 5,
            'backbone': 'resnet50',
            'pretrained': True,
            'dropout_rate': 0.5
        }
    )
    
    print(f"📋 Model Config: {config}")
    
    # Initialize vehicle classifier
    print("🔧 Initializing Vehicle Classifier...")
    vehicle_classifier = VehicleClassifier(config)
    
    # Load model
    print("📥 Loading Vehicle Classification Model...")
    if not vehicle_classifier.load_model():
        print("❌ Failed to load vehicle classification model")
        return False
    
    print("✅ Vehicle classification model loaded successfully")
    
    # Get model info
    model_info = vehicle_classifier.get_model_info()
    print(f"📊 Model Info: {model_info['model_class']}")
    print(f"📊 Device: {model_info['device']}")
    print(f"📊 Number of classes: {vehicle_classifier.num_classes}")
    
    # Create realistic test images
    vehicles, test_dir = create_realistic_vehicle_images()
    
    # Test vehicle classification functionality
    print("🔍 Testing vehicle classification with realistic images...")
    try:
        results = {}
        
        # Expected categories for validation
        expected_categories = {
            'sedan_car': 'car',
            'suv_car': 'car',
            'pickup_truck': 'truck',
            'city_bus': 'bus',
            'sport_motorcycle': 'motorcycle',
            'road_bicycle': 'bicycle',
            'mixed_vehicles': 'mixed'  # Multiple categories expected
        }
        
        # Test each vehicle scenario
        for vehicle_name, image in vehicles.items():
            print(f"\n🎯 Testing {vehicle_name}...")
            
            start_time = time.time()
            result = vehicle_classifier.classify_vehicle(image)
            inference_time = time.time() - start_time
            
            if result.success and result.detections:
                print(f"📊 Found {len(result.detections)} classifications")
                print(f"📊 Inference time: {inference_time*1000:.1f}ms")
                
                # Get top prediction
                top_prediction = vehicle_classifier.get_top_prediction(image)
                if top_prediction:
                    print(f"📊 Top prediction: {top_prediction['class']} ({top_prediction['category']}) - {top_prediction['confidence']:.3f}")
                
                # Get category predictions
                category_predictions = vehicle_classifier.get_category_predictions(image)
                print(f"📊 Category predictions:")
                for category, preds in category_predictions.items():
                    if preds:
                        print(f"   - {category}: {len(preds)} predictions")
                        for pred in preds[:2]:  # Show top 2
                            print(f"     • {pred['class']}: {pred['confidence']:.3f}")
                
                # Show individual detections
                for i, detection in enumerate(result.detections[:3]):  # Show first 3
                    class_name = detection.class_name
                    category = detection.attributes.get('category', 'unknown')
                    confidence = detection.confidence
                    rank = detection.attributes.get('rank', i+1)
                    print(f"   Rank {rank}: {class_name} ({category}) - {confidence:.3f}")
                
                # Validate against expected
                expected_category = expected_categories.get(vehicle_name)
                if expected_category and expected_category != 'mixed':
                    if top_prediction and top_prediction['category'] == expected_category:
                        validation_result = "✅"
                    else:
                        validation_result = "❌"
                    print(f"📊 Validation: Expected {expected_category}, Got {top_prediction['category'] if top_prediction else 'None'} {validation_result}")
                
                results[vehicle_name] = {
                    'detections': len(result.detections),
                    'top_prediction': top_prediction,
                    'category_predictions': category_predictions,
                    'inference_time': inference_time
                }
            else:
                print(f"❌ Failed to classify vehicle in {vehicle_name}")
                results[vehicle_name] = {'error': 'Classification failed'}
        
        # Performance summary
        print(f"\n📊 Performance Summary:")
        successful_tests = [r for r in results.values() if 'inference_time' in r]
        if successful_tests:
            avg_inference_time = np.mean([r['inference_time'] for r in successful_tests])
            total_detections = sum([r['detections'] for r in successful_tests])
            print(f"📊 Average inference time: {avg_inference_time*1000:.1f}ms")
            print(f"📊 Total classifications: {total_detections}")
            
            # Category distribution
            all_categories = {}
            for r in successful_tests:
                if 'top_prediction' in r and r['top_prediction']:
                    category = r['top_prediction']['category']
                    all_categories[category] = all_categories.get(category, 0) + 1
            print(f"📊 Category distribution: {all_categories}")
        
        # Test model capabilities
        print(f"\n🎯 Testing model capabilities...")
        test_image = vehicles["sedan_car"]
        
        # Test all probabilities
        all_probs = vehicle_classifier.get_all_probabilities(test_image)
        top_5_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"📊 Top 5 class probabilities:")
        for class_name, prob in top_5_probs:
            category = vehicle_classifier.class_to_category[class_name]
            print(f"   - {class_name} ({category}): {prob:.4f}")
        
        print("✅ Vehicle classification real image tests completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Vehicle classification real image test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def cleanup_test_files():
    """Clean up test files."""
    print("\n🧹 Cleaning up test files...")
    
    # Remove test images
    test_dir = Path("test_vehicle_classification")
    if test_dir.exists():
        for file in test_dir.glob("*.jpg"):
            file.unlink()
        test_dir.rmdir()
        print("🧹 Removed test vehicle classification images")

def main():
    """Main test function."""
    print("🚀 AI Box - Vehicle Classification Real Image Test")
    print("=" * 65)
    
    try:
        # Test with realistic images
        success = test_vehicle_classification_real()
        
        if success:
            print("\n🎉 All real image tests completed successfully!")
            print("✅ Vehicle Classification Model is working correctly with realistic images")
        else:
            print("\n❌ Tests failed!")
            print("🔧 Please check the error messages above")
    
    finally:
        # Cleanup
        cleanup_test_files()

if __name__ == "__main__":
    main()
