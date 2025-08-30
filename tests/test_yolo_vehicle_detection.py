# ==========================================
# AI Box - Test YOLOv8 Vehicle Detection
# Debug vehicle detection issues
# ==========================================

import sys
import os
from pathlib import Path
import numpy as np
import cv2

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ai_models.common.model_config import ModelConfig
from ai_models.vehicle_analysis.vehicle_detector import VehicleDetector

def test_yolo_classes():
    """Test YOLOv8 class detection."""
    print("🧪 Testing YOLOv8 Vehicle Class Detection...")
    
    # Create config
    config = ModelConfig(
        model_name="yolov8n",
        model_type="vehicle_detection",
        confidence_threshold=0.1,  # Lower threshold
        input_size=(640, 640),
        use_gpu=False,
        model_params={
            'vehicle_class_ids': [2, 3, 5, 7, 1],  # car, motorcycle, bus, truck, bicycle
            'min_vehicle_size': 50,  # Lower minimum size
            'vehicle_confidence_threshold': 0.1
        }
    )
    
    # Initialize detector
    detector = VehicleDetector(config)
    if not detector.load_model():
        print("❌ Failed to load detector")
        return False
    
    print("✅ Detector loaded")
    print(f"📊 Vehicle classes: {detector.vehicle_classes}")
    print(f"📊 Vehicle class IDs: {detector.vehicle_class_ids}")
    
    # Create a simple test image with a large rectangle
    print("\n🖼️ Creating test image with large rectangle...")
    test_image = np.zeros((640, 640, 3), dtype=np.uint8)
    test_image[:, :] = [100, 100, 100]  # Gray background
    
    # Draw a very large rectangle that should be detected as something
    cv2.rectangle(test_image, (100, 200), (500, 400), (0, 0, 255), -1)  # Large red rectangle
    cv2.rectangle(test_image, (100, 200), (500, 400), (255, 255, 255), 3)  # White border
    
    # Add some car-like features
    # Windows
    cv2.rectangle(test_image, (150, 230), (450, 280), (50, 50, 150), -1)
    # Wheels
    cv2.circle(test_image, (180, 400), 30, (0, 0, 0), -1)
    cv2.circle(test_image, (420, 400), 30, (0, 0, 0), -1)
    
    print(f"✅ Test image created: {test_image.shape}")
    
    # Test raw YOLO detection
    print("\n🔍 Testing raw YOLO detection...")
    try:
        # Use the model directly
        results = detector.model(test_image, conf=0.1, verbose=False)
        
        print(f"📊 Raw YOLO results: {len(results)} result sets")
        
        for i, result in enumerate(results):
            print(f"   Result {i}:")
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes
                print(f"     Boxes detected: {len(boxes)}")
                
                for j, box in enumerate(boxes):
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    bbox = box.xyxy[0].tolist()
                    
                    # Get class name from COCO classes
                    coco_classes = {
                        0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
                        5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light'
                    }
                    class_name = coco_classes.get(class_id, f'class_{class_id}')
                    
                    print(f"       Box {j}: class_id={class_id} ({class_name}), conf={confidence:.3f}, bbox={bbox}")
                    
                    # Check if it's a vehicle class
                    if class_id in detector.vehicle_class_ids:
                        print(f"         ✅ This is a vehicle class!")
                    else:
                        print(f"         ❌ This is NOT a vehicle class")
            else:
                print("     No boxes detected")
    
    except Exception as e:
        print(f"❌ Raw YOLO detection failed: {e}")
        return False
    
    # Test detector method
    print("\n🚗 Testing VehicleDetector.detect_vehicles()...")
    try:
        result = detector.detect_vehicles(test_image)
        
        print(f"📊 Detection result:")
        print(f"   Success: {result.success}")
        print(f"   Detections: {len(result.detections)}")
        
        for i, detection in enumerate(result.detections):
            print(f"   Detection {i}:")
            print(f"     Class: {detection.class_name}")
            print(f"     Confidence: {detection.confidence:.3f}")
            print(f"     BBox: {detection.bbox}")
            print(f"     Attributes: {detection.attributes}")
        
        return len(result.detections) > 0
        
    except Exception as e:
        print(f"❌ VehicleDetector.detect_vehicles() failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_real_car_image():
    """Test with a more realistic car image."""
    print("\n🧪 Testing with Realistic Car Image...")
    
    # Create a more realistic car
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    img[:, :] = [120, 120, 120]  # Road color
    
    # Draw road
    cv2.line(img, (0, 500), (640, 500), (255, 255, 255), 4)
    
    # Draw a car that looks more like a real car
    car_x, car_y = 200, 350
    car_w, car_h = 240, 150
    
    # Car body (blue)
    cv2.rectangle(img, (car_x, car_y), (car_x + car_w, car_y + car_h), (100, 50, 200), -1)
    
    # Car roof (darker)
    cv2.rectangle(img, (car_x + 30, car_y + 20), (car_x + car_w - 30, car_y + 80), (80, 40, 160), -1)
    
    # Windshield
    cv2.rectangle(img, (car_x + 40, car_y + 30), (car_x + car_w - 40, car_y + 70), (200, 200, 250), -1)
    
    # Wheels
    cv2.circle(img, (car_x + 50, car_y + car_h), 35, (0, 0, 0), -1)
    cv2.circle(img, (car_x + car_w - 50, car_y + car_h), 35, (0, 0, 0), -1)
    cv2.circle(img, (car_x + 50, car_y + car_h), 25, (100, 100, 100), -1)
    cv2.circle(img, (car_x + car_w - 50, car_y + car_h), 25, (100, 100, 100), -1)
    
    # Headlights
    cv2.circle(img, (car_x + 20, car_y + 60), 15, (255, 255, 200), -1)
    cv2.circle(img, (car_x + 20, car_y + 90), 15, (255, 255, 200), -1)
    
    # License plate
    cv2.rectangle(img, (car_x + car_w - 80, car_y + car_h - 30), 
                  (car_x + car_w - 20, car_y + car_h - 10), (255, 255, 255), -1)
    
    # Add some noise for realism
    noise = np.random.randint(0, 30, img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)
    
    # Save for inspection
    cv2.imwrite("test_car_debug.jpg", img)
    print("💾 Saved test image as test_car_debug.jpg")
    
    # Test detection
    config = ModelConfig(
        model_name="yolov8n",
        model_type="vehicle_detection",
        confidence_threshold=0.05,  # Very low threshold
        input_size=(640, 640),
        use_gpu=False,
        model_params={
            'vehicle_class_ids': [2, 3, 5, 7, 1],
            'min_vehicle_size': 100,
            'vehicle_confidence_threshold': 0.05
        }
    )
    
    detector = VehicleDetector(config)
    if not detector.load_model():
        print("❌ Failed to load detector")
        return False
    
    # Test detection
    result = detector.detect_vehicles(img)
    
    print(f"📊 Realistic car detection:")
    print(f"   Success: {result.success}")
    print(f"   Detections: {len(result.detections)}")
    
    for i, detection in enumerate(result.detections):
        print(f"   Detection {i}: {detection.class_name} (conf: {detection.confidence:.3f})")
    
    # Also test raw YOLO
    print(f"\n🔍 Raw YOLO on realistic car:")
    results = detector.model(img, conf=0.05, verbose=False)
    
    for result in results:
        if hasattr(result, 'boxes') and result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                coco_classes = {
                    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
                    5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light'
                }
                class_name = coco_classes.get(class_id, f'class_{class_id}')
                
                print(f"   Raw detection: {class_name} (id={class_id}, conf={confidence:.3f})")
    
    return True

def main():
    """Main test function."""
    print("🚀 AI Box - YOLOv8 Vehicle Detection Debug")
    print("=" * 60)
    
    tests = [
        ("YOLOv8 Class Detection", test_yolo_classes),
        ("Realistic Car Image", test_with_real_car_image)
    ]
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            test_func()
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    # Cleanup
    try:
        if os.path.exists("test_car_debug.jpg"):
            os.remove("test_car_debug.jpg")
            print("\n🧹 Cleaned up debug image")
    except:
        pass

if __name__ == "__main__":
    main()
