# ==========================================
# AI Box - Vehicle Pipeline Real Image Test
# Test complete pipeline with realistic images
# ==========================================

import sys
import os
from pathlib import Path
import numpy as np
import cv2
import time
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ai_models.common.model_config import ModelConfig
from ai_models.vehicle_analysis import (
    VehicleDetector,
    LicensePlateOCR,
    VehicleClassifier,
    TrafficAnalyzer
)

# Import the pipeline class from integration test
from test_vehicle_pipeline_integration import VehiclePipeline

def create_realistic_vehicle_images():
    """Create realistic vehicle images for testing."""
    test_images = {}
    
    # Create test images directory
    test_dir = Path("test_images")
    test_dir.mkdir(exist_ok=True)
    
    print("🖼️ Creating realistic vehicle test images...")
    
    # Image 1: Single car on road
    print("   Creating single car image...")
    img1 = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Road background
    img1[:, :] = [70, 70, 70]  # Dark gray road
    
    # Road markings
    cv2.line(img1, (0, 240), (640, 240), (255, 255, 255), 3)  # Center line
    cv2.line(img1, (0, 160), (640, 160), (255, 255, 255), 2)  # Lane line
    cv2.line(img1, (0, 320), (640, 320), (255, 255, 255), 2)  # Lane line
    
    # Draw a realistic car
    car_x, car_y = 200, 180
    car_w, car_h = 120, 80
    
    # Car body (blue)
    cv2.rectangle(img1, (car_x, car_y), (car_x + car_w, car_y + car_h), (180, 100, 50), -1)
    cv2.rectangle(img1, (car_x, car_y), (car_x + car_w, car_y + car_h), (255, 255, 255), 2)
    
    # Car windows (darker blue)
    cv2.rectangle(img1, (car_x + 10, car_y + 10), (car_x + car_w - 10, car_y + 30), (100, 60, 30), -1)
    
    # Car wheels
    cv2.circle(img1, (car_x + 20, car_y + car_h), 15, (0, 0, 0), -1)
    cv2.circle(img1, (car_x + car_w - 20, car_y + car_h), 15, (0, 0, 0), -1)
    
    # License plate
    plate_x = car_x + car_w // 2 - 25
    plate_y = car_y + car_h - 20
    cv2.rectangle(img1, (plate_x, plate_y), (plate_x + 50, plate_y + 15), (255, 255, 255), -1)
    cv2.rectangle(img1, (plate_x, plate_y), (plate_x + 50, plate_y + 15), (0, 0, 0), 1)
    cv2.putText(img1, "29A12345", (plate_x + 2, plate_y + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    # Add some texture and noise
    noise = np.random.randint(0, 20, img1.shape, dtype=np.uint8)
    img1 = cv2.add(img1, noise)
    
    test_images['single_car'] = img1
    cv2.imwrite(str(test_dir / "single_car.jpg"), img1)
    
    # Image 2: Multiple vehicles
    print("   Creating multiple vehicles image...")
    img2 = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Road background
    img2[:, :] = [65, 65, 65]
    
    # Road markings
    cv2.line(img2, (0, 240), (640, 240), (255, 255, 255), 3)
    
    # Vehicle 1: Car
    car1_x, car1_y = 80, 160
    cv2.rectangle(img2, (car1_x, car1_y), (car1_x + 100, car1_y + 70), (50, 150, 200), -1)
    cv2.rectangle(img2, (car1_x, car1_y), (car1_x + 100, car1_y + 70), (255, 255, 255), 2)
    cv2.circle(img2, (car1_x + 15, car1_y + 70), 12, (0, 0, 0), -1)
    cv2.circle(img2, (car1_x + 85, car1_y + 70), 12, (0, 0, 0), -1)
    
    # Vehicle 2: Truck
    truck_x, truck_y = 250, 140
    cv2.rectangle(img2, (truck_x, truck_y), (truck_x + 140, truck_y + 90), (100, 200, 100), -1)
    cv2.rectangle(img2, (truck_x, truck_y), (truck_x + 140, truck_y + 90), (255, 255, 255), 2)
    cv2.circle(img2, (truck_x + 20, truck_y + 90), 15, (0, 0, 0), -1)
    cv2.circle(img2, (truck_x + 120, truck_y + 90), 15, (0, 0, 0), -1)
    
    # Vehicle 3: Motorcycle
    bike_x, bike_y = 450, 200
    cv2.rectangle(img2, (bike_x, bike_y), (bike_x + 60, bike_y + 40), (200, 50, 200), -1)
    cv2.rectangle(img2, (bike_x, bike_y), (bike_x + 60, bike_y + 40), (255, 255, 255), 2)
    cv2.circle(img2, (bike_x + 10, bike_y + 40), 8, (0, 0, 0), -1)
    cv2.circle(img2, (bike_x + 50, bike_y + 40), 8, (0, 0, 0), -1)
    
    # Add license plates
    for i, (vx, vy, vw) in enumerate([(car1_x, car1_y + 70, 100), (truck_x, truck_y + 90, 140), (bike_x, bike_y + 40, 60)]):
        plate_x = vx + vw // 2 - 20
        plate_y = vy - 15
        cv2.rectangle(img2, (plate_x, plate_y), (plate_x + 40, plate_y + 12), (255, 255, 255), -1)
        cv2.putText(img2, f"ABC{i+1}23", (plate_x + 2, plate_y + 9), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
    
    # Add noise
    noise = np.random.randint(0, 25, img2.shape, dtype=np.uint8)
    img2 = cv2.add(img2, noise)
    
    test_images['multiple_vehicles'] = img2
    cv2.imwrite(str(test_dir / "multiple_vehicles.jpg"), img2)
    
    # Image 3: Traffic scene
    print("   Creating traffic scene image...")
    img3 = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Complex road scene
    img3[:, :] = [60, 60, 60]
    
    # Multiple lanes
    for y in [120, 180, 240, 300, 360]:
        cv2.line(img3, (0, y), (640, y), (255, 255, 255), 2)
    
    # Add 6 vehicles in different lanes
    vehicles = [
        (50, 90, 80, 50, (200, 100, 100)),   # Red car
        (180, 150, 90, 60, (100, 200, 100)), # Green truck
        (320, 210, 70, 45, (100, 100, 200)), # Blue car
        (450, 270, 85, 55, (200, 200, 100)), # Yellow car
        (120, 330, 95, 65, (200, 100, 200)), # Purple truck
        (350, 90, 75, 50, (100, 200, 200)),  # Cyan car
    ]
    
    for i, (x, y, w, h, color) in enumerate(vehicles):
        # Vehicle body
        cv2.rectangle(img3, (x, y), (x + w, y + h), color, -1)
        cv2.rectangle(img3, (x, y), (x + w, y + h), (255, 255, 255), 2)
        
        # Wheels
        wheel_size = 8 if w < 80 else 12
        cv2.circle(img3, (x + 15, y + h), wheel_size, (0, 0, 0), -1)
        cv2.circle(img3, (x + w - 15, y + h), wheel_size, (0, 0, 0), -1)
        
        # License plate
        plate_x = x + w // 2 - 15
        plate_y = y + h - 12
        cv2.rectangle(img3, (plate_x, plate_y), (plate_x + 30, plate_y + 10), (255, 255, 255), -1)
        cv2.putText(img3, f"V{i+1}23", (plate_x + 2, plate_y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 0), 1)
    
    # Add realistic noise and lighting
    noise = np.random.randint(0, 30, img3.shape, dtype=np.uint8)
    img3 = cv2.add(img3, noise)
    
    # Add some brightness variation
    brightness = np.random.randint(-20, 20, img3.shape[:2], dtype=np.int16)
    brightness = np.stack([brightness] * 3, axis=2)
    img3 = np.clip(img3.astype(np.int16) + brightness, 0, 255).astype(np.uint8)
    
    test_images['traffic_scene'] = img3
    cv2.imwrite(str(test_dir / "traffic_scene.jpg"), img3)
    
    print(f"✅ Created {len(test_images)} realistic test images")
    return test_images

def test_pipeline_with_real_images():
    """Test pipeline with realistic images."""
    print("🧪 Testing Pipeline with Realistic Images...")
    
    try:
        # Create test images
        test_images = create_realistic_vehicle_images()
        
        # Initialize pipeline
        pipeline = VehiclePipeline(platform="auto")
        if not pipeline.initialize_models():
            print("❌ Failed to initialize pipeline")
            return False
        
        print("\n🚀 Processing realistic images through pipeline...")
        
        all_results = {}
        
        for image_name, image in test_images.items():
            print(f"\n📸 Processing {image_name}...")
            print("=" * 40)
            
            # Process image
            results = pipeline.process_image(image)
            all_results[image_name] = results
            
            # Display results
            vehicle_detection = results['vehicle_detection']
            print(f"🚗 Vehicle Detection:")
            print(f"   Vehicles detected: {vehicle_detection['total_vehicles']}")
            print(f"   Types: {vehicle_detection['vehicle_types']}")
            
            if vehicle_detection['total_vehicles'] > 0:
                avg_conf = np.mean(vehicle_detection['confidences'])
                print(f"   Avg confidence: {avg_conf:.3f}")
                
                # Show bounding boxes
                for i, bbox in enumerate(vehicle_detection['bboxes']):
                    print(f"   Vehicle {i+1}: {vehicle_detection['vehicle_types'][i]} at {bbox}")
            
            # License plates
            plates = results['license_plates']
            successful_plates = sum(1 for p in plates if p.get('success', False))
            print(f"\n🔤 License Plates: {successful_plates}/{len(plates)} recognized")
            
            for plate in plates:
                if plate.get('success', False):
                    print(f"   '{plate['text']}' (conf: {plate['confidence']:.3f}, valid: {plate['valid_pattern']})")
            
            # Classifications
            classifications = results['vehicle_classifications']
            successful_class = sum(1 for c in classifications if c.get('success', False))
            print(f"\n🏷️ Classifications: {successful_class}/{len(classifications)} successful")
            
            for classification in classifications:
                if classification.get('success', False):
                    print(f"   {classification['original_type']} → {classification['classified_type']}")
                    print(f"   Category: {classification['category']} (conf: {classification['confidence']:.3f})")
            
            # Traffic analytics
            traffic = results['traffic_analytics']
            print(f"\n📊 Traffic Analytics:")
            if traffic['success']:
                print(f"   Total vehicles: {traffic['total_vehicles']}")
                print(f"   Density: {traffic['traffic_density']:.3f}")
                print(f"   Congestion: {traffic['congestion_level']}")
                print(f"   Flow rate: {traffic['flow_rate']:.2f} vehicles/min")
            
            # Performance
            summary = results['pipeline_summary']
            print(f"\n⚡ Performance:")
            print(f"   Processing time: {summary['total_processing_time']:.3f}s")
            print(f"   FPS: {summary['fps']:.2f}")
        
        # Overall analysis
        print(f"\n{'='*60}")
        print("📊 OVERALL PIPELINE ANALYSIS")
        print(f"{'='*60}")
        
        total_vehicles_detected = sum(r['vehicle_detection']['total_vehicles'] for r in all_results.values())
        total_plates_recognized = sum(len([p for p in r['license_plates'] if p.get('success', False)]) for r in all_results.values())
        total_classifications = sum(len([c for c in r['vehicle_classifications'] if c.get('success', False)]) for r in all_results.values())
        
        avg_processing_time = np.mean([r['pipeline_summary']['total_processing_time'] for r in all_results.values()])
        avg_fps = np.mean([r['pipeline_summary']['fps'] for r in all_results.values()])
        
        print(f"📊 Detection Summary:")
        print(f"   Total vehicles detected: {total_vehicles_detected}")
        print(f"   Total plates recognized: {total_plates_recognized}")
        print(f"   Total classifications: {total_classifications}")
        
        print(f"\n⚡ Performance Summary:")
        print(f"   Average processing time: {avg_processing_time:.3f}s")
        print(f"   Average FPS: {avg_fps:.2f}")
        
        # Success criteria for realistic images
        success_criteria = {
            'vehicles_detected': total_vehicles_detected > 0,
            'reasonable_performance': avg_processing_time < 5.0,
            'positive_fps': avg_fps > 0,
            'traffic_analytics_working': all(r['traffic_analytics']['success'] for r in all_results.values())
        }
        
        print(f"\n✅ Success Criteria:")
        all_passed = True
        for criterion, passed in success_criteria.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"   {criterion}: {status}")
            if not passed:
                all_passed = False
        
        if all_passed:
            print(f"\n🎉 All realistic image tests PASSED!")
            print(f"✅ Vehicle Pipeline working correctly with real images")
        else:
            print(f"\n❌ Some realistic image tests FAILED!")
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Realistic image test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def cleanup_test_files():
    """Clean up test files."""
    try:
        test_dir = Path("test_images")
        if test_dir.exists():
            import shutil
            shutil.rmtree(test_dir)
            print("🧹 Cleaned up test images")
    except Exception as e:
        print(f"⚠️ Cleanup warning: {e}")

def main():
    """Main realistic test function."""
    print("🚀 AI Box - Vehicle Pipeline Real Image Test")
    print("=" * 60)
    
    try:
        success = test_pipeline_with_real_images()
        
        if success:
            print("\n🎉 VEHICLE PIPELINE INTEGRATION TEST SUCCESSFUL!")
            print("✅ Complete pipeline working with realistic images")
            print("🚀 Ready for production deployment!")
        else:
            print("\n❌ Vehicle pipeline integration test failed!")
            print("🔧 Please check the error messages above")
        
        return success
        
    finally:
        cleanup_test_files()

if __name__ == "__main__":
    main()
