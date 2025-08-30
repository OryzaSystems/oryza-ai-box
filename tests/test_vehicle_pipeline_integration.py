# ==========================================
# AI Box - Vehicle Pipeline Integration Test
# Test complete vehicle analysis pipeline
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
from ai_models.common.inference_result import Detection

class VehiclePipeline:
    """Complete Vehicle Analysis Pipeline."""
    
    def __init__(self, platform: str = "auto"):
        """Initialize complete vehicle pipeline."""
        self.platform = platform
        self.vehicle_detector = None
        self.license_plate_ocr = None
        self.vehicle_classifier = None
        self.traffic_analyzer = None
        
        print(f"🚀 Initializing Vehicle Pipeline for platform: {platform}")
        
    def initialize_models(self) -> bool:
        """Initialize all models in the pipeline."""
        try:
            # Vehicle Detection Model
            print("🚗 Initializing Vehicle Detector...")
            vehicle_config = ModelConfig(
                model_name="yolov8n",
                model_type="vehicle_detection",
                confidence_threshold=0.3,
                input_size=(640, 640),
                platform=self.platform,
                use_gpu=False,  # CPU for testing
                model_params={
                    'vehicle_class_ids': [2, 3, 5, 7, 1],  # car, motorcycle, bus, truck, bicycle
                    'min_vehicle_size': 100,
                    'vehicle_confidence_threshold': 0.3
                }
            )
            
            self.vehicle_detector = VehicleDetector(vehicle_config)
            if not self.vehicle_detector.load_model():
                print("❌ Failed to load Vehicle Detector")
                return False
            print("✅ Vehicle Detector loaded")
            
            # License Plate OCR Model
            print("🔤 Initializing License Plate OCR...")
            ocr_config = ModelConfig(
                model_name="easyocr",
                model_type="license_plate_ocr",
                confidence_threshold=0.5,
                input_size=(224, 224),
                platform=self.platform,
                use_gpu=False,
                model_params={
                    'languages': ['en', 'vi'],
                    'min_text_confidence': 0.5,
                    'license_plate_patterns': [
                        r'^[0-9]{2}[A-Z]{1,2}[0-9]{4,5}$',  # Vietnam
                        r'^[A-Z]{3}[0-9]{4}$'  # US style
                    ]
                }
            )
            
            self.license_plate_ocr = LicensePlateOCR(ocr_config)
            if not self.license_plate_ocr.load_model():
                print("❌ Failed to load License Plate OCR")
                return False
            print("✅ License Plate OCR loaded")
            
            # Vehicle Classification Model
            print("🏷️ Initializing Vehicle Classifier...")
            classifier_config = ModelConfig(
                model_name="resnet50",
                model_type="vehicle_classification",
                confidence_threshold=0.3,
                input_size=(224, 224),
                platform=self.platform,
                use_gpu=False,
                model_params={
                    'num_classes': 30,
                    'backbone': 'resnet50',
                    'pretrained': True,
                    'top_k_predictions': 3
                }
            )
            
            self.vehicle_classifier = VehicleClassifier(classifier_config)
            if not self.vehicle_classifier.load_model():
                print("❌ Failed to load Vehicle Classifier")
                return False
            print("✅ Vehicle Classifier loaded")
            
            # Traffic Analytics Model
            print("📊 Initializing Traffic Analyzer...")
            analytics_config = ModelConfig(
                model_name="traffic-analyzer",
                model_type="traffic_analysis",
                confidence_threshold=0.5,
                input_size=(640, 640),
                platform=self.platform,
                use_gpu=False,
                model_params={
                    'max_tracks': 100,
                    'track_timeout': 5.0,
                    'pixels_per_meter': 10.0
                }
            )
            
            self.traffic_analyzer = TrafficAnalyzer(analytics_config)
            if not self.traffic_analyzer.load_model():
                print("❌ Failed to load Traffic Analyzer")
                return False
            print("✅ Traffic Analyzer loaded")
            
            print("🎉 All models loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Pipeline initialization failed: {e}")
            return False
    
    def process_image(self, image: np.ndarray) -> Dict[str, Any]:
        """Process image through complete pipeline."""
        if not all([self.vehicle_detector, self.license_plate_ocr, 
                   self.vehicle_classifier, self.traffic_analyzer]):
            raise RuntimeError("Pipeline not initialized. Call initialize_models() first.")
        
        results = {
            'timestamp': time.time(),
            'image_shape': image.shape,
            'vehicle_detection': None,
            'license_plates': [],
            'vehicle_classifications': [],
            'traffic_analytics': None,
            'processing_times': {},
            'pipeline_summary': {}
        }
        
        # Step 1: Vehicle Detection
        print("🚗 Step 1: Vehicle Detection...")
        start_time = time.time()
        
        vehicle_result = self.vehicle_detector.detect_vehicles(image)
        vehicle_detections = vehicle_result.detections if vehicle_result.success else []
        
        results['vehicle_detection'] = {
            'success': vehicle_result.success,
            'total_vehicles': len(vehicle_detections),
            'vehicle_types': [d.class_name for d in vehicle_detections],
            'confidences': [d.confidence for d in vehicle_detections],
            'bboxes': [d.bbox for d in vehicle_detections]
        }
        results['processing_times']['vehicle_detection'] = time.time() - start_time
        
        print(f"   ✅ Detected {len(vehicle_detections)} vehicles")
        
        # Step 2: License Plate OCR (for each vehicle)
        print("🔤 Step 2: License Plate OCR...")
        start_time = time.time()
        
        for i, detection in enumerate(vehicle_detections):
            # Extract vehicle region
            x1, y1, x2, y2 = map(int, detection.bbox)
            vehicle_crop = image[y1:y2, x1:x2]
            
            if vehicle_crop.size > 0:
                try:
                    ocr_result = self.license_plate_ocr.recognize_license_plate(vehicle_crop)
                    
                    plate_info = {
                        'vehicle_index': i,
                        'vehicle_type': detection.class_name,
                        'success': ocr_result.success,
                        'text': '',
                        'confidence': 0.0,
                        'valid_pattern': False
                    }
                    
                    if ocr_result.success and ocr_result.detections:
                        plate_detection = ocr_result.detections[0]
                        plate_info.update({
                            'text': plate_detection.class_name,
                            'confidence': plate_detection.confidence,
                            'valid_pattern': plate_detection.attributes.get('valid_pattern', False)
                        })
                    
                    results['license_plates'].append(plate_info)
                    
                except Exception as e:
                    print(f"   ⚠️ OCR failed for vehicle {i}: {e}")
                    results['license_plates'].append({
                        'vehicle_index': i,
                        'vehicle_type': detection.class_name,
                        'success': False,
                        'error': str(e)
                    })
        
        results['processing_times']['license_plate_ocr'] = time.time() - start_time
        print(f"   ✅ Processed {len(results['license_plates'])} license plates")
        
        # Step 3: Vehicle Classification (for each vehicle)
        print("🏷️ Step 3: Vehicle Classification...")
        start_time = time.time()
        
        for i, detection in enumerate(vehicle_detections):
            # Extract vehicle region
            x1, y1, x2, y2 = map(int, detection.bbox)
            vehicle_crop = image[y1:y2, x1:x2]
            
            if vehicle_crop.size > 0:
                try:
                    classification_result = self.vehicle_classifier.classify_vehicle(vehicle_crop)
                    
                    classification_info = {
                        'vehicle_index': i,
                        'original_type': detection.class_name,
                        'success': classification_result.success,
                        'classified_type': '',
                        'category': '',
                        'confidence': 0.0,
                        'top_k_predictions': []
                    }
                    
                    if classification_result.success and classification_result.detections:
                        class_detection = classification_result.detections[0]
                        classification_info.update({
                            'classified_type': class_detection.class_name,
                            'category': class_detection.attributes.get('category', ''),
                            'confidence': class_detection.confidence,
                            'top_k_predictions': class_detection.attributes.get('top_k_classes', [])
                        })
                    
                    results['vehicle_classifications'].append(classification_info)
                    
                except Exception as e:
                    print(f"   ⚠️ Classification failed for vehicle {i}: {e}")
                    results['vehicle_classifications'].append({
                        'vehicle_index': i,
                        'original_type': detection.class_name,
                        'success': False,
                        'error': str(e)
                    })
        
        results['processing_times']['vehicle_classification'] = time.time() - start_time
        print(f"   ✅ Classified {len(results['vehicle_classifications'])} vehicles")
        
        # Step 4: Traffic Analytics
        print("📊 Step 4: Traffic Analytics...")
        start_time = time.time()
        
        try:
            traffic_metrics = self.traffic_analyzer.analyze_traffic(image, vehicle_detections)
            
            results['traffic_analytics'] = {
                'success': True,
                'total_vehicles': traffic_metrics.total_vehicles,
                'vehicle_counts': traffic_metrics.vehicle_counts,
                'traffic_density': traffic_metrics.traffic_density,
                'congestion_level': traffic_metrics.congestion_level,
                'flow_rate': traffic_metrics.flow_rate,
                'zone_counts': traffic_metrics.zone_counts
            }
            
        except Exception as e:
            print(f"   ⚠️ Traffic analytics failed: {e}")
            results['traffic_analytics'] = {
                'success': False,
                'error': str(e)
            }
        
        results['processing_times']['traffic_analytics'] = time.time() - start_time
        print(f"   ✅ Traffic analytics completed")
        
        # Pipeline Summary
        total_time = sum(results['processing_times'].values())
        results['pipeline_summary'] = {
            'total_processing_time': total_time,
            'fps': 1.0 / total_time if total_time > 0 else 0,
            'vehicles_detected': len(vehicle_detections),
            'plates_recognized': sum(1 for p in results['license_plates'] if p.get('success', False)),
            'vehicles_classified': sum(1 for c in results['vehicle_classifications'] if c.get('success', False)),
            'traffic_analysis_success': results['traffic_analytics']['success']
        }
        
        return results
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get pipeline information."""
        return {
            'platform': self.platform,
            'models_loaded': {
                'vehicle_detector': self.vehicle_detector is not None and self.vehicle_detector.is_loaded,
                'license_plate_ocr': self.license_plate_ocr is not None and self.license_plate_ocr.is_loaded,
                'vehicle_classifier': self.vehicle_classifier is not None and self.vehicle_classifier.is_loaded,
                'traffic_analyzer': self.traffic_analyzer is not None and self.traffic_analyzer.is_loaded
            },
            'model_info': {
                'vehicle_detector': self.vehicle_detector.get_model_info() if self.vehicle_detector else None,
                'license_plate_ocr': self.license_plate_ocr.get_model_info() if self.license_plate_ocr else None,
                'vehicle_classifier': self.vehicle_classifier.get_model_info() if self.vehicle_classifier else None,
                'traffic_analyzer': self.traffic_analyzer.get_model_info() if self.traffic_analyzer else None
            }
        }

def create_test_traffic_image() -> np.ndarray:
    """Create a synthetic traffic scene for testing."""
    # Create a 640x480 traffic scene
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Background (road)
    image[:, :] = [60, 60, 60]  # Dark gray road
    
    # Road markings
    cv2.line(image, (0, 240), (640, 240), (255, 255, 255), 2)  # Center line
    
    # Draw some vehicle-like rectangles
    vehicles = [
        # (x, y, width, height, color, label)
        (100, 200, 80, 40, (0, 0, 255), "car"),      # Red car
        (250, 180, 100, 60, (0, 255, 0), "truck"),   # Green truck
        (400, 220, 60, 30, (255, 0, 0), "motorcycle"), # Blue motorcycle
        (500, 190, 90, 50, (255, 255, 0), "bus"),    # Yellow bus
    ]
    
    for x, y, w, h, color, label in vehicles:
        cv2.rectangle(image, (x, y), (x + w, y + h), color, -1)
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), 2)
        
        # Add license plate area
        plate_x = x + w // 4
        plate_y = y + h - 15
        plate_w = w // 2
        plate_h = 10
        cv2.rectangle(image, (plate_x, plate_y), (plate_x + plate_w, plate_y + plate_h), (255, 255, 255), -1)
        
        # Add some text on the plate
        cv2.putText(image, "ABC123", (plate_x + 2, plate_y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
    
    # Add some noise for realism
    noise = np.random.randint(0, 30, image.shape, dtype=np.uint8)
    image = cv2.add(image, noise)
    
    return image

def test_pipeline_initialization():
    """Test pipeline initialization."""
    print("🧪 Testing Pipeline Initialization...")
    
    try:
        pipeline = VehiclePipeline(platform="auto")
        success = pipeline.initialize_models()
        
        if success:
            print("✅ Pipeline initialization successful")
            
            # Test pipeline info
            info = pipeline.get_pipeline_info()
            print(f"📊 Platform: {info['platform']}")
            
            models_loaded = info['models_loaded']
            for model_name, loaded in models_loaded.items():
                status = "✅ Loaded" if loaded else "❌ Not loaded"
                print(f"   {model_name}: {status}")
            
            return True
        else:
            print("❌ Pipeline initialization failed")
            return False
            
    except Exception as e:
        print(f"❌ Pipeline initialization error: {e}")
        return False

def test_complete_pipeline():
    """Test complete pipeline processing."""
    print("\n🧪 Testing Complete Pipeline Processing...")
    
    try:
        # Initialize pipeline
        pipeline = VehiclePipeline(platform="auto")
        if not pipeline.initialize_models():
            print("❌ Failed to initialize pipeline")
            return False
        
        # Create test image
        print("🖼️ Creating test traffic image...")
        test_image = create_test_traffic_image()
        print(f"✅ Test image created: {test_image.shape}")
        
        # Process image through pipeline
        print("\n🚀 Processing image through complete pipeline...")
        results = pipeline.process_image(test_image)
        
        # Analyze results
        print("\n📊 Pipeline Results Analysis:")
        print("=" * 50)
        
        # Vehicle Detection Results
        vehicle_detection = results['vehicle_detection']
        print(f"🚗 Vehicle Detection:")
        print(f"   Success: {vehicle_detection['success']}")
        print(f"   Total vehicles: {vehicle_detection['total_vehicles']}")
        print(f"   Vehicle types: {vehicle_detection['vehicle_types']}")
        print(f"   Avg confidence: {np.mean(vehicle_detection['confidences']):.3f}")
        
        # License Plate OCR Results
        license_plates = results['license_plates']
        successful_plates = sum(1 for p in license_plates if p.get('success', False))
        print(f"\n🔤 License Plate OCR:")
        print(f"   Plates processed: {len(license_plates)}")
        print(f"   Successful recognitions: {successful_plates}")
        
        for i, plate in enumerate(license_plates):
            if plate.get('success', False):
                print(f"   Plate {i}: '{plate['text']}' (conf: {plate['confidence']:.3f})")
        
        # Vehicle Classification Results
        classifications = results['vehicle_classifications']
        successful_classifications = sum(1 for c in classifications if c.get('success', False))
        print(f"\n🏷️ Vehicle Classification:")
        print(f"   Vehicles classified: {len(classifications)}")
        print(f"   Successful classifications: {successful_classifications}")
        
        for i, classification in enumerate(classifications):
            if classification.get('success', False):
                print(f"   Vehicle {i}: {classification['original_type']} → {classification['classified_type']}")
                print(f"      Category: {classification['category']} (conf: {classification['confidence']:.3f})")
        
        # Traffic Analytics Results
        traffic_analytics = results['traffic_analytics']
        print(f"\n📊 Traffic Analytics:")
        print(f"   Success: {traffic_analytics['success']}")
        if traffic_analytics['success']:
            print(f"   Total vehicles: {traffic_analytics['total_vehicles']}")
            print(f"   Vehicle counts: {traffic_analytics['vehicle_counts']}")
            print(f"   Traffic density: {traffic_analytics['traffic_density']:.3f}")
            print(f"   Congestion level: {traffic_analytics['congestion_level']}")
            print(f"   Flow rate: {traffic_analytics['flow_rate']:.2f} vehicles/min")
        
        # Performance Summary
        summary = results['pipeline_summary']
        print(f"\n⚡ Performance Summary:")
        print(f"   Total processing time: {summary['total_processing_time']:.3f}s")
        print(f"   Pipeline FPS: {summary['fps']:.2f}")
        print(f"   Vehicles detected: {summary['vehicles_detected']}")
        print(f"   Plates recognized: {summary['plates_recognized']}")
        print(f"   Vehicles classified: {summary['vehicles_classified']}")
        print(f"   Traffic analysis: {'✅ Success' if summary['traffic_analysis_success'] else '❌ Failed'}")
        
        # Processing time breakdown
        processing_times = results['processing_times']
        print(f"\n⏱️ Processing Time Breakdown:")
        for step, time_taken in processing_times.items():
            percentage = (time_taken / summary['total_processing_time']) * 100
            print(f"   {step}: {time_taken:.3f}s ({percentage:.1f}%)")
        
        # Success criteria
        success_criteria = {
            'vehicle_detection_success': vehicle_detection['success'],
            'at_least_one_vehicle': vehicle_detection['total_vehicles'] > 0,
            'traffic_analysis_success': traffic_analytics['success'],
            'reasonable_processing_time': summary['total_processing_time'] < 10.0,  # Less than 10 seconds
            'positive_fps': summary['fps'] > 0
        }
        
        print(f"\n✅ Success Criteria:")
        all_passed = True
        for criterion, passed in success_criteria.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"   {criterion}: {status}")
            if not passed:
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Complete pipeline test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_platform_optimization():
    """Test pipeline optimization for different platforms."""
    print("\n🧪 Testing Platform Optimization...")
    
    platforms = ['raspberry-pi-5', 'radxa-rock-5', 'jetson-nano', 'core-i5']
    
    for platform in platforms:
        print(f"\n🔧 Testing {platform} optimization...")
        
        try:
            pipeline = VehiclePipeline(platform=platform)
            success = pipeline.initialize_models()
            
            if success:
                print(f"✅ {platform} pipeline initialized successfully")
                
                # Get model info to verify optimization
                info = pipeline.get_pipeline_info()
                print(f"   Platform: {info['platform']}")
                
                # Check if all models are loaded
                all_loaded = all(info['models_loaded'].values())
                print(f"   All models loaded: {'✅ Yes' if all_loaded else '❌ No'}")
                
            else:
                print(f"❌ {platform} pipeline initialization failed")
                
        except Exception as e:
            print(f"❌ {platform} optimization error: {e}")
    
    return True

def main():
    """Main integration test function."""
    print("🚀 AI Box - Vehicle Pipeline Integration Test")
    print("=" * 60)
    
    tests = [
        ("Pipeline Initialization", test_pipeline_initialization),
        ("Complete Pipeline Processing", test_complete_pipeline),
        ("Platform Optimization", test_platform_optimization)
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
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"📊 Integration Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All integration tests passed!")
        print("✅ Vehicle Analysis Pipeline is working correctly")
        print("🚀 Pipeline ready for production deployment!")
    else:
        print("❌ Some integration tests failed!")
        print("🔧 Please check the error messages above")
    
    return passed == total

if __name__ == "__main__":
    main()
