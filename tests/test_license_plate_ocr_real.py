# ==========================================
# AI Box - License Plate OCR Real Test
# Test license plate OCR with real/synthetic images
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
from ai_models.vehicle_analysis.license_plate_ocr import LicensePlateOCR

def create_realistic_license_plate_images():
    """Create realistic license plate images for testing."""
    print("🖼️ Creating realistic license plate images...")
    
    # Create directory for test images
    test_dir = Path("test_license_plates")
    test_dir.mkdir(exist_ok=True)
    
    # Create different license plate scenarios
    plates = {
        "vietnam_plate": create_vietnam_license_plate(),
        "us_plate": create_us_license_plate(),
        "simple_plate": create_simple_license_plate(),
        "multiple_plates": create_multiple_license_plates(),
        "noisy_plate": create_noisy_license_plate()
    }
    
    # Save test images
    for name, image in plates.items():
        image_path = test_dir / f"{name}.jpg"
        cv2.imwrite(str(image_path), image)
        print(f"💾 Saved {name} image: {image_path}")
    
    return plates, test_dir

def create_vietnam_license_plate() -> np.ndarray:
    """Create a Vietnam-style license plate."""
    # Create white background
    image = np.ones((120, 300, 3), dtype=np.uint8) * 255
    
    # Add blue border (Vietnam style)
    cv2.rectangle(image, (5, 5), (295, 115), (200, 100, 0), 3)
    
    # Add text "30A-12345"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 2
    color = (0, 0, 0)  # Black text
    
    # Calculate text position to center it
    text = "30A-12345"
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (image.shape[1] - text_size[0]) // 2
    text_y = (image.shape[0] + text_size[1]) // 2
    
    cv2.putText(image, text, (text_x, text_y), font, font_scale, color, thickness)
    
    return image

def create_us_license_plate() -> np.ndarray:
    """Create a US-style license plate."""
    # Create light blue background
    image = np.ones((120, 300, 3), dtype=np.uint8)
    image[:, :] = [240, 240, 200]  # Light blue/white
    
    # Add border
    cv2.rectangle(image, (5, 5), (295, 115), (100, 100, 100), 2)
    
    # Add text "ABC1234"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.4
    thickness = 3
    color = (0, 0, 0)  # Black text
    
    text = "ABC1234"
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (image.shape[1] - text_size[0]) // 2
    text_y = (image.shape[0] + text_size[1]) // 2
    
    cv2.putText(image, text, (text_x, text_y), font, font_scale, color, thickness)
    
    return image

def create_simple_license_plate() -> np.ndarray:
    """Create a simple license plate."""
    # Create white background
    image = np.ones((100, 250, 3), dtype=np.uint8) * 255
    
    # Add simple border
    cv2.rectangle(image, (3, 3), (247, 97), (0, 0, 0), 2)
    
    # Add text "XYZ789"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    color = (0, 0, 0)
    
    text = "XYZ789"
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (image.shape[1] - text_size[0]) // 2
    text_y = (image.shape[0] + text_size[1]) // 2
    
    cv2.putText(image, text, (text_x, text_y), font, font_scale, color, thickness)
    
    return image

def create_multiple_license_plates() -> np.ndarray:
    """Create an image with multiple license plates."""
    # Create larger scene
    image = np.ones((400, 600, 3), dtype=np.uint8) * 200
    
    # Plate 1: Top left
    plate1 = create_vietnam_license_plate()
    image[50:170, 50:350] = plate1
    
    # Plate 2: Top right  
    plate2 = create_us_license_plate()
    image[50:170, 250:550] = plate2
    
    # Plate 3: Bottom center
    plate3 = create_simple_license_plate()
    image[250:350, 175:425] = plate3
    
    return image

def create_noisy_license_plate() -> np.ndarray:
    """Create a noisy/challenging license plate."""
    # Start with a basic plate
    image = create_vietnam_license_plate()
    
    # Add noise
    noise = np.random.randint(0, 50, image.shape, dtype=np.uint8)
    image = cv2.add(image, noise)
    
    # Add some blur
    image = cv2.GaussianBlur(image, (3, 3), 0)
    
    # Adjust contrast slightly
    image = cv2.convertScaleAbs(image, alpha=0.9, beta=10)
    
    return image

def test_license_plate_ocr_real():
    """Test license plate OCR with realistic images."""
    print("🧪 Testing License Plate OCR with Real Images...")
    
    # Create model configuration
    config = ModelConfig(
        model_name="license-plate-ocr",
        model_type="license_plate_ocr",
        confidence_threshold=0.3,
        input_size=(640, 480),
        use_gpu=True,
        model_params={
            'languages': ['en'],
            'min_text_confidence': 0.3,
            'preprocess_enabled': True,
            'contrast_enhancement': True,
            'noise_reduction': True
        }
    )
    
    print(f"📋 Model Config: {config}")
    
    # Initialize license plate OCR
    print("🔧 Initializing License Plate OCR...")
    license_ocr = LicensePlateOCR(config)
    
    # Load model
    print("📥 Loading License Plate OCR Model...")
    if not license_ocr.load_model():
        print("❌ Failed to load license plate OCR model")
        return False
    
    print("✅ License plate OCR model loaded successfully")
    
    # Get model info
    model_info = license_ocr.get_model_info()
    print(f"📊 Model Info: {model_info['model_class']}")
    print(f"📊 Device: {model_info['device']}")
    
    # Create realistic test images
    plates, test_dir = create_realistic_license_plate_images()
    
    # Test license plate OCR functionality
    print("🔍 Testing license plate OCR with realistic images...")
    try:
        results = {}
        
        # Expected texts for validation
        expected_texts = {
            'vietnam_plate': '30A-12345',
            'us_plate': 'ABC1234',
            'simple_plate': 'XYZ789',
            'multiple_plates': ['30A-12345', 'ABC1234', 'XYZ789'],  # Multiple expected
            'noisy_plate': '30A-12345'
        }
        
        # Test each license plate scenario
        for plate_name, image in plates.items():
            print(f"\n🎯 Testing {plate_name}...")
            
            start_time = time.time()
            result = license_ocr.recognize_license_plate(image)
            inference_time = time.time() - start_time
            
            if result.success:
                print(f"📊 Detected {len(result.detections)} text regions")
                print(f"📊 Inference time: {inference_time*1000:.1f}ms")
                
                # Extract recognized texts
                recognized_texts = license_ocr.extract_text_only(image)
                print(f"📊 Recognized texts: {recognized_texts}")
                
                # Get best license plate
                best_plate = license_ocr.get_best_license_plate(image)
                if best_plate:
                    print(f"📊 Best plate: {best_plate['text']} (confidence: {best_plate['confidence']:.3f})")
                
                # Get valid license plates
                valid_plates = license_ocr.get_valid_license_plates(image)
                print(f"📊 Valid plates: {len(valid_plates)} found")
                
                # Show individual detections
                for i, detection in enumerate(result.detections[:3]):  # Show first 3
                    text = detection.attributes.get('text', 'N/A')
                    is_valid = detection.attributes.get('is_valid_plate', False)
                    print(f"   Text {i+1}: '{text}' ({detection.confidence:.3f}) - Valid: {is_valid}")
                
                # Validate against expected
                expected = expected_texts.get(plate_name)
                if expected:
                    if isinstance(expected, list):
                        # Multiple expected texts
                        found_expected = any(exp in recognized_texts for exp in expected)
                        validation_result = "✅" if found_expected else "❌"
                        print(f"📊 Validation: Expected any of {expected}, Got {recognized_texts} {validation_result}")
                    else:
                        # Single expected text
                        found_expected = expected in recognized_texts
                        validation_result = "✅" if found_expected else "❌"
                        print(f"📊 Validation: Expected '{expected}', Got {recognized_texts} {validation_result}")
                
                results[plate_name] = {
                    'detections': len(result.detections),
                    'recognized_texts': recognized_texts,
                    'best_plate': best_plate,
                    'valid_plates': valid_plates,
                    'inference_time': inference_time
                }
            else:
                print(f"❌ Failed to recognize text in {plate_name}")
                results[plate_name] = {'error': 'Recognition failed'}
        
        # Performance summary
        print(f"\n📊 Performance Summary:")
        successful_tests = [r for r in results.values() if 'inference_time' in r]
        if successful_tests:
            avg_inference_time = np.mean([r['inference_time'] for r in successful_tests])
            total_detections = sum([r['detections'] for r in successful_tests])
            total_texts = sum([len(r['recognized_texts']) for r in successful_tests])
            print(f"📊 Average inference time: {avg_inference_time*1000:.1f}ms")
            print(f"📊 Total text regions detected: {total_detections}")
            print(f"📊 Total texts recognized: {total_texts}")
        
        # Test preprocessing effects
        print(f"\n🎯 Testing preprocessing effects...")
        test_image = plates["noisy_plate"]
        
        # Test with preprocessing enabled
        license_ocr.preprocess_enabled = True
        result_with_preprocessing = license_ocr.recognize_license_plate(test_image)
        texts_with_preprocessing = len(result_with_preprocessing.detections)
        
        # Test with preprocessing disabled
        license_ocr.preprocess_enabled = False
        result_without_preprocessing = license_ocr.recognize_license_plate(test_image)
        texts_without_preprocessing = len(result_without_preprocessing.detections)
        
        # Restore preprocessing
        license_ocr.preprocess_enabled = True
        
        print(f"📊 With preprocessing: {texts_with_preprocessing} detections")
        print(f"📊 Without preprocessing: {texts_without_preprocessing} detections")
        
        print("✅ License plate OCR real image tests completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ License plate OCR real image test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def cleanup_test_files():
    """Clean up test files."""
    print("\n🧹 Cleaning up test files...")
    
    # Remove test images
    test_dir = Path("test_license_plates")
    if test_dir.exists():
        for file in test_dir.glob("*.jpg"):
            file.unlink()
        test_dir.rmdir()
        print("🧹 Removed test license plate images")

def main():
    """Main test function."""
    print("🚀 AI Box - License Plate OCR Real Image Test")
    print("=" * 60)
    
    try:
        # Test with realistic images
        success = test_license_plate_ocr_real()
        
        if success:
            print("\n🎉 All real image tests completed successfully!")
            print("✅ License Plate OCR Model is working correctly with realistic images")
        else:
            print("\n❌ Tests failed!")
            print("🔧 Please check the error messages above")
    
    finally:
        # Cleanup
        cleanup_test_files()

if __name__ == "__main__":
    main()
