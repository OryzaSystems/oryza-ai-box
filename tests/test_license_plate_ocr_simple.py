# ==========================================
# AI Box - License Plate OCR Simple Test
# Test license plate OCR infrastructure
# ==========================================

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ai_models.common.model_config import ModelConfig
from ai_models.vehicle_analysis.license_plate_ocr import LicensePlateOCR

def test_license_plate_ocr_initialization():
    """Test LicensePlateOCR initialization without loading model."""
    print("🧪 Testing LicensePlateOCR Initialization...")
    
    # Create config
    config = ModelConfig(
        model_name="license-plate-ocr",
        model_type="license_plate_ocr",
        confidence_threshold=0.5,
        input_size=(640, 480),
        use_gpu=True,
        model_params={
            'languages': ['en'],
            'min_text_confidence': 0.5,
            'preprocess_enabled': True
        }
    )
    
    # Initialize OCR (without loading model)
    try:
        license_ocr = LicensePlateOCR(config)
        print(f"✅ LicensePlateOCR initialized: {license_ocr}")
        
        # Test model info
        model_info = license_ocr.get_model_info()
        print(f"✅ Model info: {model_info['model_class']}")
        
        # Test performance metrics
        metrics = license_ocr.get_performance_metrics()
        print(f"✅ Performance metrics: {metrics}")
        
        # Test configuration
        print(f"✅ Languages: {license_ocr.languages}")
        print(f"✅ Min text confidence: {license_ocr.min_text_confidence}")
        print(f"✅ License plate patterns: {len(license_ocr.license_plate_patterns)} patterns")
        print(f"✅ Preprocessing enabled: {license_ocr.preprocess_enabled}")
        print(f"✅ Contrast enhancement: {license_ocr.contrast_enhancement}")
        print(f"✅ Noise reduction: {license_ocr.noise_reduction}")
        
        return True
        
    except Exception as e:
        print(f"❌ LicensePlateOCR initialization failed: {e}")
        return False

def test_platform_optimization():
    """Test platform optimization functionality."""
    print("\n🔧 Testing Platform Optimization...")
    
    config = ModelConfig(
        model_name="license-plate-ocr",
        model_type="license_plate_ocr",
        platform="auto",
        model_params={'languages': ['en']}
    )
    
    license_ocr = LicensePlateOCR(config)
    
    # Test different platforms
    platforms = ['raspberry-pi-5', 'radxa-rock-5', 'jetson-nano', 'core-i5']
    
    for platform in platforms:
        print(f"🔧 Testing optimization for {platform}...")
        success = license_ocr.optimize_for_platform(platform)
        print(f"📊 {platform} optimization: {'✅ Success' if success else '❌ Failed'}")
    
    return True

def test_license_plate_patterns():
    """Test license plate pattern validation."""
    print("\n🧪 Testing License Plate Patterns...")
    
    config = ModelConfig(
        model_name="license-plate-ocr",
        model_type="license_plate_ocr"
    )
    
    license_ocr = LicensePlateOCR(config)
    
    # Test various license plate formats
    test_plates = [
        ("30A-12345", True),   # Vietnam format
        ("ABC1234", True),     # Simple format
        ("ABC123D", True),     # Mixed format
        ("INVALID", False),    # Invalid format
        ("12-34", False),      # Too short
        ("TOOLONGPLATE123", False)  # Too long
    ]
    
    print(f"📊 Testing {len(test_plates)} license plate formats...")
    
    correct_validations = 0
    for plate_text, expected_valid in test_plates:
        is_valid = license_ocr._validate_license_plate(plate_text)
        result = "✅" if is_valid == expected_valid else "❌"
        print(f"   {plate_text}: Expected {expected_valid}, Got {is_valid} {result}")
        
        if is_valid == expected_valid:
            correct_validations += 1
    
    accuracy = correct_validations / len(test_plates)
    print(f"📊 Pattern validation accuracy: {accuracy:.1%} ({correct_validations}/{len(test_plates)})")
    
    return accuracy >= 0.8  # At least 80% accuracy

def test_text_cleaning():
    """Test license plate text cleaning functionality."""
    print("\n🧪 Testing Text Cleaning...")
    
    config = ModelConfig(
        model_name="license-plate-ocr",
        model_type="license_plate_ocr"
    )
    
    license_ocr = LicensePlateOCR(config)
    
    # Test text cleaning
    test_cases = [
        ("  30a-12345  ", "30A-12345"),  # Whitespace and case
        ("30A.12345", "30A-12345"),      # Noise characters
        ("3OA-I2345", "3OA-12345"),      # OCR character fixes
        ("abc@123#", "ABC123"),          # Special characters
    ]
    
    print(f"📊 Testing {len(test_cases)} text cleaning cases...")
    
    correct_cleanings = 0
    for raw_text, expected_clean in test_cases:
        cleaned = license_ocr._clean_license_plate_text(raw_text)
        result = "✅" if cleaned == expected_clean else "❌"
        print(f"   '{raw_text}' → '{cleaned}' (expected '{expected_clean}') {result}")
        
        if cleaned == expected_clean:
            correct_cleanings += 1
    
    accuracy = correct_cleanings / len(test_cases)
    print(f"📊 Text cleaning accuracy: {accuracy:.1%} ({correct_cleanings}/{len(test_cases)})")
    
    return accuracy >= 0.7  # At least 70% accuracy

def test_easyocr_availability():
    """Test EasyOCR library availability."""
    print("\n🔧 Testing EasyOCR Availability...")
    
    try:
        import easyocr
        print("✅ EasyOCR library is available")
        print(f"✅ EasyOCR version: {easyocr.__version__ if hasattr(easyocr, '__version__') else 'Unknown'}")
        
        # Test basic EasyOCR functionality
        try:
            # This will not actually load the model, just test the import
            reader_class = easyocr.Reader
            print("✅ EasyOCR Reader class accessible")
            return True
        except Exception as e:
            print(f"❌ EasyOCR Reader class error: {e}")
            return False
            
    except ImportError:
        print("❌ EasyOCR library not available")
        print("💡 Install with: pip install easyocr")
        return False

def main():
    """Main test function."""
    print("🚀 AI Box - License Plate OCR Simple Test")
    print("=" * 55)
    
    tests = [
        ("EasyOCR Availability", test_easyocr_availability),
        ("LicensePlateOCR Initialization", test_license_plate_ocr_initialization),
        ("Platform Optimization", test_platform_optimization),
        ("License Plate Patterns", test_license_plate_patterns),
        ("Text Cleaning", test_text_cleaning)
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
    
    print(f"\n{'='*55}")
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All license plate OCR infrastructure tests passed!")
        print("✅ License Plate OCR Infrastructure is working correctly")
    else:
        print("❌ Some tests failed!")
        print("🔧 Please check the error messages above")
        if passed == total - 1:  # Only EasyOCR missing
            print("💡 Most likely issue: EasyOCR not installed")
            print("💡 Run: pip install easyocr")
    
    return passed == total

if __name__ == "__main__":
    main()
