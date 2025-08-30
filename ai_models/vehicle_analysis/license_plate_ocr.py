# ==========================================
# AI Box - License Plate OCR Model
# PaddleOCR/EasyOCR-based license plate recognition
# ==========================================

import cv2
import numpy as np
import torch
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
import logging
import re

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    easyocr = None

from ..common.base_model import BaseModel
from ..common.model_config import ModelConfig
from ..common.inference_result import InferenceResult, Detection

logger = logging.getLogger(__name__)

class LicensePlateOCR(BaseModel):
    """
    License Plate OCR Model using EasyOCR.
    
    This model recognizes text from license plates with high accuracy:
    - Multi-language support (English, Vietnamese, etc.)
    - License plate format validation
    - Text confidence scoring
    - Preprocessing for better OCR accuracy
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize License Plate OCR.
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        
        # OCR specific attributes
        self.languages = config.model_params.get('languages', ['en'])  # English by default
        self.min_text_confidence = config.model_params.get('min_text_confidence', 0.5)
        self.license_plate_patterns = config.model_params.get('license_plate_patterns', [
            r'^[A-Z0-9]{2,3}-[A-Z0-9]{4,5}$',  # Vietnam format: 30A-12345
            r'^[A-Z0-9]{6,8}$',                 # Simple format: ABC1234
            r'^[A-Z]{1,3}[0-9]{3,4}[A-Z]{0,2}$' # Mixed format: ABC123D
        ])
        
        # Text preprocessing parameters
        self.preprocess_enabled = config.model_params.get('preprocess_enabled', True)
        self.contrast_enhancement = config.model_params.get('contrast_enhancement', True)
        self.noise_reduction = config.model_params.get('noise_reduction', True)
        
        # Model metadata
        self.metadata = {
            'name': 'EasyOCR-License-Plate',
            'version': '1.0.0',
            'description': 'EasyOCR-based license plate text recognition model',
            'author': 'Oryza AI Team',
            'framework': 'EasyOCR',
            'input_shape': self.config.input_size,
            'output_shape': None,  # Dynamic based on text
            'model_size_mb': 0.0,  # Will be updated after loading
            'parameters': 0,  # Will be updated after loading
            'created_date': '2025-08-30',
            'last_updated': '2025-08-30',
            'supported_languages': self.languages
        }
        
        logger.info(f"Initialized LicensePlateOCR with languages: {self.languages}")
    
    def load_model(self) -> bool:
        """
        Load the EasyOCR model.
        
        Returns:
            bool: True if model loaded successfully
        """
        try:
            if not EASYOCR_AVAILABLE:
                logger.error("EasyOCR not available. Install with: pip install easyocr")
                return False
            
            logger.info("Loading EasyOCR model...")
            
            # Initialize EasyOCR reader
            self.model = easyocr.Reader(
                lang_list=self.languages,
                gpu=self.device.type == 'cuda'
            )
            
            # Update metadata
            self._update_model_metadata()
            
            self.is_loaded = True
            logger.info("License plate OCR model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load license plate OCR model: {str(e)}")
            return False
    
    def _update_model_metadata(self):
        """Update model metadata with actual model information."""
        if self.model is None:
            return
        
        try:
            # EasyOCR doesn't provide direct model info, so we estimate
            self.metadata.update({
                'model_size_mb': 50.0,  # Approximate size for EasyOCR models
                'parameters': 1000000,  # Approximate parameter count
                'output_shape': 'variable_text'
            })
            
        except Exception as e:
            logger.warning(f"Could not update model metadata: {e}")
    
    def preprocess(self, input_data: Union[str, np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Preprocess input image for license plate OCR.
        
        Args:
            input_data: Input image (file path, numpy array, or tensor)
            
        Returns:
            np.ndarray: Preprocessed image
        """
        try:
            # Handle different input types
            if isinstance(input_data, str):
                # File path
                image = cv2.imread(input_data)
                if image is None:
                    raise ValueError(f"Could not load image from path: {input_data}")
            elif isinstance(input_data, np.ndarray):
                # Numpy array
                image = input_data.copy()
            elif isinstance(input_data, torch.Tensor):
                # PyTorch tensor
                image = input_data.cpu().numpy()
                if image.ndim == 3 and image.shape[0] in [1, 3]:
                    image = np.transpose(image, (1, 2, 0))
            else:
                raise ValueError(f"Unsupported input type: {type(input_data)}")
            
            # Apply preprocessing if enabled
            if self.preprocess_enabled:
                image = self._preprocess_for_ocr(image)
            
            return image
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            raise
    
    def _preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """
        Apply OCR-specific preprocessing to improve text recognition.
        
        Args:
            image: Input image
            
        Returns:
            np.ndarray: Preprocessed image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Contrast enhancement
        if self.contrast_enhancement:
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        
        # Noise reduction
        if self.noise_reduction:
            # Apply bilateral filter to reduce noise while preserving edges
            gray = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Resize if too small (OCR works better on larger images)
        height, width = gray.shape
        if height < 32 or width < 128:
            scale_factor = max(32 / height, 128 / width)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Convert back to BGR for EasyOCR
        if len(image.shape) == 3:
            processed = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        else:
            processed = gray
        
        return processed
    
    def _inference_step(self, preprocessed_data: np.ndarray) -> Any:
        """
        Execute inference using EasyOCR model.
        
        Args:
            preprocessed_data: Preprocessed image
            
        Returns:
            Raw EasyOCR output
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # EasyOCR expects image as numpy array
        # readtext returns list of [bbox_points, text, confidence]
        results = self.model.readtext(preprocessed_data)
        return results
    
    def postprocess(self, model_output: Any) -> InferenceResult:
        """
        Postprocess EasyOCR output into standardized format.
        
        Args:
            model_output: Raw EasyOCR model output
            
        Returns:
            InferenceResult: Standardized inference result
        """
        try:
            # Create result object
            result = InferenceResult(
                success=True,
                model_name=self.metadata['name'],
                model_type='license_plate_ocr'
            )
            
            # Process EasyOCR results
            if model_output:
                for detection in model_output:
                    # EasyOCR returns: [bbox_points, text, confidence]
                    bbox_points, text, confidence = detection
                    
                    # Filter by confidence threshold
                    if confidence >= self.min_text_confidence:
                        # Convert bbox points to standard format [x1, y1, x2, y2]
                        bbox_array = np.array(bbox_points)
                        x1, y1 = bbox_array.min(axis=0)
                        x2, y2 = bbox_array.max(axis=0)
                        bbox = [float(x1), float(y1), float(x2), float(y2)]
                        
                        # Clean and validate text
                        cleaned_text = self._clean_license_plate_text(text)
                        is_valid_plate = self._validate_license_plate(cleaned_text)
                        
                        # Add detection to result
                        result.add_detection(
                            bbox=bbox,
                            confidence=float(confidence),
                            class_id=0,  # Single class for license plate text
                            class_name='license_plate',
                            attributes={
                                'text': cleaned_text,
                                'raw_text': text,
                                'is_valid_plate': is_valid_plate,
                                'text_length': len(cleaned_text),
                                'bbox_points': bbox_points
                            }
                        )
            
            # Add processing metadata
            result.input_shape = self.config.input_size
            result.raw_output = model_output
            
            return result
            
        except Exception as e:
            logger.error(f"Postprocessing failed: {str(e)}")
            # Return error result
            return InferenceResult(
                success=False,
                model_name=self.metadata['name'],
                model_type='license_plate_ocr',
                metadata={'error': str(e)}
            )
    
    def _clean_license_plate_text(self, text: str) -> str:
        """
        Clean and normalize license plate text.
        
        Args:
            text: Raw OCR text
            
        Returns:
            str: Cleaned text
        """
        # Remove extra whitespace and convert to uppercase
        cleaned = text.strip().upper()
        
        # Replace common noise characters with appropriate ones
        replacements = {
            '.': '-',  # Dot to dash
            '_': '-',  # Underscore to dash
            ' ': '',   # Remove spaces
            '@': '',   # Remove special chars
            '#': '',   # Remove special chars
        }
        
        for old_char, new_char in replacements.items():
            cleaned = cleaned.replace(old_char, new_char)
        
        # Remove any remaining non-alphanumeric characters except dash
        cleaned = re.sub(r'[^\w\-]', '', cleaned)
        
        # Fix common OCR character confusions (more conservative)
        char_fixes = {
            'O': '0',  # O to 0 in number contexts
            'I': '1',  # I to 1 in number contexts
        }
        
        # Apply fixes only in specific contexts
        if len(cleaned) >= 6:  # Only for longer license plates
            # Fix O to 0 if surrounded by numbers
            cleaned = re.sub(r'(?<=\d)O(?=\d)', '0', cleaned)
            # Fix I to 1 if surrounded by numbers  
            cleaned = re.sub(r'(?<=\d)I(?=\d)', '1', cleaned)
        
        return cleaned
    
    def _validate_license_plate(self, text: str) -> bool:
        """
        Validate if text matches license plate patterns.
        
        Args:
            text: License plate text
            
        Returns:
            bool: True if text matches a valid pattern
        """
        for pattern in self.license_plate_patterns:
            if re.match(pattern, text):
                return True
        return False
    
    def recognize_license_plate(self, image: Union[str, np.ndarray], 
                              min_confidence: Optional[float] = None) -> InferenceResult:
        """
        Recognize license plate text in an image.
        
        Args:
            image: Input image (file path or numpy array)
            min_confidence: Minimum confidence threshold (overrides config)
            
        Returns:
            InferenceResult: License plate recognition results
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Use custom confidence threshold if provided
        original_confidence = self.min_text_confidence
        if min_confidence is not None:
            self.min_text_confidence = min_confidence
        
        try:
            # Execute inference
            result = self.predict(image)
            return result
            
        finally:
            # Restore original confidence threshold
            self.min_text_confidence = original_confidence
    
    def extract_text_only(self, image: Union[str, np.ndarray]) -> List[str]:
        """
        Extract only the text from license plates (no bounding boxes).
        
        Args:
            image: Input image
            
        Returns:
            List of recognized text strings
        """
        result = self.recognize_license_plate(image)
        return [detection.attributes['text'] for detection in result.detections]
    
    def get_best_license_plate(self, image: Union[str, np.ndarray]) -> Optional[Dict[str, Any]]:
        """
        Get the best (highest confidence) license plate from an image.
        
        Args:
            image: Input image
            
        Returns:
            Dict with best license plate info or None
        """
        result = self.recognize_license_plate(image)
        
        if not result.detections:
            return None
        
        # Find detection with highest confidence
        best_detection = max(result.detections, key=lambda d: d.confidence)
        
        return {
            'text': best_detection.attributes['text'],
            'confidence': best_detection.confidence,
            'bbox': best_detection.bbox,
            'is_valid': best_detection.attributes['is_valid_plate']
        }
    
    def get_valid_license_plates(self, image: Union[str, np.ndarray]) -> List[Dict[str, Any]]:
        """
        Get only valid license plates (matching known patterns).
        
        Args:
            image: Input image
            
        Returns:
            List of valid license plate info
        """
        result = self.recognize_license_plate(image)
        
        valid_plates = []
        for detection in result.detections:
            if detection.attributes['is_valid_plate']:
                valid_plates.append({
                    'text': detection.attributes['text'],
                    'confidence': detection.confidence,
                    'bbox': detection.bbox,
                    'pattern_matched': True
                })
        
        return valid_plates
    
    def _optimize_for_raspberry_pi(self) -> bool:
        """Optimize for Raspberry Pi 5 with Hailo-8."""
        try:
            # Use CPU for OCR on Pi (more stable)
            self.config.use_gpu = False
            self.config.input_size = (320, 240)  # Smaller input size
            self.min_text_confidence = 0.6  # Higher threshold
            self.preprocess_enabled = True  # Enable preprocessing
            return True
        except Exception as e:
            logger.error(f"Raspberry Pi optimization failed: {e}")
            return False
    
    def _optimize_for_radxa_rock(self) -> bool:
        """Optimize for Radxa Rock 5 with NPU."""
        try:
            # Use CPU for OCR (NPU not supported by EasyOCR)
            self.config.use_gpu = False
            self.config.input_size = (640, 480)
            self.min_text_confidence = 0.5
            self.preprocess_enabled = True
            return True
        except Exception as e:
            logger.error(f"Radxa Rock optimization failed: {e}")
            return False
    
    def _optimize_for_jetson(self) -> bool:
        """Optimize for Jetson Nano with GPU."""
        try:
            # Use GPU if available
            self.config.use_gpu = True
            self.config.input_size = (640, 480)
            self.min_text_confidence = 0.4
            self.preprocess_enabled = True
            return True
        except Exception as e:
            logger.error(f"Jetson optimization failed: {e}")
            return False
    
    def _optimize_for_core_i5(self) -> bool:
        """Optimize for Core i5 with CUDA."""
        try:
            # Use GPU for faster processing
            if self.device.type == 'cuda':
                self.config.use_gpu = True
                self.config.input_size = (800, 600)  # Higher resolution
                self.min_text_confidence = 0.3  # Lower threshold (more sensitive)
                self.preprocess_enabled = True
            
            return True
        except Exception as e:
            logger.error(f"Core i5 optimization failed: {e}")
            return False