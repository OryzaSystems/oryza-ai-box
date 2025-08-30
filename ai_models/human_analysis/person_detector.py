# ==========================================
# AI Box - Person Detection Model
# YOLOv8-based person detection implementation
# ==========================================

import cv2
import numpy as np
import torch
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
import logging

from ultralytics import YOLO
from ..common.base_model import BaseModel
from ..common.model_config import ModelConfig
from ..common.inference_result import InferenceResult, Detection

logger = logging.getLogger(__name__)

class PersonDetector(BaseModel):
    """
    Person Detection Model using YOLOv8.
    
    This model detects people in images with high accuracy and speed.
    Supports multiple person detection scenarios including:
    - Single person detection
    - Multiple person detection
    - Person counting
    - Person tracking (basic)
    - Confidence-based filtering
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize Person Detector.
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        
        # Person detection specific attributes
        self.person_class_id = 0  # COCO dataset person class
        self.person_classes = ['person']
        self.min_person_size = config.model_params.get('min_person_size', 50)
        self.max_persons = config.model_params.get('max_persons', 100)
        
        # Tracking attributes
        self.track_history: Dict[int, List[Tuple[int, int]]] = {}
        self.next_track_id = 1
        
        # Model metadata
        self.metadata = {
            'name': 'YOLOv8-Person-Detector',
            'version': '1.0.0',
            'description': 'YOLOv8-based person detection model',
            'author': 'Oryza AI Team',
            'framework': 'Ultralytics YOLOv8',
            'input_shape': self.config.input_size,
            'output_shape': None,  # Dynamic based on detections
            'model_size_mb': 0.0,  # Will be updated after loading
            'parameters': 0,  # Will be updated after loading
            'created_date': '2025-08-30',
            'last_updated': '2025-08-30'
        }
        
        logger.info(f"Initialized PersonDetector with config: {config}")
    
    def load_model(self) -> bool:
        """
        Load the YOLOv8 person detection model.
        
        Returns:
            bool: True if model loaded successfully
        """
        try:
            logger.info("Loading YOLOv8 person detection model...")
            
            # Determine model path
            model_path = self._get_model_path()
            
            # Load YOLOv8 model
            self.model = YOLO(model_path)
            
            # Set model parameters
            self.model.conf = self.config.confidence_threshold
            self.model.iou = self.config.nms_threshold
            self.model.max_det = self.config.max_detections
            
            # Move to device
            if self.device.type == 'cuda':
                self.model.to(self.device)
            
            # Update metadata
            self._update_model_metadata()
            
            self.is_loaded = True
            logger.info("Person detection model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load person detection model: {str(e)}")
            return False
    
    def _get_model_path(self) -> str:
        """
        Get the model file path.
        
        Returns:
            str: Path to the model file
        """
        # Check if custom model path is provided
        if self.config.model_path and Path(self.config.model_path).exists():
            return self.config.model_path
        
        # Use default YOLOv8 person detection model
        model_name = "yolov8n.pt"  # YOLOv8n includes person detection
        
        # Check cache directory
        cache_path = Path(self.config.cache_dir) / model_name
        if cache_path.exists():
            return str(cache_path)
        
        # Download from Ultralytics hub
        logger.info(f"Downloading {model_name} from Ultralytics hub...")
        return model_name  # YOLO will auto-download
    
    def _update_model_metadata(self):
        """Update model metadata with actual model information."""
        if self.model is None:
            return
        
        try:
            # Get model info
            model_info = self.model.info()
            
            # Update metadata
            self.metadata.update({
                'model_size_mb': model_info.get('model_size_mb', 0.0),
                'parameters': model_info.get('parameters', 0),
                'output_shape': self.config.input_size  # YOLOv8 outputs detections
            })
            
        except Exception as e:
            logger.warning(f"Could not update model metadata: {e}")
    
    def preprocess(self, input_data: Union[str, np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Preprocess input image for person detection.
        
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
            
            # Ensure BGR format for OpenCV
            if image.ndim == 3 and image.shape[2] == 3:
                # Convert RGB to BGR if needed
                if hasattr(self, '_last_image') and np.array_equal(image, self._last_image):
                    # Skip conversion if same image
                    return image
                
                # Check if RGB (common in PIL/numpy) and convert to BGR
                if image.dtype == np.uint8:
                    # Simple heuristic: if red channel > blue channel, likely RGB
                    if np.mean(image[:, :, 0]) > np.mean(image[:, :, 2]):
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            self._last_image = image.copy()
            return image
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            raise
    
    def postprocess(self, model_output: Any) -> InferenceResult:
        """
        Postprocess YOLOv8 output into standardized format.
        
        Args:
            model_output: Raw YOLOv8 model output
            
        Returns:
            InferenceResult: Standardized inference result
        """
        try:
            # Create result object
            result = InferenceResult(
                success=True,
                model_name=self.metadata['name'],
                model_type='person_detection'
            )
            
            # Process YOLOv8 results
            if hasattr(model_output, 'boxes') and model_output.boxes is not None:
                boxes = model_output.boxes
                
                # Get detections
                if boxes.xyxy is not None:
                    bboxes = boxes.xyxy.cpu().numpy()
                    confidences = boxes.conf.cpu().numpy()
                    class_ids = boxes.cls.cpu().numpy().astype(int)
                    
                    # Filter for person class only
                    person_mask = class_ids == self.person_class_id
                    person_bboxes = bboxes[person_mask]
                    person_confidences = confidences[person_mask]
                    person_class_ids = class_ids[person_mask]
                    
                    # Add each person detection
                    for i in range(len(person_bboxes)):
                        bbox = person_bboxes[i].tolist()
                        confidence = float(person_confidences[i])
                        class_id = int(person_class_ids[i])
                        class_name = self.person_classes[0]  # 'person'
                        
                        # Filter by minimum person size
                        x1, y1, x2, y2 = bbox
                        person_width = x2 - x1
                        person_height = y2 - y1
                        
                        if person_width >= self.min_person_size and person_height >= self.min_person_size:
                            # Add detection to result
                            result.add_detection(
                                bbox=bbox,
                                confidence=confidence,
                                class_id=class_id,
                                class_name=class_name,
                                attributes={
                                    'width': person_width,
                                    'height': person_height,
                                    'area': person_width * person_height
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
                model_type='person_detection',
                metadata={'error': str(e)}
            )
    
    def detect_persons(self, image: Union[str, np.ndarray], 
                      min_confidence: Optional[float] = None) -> InferenceResult:
        """
        Detect persons in an image.
        
        Args:
            image: Input image (file path or numpy array)
            min_confidence: Minimum confidence threshold (overrides config)
            
        Returns:
            InferenceResult: Person detection results
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Use custom confidence threshold if provided
        original_conf = self.model.conf
        if min_confidence is not None:
            self.model.conf = min_confidence
        
        try:
            # Execute inference
            result = self.predict(image)
            
            # Apply NMS if configured
            if self.config.nms_threshold > 0:
                result = result.filter_by_nms(self.config.nms_threshold)
            
            return result
            
        finally:
            # Restore original confidence threshold
            self.model.conf = original_conf
    
    def count_persons(self, image: Union[str, np.ndarray]) -> int:
        """
        Count the number of persons in an image.
        
        Args:
            image: Input image
            
        Returns:
            int: Number of persons detected
        """
        result = self.detect_persons(image)
        return result.count_detections('person')
    
    def get_person_bboxes(self, image: Union[str, np.ndarray]) -> List[List[float]]:
        """
        Get bounding boxes of all detected persons.
        
        Args:
            image: Input image
            
        Returns:
            List of bounding boxes [x1, y1, x2, y2]
        """
        result = self.detect_persons(image)
        return result.get_bboxes()
    
    def get_person_confidences(self, image: Union[str, np.ndarray]) -> List[float]:
        """
        Get confidence scores of all detected persons.
        
        Args:
            image: Input image
            
        Returns:
            List of confidence scores
        """
        result = self.detect_persons(image)
        return result.get_confidences()
    
    def detect_single_person(self, image: Union[str, np.ndarray]) -> Optional[Detection]:
        """
        Detect the most confident single person in an image.
        
        Args:
            image: Input image
            
        Returns:
            Detection: Best person detection or None
        """
        result = self.detect_persons(image)
        return result.get_best_detection()
    
    def track_persons(self, image: Union[str, np.ndarray]) -> Dict[int, List[Tuple[int, int]]]:
        """
        Track persons across frames (basic implementation).
        
        Args:
            image: Input image
            
        Returns:
            Dict of track_id -> list of (x, y) positions
        """
        result = self.detect_persons(image)
        
        # Simple tracking based on center points
        current_centers = []
        for detection in result.detections:
            x1, y1, x2, y2 = detection.bbox
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            current_centers.append((center_x, center_y))
        
        # Update tracking history
        for center in current_centers:
            # Simple assignment - in real implementation, use more sophisticated tracking
            track_id = self.next_track_id
            if track_id not in self.track_history:
                self.track_history[track_id] = []
            self.track_history[track_id].append(center)
            self.next_track_id += 1
        
        return self.track_history
    
    def get_person_areas(self, image: Union[str, np.ndarray]) -> List[float]:
        """
        Get areas of all detected persons.
        
        Args:
            image: Input image
            
        Returns:
            List of person areas (width * height)
        """
        result = self.detect_persons(image)
        areas = []
        for detection in result.detections:
            x1, y1, x2, y2 = detection.bbox
            area = (x2 - x1) * (y2 - y1)
            areas.append(area)
        return areas
    
    def filter_by_size(self, image: Union[str, np.ndarray], 
                      min_size: int, max_size: Optional[int] = None) -> InferenceResult:
        """
        Filter person detections by size.
        
        Args:
            image: Input image
            min_size: Minimum person size (width or height)
            max_size: Maximum person size (width or height)
            
        Returns:
            InferenceResult: Filtered person detection results
        """
        result = self.detect_persons(image)
        
        # Filter detections by size
        filtered_detections = []
        for detection in result.detections:
            x1, y1, x2, y2 = detection.bbox
            width = x2 - x1
            height = y2 - y1
            
            # Check size constraints
            if width >= min_size and height >= min_size:
                if max_size is None or (width <= max_size and height <= max_size):
                    filtered_detections.append(detection)
        
        # Create new result with filtered detections
        filtered_result = InferenceResult(
            success=result.success,
            model_name=result.model_name,
            model_type=result.model_type,
            detections=filtered_detections,
            raw_output=result.raw_output,
            input_shape=result.input_shape,
            output_shape=result.output_shape,
            processing_time=result.processing_time,
            metadata=result.metadata,
            timestamp=result.timestamp
        )
        
        return filtered_result
    
    def _optimize_for_raspberry_pi(self) -> bool:
        """Optimize for Raspberry Pi 5 with Hailo-8."""
        try:
            # Use smaller model for Pi 5
            if 'yolov8n' not in str(self.config.model_path):
                logger.info("Switching to YOLOv8n for Raspberry Pi 5 optimization")
                self.config.model_path = "yolov8n.pt"
                self.config.input_size = (320, 320)  # Smaller input size
                self.config.batch_size = 1
                self.config.num_threads = 2
                self.min_person_size = 30  # Smaller minimum size
            
            return True
        except Exception as e:
            logger.error(f"Raspberry Pi optimization failed: {e}")
            return False
    
    def _optimize_for_radxa_rock(self) -> bool:
        """Optimize for Radxa Rock 5 with NPU."""
        try:
            # Optimize for NPU
            self.config.input_size = (416, 416)
            self.config.batch_size = 1
            self.config.num_threads = 4
            self.min_person_size = 40
            return True
        except Exception as e:
            logger.error(f"Radxa Rock optimization failed: {e}")
            return False
    
    def _optimize_for_jetson(self) -> bool:
        """Optimize for Jetson Nano with TensorRT."""
        try:
            # Enable TensorRT optimization
            if hasattr(self.model, 'fuse'):
                self.model.fuse()
            
            self.config.input_size = (640, 640)
            self.config.batch_size = 1
            self.min_person_size = 50
            return True
        except Exception as e:
            logger.error(f"Jetson optimization failed: {e}")
            return False
    
    def _optimize_for_core_i5(self) -> bool:
        """Optimize for Core i5 with CUDA."""
        try:
            # Enable CUDA optimizations
            if self.device.type == 'cuda':
                self.config.input_size = (640, 640)
                self.config.batch_size = 4
                self.config.use_fp16 = True
                self.min_person_size = 50
            
            return True
        except Exception as e:
            logger.error(f"Core i5 optimization failed: {e}")
            return False
