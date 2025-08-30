# ==========================================
# AI Box - Face Detection Model
# YOLOv8-based face detection implementation
# ==========================================

import cv2
import numpy as np
import torch
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import logging

from ultralytics import YOLO
from ..common.base_model import BaseModel
from ..common.model_config import ModelConfig
from ..common.inference_result import InferenceResult, Detection

logger = logging.getLogger(__name__)

class FaceDetector(BaseModel):
    """
    Face Detection Model using YOLOv8.
    
    This model detects faces in images with high accuracy and speed.
    Supports multiple face detection scenarios including:
    - Single face detection
    - Multiple face detection
    - Face landmark detection (optional)
    - Confidence-based filtering
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize Face Detector.
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        
        # Face detection specific attributes
        self.face_classes = ['face']
        self.landmark_points = 5  # Number of facial landmarks
        
        # Model metadata
        self.metadata = {
            'name': 'YOLOv8-Face-Detector',
            'version': '1.0.0',
            'description': 'YOLOv8-based face detection model',
            'author': 'Oryza AI Team',
            'framework': 'Ultralytics YOLOv8',
            'input_shape': self.config.input_size,
            'output_shape': None,  # Dynamic based on detections
            'model_size_mb': 0.0,  # Will be updated after loading
            'parameters': 0,  # Will be updated after loading
            'created_date': '2025-08-30',
            'last_updated': '2025-08-30'
        }
        
        logger.info(f"Initialized FaceDetector with config: {config}")
    
    def load_model(self) -> bool:
        """
        Load the YOLOv8 face detection model.
        
        Returns:
            bool: True if model loaded successfully
        """
        try:
            logger.info("Loading YOLOv8 face detection model...")
            
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
            logger.info("Face detection model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load face detection model: {str(e)}")
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
        
        # Use default YOLOv8 face detection model
        model_name = "yolov8n-face.pt"
        
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
        Preprocess input image for face detection.
        
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
                model_type='face_detection'
            )
            
            # Process YOLOv8 results
            if hasattr(model_output, 'boxes') and model_output.boxes is not None:
                boxes = model_output.boxes
                
                # Get detections
                if boxes.xyxy is not None:
                    bboxes = boxes.xyxy.cpu().numpy()
                    confidences = boxes.conf.cpu().numpy()
                    class_ids = boxes.cls.cpu().numpy().astype(int)
                    
                    # Add each detection
                    for i in range(len(bboxes)):
                        bbox = bboxes[i].tolist()
                        confidence = float(confidences[i])
                        class_id = int(class_ids[i])
                        class_name = self.face_classes[0]  # 'face'
                        
                        # Get landmarks if available
                        landmarks = None
                        if hasattr(boxes, 'keypoints') and boxes.keypoints is not None:
                            keypoints = boxes.keypoints[i].cpu().numpy()
                            if len(keypoints) > 0:
                                landmarks = keypoints.tolist()
                        
                        # Add detection to result
                        result.add_detection(
                            bbox=bbox,
                            confidence=confidence,
                            class_id=class_id,
                            class_name=class_name,
                            landmarks=landmarks
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
                model_type='face_detection',
                metadata={'error': str(e)}
            )
    
    def detect_faces(self, image: Union[str, np.ndarray], 
                    min_confidence: Optional[float] = None) -> InferenceResult:
        """
        Detect faces in an image.
        
        Args:
            image: Input image (file path or numpy array)
            min_confidence: Minimum confidence threshold (overrides config)
            
        Returns:
            InferenceResult: Face detection results
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
    
    def detect_single_face(self, image: Union[str, np.ndarray]) -> Optional[Detection]:
        """
        Detect the most confident single face in an image.
        
        Args:
            image: Input image
            
        Returns:
            Detection: Best face detection or None
        """
        result = self.detect_faces(image)
        return result.get_best_detection()
    
    def count_faces(self, image: Union[str, np.ndarray]) -> int:
        """
        Count the number of faces in an image.
        
        Args:
            image: Input image
            
        Returns:
            int: Number of faces detected
        """
        result = self.detect_faces(image)
        return result.count_detections('face')
    
    def get_face_bboxes(self, image: Union[str, np.ndarray]) -> List[List[float]]:
        """
        Get bounding boxes of all detected faces.
        
        Args:
            image: Input image
            
        Returns:
            List of bounding boxes [x1, y1, x2, y2]
        """
        result = self.detect_faces(image)
        return result.get_bboxes()
    
    def get_face_landmarks(self, image: Union[str, np.ndarray]) -> List[List[List[float]]]:
        """
        Get facial landmarks for all detected faces.
        
        Args:
            image: Input image
            
        Returns:
            List of landmark points for each face
        """
        result = self.detect_faces(image)
        landmarks = []
        for detection in result.detections:
            if detection.landmarks is not None:
                landmarks.append(detection.landmarks)
        return landmarks
    
    def _optimize_for_raspberry_pi(self) -> bool:
        """Optimize for Raspberry Pi 5 with Hailo-8."""
        try:
            # Use smaller model for Pi 5
            if 'yolov8n' not in str(self.config.model_path):
                logger.info("Switching to YOLOv8n for Raspberry Pi 5 optimization")
                self.config.model_path = "yolov8n-face.pt"
                self.config.input_size = (320, 320)  # Smaller input size
                self.config.batch_size = 1
                self.config.num_threads = 2
            
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
            
            return True
        except Exception as e:
            logger.error(f"Core i5 optimization failed: {e}")
            return False
