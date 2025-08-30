# ==========================================
# AI Box - Vehicle Detection Model
# YOLOv8-based vehicle detection implementation
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

class VehicleDetector(BaseModel):
    """Vehicle Detection Model using YOLOv8."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        # Vehicle classes from COCO dataset
        self.vehicle_classes = {
            2: 'car',
            3: 'motorcycle', 
            5: 'bus',
            7: 'truck',
            1: 'bicycle'
        }
        self.vehicle_class_ids = list(self.vehicle_classes.keys())
        
        # Vehicle analysis parameters
        self.min_vehicle_size = config.model_params.get('min_vehicle_size', 100)
        self.vehicle_confidence_threshold = config.model_params.get('vehicle_confidence', 0.3)
        
        # Model metadata
        self.metadata = {
            'name': 'YOLOv8-Vehicle-Detector',
            'version': '1.0.0',
            'description': 'YOLOv8-based vehicle detection model',
            'author': 'Oryza AI Team',
            'framework': 'Ultralytics YOLOv8',
            'supported_vehicles': list(self.vehicle_classes.values())
        }
        
        logger.info(f"Initialized VehicleDetector: {list(self.vehicle_classes.values())}")
    
    def load_model(self) -> bool:
        """Load the YOLOv8 vehicle detection model."""
        try:
            logger.info("Loading YOLOv8 vehicle detection model...")
            
            # Use YOLOv8n model
            self.model = YOLO("yolov8n.pt")
            
            # Set model parameters
            self.model.conf = max(self.config.confidence_threshold, self.vehicle_confidence_threshold)
            self.model.iou = self.config.nms_threshold
            self.model.max_det = self.config.max_detections
            
            # Move to device
            if self.device.type == 'cuda':
                self.model.to(self.device)
            
            self.is_loaded = True
            logger.info("Vehicle detection model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load vehicle detection model: {str(e)}")
            return False
    
    def preprocess(self, input_data: Union[str, np.ndarray, torch.Tensor]) -> np.ndarray:
        """Preprocess input image for vehicle detection."""
        try:
            if isinstance(input_data, str):
                image = cv2.imread(input_data)
                if image is None:
                    raise ValueError(f"Could not load image from path: {input_data}")
            elif isinstance(input_data, np.ndarray):
                image = input_data.copy()
            elif isinstance(input_data, torch.Tensor):
                image = input_data.cpu().numpy()
                if image.ndim == 3 and image.shape[0] in [1, 3]:
                    image = np.transpose(image, (1, 2, 0))
            else:
                raise ValueError(f"Unsupported input type: {type(input_data)}")
            
            return image
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            raise
    
    def postprocess(self, model_output: Any) -> InferenceResult:
        """Postprocess YOLOv8 output into standardized format."""
        try:
            result = InferenceResult(
                success=True,
                model_name=self.metadata['name'],
                model_type='vehicle_detection'
            )
            
            # Process YOLOv8 results
            if hasattr(model_output, 'boxes') and model_output.boxes is not None:
                boxes = model_output.boxes
                
                if boxes.xyxy is not None:
                    bboxes = boxes.xyxy.cpu().numpy()
                    confidences = boxes.conf.cpu().numpy()
                    class_ids = boxes.cls.cpu().numpy().astype(int)
                    
                    # Filter for vehicle classes only
                    vehicle_mask = np.isin(class_ids, self.vehicle_class_ids)
                    vehicle_bboxes = bboxes[vehicle_mask]
                    vehicle_confidences = confidences[vehicle_mask]
                    vehicle_class_ids = class_ids[vehicle_mask]
                    
                    # Add each vehicle detection
                    for i in range(len(vehicle_bboxes)):
                        bbox = vehicle_bboxes[i].tolist()
                        confidence = float(vehicle_confidences[i])
                        class_id = int(vehicle_class_ids[i])
                        class_name = self.vehicle_classes.get(class_id, 'unknown_vehicle')
                        
                        # Filter by minimum vehicle size
                        x1, y1, x2, y2 = bbox
                        vehicle_area = (x2 - x1) * (y2 - y1)
                        
                        if vehicle_area >= self.min_vehicle_size:
                            result.add_detection(
                                bbox=bbox,
                                confidence=confidence,
                                class_id=class_id,
                                class_name=class_name,
                                attributes={
                                    'area': vehicle_area,
                                    'vehicle_type': class_name
                                }
                            )
            
            result.input_shape = self.config.input_size
            result.raw_output = model_output
            
            return result
            
        except Exception as e:
            logger.error(f"Postprocessing failed: {str(e)}")
            return InferenceResult(
                success=False,
                model_name=self.metadata['name'],
                model_type='vehicle_detection',
                metadata={'error': str(e)}
            )
    
    def detect_vehicles(self, image: Union[str, np.ndarray]) -> InferenceResult:
        """Detect vehicles in an image."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        return self.predict(image)
    
    def count_vehicles(self, image: Union[str, np.ndarray]) -> Dict[str, int]:
        """Count vehicles by type in an image."""
        result = self.detect_vehicles(image)
        
        vehicle_counts = {vehicle_type: 0 for vehicle_type in self.vehicle_classes.values()}
        vehicle_counts['total'] = 0
        
        for detection in result.detections:
            vehicle_type = detection.class_name
            if vehicle_type in vehicle_counts:
                vehicle_counts[vehicle_type] += 1
                vehicle_counts['total'] += 1
        
        return vehicle_counts
    
    def get_vehicles_by_type(self, image: Union[str, np.ndarray], 
                           vehicle_type: str) -> List[Detection]:
        """Get all vehicles of a specific type."""
        result = self.detect_vehicles(image)
        return [detection for detection in result.detections 
                if detection.class_name == vehicle_type]
    
    def analyze_traffic_flow(self, image: Union[str, np.ndarray]) -> Dict[str, Any]:
        """Analyze traffic flow in an image."""
        result = self.detect_vehicles(image)
        
        # Count vehicles by type
        vehicle_counts = self.count_vehicles(image)
        
        # Calculate traffic density
        image_area = self.config.input_size[0] * self.config.input_size[1]
        total_vehicle_area = sum(
            detection.attributes.get('area', 0) 
            for detection in result.detections
        )
        traffic_density = total_vehicle_area / image_area if image_area > 0 else 0
        
        # Calculate average vehicle size
        total_area = sum(detection.attributes.get('area', 0) for detection in result.detections)
        avg_vehicle_size = total_area / len(result.detections) if result.detections else 0
        
        return {
            'vehicle_counts': vehicle_counts,
            'traffic_density': traffic_density,
            'average_vehicle_size': avg_vehicle_size,
            'total_vehicles': len(result.detections),
            'detection_confidence': np.mean([d.confidence for d in result.detections]) if result.detections else 0
        }
    
    def _optimize_for_raspberry_pi(self) -> bool:
        """Optimize for Raspberry Pi 5 with Hailo-8."""
        try:
            self.config.input_size = (416, 416)
            self.config.batch_size = 1
            self.min_vehicle_size = 50
            self.vehicle_confidence_threshold = 0.4
            return True
        except Exception as e:
            logger.error(f"Raspberry Pi optimization failed: {e}")
            return False
    
    def _optimize_for_radxa_rock(self) -> bool:
        """Optimize for Radxa Rock 5 with NPU."""
        try:
            self.config.input_size = (640, 640)
            self.config.batch_size = 1
            self.min_vehicle_size = 100
            self.vehicle_confidence_threshold = 0.35
            return True
        except Exception as e:
            logger.error(f"Radxa Rock optimization failed: {e}")
            return False
    
    def _optimize_for_jetson(self) -> bool:
        """Optimize for Jetson Nano with TensorRT."""
        try:
            self.config.input_size = (640, 640)
            self.config.batch_size = 1
            self.min_vehicle_size = 100
            self.vehicle_confidence_threshold = 0.3
            return True
        except Exception as e:
            logger.error(f"Jetson optimization failed: {e}")
            return False
    
    def _optimize_for_core_i5(self) -> bool:
        """Optimize for Core i5 with CUDA."""
        try:
            if self.device.type == 'cuda':
                self.config.input_size = (640, 640)
                self.config.batch_size = 4
                self.config.use_fp16 = True
                self.min_vehicle_size = 100
                self.vehicle_confidence_threshold = 0.25
            return True
        except Exception as e:
            logger.error(f"Core i5 optimization failed: {e}")
            return False
