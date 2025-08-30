# ==========================================
# AI Box - Inference Result
# Standardized inference result format
# ==========================================

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import numpy as np
from datetime import datetime

@dataclass
class Detection:
    """Single detection result"""
    bbox: List[float]  # [x1, y1, x2, y2] or [x, y, width, height]
    confidence: float
    class_id: int
    class_name: str
    landmarks: Optional[List[List[float]]] = None  # Face landmarks
    attributes: Optional[Dict[str, Any]] = None  # Additional attributes

@dataclass
class InferenceResult:
    """
    Standardized inference result for all AI models.
    
    This class provides a consistent format for inference results
    across different model types (detection, classification, etc.)
    """
    
    # Basic result information
    success: bool
    model_name: str
    model_type: str
    
    # Detection results
    detections: List[Detection] = field(default_factory=list)
    
    # Classification results
    classifications: List[Dict[str, Any]] = field(default_factory=list)
    
    # Raw model output (for debugging)
    raw_output: Optional[Any] = None
    
    # Processing information
    input_shape: Optional[tuple] = None
    output_shape: Optional[tuple] = None
    processing_time: Optional[float] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Timestamp
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Post-initialization setup."""
        if not self.metadata:
            self.metadata = {}
    
    def add_detection(self, bbox: List[float], confidence: float, 
                     class_id: int, class_name: str, 
                     landmarks: Optional[List[List[float]]] = None,
                     attributes: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a detection to the result.
        
        Args:
            bbox: Bounding box coordinates
            confidence: Detection confidence
            class_id: Class ID
            class_name: Class name
            landmarks: Optional landmarks (for faces)
            attributes: Optional additional attributes
        """
        detection = Detection(
            bbox=bbox,
            confidence=confidence,
            class_id=class_id,
            class_name=class_name,
            landmarks=landmarks,
            attributes=attributes
        )
        self.detections.append(detection)
    
    def add_classification(self, class_id: int, class_name: str, 
                          confidence: float, attributes: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a classification result.
        
        Args:
            class_id: Class ID
            class_name: Class name
            confidence: Classification confidence
            attributes: Optional additional attributes
        """
        classification = {
            'class_id': class_id,
            'class_name': class_name,
            'confidence': confidence,
            'attributes': attributes or {}
        }
        self.classifications.append(classification)
    
    def get_detections_by_class(self, class_name: str) -> List[Detection]:
        """
        Get detections filtered by class name.
        
        Args:
            class_name: Class name to filter by
            
        Returns:
            List of detections for the specified class
        """
        return [d for d in self.detections if d.class_name == class_name]
    
    def get_detections_by_confidence(self, min_confidence: float) -> List[Detection]:
        """
        Get detections filtered by minimum confidence.
        
        Args:
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of detections above the confidence threshold
        """
        return [d for d in self.detections if d.confidence >= min_confidence]
    
    def get_best_detection(self) -> Optional[Detection]:
        """
        Get the detection with highest confidence.
        
        Returns:
            Detection with highest confidence or None
        """
        if not self.detections:
            return None
        return max(self.detections, key=lambda d: d.confidence)
    
    def get_best_classification(self) -> Optional[Dict[str, Any]]:
        """
        Get the classification with highest confidence.
        
        Returns:
            Classification with highest confidence or None
        """
        if not self.classifications:
            return None
        return max(self.classifications, key=lambda c: c['confidence'])
    
    def count_detections(self, class_name: Optional[str] = None) -> int:
        """
        Count total detections or detections by class.
        
        Args:
            class_name: Optional class name to filter by
            
        Returns:
            Number of detections
        """
        if class_name:
            return len(self.get_detections_by_class(class_name))
        return len(self.detections)
    
    def get_bboxes(self) -> List[List[float]]:
        """
        Get all bounding boxes.
        
        Returns:
            List of bounding boxes
        """
        return [d.bbox for d in self.detections]
    
    def get_confidences(self) -> List[float]:
        """
        Get all confidence scores.
        
        Returns:
            List of confidence scores
        """
        return [d.confidence for d in self.detections]
    
    def get_class_names(self) -> List[str]:
        """
        Get all class names.
        
        Returns:
            List of class names
        """
        return [d.class_name for d in self.detections]
    
    def filter_by_nms(self, iou_threshold: float = 0.5) -> 'InferenceResult':
        """
        Apply Non-Maximum Suppression to remove overlapping detections.
        
        Args:
            iou_threshold: IoU threshold for NMS
            
        Returns:
            New InferenceResult with NMS applied
        """
        if not self.detections:
            return self
        
        # Sort by confidence
        sorted_detections = sorted(self.detections, key=lambda d: d.confidence, reverse=True)
        
        # Apply NMS
        kept_detections = []
        for detection in sorted_detections:
            should_keep = True
            for kept in kept_detections:
                if self._calculate_iou(detection.bbox, kept.bbox) > iou_threshold:
                    should_keep = False
                    break
            if should_keep:
                kept_detections.append(detection)
        
        # Create new result
        new_result = InferenceResult(
            success=self.success,
            model_name=self.model_name,
            model_type=self.model_type,
            detections=kept_detections,
            classifications=self.classifications,
            raw_output=self.raw_output,
            input_shape=self.input_shape,
            output_shape=self.output_shape,
            processing_time=self.processing_time,
            metadata=self.metadata.copy(),
            timestamp=self.timestamp
        )
        
        return new_result
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """
        Calculate Intersection over Union between two bounding boxes.
        
        Args:
            bbox1: First bounding box [x1, y1, x2, y2]
            bbox2: Second bounding box [x1, y1, x2, y2]
            
        Returns:
            IoU value
        """
        # Convert to [x1, y1, x2, y2] format if needed
        if len(bbox1) == 4 and len(bbox2) == 4:
            x1_1, y1_1, x2_1, y2_1 = bbox1
            x1_2, y1_2, x2_2, y2_2 = bbox2
        else:
            # Assume [x, y, width, height] format
            x1_1, y1_1, w1, h1 = bbox1
            x2_1, y2_1 = x1_1 + w1, y1_1 + h1
            x1_2, y1_2, w2, h2 = bbox2
            x2_2, y2_2 = x1_2 + w2, y1_2 + h2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert result to dictionary.
        
        Returns:
            Dictionary representation of the result
        """
        return {
            'success': self.success,
            'model_name': self.model_name,
            'model_type': self.model_type,
            'detections': [
                {
                    'bbox': d.bbox,
                    'confidence': d.confidence,
                    'class_id': d.class_id,
                    'class_name': d.class_name,
                    'landmarks': d.landmarks,
                    'attributes': d.attributes
                }
                for d in self.detections
            ],
            'classifications': self.classifications,
            'input_shape': self.input_shape,
            'output_shape': self.output_shape,
            'processing_time': self.processing_time,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat()
        }
    
    def __str__(self) -> str:
        """String representation."""
        return f"InferenceResult(model={self.model_name}, detections={len(self.detections)}, success={self.success})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"InferenceResult(\n" + \
               f"  success={self.success},\n" + \
               f"  model_name='{self.model_name}',\n" + \
               f"  model_type='{self.model_type}',\n" + \
               f"  detections={len(self.detections)},\n" + \
               f"  classifications={len(self.classifications)},\n" + \
               f"  processing_time={self.processing_time},\n" + \
               f"  timestamp={self.timestamp}\n" + \
               f")"
