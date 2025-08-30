# ==========================================
# AI Box - AI Models Package
# Human and Vehicle Analysis Models
# ==========================================

"""
AI Box - AI Models Package

This package contains all AI models for human and vehicle analysis:

Human Analysis:
- Face Detection (YOLOv8-Face)
- Face Recognition (FaceNet/ArcFace)
- Person Detection (YOLOv8-Person)
- Behavior Analysis (Custom CNN)

Vehicle Analysis:
- Vehicle Detection (YOLOv8-Vehicle)
- License Plate OCR (PaddleOCR)
- Vehicle Classification (ResNet50)
- Traffic Analytics (Custom algorithms)
"""

__version__ = "1.0.0"
__author__ = "Oryza AI Team"
__email__ = "ai@oryza.vn"

# Import main model classes
from .human_analysis import (
    FaceDetector,
    FaceRecognizer,
    PersonDetector,
    BehaviorAnalyzer
)

from .vehicle_analysis import (
    VehicleDetector,
    LicensePlateOCR,
    VehicleClassifier,
    TrafficAnalyzer
)

from .common import (
    BaseModel,
    ModelConfig,
    InferenceResult,
    ModelManager
)

__all__ = [
    # Human Analysis
    "FaceDetector",
    "FaceRecognizer", 
    "PersonDetector",
    "BehaviorAnalyzer",
    
    # Vehicle Analysis
    "VehicleDetector",
    "LicensePlateOCR",
    "VehicleClassifier",
    "TrafficAnalyzer",
    
    # Common
    "BaseModel",
    "ModelConfig",
    "InferenceResult",
    "ModelManager"
]
