# ==========================================
# AI Box - Vehicle Analysis Module
# Vehicle detection, classification, and analytics
# ==========================================

from .vehicle_detector import VehicleDetector
from .license_plate_ocr import LicensePlateOCR

__all__ = [
    'VehicleDetector',
    'LicensePlateOCR'
]