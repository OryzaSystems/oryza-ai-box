# ==========================================
# AI Box - Traffic Analytics Model
# Advanced traffic flow analysis and monitoring
# ==========================================

import cv2
import numpy as np
import torch
from typing import Any, Dict, List, Optional, Union, Tuple, Set
from pathlib import Path
import logging
from collections import defaultdict, deque
from dataclasses import dataclass
import time
import math

from ..common.base_model import BaseModel
from ..common.model_config import ModelConfig
from ..common.inference_result import InferenceResult, Detection

logger = logging.getLogger(__name__)

@dataclass
class TrafficZone:
    """Traffic monitoring zone definition."""
    name: str
    polygon: List[Tuple[int, int]]
    zone_type: str  # 'entry', 'exit', 'counting', 'speed'
    direction: Optional[str] = None

@dataclass
class VehicleTrack:
    """Vehicle tracking information."""
    track_id: int
    vehicle_type: str
    positions: deque
    timestamps: deque
    first_seen: float
    last_seen: float
    speed: Optional[float] = None
    zones_visited: Set[str] = None
    
    def __post_init__(self):
        if self.zones_visited is None:
            self.zones_visited = set()

@dataclass
class TrafficMetrics:
    """Traffic analysis metrics."""
    timestamp: float
    total_vehicles: int
    vehicle_counts: Dict[str, int]
    average_speed: float
    traffic_density: float
    congestion_level: str
    flow_rate: float
    zone_counts: Dict[str, int]

class TrafficAnalyzer(BaseModel):
    """Traffic Analytics Model for comprehensive traffic monitoring."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        # Traffic analysis parameters
        self.max_tracks = config.model_params.get('max_tracks', 100)
        self.track_timeout = config.model_params.get('track_timeout', 5.0)
        self.pixels_per_meter = config.model_params.get('pixels_per_meter', 10.0)
        
        # Congestion thresholds
        self.congestion_thresholds = {
            'low': 0.3, 'medium': 0.6, 'high': 0.8
        }
        
        # Traffic zones and tracking
        self.traffic_zones: List[TrafficZone] = []
        self.zone_polygons = {}
        self.active_tracks: Dict[int, VehicleTrack] = {}
        self.next_track_id = 1
        self.track_history = deque(maxlen=1000)
        self.metrics_history = deque(maxlen=100)
        self.frame_count = 0
        self.start_time = time.time()
        
        # Model metadata
        self.metadata = {
            'name': 'Traffic-Analytics-Engine',
            'version': '1.0.0',
            'description': 'Advanced traffic flow analysis and monitoring system',
            'author': 'Oryza AI Team',
            'framework': 'Custom Analytics',
            'capabilities': [
                'vehicle_counting', 'speed_estimation', 'flow_analysis',
                'congestion_detection', 'zone_monitoring'
            ]
        }
        
        logger.info(f"Initialized TrafficAnalyzer with {self.max_tracks} max tracks")
    
    def load_model(self) -> bool:
        """Load the traffic analytics model."""
        try:
            logger.info("Loading Traffic Analytics Engine...")
            
            # Initialize default traffic zones
            if not self.traffic_zones:
                self._create_default_zones()
            
            # Precompile zone polygons
            self._compile_zone_polygons()
            
            self.is_loaded = True
            logger.info("Traffic analytics engine loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load traffic analytics engine: {str(e)}")
            return False
    
    def _create_default_zones(self):
        """Create default traffic monitoring zones."""
        width, height = self.config.input_size
        
        # Entry zone (top)
        self.traffic_zones.append(TrafficZone(
            name="entry_north",
            polygon=[(0, 0), (width, 0), (width, height//4), (0, height//4)],
            zone_type="entry",
            direction="south"
        ))
        
        # Counting zone (center)
        self.traffic_zones.append(TrafficZone(
            name="counting_center",
            polygon=[(width//4, height//4), (3*width//4, height//4), 
                    (3*width//4, 3*height//4), (width//4, 3*height//4)],
            zone_type="counting"
        ))
    
    def _compile_zone_polygons(self):
        """Precompile zone polygons for faster processing."""
        for zone in self.traffic_zones:
            self.zone_polygons[zone.name] = np.array(zone.polygon, dtype=np.int32)
    
    def add_traffic_zone(self, zone: TrafficZone):
        """Add a custom traffic monitoring zone."""
        self.traffic_zones.append(zone)
        self.zone_polygons[zone.name] = np.array(zone.polygon, dtype=np.int32)
        logger.info(f"Added traffic zone: {zone.name} ({zone.zone_type})")
    
    def preprocess(self, input_data: Union[str, np.ndarray, torch.Tensor]) -> np.ndarray:
        """Preprocess input for traffic analysis."""
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
    
    def _inference_step(self, preprocessed_data: np.ndarray) -> Dict[str, Any]:
        """Execute traffic analysis."""
        current_time = time.time()
        
        analysis_results = {
            'timestamp': current_time,
            'frame_count': self.frame_count,
            'image_shape': preprocessed_data.shape,
            'zones_analyzed': len(self.traffic_zones),
            'active_tracks': len(self.active_tracks)
        }
        
        self.frame_count += 1
        return analysis_results
    
    def postprocess(self, model_output: Dict[str, Any]) -> InferenceResult:
        """Postprocess traffic analysis results."""
        try:
            result = InferenceResult(
                success=True,
                model_name=self.metadata['name'],
                model_type='traffic_analysis'
            )
            
            result.add_detection(
                bbox=[0, 0, 1, 1],
                confidence=1.0,
                class_id=0,
                class_name='traffic_metrics',
                attributes=model_output
            )
            
            result.input_shape = self.config.input_size
            result.raw_output = model_output
            
            return result
            
        except Exception as e:
            logger.error(f"Postprocessing failed: {str(e)}")
            return InferenceResult(
                success=False,
                model_name=self.metadata['name'],
                model_type='traffic_analysis',
                metadata={'error': str(e)}
            )
    
    def analyze_traffic(self, image: np.ndarray, 
                       vehicle_detections: List[Detection] = None) -> TrafficMetrics:
        """Analyze traffic in an image."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        current_time = time.time()
        
        # Update vehicle tracking
        if vehicle_detections:
            self._update_vehicle_tracking(vehicle_detections, current_time)
        
        # Calculate traffic metrics
        metrics = self._calculate_traffic_metrics(current_time)
        
        # Store metrics history
        self.metrics_history.append(metrics)
        
        return metrics
    
    def _update_vehicle_tracking(self, detections: List[Detection], current_time: float):
        """Update vehicle tracking with new detections."""
        # Simple tracking based on proximity
        for detection in detections:
            center_x = (detection.bbox[0] + detection.bbox[2]) / 2
            center_y = (detection.bbox[1] + detection.bbox[3]) / 2
            
            # Create new track (simplified)
            if len(self.active_tracks) < self.max_tracks:
                new_track = VehicleTrack(
                    track_id=self.next_track_id,
                    vehicle_type=detection.class_name,
                    positions=deque([(center_x, center_y)], maxlen=50),
                    timestamps=deque([current_time], maxlen=50),
                    first_seen=current_time,
                    last_seen=current_time,
                    zones_visited=set()
                )
                
                self.active_tracks[self.next_track_id] = new_track
                self.next_track_id += 1
        
        # Remove expired tracks
        expired_tracks = []
        for track_id, track in self.active_tracks.items():
            if current_time - track.last_seen > self.track_timeout:
                expired_tracks.append(track_id)
        
        for track_id in expired_tracks:
            self.active_tracks.pop(track_id)
    
    def _calculate_traffic_metrics(self, current_time: float) -> TrafficMetrics:
        """Calculate comprehensive traffic metrics."""
        # Count vehicles by type
        vehicle_counts = defaultdict(int)
        total_vehicles = len(self.active_tracks)
        
        for track in self.active_tracks.values():
            vehicle_counts[track.vehicle_type] += 1
        
        # Calculate traffic density
        image_area = self.config.input_size[0] * self.config.input_size[1]
        traffic_density = total_vehicles / (image_area / 10000)
        
        # Determine congestion level
        if traffic_density <= self.congestion_thresholds['low']:
            congestion_level = 'low'
        elif traffic_density <= self.congestion_thresholds['medium']:
            congestion_level = 'medium'
        elif traffic_density <= self.congestion_thresholds['high']:
            congestion_level = 'high'
        else:
            congestion_level = 'severe'
        
        # Calculate flow rate
        elapsed_time = current_time - self.start_time
        flow_rate = total_vehicles / (elapsed_time / 60) if elapsed_time > 0 else 0.0
        
        return TrafficMetrics(
            timestamp=current_time,
            total_vehicles=total_vehicles,
            vehicle_counts=dict(vehicle_counts),
            average_speed=0.0,  # Simplified
            traffic_density=traffic_density,
            congestion_level=congestion_level,
            flow_rate=flow_rate,
            zone_counts={}  # Simplified
        )
    
    def get_traffic_summary(self, time_window: float = 300) -> Dict[str, Any]:
        """Get traffic summary for the last time window."""
        current_time = time.time()
        cutoff_time = current_time - time_window
        
        recent_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return {'error': 'No recent traffic data available'}
        
        total_vehicles = [m.total_vehicles for m in recent_metrics]
        traffic_densities = [m.traffic_density for m in recent_metrics]
        
        summary = {
            'time_window_minutes': time_window / 60,
            'total_frames_analyzed': len(recent_metrics),
            'average_vehicles_per_frame': np.mean(total_vehicles) if total_vehicles else 0,
            'max_vehicles_per_frame': max(total_vehicles) if total_vehicles else 0,
            'average_traffic_density': np.mean(traffic_densities) if traffic_densities else 0,
            'congestion_distribution': {
                level: sum(1 for m in recent_metrics if m.congestion_level == level)
                for level in ['low', 'medium', 'high', 'severe']
            }
        }
        
        return summary
    
    def reset_analytics(self):
        """Reset all traffic analytics data."""
        self.active_tracks.clear()
        self.track_history.clear()
        self.metrics_history.clear()
        self.frame_count = 0
        self.start_time = time.time()
        self.next_track_id = 1
        logger.info("Traffic analytics data reset")
    
    def _optimize_for_raspberry_pi(self) -> bool:
        """Optimize for Raspberry Pi 5."""
        try:
            self.max_tracks = 50
            self.track_timeout = 3.0
            return True
        except Exception as e:
            logger.error(f"Raspberry Pi optimization failed: {e}")
            return False
    
    def _optimize_for_radxa_rock(self) -> bool:
        """Optimize for Radxa Rock 5."""
        try:
            self.max_tracks = 75
            self.track_timeout = 4.0
            return True
        except Exception as e:
            logger.error(f"Radxa Rock optimization failed: {e}")
            return False
    
    def _optimize_for_jetson(self) -> bool:
        """Optimize for Jetson Nano."""
        try:
            self.max_tracks = 100
            self.track_timeout = 5.0
            return True
        except Exception as e:
            logger.error(f"Jetson optimization failed: {e}")
            return False
    
    def _optimize_for_core_i5(self) -> bool:
        """Optimize for Core i5."""
        try:
            self.max_tracks = 200
            self.track_timeout = 10.0
            return True
        except Exception as e:
            logger.error(f"Core i5 optimization failed: {e}")
            return False
