# ==========================================
# AI Box - Behavior Analysis Model
# Custom CNN-based behavior analysis implementation
# ==========================================

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
import logging
from collections import deque
import time

from ..common.base_model import BaseModel
from ..common.model_config import ModelConfig
from ..common.inference_result import InferenceResult, Detection

logger = logging.getLogger(__name__)

class BehaviorCNN(nn.Module):
    """Custom CNN for behavior analysis."""
    
    def __init__(self, num_classes: int = 4, dropout_rate: float = 0.5):
        super(BehaviorCNN, self).__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
        # Classifier layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class BehaviorAnalyzer(BaseModel):
    """
    Behavior Analysis Model using Custom CNN.
    
    Analyzes human behavior: standing, walking, running, sitting
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        # Behavior classes
        self.behavior_classes = ['standing', 'walking', 'running', 'sitting']
        self.num_classes = len(self.behavior_classes)
        
        # Temporal analysis
        self.temporal_window = config.model_params.get('temporal_window', 5)
        self.behavior_history: deque = deque(maxlen=self.temporal_window)
        self.smoothing_factor = config.model_params.get('smoothing_factor', 0.7)
        
        # Model parameters
        self.input_size = config.input_size or (224, 224)
        self.dropout_rate = config.model_params.get('dropout_rate', 0.5)
        
        # Model metadata
        self.metadata = {
            'name': 'Custom-CNN-Behavior-Analyzer',
            'version': '1.0.0',
            'description': 'Custom CNN-based behavior analysis model',
            'author': 'Oryza AI Team',
            'framework': 'PyTorch',
            'input_shape': (*self.input_size, 3),
            'output_shape': (self.num_classes,),
            'model_size_mb': 0.0,
            'parameters': 0,
            'created_date': '2025-08-30',
            'last_updated': '2025-08-30'
        }
        
        logger.info(f"Initialized BehaviorAnalyzer with config: {config}")
    
    def load_model(self) -> bool:
        """Load the behavior analysis model."""
        try:
            logger.info("Loading behavior analysis model...")
            
            # Create model
            self.model = BehaviorCNN(
                num_classes=self.num_classes,
                dropout_rate=self.dropout_rate
            )
            
            # Load pretrained weights if available
            model_path = self._get_model_path()
            if model_path and Path(model_path).exists():
                logger.info(f"Loading pretrained weights from {model_path}")
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                logger.info("No pretrained weights found, using random initialization")
                self._initialize_weights()
            
            # Move to device
            self.model.to(self.device)
            self.model.eval()
            
            # Update metadata
            self._update_model_metadata()
            
            self.is_loaded = True
            logger.info("Behavior analysis model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load behavior analysis model: {str(e)}")
            return False
    
    def _get_model_path(self) -> Optional[str]:
        """Get the model file path."""
        if self.config.model_path and Path(self.config.model_path).exists():
            return self.config.model_path
        
        cache_path = Path(self.config.cache_dir) / "behavior_analyzer.pth"
        if cache_path.exists():
            return str(cache_path)
        
        return None
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def _update_model_metadata(self):
        """Update model metadata."""
        if self.model is None:
            return
        
        try:
            param_count = sum(p.numel() for p in self.model.parameters())
            model_size_mb = param_count * 4 / (1024 * 1024)
            
            self.metadata.update({
                'model_size_mb': model_size_mb,
                'parameters': param_count,
                'output_shape': (self.num_classes,)
            })
            
        except Exception as e:
            logger.warning(f"Could not update model metadata: {e}")
    
    def preprocess(self, input_data: Union[str, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Preprocess input image for behavior analysis."""
        try:
            # Handle different input types
            if isinstance(input_data, str):
                image = cv2.imread(input_data)
                if image is None:
                    raise ValueError(f"Could not load image from path: {input_data}")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif isinstance(input_data, np.ndarray):
                image = input_data.copy()
                if image.ndim == 3 and image.shape[2] == 3:
                    if np.mean(image[:, :, 0]) < np.mean(image[:, :, 2]):
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif isinstance(input_data, torch.Tensor):
                if input_data.dim() == 4:
                    return input_data.to(self.device)
                image = input_data.cpu().numpy()
                if image.ndim == 3 and image.shape[0] in [1, 3]:
                    image = np.transpose(image, (1, 2, 0))
            else:
                raise ValueError(f"Unsupported input type: {type(input_data)}")
            
            # Resize to model input size
            image = cv2.resize(image, self.input_size)
            
            # Normalize to [0, 1]
            image = image.astype(np.float32) / 255.0
            
            # Normalize with ImageNet stats
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = (image - mean) / std
            
            # Convert to tensor and add batch dimension
            tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
            
            return tensor.to(self.device)
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            raise
    
    def postprocess(self, model_output: torch.Tensor) -> InferenceResult:
        """Postprocess model output into standardized format."""
        try:
            # Apply softmax to get probabilities
            probabilities = F.softmax(model_output, dim=1)
            probs = probabilities.cpu().numpy()[0]
            
            # Get predicted class
            predicted_class_id = np.argmax(probs)
            predicted_class_name = self.behavior_classes[predicted_class_id]
            confidence = float(probs[predicted_class_id])
            
            # Create result object
            result = InferenceResult(
                success=True,
                model_name=self.metadata['name'],
                model_type='behavior_analysis'
            )
            
            # Add behavior detection
            result.add_detection(
                bbox=[0, 0, self.input_size[0], self.input_size[1]],
                confidence=confidence,
                class_id=predicted_class_id,
                class_name=predicted_class_name,
                attributes={
                    'behavior_probabilities': {
                        self.behavior_classes[i]: float(probs[i]) 
                        for i in range(len(self.behavior_classes))
                    },
                    'temporal_smoothed': False
                }
            )
            
            result.input_shape = (*self.input_size, 3)
            result.raw_output = model_output.cpu().numpy()
            
            return result
            
        except Exception as e:
            logger.error(f"Postprocessing failed: {str(e)}")
            return InferenceResult(
                success=False,
                model_name=self.metadata['name'],
                model_type='behavior_analysis',
                metadata={'error': str(e)}
            )
    
    def analyze_behavior(self, image: Union[str, np.ndarray], 
                        use_temporal: bool = True) -> InferenceResult:
        """Analyze behavior in an image."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            result = self.predict(image)
            
            # Apply temporal smoothing if enabled
            if use_temporal and len(self.behavior_history) > 0:
                result = self._apply_temporal_smoothing(result)
            
            # Update behavior history
            if result.success and result.detections:
                detection = result.detections[0]
                self.behavior_history.append({
                    'behavior': detection.class_name,
                    'confidence': detection.confidence,
                    'timestamp': time.time(),
                    'probabilities': detection.attributes.get('behavior_probabilities', {})
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Behavior analysis failed: {e}")
            raise
    
    def _apply_temporal_smoothing(self, current_result: InferenceResult) -> InferenceResult:
        """Apply temporal smoothing to behavior predictions."""
        if not current_result.success or not current_result.detections:
            return current_result
        
        current_detection = current_result.detections[0]
        current_probs = current_detection.attributes.get('behavior_probabilities', {})
        
        # Calculate weighted average with history
        smoothed_probs = {}
        for behavior in self.behavior_classes:
            current_prob = current_probs.get(behavior, 0.0)
            
            historical_probs = [
                entry['probabilities'].get(behavior, 0.0) 
                for entry in self.behavior_history
            ]
            
            if historical_probs:
                weights = np.exp(np.linspace(-1, 0, len(historical_probs)))
                weights = weights / np.sum(weights)
                historical_avg = np.average(historical_probs, weights=weights)
                
                smoothed_prob = (self.smoothing_factor * current_prob + 
                               (1 - self.smoothing_factor) * historical_avg)
            else:
                smoothed_prob = current_prob
            
            smoothed_probs[behavior] = smoothed_prob
        
        # Find new predicted behavior
        predicted_behavior = max(smoothed_probs, key=smoothed_probs.get)
        predicted_confidence = smoothed_probs[predicted_behavior]
        predicted_class_id = self.behavior_classes.index(predicted_behavior)
        
        # Create new result with smoothed predictions
        smoothed_result = InferenceResult(
            success=True,
            model_name=self.metadata['name'],
            model_type='behavior_analysis'
        )
        
        smoothed_result.add_detection(
            bbox=current_detection.bbox,
            confidence=predicted_confidence,
            class_id=predicted_class_id,
            class_name=predicted_behavior,
            attributes={
                'behavior_probabilities': smoothed_probs,
                'temporal_smoothed': True,
                'original_prediction': current_detection.class_name,
                'original_confidence': current_detection.confidence
            }
        )
        
        smoothed_result.input_shape = current_result.input_shape
        smoothed_result.raw_output = current_result.raw_output
        
        return smoothed_result
    
    def get_behavior_probabilities(self, image: Union[str, np.ndarray]) -> Dict[str, float]:
        """Get behavior probabilities for all classes."""
        result = self.analyze_behavior(image, use_temporal=False)
        if result.success and result.detections:
            return result.detections[0].attributes.get('behavior_probabilities', {})
        return {}
    
    def get_dominant_behavior(self, image: Union[str, np.ndarray]) -> Tuple[str, float]:
        """Get the dominant behavior and its confidence."""
        result = self.analyze_behavior(image, use_temporal=False)
        if result.success and result.detections:
            detection = result.detections[0]
            return detection.class_name, detection.confidence
        return "unknown", 0.0
    
    def analyze_behavior_sequence(self, images: List[Union[str, np.ndarray]]) -> List[InferenceResult]:
        """Analyze behavior in a sequence of images."""
        results = []
        self.behavior_history.clear()
        
        for image in images:
            result = self.analyze_behavior(image, use_temporal=True)
            results.append(result)
        
        return results
    
    def get_behavior_trends(self) -> Dict[str, Any]:
        """Get behavior trends from history."""
        if len(self.behavior_history) < 2:
            return {'trend': 'insufficient_data', 'history_length': len(self.behavior_history)}
        
        behaviors = [entry['behavior'] for entry in self.behavior_history]
        confidences = [entry['confidence'] for entry in self.behavior_history]
        
        from collections import Counter
        behavior_counts = Counter(behaviors)
        most_common = behavior_counts.most_common(1)[0]
        
        confidence_trend = 'stable'
        if len(confidences) >= 3:
            recent_conf = np.mean(confidences[-3:])
            earlier_conf = np.mean(confidences[:-3])
            if recent_conf > earlier_conf + 0.1:
                confidence_trend = 'increasing'
            elif recent_conf < earlier_conf - 0.1:
                confidence_trend = 'decreasing'
        
        return {
            'most_common_behavior': most_common[0],
            'most_common_count': most_common[1],
            'confidence_trend': confidence_trend,
            'average_confidence': np.mean(confidences),
            'behavior_changes': len(set(behaviors)),
            'history_length': len(self.behavior_history),
            'recent_behaviors': behaviors[-3:] if len(behaviors) >= 3 else behaviors
        }
    
    def reset_temporal_history(self):
        """Reset temporal behavior history."""
        self.behavior_history.clear()
        logger.info("Temporal behavior history reset")
    
    def _optimize_for_raspberry_pi(self) -> bool:
        """Optimize for Raspberry Pi 5 with Hailo-8."""
        try:
            self.config.input_size = (160, 160)
            self.input_size = (160, 160)
            self.config.batch_size = 1
            self.dropout_rate = 0.3
            logger.info("Optimized for Raspberry Pi 5: Smaller input size")
            return True
        except Exception as e:
            logger.error(f"Raspberry Pi optimization failed: {e}")
            return False
    
    def _optimize_for_radxa_rock(self) -> bool:
        """Optimize for Radxa Rock 5 with NPU."""
        try:
            self.config.input_size = (224, 224)
            self.input_size = (224, 224)
            self.config.batch_size = 1
            self.dropout_rate = 0.4
            return True
        except Exception as e:
            logger.error(f"Radxa Rock optimization failed: {e}")
            return False
    
    def _optimize_for_jetson(self) -> bool:
        """Optimize for Jetson Nano with TensorRT."""
        try:
            self.config.input_size = (224, 224)
            self.input_size = (224, 224)
            self.config.batch_size = 1
            self.config.use_fp16 = True
            return True
        except Exception as e:
            logger.error(f"Jetson optimization failed: {e}")
            return False
    
    def _optimize_for_core_i5(self) -> bool:
        """Optimize for Core i5 with CUDA."""
        try:
            if self.device.type == 'cuda':
                self.config.input_size = (224, 224)
                self.input_size = (224, 224)
                self.config.batch_size = 4
                self.config.use_fp16 = True
            return True
        except Exception as e:
            logger.error(f"Core i5 optimization failed: {e}")
            return False
