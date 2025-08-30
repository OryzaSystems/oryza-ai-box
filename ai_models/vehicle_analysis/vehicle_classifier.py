# ==========================================
# AI Box - Vehicle Classification Model
# ResNet50-based vehicle classification implementation
# ==========================================

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
import logging

from ..common.base_model import BaseModel
from ..common.model_config import ModelConfig
from ..common.inference_result import InferenceResult, Detection

logger = logging.getLogger(__name__)

class VehicleClassifier(BaseModel):
    """Vehicle Classification Model using ResNet50."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        # Vehicle classification categories
        self.vehicle_categories = {
            'car': ['sedan', 'suv', 'hatchback', 'coupe', 'convertible', 'wagon'],
            'truck': ['pickup', 'delivery', 'semi', 'dump', 'box_truck', 'flatbed'],
            'bus': ['city_bus', 'school_bus', 'coach', 'minibus', 'double_decker', 'shuttle'],
            'motorcycle': ['sport', 'cruiser', 'touring', 'scooter', 'dirt_bike', 'chopper'],
            'bicycle': ['road', 'mountain', 'hybrid', 'electric', 'bmx', 'folding']
        }
        
        # Flatten all classes
        self.all_classes = []
        self.class_to_category = {}
        for category, classes in self.vehicle_categories.items():
            for cls in classes:
                self.all_classes.append(cls)
                self.class_to_category[cls] = category
        
        self.num_classes = len(self.all_classes)
        
        # Classification parameters
        self.min_classification_confidence = config.model_params.get('min_classification_confidence', 0.3)
        self.top_k_predictions = config.model_params.get('top_k_predictions', 3)
        self.image_size = config.model_params.get('image_size', 224)
        self.backbone = config.model_params.get('backbone', 'resnet50')
        self.pretrained = config.model_params.get('pretrained', True)
        self.dropout_rate = config.model_params.get('dropout_rate', 0.5)
        
        # Model metadata
        self.metadata = {
            'name': 'ResNet50-Vehicle-Classifier',
            'version': '1.0.0',
            'description': 'ResNet50-based vehicle classification model',
            'author': 'Oryza AI Team',
            'framework': 'PyTorch',
            'num_classes': self.num_classes,
            'vehicle_categories': list(self.vehicle_categories.keys()),
            'all_classes': self.all_classes
        }
        
        logger.info(f"Initialized VehicleClassifier with {self.num_classes} classes")
    
    def load_model(self) -> bool:
        """Load the ResNet50 vehicle classification model."""
        try:
            logger.info("Loading ResNet50 vehicle classification model...")
            
            # Create ResNet50 model
            if self.backbone == 'resnet50':
                self.model = models.resnet50(pretrained=self.pretrained)
                num_features = self.model.fc.in_features
                self.model.fc = nn.Sequential(
                    nn.Dropout(self.dropout_rate),
                    nn.Linear(num_features, self.num_classes)
                )
            elif self.backbone == 'resnet34':
                self.model = models.resnet34(pretrained=self.pretrained)
                num_features = self.model.fc.in_features
                self.model.fc = nn.Sequential(
                    nn.Dropout(self.dropout_rate),
                    nn.Linear(num_features, self.num_classes)
                )
            else:
                raise ValueError(f"Unsupported backbone: {self.backbone}")
            
            # Move to device and set eval mode
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Create preprocessing transforms
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            self.is_loaded = True
            logger.info("Vehicle classification model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load vehicle classification model: {str(e)}")
            return False
    
    def preprocess(self, input_data: Union[str, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Preprocess input image for vehicle classification."""
        try:
            if isinstance(input_data, str):
                image = cv2.imread(input_data)
                if image is None:
                    raise ValueError(f"Could not load image from path: {input_data}")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif isinstance(input_data, np.ndarray):
                image = input_data.copy()
                if image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif isinstance(input_data, torch.Tensor):
                image = input_data.cpu().numpy()
                if image.ndim == 3 and image.shape[0] in [1, 3]:
                    image = np.transpose(image, (1, 2, 0))
                if image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                raise ValueError(f"Unsupported input type: {type(input_data)}")
            
            # Apply transforms
            tensor = self.transform(image)
            tensor = tensor.unsqueeze(0)  # Add batch dimension
            
            return tensor.to(self.device)
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            raise
    
    def _inference_step(self, preprocessed_data: torch.Tensor) -> torch.Tensor:
        """Execute inference using ResNet50 model."""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        with torch.no_grad():
            logits = self.model(preprocessed_data)
            return logits
    
    def postprocess(self, model_output: torch.Tensor) -> InferenceResult:
        """Postprocess ResNet50 output into standardized format."""
        try:
            result = InferenceResult(
                success=True,
                model_name=self.metadata['name'],
                model_type='vehicle_classification'
            )
            
            # Convert logits to probabilities
            probabilities = torch.softmax(model_output, dim=1)
            probs = probabilities.cpu().numpy()[0]  # Remove batch dimension
            
            # Get top-k predictions
            top_k_indices = np.argsort(probs)[::-1][:self.top_k_predictions]
            
            # Create detections for top predictions
            for i, class_idx in enumerate(top_k_indices):
                confidence = float(probs[class_idx])
                
                if confidence >= self.min_classification_confidence:
                    class_name = self.all_classes[class_idx]
                    category = self.class_to_category[class_name]
                    
                    result.add_detection(
                        bbox=[0, 0, 1, 1],  # Dummy bbox for classification
                        confidence=confidence,
                        class_id=class_idx,
                        class_name=class_name,
                        attributes={
                            'category': category,
                            'rank': i + 1,
                            'probability': confidence,
                            'all_probabilities': probs.tolist(),
                            'top_k_classes': [self.all_classes[idx] for idx in top_k_indices],
                            'top_k_probabilities': [float(probs[idx]) for idx in top_k_indices]
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
                model_type='vehicle_classification',
                metadata={'error': str(e)}
            )
    
    def classify_vehicle(self, image: Union[str, np.ndarray]) -> InferenceResult:
        """Classify vehicle type in an image."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        return self.predict(image)
    
    def get_top_prediction(self, image: Union[str, np.ndarray]) -> Optional[Dict[str, Any]]:
        """Get the top vehicle classification."""
        result = self.classify_vehicle(image)
        
        if not result.detections:
            return None
        
        top_detection = result.detections[0]
        
        return {
            'class': top_detection.class_name,
            'category': top_detection.attributes['category'],
            'confidence': top_detection.confidence,
            'rank': top_detection.attributes['rank']
        }
    
    def get_category_predictions(self, image: Union[str, np.ndarray]) -> Dict[str, List[Dict[str, Any]]]:
        """Get predictions grouped by vehicle category."""
        result = self.classify_vehicle(image)
        
        category_predictions = {category: [] for category in self.vehicle_categories.keys()}
        
        for detection in result.detections:
            category = detection.attributes['category']
            category_predictions[category].append({
                'class': detection.class_name,
                'confidence': detection.confidence,
                'rank': detection.attributes['rank']
            })
        
        return category_predictions
    
    def get_all_probabilities(self, image: Union[str, np.ndarray]) -> Dict[str, float]:
        """Get probabilities for all vehicle classes."""
        result = self.classify_vehicle(image)
        
        if not result.detections:
            # If no detections, run inference to get raw probabilities
            try:
                tensor = self.preprocess(image)
                logits = self._inference_step(tensor)
                probabilities = torch.softmax(logits, dim=1)
                probs = probabilities.cpu().numpy()[0]
                return {self.all_classes[i]: float(probs[i]) for i in range(len(self.all_classes))}
            except:
                return {}
        
        # Get all probabilities from the first detection
        if 'all_probabilities' in result.detections[0].attributes:
            all_probs = result.detections[0].attributes['all_probabilities']
            return {self.all_classes[i]: prob for i, prob in enumerate(all_probs)}
        else:
            return {}
    
    def _optimize_for_raspberry_pi(self) -> bool:
        """Optimize for Raspberry Pi 5 with Hailo-8."""
        try:
            self.backbone = 'resnet34'
            self.image_size = 224
            self.config.batch_size = 1
            self.min_classification_confidence = 0.4
            self.top_k_predictions = 2
            return True
        except Exception as e:
            logger.error(f"Raspberry Pi optimization failed: {e}")
            return False
    
    def _optimize_for_radxa_rock(self) -> bool:
        """Optimize for Radxa Rock 5 with NPU."""
        try:
            self.backbone = 'resnet50'
            self.image_size = 224
            self.config.batch_size = 1
            self.min_classification_confidence = 0.35
            self.top_k_predictions = 3
            return True
        except Exception as e:
            logger.error(f"Radxa Rock optimization failed: {e}")
            return False
    
    def _optimize_for_jetson(self) -> bool:
        """Optimize for Jetson Nano with TensorRT."""
        try:
            self.backbone = 'resnet50'
            self.image_size = 224
            self.config.batch_size = 1
            self.config.use_fp16 = True
            self.min_classification_confidence = 0.3
            self.top_k_predictions = 3
            return True
        except Exception as e:
            logger.error(f"Jetson optimization failed: {e}")
            return False
    
    def _optimize_for_core_i5(self) -> bool:
        """Optimize for Core i5 with CUDA."""
        try:
            if self.device.type == 'cuda':
                self.backbone = 'resnet50'
                self.image_size = 224
                self.config.batch_size = 4
                self.config.use_fp16 = True
                self.min_classification_confidence = 0.25
                self.top_k_predictions = 5
            return True
        except Exception as e:
            logger.error(f"Core i5 optimization failed: {e}")
            return False