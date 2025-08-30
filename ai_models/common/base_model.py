# ==========================================
# AI Box - Base Model Class
# Abstract base class for all AI models
# ==========================================

import time
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import torch
import numpy as np
from dataclasses import dataclass

from .model_config import ModelConfig
from .inference_result import InferenceResult

logger = logging.getLogger(__name__)

@dataclass
class ModelMetadata:
    """Model metadata information"""
    name: str
    version: str
    description: str
    author: str
    framework: str
    input_shape: tuple
    output_shape: tuple
    model_size_mb: float
    parameters: int
    created_date: str
    last_updated: str

class BaseModel(ABC):
    """
    Abstract base class for all AI models in the AI Box system.
    
    This class provides common functionality for:
    - Model loading and initialization
    - Inference execution
    - Performance monitoring
    - Error handling
    - Platform optimization
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize the base model.
        
        Args:
            config: Model configuration object
        """
        self.config = config
        self.model = None
        self.device = self._setup_device()
        self.metadata = None
        self.is_loaded = False
        self.inference_count = 0
        self.total_inference_time = 0.0
        
        # Performance tracking
        self.performance_metrics = {
            'avg_inference_time': 0.0,
            'min_inference_time': float('inf'),
            'max_inference_time': 0.0,
            'total_inferences': 0,
            'errors': 0
        }
        
        logger.info(f"Initializing {self.__class__.__name__} with config: {config}")
    
    def _setup_device(self) -> torch.device:
        """
        Setup the computation device (CPU/GPU) based on configuration.
        
        Returns:
            torch.device: The device to use for computation
        """
        if self.config.use_gpu and torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        else:
            device = torch.device('cpu')
            logger.info("Using CPU device")
        
        return device
    
    @abstractmethod
    def load_model(self) -> bool:
        """
        Load the AI model into memory.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def preprocess(self, input_data: Any) -> Any:
        """
        Preprocess input data for inference.
        
        Args:
            input_data: Raw input data
            
        Returns:
            Preprocessed data ready for model inference
        """
        pass
    
    @abstractmethod
    def postprocess(self, model_output: Any) -> InferenceResult:
        """
        Postprocess model output into standardized format.
        
        Args:
            model_output: Raw model output
            
        Returns:
            InferenceResult: Standardized inference result
        """
        pass
    
    def predict(self, input_data: Any) -> InferenceResult:
        """
        Execute inference on input data.
        
        Args:
            input_data: Input data for inference
            
        Returns:
            InferenceResult: Inference results with metadata
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        start_time = time.time()
        
        try:
            # Preprocess input
            preprocessed_data = self.preprocess(input_data)
            
            # Execute inference
            with torch.no_grad():
                model_output = self._inference_step(preprocessed_data)
            
            # Postprocess output
            result = self.postprocess(model_output)
            
            # Update performance metrics
            inference_time = time.time() - start_time
            self._update_performance_metrics(inference_time)
            
            # Add metadata to result
            result.metadata.update({
                'inference_time': inference_time,
                'device': str(self.device),
                'model_name': self.__class__.__name__,
                'inference_count': self.inference_count
            })
            
            self.inference_count += 1
            
            return result
            
        except Exception as e:
            self.performance_metrics['errors'] += 1
            logger.error(f"Inference failed: {str(e)}")
            raise
    
    def _inference_step(self, preprocessed_data: Any) -> Any:
        """
        Execute the actual model inference step.
        
        Args:
            preprocessed_data: Preprocessed input data
            
        Returns:
            Raw model output
        """
        if hasattr(self.model, 'predict'):
            # For models with predict method (e.g., scikit-learn)
            return self.model.predict(preprocessed_data)
        elif hasattr(self.model, '__call__'):
            # For PyTorch models
            return self.model(preprocessed_data)
        else:
            raise NotImplementedError("Model must have predict() or __call__() method")
    
    def _update_performance_metrics(self, inference_time: float):
        """Update performance tracking metrics."""
        self.total_inference_time += inference_time
        self.performance_metrics['total_inferences'] += 1
        
        # Update average
        total = self.performance_metrics['total_inferences']
        self.performance_metrics['avg_inference_time'] = self.total_inference_time / total
        
        # Update min/max
        self.performance_metrics['min_inference_time'] = min(
            self.performance_metrics['min_inference_time'], 
            inference_time
        )
        self.performance_metrics['max_inference_time'] = max(
            self.performance_metrics['max_inference_time'], 
            inference_time
        )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get current performance metrics.
        
        Returns:
            Dict containing performance metrics
        """
        return self.performance_metrics.copy()
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information and metadata.
        
        Returns:
            Dict containing model information
        """
        info = {
            'model_class': self.__class__.__name__,
            'config': self.config.__dict__,
            'device': str(self.device),
            'is_loaded': self.is_loaded,
            'inference_count': self.inference_count,
            'performance_metrics': self.performance_metrics
        }
        
        if self.metadata:
            info['metadata'] = self.metadata
        
        return info
    
    def optimize_for_platform(self, platform: str) -> bool:
        """
        Optimize model for specific platform.
        
        Args:
            platform: Target platform ('raspberry-pi-5', 'radxa-rock-5', 'jetson-nano', 'core-i5')
            
        Returns:
            bool: True if optimization successful
        """
        logger.info(f"Optimizing model for platform: {platform}")
        
        try:
            if platform == 'raspberry-pi-5':
                return self._optimize_for_raspberry_pi()
            elif platform == 'radxa-rock-5':
                return self._optimize_for_radxa_rock()
            elif platform == 'jetson-nano':
                return self._optimize_for_jetson()
            elif platform == 'core-i5':
                return self._optimize_for_core_i5()
            else:
                logger.warning(f"Unknown platform: {platform}")
                return False
                
        except Exception as e:
            logger.error(f"Platform optimization failed: {str(e)}")
            return False
    
    def _optimize_for_raspberry_pi(self) -> bool:
        """Optimize for Raspberry Pi 5 with Hailo-8."""
        # TODO: Implement Hailo-8 optimization
        logger.info("Raspberry Pi 5 optimization not yet implemented")
        return True
    
    def _optimize_for_radxa_rock(self) -> bool:
        """Optimize for Radxa Rock 5 with NPU."""
        # TODO: Implement RKNN optimization
        logger.info("Radxa Rock 5 optimization not yet implemented")
        return True
    
    def _optimize_for_jetson(self) -> bool:
        """Optimize for Jetson Nano with TensorRT."""
        # TODO: Implement TensorRT optimization
        logger.info("Jetson Nano optimization not yet implemented")
        return True
    
    def _optimize_for_core_i5(self) -> bool:
        """Optimize for Core i5 with CUDA."""
        # TODO: Implement CUDA optimization
        logger.info("Core i5 optimization not yet implemented")
        return True
    
    def save_model(self, path: Union[str, Path]) -> bool:
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model
            
        Returns:
            bool: True if save successful
        """
        try:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            if hasattr(self.model, 'save'):
                self.model.save(str(path))
            elif hasattr(self.model, 'state_dict'):
                torch.save(self.model.state_dict(), str(path))
            else:
                logger.warning("Model does not have save method")
                return False
            
            logger.info(f"Model saved to: {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            return False
    
    def load_model_from_path(self, path: Union[str, Path]) -> bool:
        """
        Load model from disk.
        
        Args:
            path: Path to the model file
            
        Returns:
            bool: True if load successful
        """
        try:
            path = Path(path)
            if not path.exists():
                logger.error(f"Model file not found: {path}")
                return False
            
            # Load model based on file extension
            if path.suffix == '.pt' or path.suffix == '.pth':
                # PyTorch model
                state_dict = torch.load(str(path), map_location=self.device)
                if hasattr(self.model, 'load_state_dict'):
                    self.model.load_state_dict(state_dict)
                else:
                    self.model = state_dict
            elif path.suffix == '.onnx':
                # ONNX model
                import onnxruntime as ort
                self.model = ort.InferenceSession(str(path))
            else:
                logger.error(f"Unsupported model format: {path.suffix}")
                return False
            
            self.is_loaded = True
            logger.info(f"Model loaded from: {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return False
    
    def __del__(self):
        """Cleanup when model is destroyed."""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
