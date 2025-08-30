# ==========================================
# AI Box - Model Manager
# Centralized model management and orchestration
# ==========================================

import logging
from typing import Dict, List, Optional, Type, Any
from pathlib import Path
import json

from .base_model import BaseModel
from .model_config import ModelConfig

logger = logging.getLogger(__name__)

class ModelManager:
    """
    Centralized model manager for the AI Box system.
    
    This class manages multiple AI models, handles model loading/unloading,
    provides model orchestration, and manages model lifecycle.
    """
    
    def __init__(self):
        """Initialize the model manager."""
        self.models: Dict[str, BaseModel] = {}
        self.model_configs: Dict[str, ModelConfig] = {}
        self.model_registry: Dict[str, Type[BaseModel]] = {}
        
        logger.info("Model Manager initialized")
    
    def register_model(self, model_name: str, model_class: Type[BaseModel]) -> bool:
        """
        Register a model class with the manager.
        
        Args:
            model_name: Name of the model
            model_class: Model class to register
            
        Returns:
            bool: True if registration successful
        """
        try:
            self.model_registry[model_name] = model_class
            logger.info(f"Registered model: {model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to register model {model_name}: {e}")
            return False
    
    def create_model(self, model_name: str, config: ModelConfig) -> Optional[BaseModel]:
        """
        Create a model instance.
        
        Args:
            model_name: Name of the model to create
            config: Model configuration
            
        Returns:
            BaseModel: Model instance or None if failed
        """
        try:
            if model_name not in self.model_registry:
                logger.error(f"Model {model_name} not registered")
                return None
            
            model_class = self.model_registry[model_name]
            model = model_class(config)
            
            # Store model and config
            self.models[model_name] = model
            self.model_configs[model_name] = config
            
            logger.info(f"Created model: {model_name}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to create model {model_name}: {e}")
            return None
    
    def load_model(self, model_name: str) -> bool:
        """
        Load a model into memory.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            bool: True if load successful
        """
        try:
            if model_name not in self.models:
                logger.error(f"Model {model_name} not found")
                return False
            
            model = self.models[model_name]
            success = model.load_model()
            
            if success:
                logger.info(f"Loaded model: {model_name}")
            else:
                logger.error(f"Failed to load model: {model_name}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return False
    
    def unload_model(self, model_name: str) -> bool:
        """
        Unload a model from memory.
        
        Args:
            model_name: Name of the model to unload
            
        Returns:
            bool: True if unload successful
        """
        try:
            if model_name not in self.models:
                logger.warning(f"Model {model_name} not found for unloading")
                return True
            
            model = self.models[model_name]
            
            # Cleanup model
            if hasattr(model, '__del__'):
                model.__del__()
            
            # Remove from manager
            del self.models[model_name]
            if model_name in self.model_configs:
                del self.model_configs[model_name]
            
            logger.info(f"Unloaded model: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error unloading model {model_name}: {e}")
            return False
    
    def get_model(self, model_name: str) -> Optional[BaseModel]:
        """
        Get a model instance.
        
        Args:
            model_name: Name of the model
            
        Returns:
            BaseModel: Model instance or None
        """
        return self.models.get(model_name)
    
    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """
        Get model configuration.
        
        Args:
            model_name: Name of the model
            
        Returns:
            ModelConfig: Model configuration or None
        """
        return self.model_configs.get(model_name)
    
    def list_models(self) -> List[str]:
        """
        Get list of all registered models.
        
        Returns:
            List of model names
        """
        return list(self.model_registry.keys())
    
    def list_loaded_models(self) -> List[str]:
        """
        Get list of currently loaded models.
        
        Returns:
            List of loaded model names
        """
        return list(self.models.keys())
    
    def is_model_loaded(self, model_name: str) -> bool:
        """
        Check if a model is loaded.
        
        Args:
            model_name: Name of the model
            
        Returns:
            bool: True if model is loaded
        """
        return model_name in self.models and self.models[model_name].is_loaded
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dict: Model information or None
        """
        model = self.get_model(model_name)
        if model:
            return model.get_model_info()
        return None
    
    def get_performance_metrics(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get performance metrics for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dict: Performance metrics or None
        """
        model = self.get_model(model_name)
        if model:
            return model.get_performance_metrics()
        return None
    
    def optimize_model(self, model_name: str, platform: str) -> bool:
        """
        Optimize a model for a specific platform.
        
        Args:
            model_name: Name of the model
            platform: Target platform
            
        Returns:
            bool: True if optimization successful
        """
        try:
            model = self.get_model(model_name)
            if not model:
                logger.error(f"Model {model_name} not found for optimization")
                return False
            
            success = model.optimize_for_platform(platform)
            
            if success:
                logger.info(f"Optimized model {model_name} for platform {platform}")
            else:
                logger.warning(f"Failed to optimize model {model_name} for platform {platform}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error optimizing model {model_name}: {e}")
            return False
    
    def save_model_configs(self, file_path: str) -> bool:
        """
        Save all model configurations to file.
        
        Args:
            file_path: Path to save configurations
            
        Returns:
            bool: True if save successful
        """
        try:
            configs = {}
            for name, config in self.model_configs.items():
                configs[name] = config.to_dict()
            
            with open(file_path, 'w') as f:
                json.dump(configs, f, indent=2)
            
            logger.info(f"Saved model configurations to: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model configurations: {e}")
            return False
    
    def load_model_configs(self, file_path: str) -> bool:
        """
        Load model configurations from file.
        
        Args:
            file_path: Path to configuration file
            
        Returns:
            bool: True if load successful
        """
        try:
            with open(file_path, 'r') as f:
                configs = json.load(f)
            
            for name, config_dict in configs.items():
                config = ModelConfig.from_dict(config_dict)
                self.model_configs[name] = config
            
            logger.info(f"Loaded model configurations from: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model configurations: {e}")
            return False
    
    def cleanup(self):
        """Cleanup all models and resources."""
        try:
            # Unload all models
            model_names = list(self.models.keys())
            for model_name in model_names:
                self.unload_model(model_name)
            
            # Clear registries
            self.models.clear()
            self.model_configs.clear()
            self.model_registry.clear()
            
            logger.info("Model Manager cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def __del__(self):
        """Destructor - cleanup on deletion."""
        self.cleanup()
