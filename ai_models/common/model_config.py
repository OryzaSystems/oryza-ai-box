# ==========================================
# AI Box - Model Configuration
# Configuration management for AI models
# ==========================================

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from pathlib import Path
import json

@dataclass
class ModelConfig:
    """
    Configuration class for AI models.
    
    This class manages all configuration parameters for AI models including:
    - Model paths and URLs
    - Inference parameters
    - Performance settings
    - Platform-specific configurations
    """
    
    # Model identification
    model_name: str
    model_type: str
    model_version: str = "1.0.0"
    
    # Model paths
    model_path: Optional[str] = None
    weights_path: Optional[str] = None
    config_path: Optional[str] = None
    
    # Inference parameters
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.4
    max_detections: int = 100
    input_size: tuple = (640, 640)
    batch_size: int = 1
    
    # Performance settings
    use_gpu: bool = True
    use_fp16: bool = False
    use_int8: bool = False
    num_threads: int = 4
    
    # Platform-specific settings
    platform: str = "auto"  # auto, raspberry-pi-5, radxa-rock-5, jetson-nano, core-i5
    optimization_level: str = "medium"  # low, medium, high
    
    # Model-specific parameters
    model_params: Dict[str, Any] = field(default_factory=dict)
    
    # Cache settings
    enable_cache: bool = True
    cache_dir: Optional[str] = None
    
    # Logging
    verbose: bool = False
    log_level: str = "INFO"
    
    def __post_init__(self):
        """Post-initialization setup."""
        if self.cache_dir is None:
            self.cache_dir = str(Path.home() / ".cache" / "ai_box" / "models")
        
        # Create cache directory
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """
        Create ModelConfig from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            ModelConfig instance
        """
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, json_path: str) -> 'ModelConfig':
        """
        Create ModelConfig from JSON file.
        
        Args:
            json_path: Path to JSON configuration file
            
        Returns:
            ModelConfig instance
        """
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Configuration dictionary
        """
        return {
            'model_name': self.model_name,
            'model_type': self.model_type,
            'model_version': self.model_version,
            'model_path': self.model_path,
            'weights_path': self.weights_path,
            'config_path': self.config_path,
            'confidence_threshold': self.confidence_threshold,
            'nms_threshold': self.nms_threshold,
            'max_detections': self.max_detections,
            'input_size': self.input_size,
            'batch_size': self.batch_size,
            'use_gpu': self.use_gpu,
            'use_fp16': self.use_fp16,
            'use_int8': self.use_int8,
            'num_threads': self.num_threads,
            'platform': self.platform,
            'optimization_level': self.optimization_level,
            'model_params': self.model_params,
            'enable_cache': self.enable_cache,
            'cache_dir': self.cache_dir,
            'verbose': self.verbose,
            'log_level': self.log_level
        }
    
    def to_json(self, json_path: str) -> bool:
        """
        Save configuration to JSON file.
        
        Args:
            json_path: Path to save JSON file
            
        Returns:
            True if save successful
        """
        try:
            with open(json_path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            return True
        except Exception as e:
            print(f"Failed to save config to JSON: {e}")
            return False
    
    def update(self, **kwargs) -> None:
        """
        Update configuration parameters.
        
        Args:
            **kwargs: Parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")
    
    def get_model_file_path(self) -> Optional[str]:
        """
        Get the primary model file path.
        
        Returns:
            Model file path or None
        """
        if self.model_path:
            return self.model_path
        elif self.weights_path:
            return self.weights_path
        else:
            return None
    
    def is_platform_supported(self, platform: str) -> bool:
        """
        Check if platform is supported.
        
        Args:
            platform: Platform to check
            
        Returns:
            True if platform is supported
        """
        supported_platforms = [
            'auto', 'raspberry-pi-5', 'radxa-rock-5', 
            'jetson-nano', 'core-i5'
        ]
        return platform in supported_platforms
    
    def get_optimization_settings(self) -> Dict[str, Any]:
        """
        Get optimization settings based on platform and level.
        
        Returns:
            Optimization settings dictionary
        """
        settings = {
            'low': {
                'use_fp16': False,
                'use_int8': False,
                'num_threads': 2,
                'batch_size': 1
            },
            'medium': {
                'use_fp16': True,
                'use_int8': False,
                'num_threads': 4,
                'batch_size': 2
            },
            'high': {
                'use_fp16': True,
                'use_int8': True,
                'num_threads': 8,
                'batch_size': 4
            }
        }
        
        return settings.get(self.optimization_level, settings['medium'])
    
    def validate(self) -> List[str]:
        """
        Validate configuration parameters.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check required fields
        if not self.model_name:
            errors.append("model_name is required")
        
        if not self.model_type:
            errors.append("model_type is required")
        
        # Check threshold values
        if not 0.0 <= self.confidence_threshold <= 1.0:
            errors.append("confidence_threshold must be between 0.0 and 1.0")
        
        if not 0.0 <= self.nms_threshold <= 1.0:
            errors.append("nms_threshold must be between 0.0 and 1.0")
        
        # Check input size
        if len(self.input_size) != 2:
            errors.append("input_size must be a tuple of 2 integers")
        elif any(size <= 0 for size in self.input_size):
            errors.append("input_size values must be positive")
        
        # Check platform
        if not self.is_platform_supported(self.platform):
            errors.append(f"Unsupported platform: {self.platform}")
        
        # Check optimization level
        if self.optimization_level not in ['low', 'medium', 'high']:
            errors.append("optimization_level must be 'low', 'medium', or 'high'")
        
        return errors
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return f"ModelConfig(model_name='{self.model_name}', model_type='{self.model_type}', platform='{self.platform}')"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"ModelConfig(\n" + \
               f"  model_name='{self.model_name}',\n" + \
               f"  model_type='{self.model_type}',\n" + \
               f"  model_version='{self.model_version}',\n" + \
               f"  platform='{self.platform}',\n" + \
               f"  optimization_level='{self.optimization_level}',\n" + \
               f"  confidence_threshold={self.confidence_threshold},\n" + \
               f"  input_size={self.input_size},\n" + \
               f"  use_gpu={self.use_gpu}\n" + \
               f")"
