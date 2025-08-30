# ==========================================
# AI Box - Face Recognition Model
# FaceNet-based face recognition implementation
# ==========================================

import cv2
import numpy as np
import face_recognition
import pickle
import os
import torch
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
import logging
from datetime import datetime

from ..common.base_model import BaseModel
from ..common.model_config import ModelConfig
from ..common.inference_result import InferenceResult, Detection

logger = logging.getLogger(__name__)

class FaceRecognizer(BaseModel):
    """
    Face Recognition Model using face-recognition library.
    
    This model provides face recognition capabilities including:
    - Face embedding extraction
    - Face similarity comparison
    - Known faces database management
    - Face identification and verification
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize Face Recognizer.
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        
        # Face recognition specific attributes
        self.known_faces: List[np.ndarray] = []
        self.known_names: List[str] = []
        self.face_database: Dict[str, Dict[str, Any]] = {}
        
        # Recognition parameters
        self.tolerance = config.model_params.get('tolerance', 0.6)
        self.min_face_size = config.model_params.get('min_face_size', 20)
        self.embedding_model = config.model_params.get('embedding_model', 'hog')  # 'hog' or 'cnn'
        
        # Database path
        self.database_path = config.model_params.get('database_path', 'face_database.pkl')
        
        # Model metadata
        self.metadata = {
            'name': 'FaceNet-Face-Recognizer',
            'version': '1.0.0',
            'description': 'FaceNet-based face recognition model',
            'author': 'Oryza AI Team',
            'framework': 'face-recognition',
            'input_shape': self.config.input_size,
            'output_shape': (128,),  # Face embedding dimension
            'model_size_mb': 0.0,
            'parameters': 0,
            'created_date': '2025-08-30',
            'last_updated': '2025-08-30'
        }
        
        logger.info(f"Initialized FaceRecognizer with config: {config}")
    
    def load_model(self) -> bool:
        """
        Load the face recognition model and known faces database.
        
        Returns:
            bool: True if model loaded successfully
        """
        try:
            logger.info("Loading face recognition model...")
            
            # Load known faces database if exists
            if os.path.exists(self.database_path):
                self._load_face_database()
                logger.info(f"Loaded {len(self.known_faces)} known faces from database")
            else:
                logger.info("No existing face database found, starting fresh")
            
            # Test model functionality
            test_embedding = face_recognition.face_encodings(
                np.zeros((100, 100, 3), dtype=np.uint8)
            )
            
            if test_embedding:
                logger.info("Face recognition model loaded successfully")
                self.is_loaded = True
                return True
            else:
                logger.error("Face recognition model test failed")
                return False
                
        except Exception as e:
            logger.error(f"Failed to load face recognition model: {str(e)}")
            return False
    
    def _load_face_database(self):
        """Load known faces database from file."""
        try:
            with open(self.database_path, 'rb') as f:
                data = pickle.load(f)
                self.known_faces = data.get('encodings', [])
                self.known_names = data.get('names', [])
                self.face_database = data.get('database', {})
                
        except Exception as e:
            logger.error(f"Failed to load face database: {e}")
            self.known_faces = []
            self.known_names = []
            self.face_database = {}
    
    def _save_face_database(self):
        """Save known faces database to file."""
        try:
            data = {
                'encodings': self.known_faces,
                'names': self.known_names,
                'database': self.face_database,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(self.database_path, 'wb') as f:
                pickle.dump(data, f)
                
            logger.info(f"Saved face database with {len(self.known_faces)} faces")
            
        except Exception as e:
            logger.error(f"Failed to save face database: {e}")
    
    def preprocess(self, input_data: Union[str, np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Preprocess input image for face recognition.
        
        Args:
            input_data: Input image (file path, numpy array, or tensor)
            
        Returns:
            np.ndarray: Preprocessed image
        """
        try:
            # Handle different input types
            if isinstance(input_data, str):
                # File path
                image = cv2.imread(input_data)
                if image is None:
                    raise ValueError(f"Could not load image from path: {input_data}")
                # Convert BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif isinstance(input_data, np.ndarray):
                # Numpy array
                image = input_data.copy()
                if image.ndim == 3 and image.shape[2] == 3:
                    # Convert BGR to RGB if needed
                    if hasattr(self, '_last_image') and np.array_equal(image, self._last_image):
                        return image
                    # Check if BGR and convert to RGB
                    if np.mean(image[:, :, 0]) > np.mean(image[:, :, 2]):
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif isinstance(input_data, torch.Tensor):
                # PyTorch tensor
                image = input_data.cpu().numpy()
                if image.ndim == 3 and image.shape[0] in [1, 3]:
                    image = np.transpose(image, (1, 2, 0))
                if image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                raise ValueError(f"Unsupported input type: {type(input_data)}")
            
            self._last_image = image.copy()
            return image
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            raise
    
    def postprocess(self, model_output: Tuple[List[np.ndarray], List[Tuple[int, int, int, int]], List[str]]) -> InferenceResult:
        """
        Postprocess face recognition output into standardized format.
        
        Args:
            model_output: Tuple of (encodings, locations, names)
            
        Returns:
            InferenceResult: Standardized inference result
        """
        try:
            encodings, locations, names = model_output
            
            # Create result object
            result = InferenceResult(
                success=True,
                model_name=self.metadata['name'],
                model_type='face_recognition'
            )
            
            # Add each recognized face
            for i, (encoding, location, name) in enumerate(zip(encodings, locations, names)):
                # Convert location to bbox format [x1, y1, x2, y2]
                top, right, bottom, left = location
                bbox = [left, top, right, bottom]
                
                # Calculate confidence based on face size and recognition confidence
                face_size = (right - left) * (bottom - top)
                confidence = min(1.0, face_size / (self.config.input_size[0] * self.config.input_size[1] * 0.1))
                
                # Add detection to result
                result.add_detection(
                    bbox=bbox,
                    confidence=confidence,
                    class_id=0,
                    class_name=name if name != "Unknown" else "unknown_face",
                    attributes={
                        'encoding': encoding.tolist(),
                        'face_size': face_size,
                        'recognition_confidence': confidence
                    }
                )
            
            # Add processing metadata
            result.input_shape = self.config.input_size
            result.raw_output = {
                'encodings': [enc.tolist() for enc in encodings],
                'locations': locations,
                'names': names
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Postprocessing failed: {str(e)}")
            # Return error result
            return InferenceResult(
                success=False,
                model_name=self.metadata['name'],
                model_type='face_recognition',
                metadata={'error': str(e)}
            )
    
    def recognize_faces(self, image: Union[str, np.ndarray]) -> InferenceResult:
        """
        Recognize faces in an image.
        
        Args:
            image: Input image (file path or numpy array)
            
        Returns:
            InferenceResult: Face recognition results
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            # Execute inference
            result = self.predict(image)
            return result
            
        except Exception as e:
            logger.error(f"Face recognition failed: {e}")
            raise
    
    def extract_face_encodings(self, image: Union[str, np.ndarray]) -> List[np.ndarray]:
        """
        Extract face encodings from an image.
        
        Args:
            image: Input image
            
        Returns:
            List of face encodings
        """
        try:
            # Preprocess image
            processed_image = self.preprocess(image)
            
            # Extract face encodings
            encodings = face_recognition.face_encodings(
                processed_image,
                model=self.embedding_model
            )
            
            return encodings
            
        except Exception as e:
            logger.error(f"Face encoding extraction failed: {e}")
            return []
    
    def compare_faces(self, face_encoding1: np.ndarray, face_encoding2: np.ndarray) -> bool:
        """
        Compare two face encodings.
        
        Args:
            face_encoding1: First face encoding
            face_encoding2: Second face encoding
            
        Returns:
            bool: True if faces match
        """
        try:
            # Compare faces using face_recognition
            matches = face_recognition.compare_faces(
                [face_encoding1], 
                face_encoding2, 
                tolerance=self.tolerance
            )
            return matches[0] if matches else False
            
        except Exception as e:
            logger.error(f"Face comparison failed: {e}")
            return False
    
    def get_face_distance(self, face_encoding1: np.ndarray, face_encoding2: np.ndarray) -> float:
        """
        Get distance between two face encodings.
        
        Args:
            face_encoding1: First face encoding
            face_encoding2: Second face encoding
            
        Returns:
            float: Distance between faces (lower = more similar)
        """
        try:
            # Calculate face distance
            distance = face_recognition.face_distance([face_encoding1], face_encoding2)
            return float(distance[0])
            
        except Exception as e:
            logger.error(f"Face distance calculation failed: {e}")
            return float('inf')
    
    def add_known_face(self, name: str, image: Union[str, np.ndarray]) -> bool:
        """
        Add a known face to the database.
        
        Args:
            name: Name of the person
            image: Image containing the face
            
        Returns:
            bool: True if face added successfully
        """
        try:
            # Extract face encodings
            encodings = self.extract_face_encodings(image)
            
            if not encodings:
                logger.warning(f"No face found in image for {name}")
                return False
            
            # Use the first face found
            encoding = encodings[0]
            
            # Add to database
            self.known_faces.append(encoding)
            self.known_names.append(name)
            
            # Update database metadata
            self.face_database[name] = {
                'encoding_index': len(self.known_faces) - 1,
                'added_date': datetime.now().isoformat(),
                'face_count': 1
            }
            
            # Save database
            self._save_face_database()
            
            logger.info(f"Added known face for {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add known face for {name}: {e}")
            return False
    
    def remove_known_face(self, name: str) -> bool:
        """
        Remove a known face from the database.
        
        Args:
            name: Name of the person to remove
            
        Returns:
            bool: True if face removed successfully
        """
        try:
            if name not in self.known_names:
                logger.warning(f"Person {name} not found in database")
                return False
            
            # Find all indices for this person
            indices = [i for i, n in enumerate(self.known_names) if n == name]
            
            # Remove in reverse order to maintain indices
            for i in reversed(indices):
                del self.known_faces[i]
                del self.known_names[i]
            
            # Remove from database metadata
            if name in self.face_database:
                del self.face_database[name]
            
            # Save database
            self._save_face_database()
            
            logger.info(f"Removed known face for {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove known face for {name}: {e}")
            return False
    
    def get_known_faces(self) -> List[str]:
        """
        Get list of known face names.
        
        Returns:
            List of known face names
        """
        return list(set(self.known_names))
    
    def get_face_count(self, name: str) -> int:
        """
        Get number of face encodings for a person.
        
        Args:
            name: Name of the person
            
        Returns:
            int: Number of face encodings
        """
        return self.known_names.count(name)
    
    def _inference_step(self, preprocessed_data: np.ndarray) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]], List[str]]:
        """
        Execute the actual face recognition inference step.
        
        Args:
            preprocessed_data: Preprocessed input image
            
        Returns:
            Tuple of (encodings, locations, names)
        """
        try:
            # Find face locations
            face_locations = face_recognition.face_locations(
                preprocessed_data,
                model=self.embedding_model
            )
            
            # Extract face encodings
            face_encodings = face_recognition.face_encodings(
                preprocessed_data,
                face_locations,
                model=self.embedding_model
            )
            
            # Recognize faces
            face_names = []
            for face_encoding in face_encodings:
                if len(self.known_faces) > 0:
                    # Compare with known faces
                    matches = face_recognition.compare_faces(
                        self.known_faces,
                        face_encoding,
                        tolerance=self.tolerance
                    )
                    
                    if True in matches:
                        # Find the first match
                        first_match_index = matches.index(True)
                        name = self.known_names[first_match_index]
                    else:
                        name = "Unknown"
                else:
                    name = "Unknown"
                
                face_names.append(name)
            
            return face_encodings, face_locations, face_names
            
        except Exception as e:
            logger.error(f"Face recognition inference failed: {e}")
            return [], [], []
    
    def _optimize_for_raspberry_pi(self) -> bool:
        """Optimize for Raspberry Pi 5 with Hailo-8."""
        try:
            # Use HOG model for faster processing on Pi
            self.embedding_model = 'hog'
            self.config.input_size = (320, 320)
            logger.info("Optimized for Raspberry Pi 5: Using HOG model")
            return True
        except Exception as e:
            logger.error(f"Raspberry Pi optimization failed: {e}")
            return False
    
    def _optimize_for_radxa_rock(self) -> bool:
        """Optimize for Radxa Rock 5 with NPU."""
        try:
            # Optimize for NPU processing
            self.embedding_model = 'hog'
            self.config.input_size = (416, 416)
            logger.info("Optimized for Radxa Rock 5: Using HOG model")
            return True
        except Exception as e:
            logger.error(f"Radxa Rock optimization failed: {e}")
            return False
    
    def _optimize_for_jetson(self) -> bool:
        """Optimize for Jetson Nano with TensorRT."""
        try:
            # Use CNN model for better accuracy on Jetson
            self.embedding_model = 'cnn'
            self.config.input_size = (640, 640)
            logger.info("Optimized for Jetson Nano: Using CNN model")
            return True
        except Exception as e:
            logger.error(f"Jetson optimization failed: {e}")
            return False
    
    def _optimize_for_core_i5(self) -> bool:
        """Optimize for Core i5 with CUDA."""
        try:
            # Use CNN model for best accuracy on powerful hardware
            self.embedding_model = 'cnn'
            self.config.input_size = (640, 640)
            logger.info("Optimized for Core i5: Using CNN model")
            return True
        except Exception as e:
            logger.error(f"Core i5 optimization failed: {e}")
            return False
