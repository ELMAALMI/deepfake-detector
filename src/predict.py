"""Prediction/inference module for deepfake detection.

This module provides:
- Single image prediction
- Batch prediction
- Confidence score calculation
- Result interpretation
"""

import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.face_extractor import FaceExtractor, extract_largest_face
from src.preprocessing import ImagePreprocessor
from src.utils import load_config, load_image

logger = logging.getLogger("deepfake_detector.predict")


class DeepfakePredictor:
    """Deepfake prediction class."""
    
    def __init__(
        self,
        model_path: str,
        config_path: str = "config/config.yaml",
        face_extractor: Optional[FaceExtractor] = None,
    ):
        """Initialize deepfake predictor.
        
        Args:
            model_path: Path to trained model
            config_path: Path to configuration file
            face_extractor: Face extractor instance (creates new one if None)
        """
        self.model_path = model_path
        self.config = load_config(config_path)
        
        # Load model
        logger.info(f"Loading model from {model_path}")
        self.model = keras.models.load_model(model_path)
        
        # Create preprocessor
        model_config = self.config.get("model", {})
        preprocessing_config = self.config.get("preprocessing", {})
        input_size = model_config.get("input_size", [299, 299, 3])
        
        self.preprocessor = ImagePreprocessor(
            target_size=(input_size[0], input_size[1]),
            normalization=preprocessing_config.get("normalization", "standard"),
        )
        
        # Create face extractor
        if face_extractor is None:
            face_detection_config = self.config.get("face_detection", {})
            self.face_extractor = FaceExtractor(
                method=face_detection_config.get("method", "mediapipe"),
                min_detection_confidence=face_detection_config.get(
                    "mediapipe_min_detection_confidence", 0.5
                ),
            )
        else:
            self.face_extractor = face_extractor
        
        # Get detection parameters
        detection_config = self.config.get("detection", {})
        self.confidence_threshold = detection_config.get("confidence_threshold", 0.5)
        self.min_face_size = detection_config.get("min_face_size", 30)
        
        logger.info("Predictor initialized successfully")
    
    def predict_image(
        self,
        image: Union[str, np.ndarray],
        return_face: bool = False,
        extract_face: bool = True,
    ) -> Dict:
        """Predict if image is real or fake.
        
        Args:
            image: Image path or numpy array
            return_face: Whether to return extracted face
            extract_face: Whether to extract face (set False if image is already a face)
            
        Returns:
            Dictionary containing:
                - 'label': Prediction label ('REAL' or 'FAKE')
                - 'confidence': Confidence score (0-100%)
                - 'probability': Raw probability from model (0-1)
                - 'face': Extracted face image (if return_face=True)
        """
        # Load image if path provided
        if isinstance(image, str):
            image = load_image(image)
        
        # Extract face if requested
        if extract_face:
            face_data = extract_largest_face(
                image,
                extractor=self.face_extractor,
                min_face_size=self.min_face_size,
            )
            
            if face_data is None:
                logger.warning("No face detected in image")
                return {
                    'label': 'UNKNOWN',
                    'confidence': 0.0,
                    'probability': 0.5,
                    'error': 'No face detected',
                }
            
            face_image = face_data['face']
        else:
            face_image = image
        
        # Preprocess
        processed = self.preprocessor.preprocess_image(face_image)
        
        # Add batch dimension
        input_batch = np.expand_dims(processed, axis=0)
        
        # Predict
        prediction = self.model.predict(input_batch, verbose=0)[0, 0]
        
        # Interpret prediction
        is_fake = prediction >= self.confidence_threshold
        label = 'FAKE' if is_fake else 'REAL'
        
        # Calculate confidence (distance from threshold)
        if is_fake:
            confidence = float(prediction * 100)
        else:
            confidence = float((1 - prediction) * 100)
        
        result = {
            'label': label,
            'confidence': confidence,
            'probability': float(prediction),
        }
        
        if return_face and extract_face:
            result['face'] = face_image
        
        return result
    
    def predict_batch(
        self,
        images: List[Union[str, np.ndarray]],
        extract_face: bool = True,
    ) -> List[Dict]:
        """Predict batch of images.
        
        Args:
            images: List of image paths or numpy arrays
            extract_face: Whether to extract faces
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for image in images:
            result = self.predict_image(image, extract_face=extract_face)
            results.append(result)
        
        return results
    
    def predict_from_directory(
        self,
        directory: str,
        extensions: Optional[List[str]] = None,
    ) -> Dict[str, Dict]:
        """Predict all images in a directory.
        
        Args:
            directory: Directory containing images
            extensions: List of file extensions to process
            
        Returns:
            Dictionary mapping file paths to prediction results
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        from pathlib import Path
        
        results = {}
        
        for ext in extensions:
            for image_path in Path(directory).rglob(f"*{ext}"):
                image_path_str = str(image_path)
                logger.info(f"Processing {image_path_str}")
                
                try:
                    result = self.predict_image(image_path_str)
                    results[image_path_str] = result
                except Exception as e:
                    logger.error(f"Error processing {image_path_str}: {e}")
                    results[image_path_str] = {
                        'label': 'ERROR',
                        'confidence': 0.0,
                        'error': str(e),
                    }
        
        return results


def predict_image(
    image_path: str,
    model_path: str,
    config_path: str = "config/config.yaml",
) -> Dict:
    """Convenience function to predict single image.
    
    Args:
        image_path: Path to image file
        model_path: Path to trained model
        config_path: Path to configuration file
        
    Returns:
        Prediction dictionary
    """
    predictor = DeepfakePredictor(model_path, config_path)
    return predictor.predict_image(image_path)


def format_prediction_result(result: Dict) -> str:
    """Format prediction result as string.
    
    Args:
        result: Prediction result dictionary
        
    Returns:
        Formatted string
    """
    if 'error' in result:
        return f"Error: {result['error']}"
    
    label = result['label']
    confidence = result['confidence']
    probability = result['probability']
    
    return (
        f"Prediction: {label}\n"
        f"Confidence: {confidence:.2f}%\n"
        f"Raw Probability: {probability:.4f}"
    )


def visualize_prediction(
    image: np.ndarray,
    result: Dict,
    save_path: Optional[str] = None,
) -> np.ndarray:
    """Visualize prediction result on image.
    
    Args:
        image: Input image
        result: Prediction result dictionary
        save_path: Optional path to save visualization
        
    Returns:
        Annotated image
    """
    annotated = image.copy()
    
    # Get label and confidence
    label = result.get('label', 'UNKNOWN')
    confidence = result.get('confidence', 0.0)
    
    # Choose color based on label
    if label == 'FAKE':
        color = (0, 0, 255)  # Red
    elif label == 'REAL':
        color = (0, 255, 0)  # Green
    else:
        color = (128, 128, 128)  # Gray
    
    # Add text
    text = f"{label}: {confidence:.1f}%"
    
    # Get text size for background
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )
    
    # Draw background rectangle
    cv2.rectangle(
        annotated,
        (10, 10),
        (20 + text_width, 20 + text_height + baseline),
        color,
        -1,
    )
    
    # Draw text
    cv2.putText(
        annotated,
        text,
        (15, 15 + text_height),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
    )
    
    # Save if requested
    if save_path:
        cv2.imwrite(save_path, annotated)
    
    return annotated
