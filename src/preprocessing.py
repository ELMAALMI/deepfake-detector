"""Image preprocessing and data augmentation module.

This module provides preprocessing functions including:
- Resizing to model input size
- Normalization (standard or min-max)
- Data augmentation for training
- Batch preprocessing
"""

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

logger = logging.getLogger("deepfake_detector.preprocessing")


class ImagePreprocessor:
    """Image preprocessing class for deepfake detection."""
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (299, 299),
        normalization: str = "standard",
        augmentation_config: Optional[dict] = None,
    ):
        """Initialize image preprocessor.
        
        Args:
            target_size: Target image size as (height, width)
            normalization: Normalization method ('standard' or 'minmax')
            augmentation_config: Configuration for data augmentation
        """
        self.target_size = target_size
        self.normalization = normalization.lower()
        self.augmentation_config = augmentation_config or {}
        
        # Create augmentation layers if enabled
        self.augmentation_enabled = self.augmentation_config.get("enabled", False)
        if self.augmentation_enabled:
            self.augmentation_layers = self._create_augmentation_layers()
    
    def _create_augmentation_layers(self) -> keras.Sequential:
        """Create data augmentation layers."""
        layers = []
        
        if self.augmentation_config.get("horizontal_flip", False):
            layers.append(keras.layers.RandomFlip("horizontal"))
        
        rotation_range = self.augmentation_config.get("rotation_range", 0)
        if rotation_range > 0:
            layers.append(keras.layers.RandomRotation(
                rotation_range / 360.0  # Convert degrees to fraction
            ))
        
        zoom_range = self.augmentation_config.get("zoom_range", 0)
        if zoom_range > 0:
            layers.append(keras.layers.RandomZoom(
                (-zoom_range, zoom_range)
            ))
        
        brightness_range = self.augmentation_config.get("brightness_range", None)
        if brightness_range:
            layers.append(keras.layers.RandomBrightness(
                (brightness_range[0] - 1.0, brightness_range[1] - 1.0)
            ))
        
        return keras.Sequential(layers) if layers else None
    
    def preprocess_image(
        self,
        image: np.ndarray,
        apply_augmentation: bool = False,
    ) -> np.ndarray:
        """Preprocess single image.
        
        Args:
            image: Input image in BGR format
            apply_augmentation: Whether to apply data augmentation
            
        Returns:
            Preprocessed image as numpy array
        """
        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to target size
        if image.shape[:2] != self.target_size:
            image = cv2.resize(image, (self.target_size[1], self.target_size[0]))
        
        # Convert to float
        image = image.astype(np.float32)
        
        # Apply augmentation if requested
        if apply_augmentation and self.augmentation_enabled and self.augmentation_layers:
            # Add batch dimension for augmentation
            image = np.expand_dims(image, axis=0)
            image = self.augmentation_layers(image, training=True)
            image = image[0].numpy()
        
        # Add Gaussian noise if configured and augmentation is enabled
        if apply_augmentation and self.augmentation_enabled:
            noise_std = self.augmentation_config.get("gaussian_noise_std", 0)
            if noise_std > 0:
                noise = np.random.normal(0, noise_std * 255, image.shape)
                image = np.clip(image + noise, 0, 255)
        
        # Normalize
        image = self._normalize(image)
        
        return image
    
    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalize image.
        
        Args:
            image: Input image as float array
            
        Returns:
            Normalized image
        """
        if self.normalization == "standard":
            # Standard normalization: (x - 127.5) / 127.5 -> [-1, 1]
            return (image - 127.5) / 127.5
        elif self.normalization == "minmax":
            # Min-max normalization: x / 255 -> [0, 1]
            return image / 255.0
        else:
            raise ValueError(f"Unknown normalization method: {self.normalization}")
    
    def preprocess_batch(
        self,
        images: List[np.ndarray],
        apply_augmentation: bool = False,
    ) -> np.ndarray:
        """Preprocess batch of images.
        
        Args:
            images: List of input images in BGR format
            apply_augmentation: Whether to apply data augmentation
            
        Returns:
            Batch of preprocessed images as numpy array
        """
        preprocessed = []
        
        for image in images:
            processed = self.preprocess_image(image, apply_augmentation)
            preprocessed.append(processed)
        
        return np.array(preprocessed)
    
    def denormalize(self, image: np.ndarray) -> np.ndarray:
        """Denormalize image for visualization.
        
        Args:
            image: Normalized image
            
        Returns:
            Denormalized image in [0, 255] range
        """
        if self.normalization == "standard":
            # Reverse standard normalization
            image = (image * 127.5) + 127.5
        elif self.normalization == "minmax":
            # Reverse min-max normalization
            image = image * 255.0
        
        return np.clip(image, 0, 255).astype(np.uint8)


def resize_image(
    image: np.ndarray,
    target_size: Tuple[int, int],
    maintain_aspect_ratio: bool = False,
) -> np.ndarray:
    """Resize image to target size.
    
    Args:
        image: Input image
        target_size: Target size as (height, width)
        maintain_aspect_ratio: Whether to maintain aspect ratio
        
    Returns:
        Resized image
    """
    if maintain_aspect_ratio:
        h, w = image.shape[:2]
        target_h, target_w = target_size
        
        # Calculate scale to fit image in target size
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize
        resized = cv2.resize(image, (new_w, new_h))
        
        # Create canvas and paste resized image
        canvas = np.zeros((target_h, target_w, image.shape[2]), dtype=image.dtype)
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return canvas
    else:
        return cv2.resize(image, (target_size[1], target_size[0]))


def apply_histogram_equalization(image: np.ndarray) -> np.ndarray:
    """Apply histogram equalization to improve contrast.
    
    Args:
        image: Input image in BGR format
        
    Returns:
        Image with equalized histogram
    """
    # Convert to YCrCb color space
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    
    # Equalize Y channel
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    
    # Convert back to BGR
    result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    
    return result


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    Args:
        image: Input image in BGR format
        clip_limit: Threshold for contrast limiting
        tile_grid_size: Size of grid for histogram equalization
        
    Returns:
        Image with CLAHE applied
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    
    # Convert back to BGR
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return result
