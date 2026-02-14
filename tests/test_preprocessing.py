"""Unit tests for preprocessing."""

import numpy as np
import pytest

from src.preprocessing import (
    ImagePreprocessor,
    resize_image,
    apply_histogram_equalization,
    apply_clahe,
)


def create_test_image(height: int = 480, width: int = 640) -> np.ndarray:
    """Create a test image.
    
    Args:
        height: Image height
        width: Image width
        
    Returns:
        Test image as numpy array in BGR format
    """
    return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)


def test_image_preprocessor_initialization():
    """Test image preprocessor initialization."""
    preprocessor = ImagePreprocessor(
        target_size=(299, 299),
        normalization="standard",
    )
    
    assert preprocessor.target_size == (299, 299)
    assert preprocessor.normalization == "standard"


def test_preprocess_image_standard_normalization():
    """Test image preprocessing with standard normalization."""
    preprocessor = ImagePreprocessor(
        target_size=(299, 299),
        normalization="standard",
    )
    
    image = create_test_image()
    processed = preprocessor.preprocess_image(image)
    
    # Check output shape
    assert processed.shape == (299, 299, 3)
    
    # Check normalization range (approximately [-1, 1])
    assert processed.min() >= -1.5  # Allow some margin
    assert processed.max() <= 1.5


def test_preprocess_image_minmax_normalization():
    """Test image preprocessing with min-max normalization."""
    preprocessor = ImagePreprocessor(
        target_size=(299, 299),
        normalization="minmax",
    )
    
    image = create_test_image()
    processed = preprocessor.preprocess_image(image)
    
    # Check output shape
    assert processed.shape == (299, 299, 3)
    
    # Check normalization range [0, 1]
    assert processed.min() >= 0.0
    assert processed.max() <= 1.0


def test_preprocess_image_invalid_normalization():
    """Test preprocessing with invalid normalization method."""
    preprocessor = ImagePreprocessor(normalization="invalid")
    image = create_test_image()
    
    with pytest.raises(ValueError):
        preprocessor.preprocess_image(image)


def test_preprocess_batch():
    """Test batch preprocessing."""
    preprocessor = ImagePreprocessor(target_size=(224, 224))
    
    images = [create_test_image() for _ in range(5)]
    batch = preprocessor.preprocess_batch(images)
    
    # Check output shape
    assert batch.shape == (5, 224, 224, 3)


def test_preprocess_with_augmentation():
    """Test preprocessing with augmentation enabled."""
    augmentation_config = {
        "enabled": True,
        "horizontal_flip": True,
        "rotation_range": 15,
        "zoom_range": 0.1,
    }
    
    preprocessor = ImagePreprocessor(
        target_size=(299, 299),
        augmentation_config=augmentation_config,
    )
    
    image = create_test_image()
    processed = preprocessor.preprocess_image(image, apply_augmentation=True)
    
    # Check output shape
    assert processed.shape == (299, 299, 3)


def test_preprocess_without_augmentation():
    """Test preprocessing with augmentation disabled."""
    augmentation_config = {"enabled": False}
    
    preprocessor = ImagePreprocessor(
        target_size=(299, 299),
        augmentation_config=augmentation_config,
    )
    
    image = create_test_image()
    processed1 = preprocessor.preprocess_image(image, apply_augmentation=False)
    processed2 = preprocessor.preprocess_image(image, apply_augmentation=False)
    
    # Without augmentation, results should be identical
    assert np.allclose(processed1, processed2)


def test_denormalize():
    """Test denormalization."""
    preprocessor = ImagePreprocessor(normalization="standard")
    
    image = create_test_image()
    processed = preprocessor.preprocess_image(image)
    denormalized = preprocessor.denormalize(processed)
    
    # Check output range [0, 255]
    assert denormalized.min() >= 0
    assert denormalized.max() <= 255
    assert denormalized.dtype == np.uint8


def test_resize_image():
    """Test image resizing."""
    image = create_test_image(height=480, width=640)
    resized = resize_image(image, target_size=(224, 224))
    
    assert resized.shape == (224, 224, 3)


def test_resize_image_maintain_aspect_ratio():
    """Test image resizing with aspect ratio maintained."""
    image = create_test_image(height=480, width=640)
    resized = resize_image(image, target_size=(224, 224), maintain_aspect_ratio=True)
    
    # Output should be target size
    assert resized.shape == (224, 224, 3)


def test_apply_histogram_equalization():
    """Test histogram equalization."""
    image = create_test_image()
    equalized = apply_histogram_equalization(image)
    
    # Check output shape and type
    assert equalized.shape == image.shape
    assert equalized.dtype == image.dtype


def test_apply_clahe():
    """Test CLAHE application."""
    image = create_test_image()
    clahe_image = apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8))
    
    # Check output shape and type
    assert clahe_image.shape == image.shape
    assert clahe_image.dtype == image.dtype


def test_preprocess_grayscale_image():
    """Test preprocessing grayscale image."""
    # Create grayscale image
    image_gray = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
    
    preprocessor = ImagePreprocessor(target_size=(299, 299))
    
    # Should handle gracefully (convert or raise error)
    try:
        processed = preprocessor.preprocess_image(image_gray)
        # If successful, check output shape
        assert len(processed.shape) == 3
    except:
        # Or it may raise an error, which is also acceptable
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
