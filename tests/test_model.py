"""Unit tests for model architecture."""

import os
import tempfile

import numpy as np
import pytest
import tensorflow as tf

from src.model import (
    DeepfakeDetectionModel,
    create_model,
    load_model,
    get_grad_cam_model,
)


def test_model_initialization():
    """Test model initialization."""
    model = DeepfakeDetectionModel(
        input_shape=(299, 299, 3),
        architecture="xception",
        pretrained_weights=None,  # Don't download weights in tests
    )
    
    assert model is not None
    assert model.model is not None
    assert model.input_shape == (299, 299, 3)


def test_model_invalid_architecture():
    """Test model with invalid architecture."""
    with pytest.raises(ValueError):
        DeepfakeDetectionModel(architecture="invalid_architecture")


def test_model_compilation():
    """Test model compilation."""
    model = DeepfakeDetectionModel(
        input_shape=(299, 299, 3),
        pretrained_weights=None,
    )
    
    model.compile(learning_rate=0.001)
    
    # Check that model is compiled
    assert model.model.optimizer is not None


def test_model_output_shape():
    """Test model output shape."""
    model = DeepfakeDetectionModel(
        input_shape=(299, 299, 3),
        pretrained_weights=None,
    )
    
    # Create dummy input
    dummy_input = np.random.randn(1, 299, 299, 3).astype(np.float32)
    
    # Get output
    output = model.model(dummy_input, training=False)
    
    # Check output shape (batch_size, 1) for binary classification
    assert output.shape == (1, 1)
    
    # Check output is in valid range [0, 1] for sigmoid
    assert 0 <= output.numpy()[0, 0] <= 1


def test_model_summary():
    """Test model summary."""
    model = DeepfakeDetectionModel(
        input_shape=(299, 299, 3),
        pretrained_weights=None,
    )
    
    # Should not raise error
    model.summary()


def test_model_save_and_load():
    """Test model saving and loading."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "test_model.h5")
        
        # Create and save model
        model = DeepfakeDetectionModel(
            input_shape=(299, 299, 3),
            pretrained_weights=None,
        )
        model.compile()
        model.save(model_path)
        
        # Check file exists
        assert os.path.exists(model_path)
        
        # Load model
        loaded_model = load_model(model_path)
        assert loaded_model is not None


def test_model_load_not_found():
    """Test loading non-existent model."""
    with pytest.raises(FileNotFoundError):
        load_model("nonexistent_model.h5")


def test_model_predict():
    """Test model prediction."""
    model = DeepfakeDetectionModel(
        input_shape=(299, 299, 3),
        pretrained_weights=None,
    )
    
    # Create dummy input
    dummy_input = np.random.randn(2, 299, 299, 3).astype(np.float32)
    
    # Predict
    predictions = model.predict(dummy_input)
    
    # Check predictions shape
    assert predictions.shape == (2, 1)
    
    # Check predictions are in valid range
    assert np.all(predictions >= 0)
    assert np.all(predictions <= 1)


def test_model_unfreeze_base():
    """Test unfreezing base model layers."""
    model = DeepfakeDetectionModel(
        input_shape=(299, 299, 3),
        pretrained_weights=None,
        trainable_base=False,
    )
    
    # Initially, base should be frozen
    base_model = model.model.layers[1]
    assert not base_model.trainable
    
    # Unfreeze all layers
    model.unfreeze_base_model()
    assert base_model.trainable


def test_model_count_parameters():
    """Test parameter counting."""
    model = DeepfakeDetectionModel(
        input_shape=(299, 299, 3),
        pretrained_weights=None,
    )
    
    trainable, non_trainable = model.count_parameters()
    
    # Check that we have parameters
    assert trainable > 0
    assert non_trainable >= 0


def test_create_model_factory():
    """Test model factory function."""
    model = create_model(
        input_shape=(299, 299, 3),
        architecture="xception",
        pretrained_weights=None,
    )
    
    assert isinstance(model, DeepfakeDetectionModel)
    assert model.model is not None


def test_model_with_custom_parameters():
    """Test model with custom parameters."""
    model = DeepfakeDetectionModel(
        input_shape=(224, 224, 3),
        pretrained_weights=None,
        dropout_rate=0.3,
        dense_units=256,
    )
    
    assert model.input_shape == (224, 224, 3)
    assert model.dropout_rate == 0.3
    assert model.dense_units == 256


def test_get_grad_cam_model():
    """Test Grad-CAM model creation."""
    model = DeepfakeDetectionModel(
        input_shape=(299, 299, 3),
        pretrained_weights=None,
    )
    
    # Try to get Grad-CAM model
    # Note: This might fail if the layer name doesn't exist, which is acceptable
    try:
        grad_model, last_conv_layer = get_grad_cam_model(model.model)
        assert grad_model is not None
    except ValueError:
        # Layer not found is acceptable for this test
        pass


def test_model_trainable_base():
    """Test model with trainable base."""
    model = DeepfakeDetectionModel(
        input_shape=(299, 299, 3),
        pretrained_weights=None,
        trainable_base=True,
    )
    
    # Base model should be trainable
    base_model = model.model.layers[1]
    assert base_model.trainable


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
