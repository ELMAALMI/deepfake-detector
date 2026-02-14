"""Deep learning model module for deepfake detection.

This module provides:
- XceptionNet-based architecture for binary classification
- Transfer learning with ImageNet pre-trained weights
- Custom classification head
- Model loading and saving utilities
"""

import logging
import os
from typing import Optional, Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import Xception

logger = logging.getLogger("deepfake_detector.model")


class DeepfakeDetectionModel:
    """Deepfake detection model based on XceptionNet."""
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (299, 299, 3),
        architecture: str = "xception",
        pretrained_weights: str = "imagenet",
        dropout_rate: float = 0.5,
        dense_units: int = 512,
        trainable_base: bool = False,
    ):
        """Initialize deepfake detection model.
        
        Args:
            input_shape: Input image shape as (height, width, channels)
            architecture: Base architecture ('xception')
            pretrained_weights: Pretrained weights ('imagenet' or None)
            dropout_rate: Dropout rate for regularization
            dense_units: Number of units in dense layer
            trainable_base: Whether to make base model trainable
        """
        self.input_shape = input_shape
        self.architecture = architecture.lower()
        self.pretrained_weights = pretrained_weights
        self.dropout_rate = dropout_rate
        self.dense_units = dense_units
        self.trainable_base = trainable_base
        
        self.model = None
        self._build_model()
    
    def _build_model(self) -> None:
        """Build the model architecture."""
        if self.architecture == "xception":
            self.model = self._build_xception_model()
        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")
        
        logger.info(f"Built {self.architecture} model with input shape {self.input_shape}")
    
    def _build_xception_model(self) -> keras.Model:
        """Build XceptionNet-based model.
        
        Returns:
            Compiled Keras model
        """
        # Load pre-trained Xception base
        if self.pretrained_weights == "imagenet":
            weights = "imagenet"
        else:
            weights = None
        
        base_model = Xception(
            input_shape=self.input_shape,
            include_top=False,
            weights=weights,
        )
        
        # Freeze base model layers initially
        base_model.trainable = self.trainable_base
        
        # Build custom classification head
        inputs = keras.Input(shape=self.input_shape)
        
        # Base model
        x = base_model(inputs, training=False)
        
        # Global average pooling
        x = layers.GlobalAveragePooling2D()(x)
        
        # Dense layer with ReLU activation
        x = layers.Dense(
            self.dense_units,
            activation="relu",
            kernel_regularizer=keras.regularizers.l2(0.01),
        )(x)
        
        # Dropout for regularization
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Output layer with sigmoid activation for binary classification
        outputs = layers.Dense(1, activation="sigmoid", name="prediction")(x)
        
        # Create model
        model = keras.Model(inputs=inputs, outputs=outputs, name="deepfake_detector")
        
        return model
    
    def compile(
        self,
        optimizer: Optional[keras.optimizers.Optimizer] = None,
        learning_rate: float = 0.0001,
        loss: str = "binary_crossentropy",
        metrics: Optional[list] = None,
    ) -> None:
        """Compile the model.
        
        Args:
            optimizer: Keras optimizer (creates Adam if None)
            learning_rate: Learning rate for optimizer
            loss: Loss function
            metrics: List of metrics to track
        """
        if optimizer is None:
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        if metrics is None:
            metrics = [
                "accuracy",
                keras.metrics.Precision(name="precision"),
                keras.metrics.Recall(name="recall"),
                keras.metrics.AUC(name="auc"),
            ]
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
        )
        
        logger.info(f"Compiled model with {optimizer.__class__.__name__} optimizer")
    
    def summary(self) -> None:
        """Print model summary."""
        if self.model:
            self.model.summary()
        else:
            logger.warning("Model not built yet")
    
    def get_model(self) -> keras.Model:
        """Get the Keras model.
        
        Returns:
            Keras model instance
        """
        return self.model
    
    def save(self, filepath: str) -> None:
        """Save model weights.
        
        Args:
            filepath: Path to save model weights
        """
        if self.model is None:
            raise ValueError("Model not built yet")
        
        # Create directory if needed
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model
        self.model.save(filepath)
        logger.info(f"Saved model to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load model weights.
        
        Args:
            filepath: Path to load model weights from
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        # Load model
        self.model = keras.models.load_model(filepath)
        logger.info(f"Loaded model from {filepath}")
    
    def predict(self, x: tf.Tensor) -> tf.Tensor:
        """Make predictions.
        
        Args:
            x: Input data
            
        Returns:
            Predictions as tensor
        """
        if self.model is None:
            raise ValueError("Model not built yet")
        
        return self.model.predict(x)
    
    def unfreeze_base_model(self, from_layer: Optional[int] = None) -> None:
        """Unfreeze base model layers for fine-tuning.
        
        Args:
            from_layer: Unfreeze layers from this index onwards (None = all layers)
        """
        if self.model is None:
            raise ValueError("Model not built yet")
        
        # Get base model (first layer after input)
        base_model = self.model.layers[1]
        
        if from_layer is None:
            # Unfreeze all layers
            base_model.trainable = True
            logger.info("Unfroze all base model layers")
        else:
            # Unfreeze layers from specified index
            base_model.trainable = True
            for layer in base_model.layers[:from_layer]:
                layer.trainable = False
            logger.info(f"Unfroze base model layers from index {from_layer}")
    
    def count_parameters(self) -> Tuple[int, int]:
        """Count trainable and non-trainable parameters.
        
        Returns:
            Tuple of (trainable_params, non_trainable_params)
        """
        if self.model is None:
            raise ValueError("Model not built yet")
        
        trainable_count = sum(
            [tf.size(w).numpy() for w in self.model.trainable_weights]
        )
        non_trainable_count = sum(
            [tf.size(w).numpy() for w in self.model.non_trainable_weights]
        )
        
        return trainable_count, non_trainable_count


def create_model(
    input_shape: Tuple[int, int, int] = (299, 299, 3),
    architecture: str = "xception",
    pretrained_weights: str = "imagenet",
    **kwargs
) -> DeepfakeDetectionModel:
    """Factory function to create a deepfake detection model.
    
    Args:
        input_shape: Input image shape
        architecture: Base architecture
        pretrained_weights: Pretrained weights
        **kwargs: Additional arguments for model initialization
        
    Returns:
        DeepfakeDetectionModel instance
    """
    model = DeepfakeDetectionModel(
        input_shape=input_shape,
        architecture=architecture,
        pretrained_weights=pretrained_weights,
        **kwargs
    )
    
    return model


def load_model(filepath: str) -> keras.Model:
    """Load a saved model.
    
    Args:
        filepath: Path to saved model
        
    Returns:
        Loaded Keras model
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    model = keras.models.load_model(filepath)
    logger.info(f"Loaded model from {filepath}")
    
    return model


def get_grad_cam_model(
    model: keras.Model,
    layer_name: str = "block14_sepconv2_act",
) -> Tuple[keras.Model, keras.Model]:
    """Create models for Grad-CAM visualization.
    
    Args:
        model: Original model
        layer_name: Name of convolutional layer for Grad-CAM
        
    Returns:
        Tuple of (grad_model, last_conv_layer)
    """
    # Find the target layer
    last_conv_layer = None
    for layer in model.layers:
        if layer.name == layer_name:
            last_conv_layer = layer
            break
    
    if last_conv_layer is None:
        # Try to find it in base model
        base_model = model.layers[1] if len(model.layers) > 1 else model
        for layer in base_model.layers:
            if layer.name == layer_name:
                last_conv_layer = layer
                break
    
    if last_conv_layer is None:
        raise ValueError(f"Layer {layer_name} not found in model")
    
    # Create a model that maps inputs to the activations of the last conv layer
    # and the output predictions
    grad_model = keras.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output]
    )
    
    return grad_model, last_conv_layer
