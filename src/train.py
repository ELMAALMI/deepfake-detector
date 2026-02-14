"""Training pipeline for deepfake detection model.

This module provides:
- Data loading from directory structure
- Training with callbacks (EarlyStopping, ModelCheckpoint, etc.)
- Training/validation split
- Metrics logging
- Support for resuming training
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.model import DeepfakeDetectionModel
from src.preprocessing import ImagePreprocessor
from src.utils import load_config, setup_logging, ensure_dir

logger = logging.getLogger("deepfake_detector.train")


class DataGenerator(keras.utils.Sequence):
    """Custom data generator for training."""
    
    def __init__(
        self,
        file_paths: List[str],
        labels: List[int],
        preprocessor: ImagePreprocessor,
        batch_size: int = 32,
        shuffle: bool = True,
        augmentation: bool = False,
    ):
        """Initialize data generator.
        
        Args:
            file_paths: List of image file paths
            labels: List of labels (0=real, 1=fake)
            preprocessor: Image preprocessor instance
            batch_size: Batch size
            shuffle: Whether to shuffle data at end of epoch
            augmentation: Whether to apply data augmentation
        """
        self.file_paths = file_paths
        self.labels = labels
        self.preprocessor = preprocessor
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augmentation = augmentation
        self.indexes = np.arange(len(self.file_paths))
        
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __len__(self) -> int:
        """Get number of batches per epoch."""
        return int(np.ceil(len(self.file_paths) / self.batch_size))
    
    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get batch at index."""
        # Get batch indexes
        batch_indexes = self.indexes[
            index * self.batch_size:(index + 1) * self.batch_size
        ]
        
        # Get batch data
        batch_paths = [self.file_paths[i] for i in batch_indexes]
        batch_labels = [self.labels[i] for i in batch_indexes]
        
        # Load and preprocess images
        batch_images = []
        for path in batch_paths:
            try:
                import cv2
                image = cv2.imread(path)
                if image is not None:
                    processed = self.preprocessor.preprocess_image(
                        image,
                        apply_augmentation=self.augmentation,
                    )
                    batch_images.append(processed)
                else:
                    logger.warning(f"Failed to load image: {path}")
                    # Skip this image rather than using blank
                    continue
            except Exception as e:
                logger.error(f"Error loading image {path}: {e}")
                continue
        
        # If no valid images, return empty batch
        if len(batch_images) == 0:
            logger.warning("No valid images in batch")
            blank = np.zeros((1, *self.preprocessor.target_size, 3), dtype=np.float32)
            return blank, np.array([0], dtype=np.float32)
        
        return np.array(batch_images), np.array([self.labels[i] for i in batch_indexes[:len(batch_images)]], dtype=np.float32)
    
    def on_epoch_end(self):
        """Called at the end of each epoch."""
        if self.shuffle:
            np.random.shuffle(self.indexes)


def load_dataset(
    data_dir: str,
    validation_split: float = 0.2,
    random_state: int = 42,
) -> Tuple[List[str], List[int], List[str], List[int]]:
    """Load dataset from directory structure.
    
    Expected structure:
    data_dir/
        real/
            img1.jpg
            img2.jpg
        fake/
            img1.jpg
            img2.jpg
    
    Args:
        data_dir: Root data directory
        validation_split: Fraction of data for validation
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_paths, train_labels, val_paths, val_labels)
    """
    file_paths = []
    labels = []
    
    # Load real images (label = 0)
    real_dir = os.path.join(data_dir, "real")
    if os.path.exists(real_dir):
        real_files = list(Path(real_dir).glob("*.[jJ][pP][gG]"))
        real_files.extend(list(Path(real_dir).glob("*.[jJ][pP][eE][gG]")))
        real_files.extend(list(Path(real_dir).glob("*.[pP][nN][gG]")))
        file_paths.extend([str(f) for f in real_files])
        labels.extend([0] * len(real_files))
        logger.info(f"Loaded {len(real_files)} real images")
    
    # Load fake images (label = 1)
    fake_dir = os.path.join(data_dir, "fake")
    if os.path.exists(fake_dir):
        fake_files = list(Path(fake_dir).glob("*.[jJ][pP][gG]"))
        fake_files.extend(list(Path(fake_dir).glob("*.[jJ][pP][eE][gG]")))
        fake_files.extend(list(Path(fake_dir).glob("*.[pP][nN][gG]")))
        file_paths.extend([str(f) for f in fake_files])
        labels.extend([1] * len(fake_files))
        logger.info(f"Loaded {len(fake_files)} fake images")
    
    if len(file_paths) == 0:
        raise ValueError(f"No images found in {data_dir}")
    
    logger.info(f"Total images: {len(file_paths)}")
    
    # Split into train and validation
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        file_paths,
        labels,
        test_size=validation_split,
        random_state=random_state,
        stratify=labels,
    )
    
    logger.info(f"Training samples: {len(train_paths)}")
    logger.info(f"Validation samples: {len(val_paths)}")
    
    return train_paths, train_labels, val_paths, val_labels


def create_callbacks(
    model_dir: str,
    early_stopping_patience: int = 10,
    reduce_lr_patience: int = 5,
    reduce_lr_factor: float = 0.5,
    min_learning_rate: float = 0.000001,
    tensorboard_dir: Optional[str] = None,
) -> List[keras.callbacks.Callback]:
    """Create training callbacks.
    
    Args:
        model_dir: Directory to save model checkpoints
        early_stopping_patience: Patience for early stopping
        reduce_lr_patience: Patience for learning rate reduction
        reduce_lr_factor: Factor to reduce learning rate by
        min_learning_rate: Minimum learning rate
        tensorboard_dir: Directory for TensorBoard logs
        
    Returns:
        List of Keras callbacks
    """
    ensure_dir(model_dir)
    
    callbacks = []
    
    # Model checkpoint
    checkpoint_path = os.path.join(model_dir, "best_model.h5")
    callbacks.append(
        keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1,
        )
    )
    
    # Early stopping
    callbacks.append(
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        )
    )
    
    # Reduce learning rate on plateau
    callbacks.append(
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=reduce_lr_factor,
            patience=reduce_lr_patience,
            min_lr=min_learning_rate,
            verbose=1,
        )
    )
    
    # TensorBoard
    if tensorboard_dir:
        ensure_dir(tensorboard_dir)
        callbacks.append(
            keras.callbacks.TensorBoard(
                log_dir=tensorboard_dir,
                histogram_freq=1,
                write_graph=True,
            )
        )
    
    return callbacks


def train_model(
    data_dir: str,
    config_path: str = "config/config.yaml",
    output_dir: str = "models",
    resume_from: Optional[str] = None,
) -> Dict:
    """Train deepfake detection model.
    
    Args:
        data_dir: Directory containing training data
        config_path: Path to configuration file
        output_dir: Directory to save trained model
        resume_from: Path to checkpoint to resume from
        
    Returns:
        Training history dictionary
    """
    # Load configuration
    config = load_config(config_path)
    
    # Setup logging
    log_dir = config.get("paths", {}).get("log_dir", "logs")
    ensure_dir(log_dir)
    setup_logging(
        level=config.get("logging", {}).get("level", "INFO"),
        log_file=os.path.join(log_dir, "training.log"),
    )
    
    logger.info("Starting training pipeline")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Create preprocessor
    model_config = config.get("model", {})
    preprocessing_config = config.get("preprocessing", {})
    
    input_size = model_config.get("input_size", [299, 299, 3])
    preprocessor = ImagePreprocessor(
        target_size=(input_size[0], input_size[1]),
        normalization=preprocessing_config.get("normalization", "standard"),
        augmentation_config=preprocessing_config.get("augmentation", {}),
    )
    
    # Load dataset
    training_config = config.get("training", {})
    validation_split = training_config.get("validation_split", 0.2)
    
    train_paths, train_labels, val_paths, val_labels = load_dataset(
        data_dir,
        validation_split=validation_split,
    )
    
    # Create data generators
    batch_size = training_config.get("batch_size", 32)
    
    train_generator = DataGenerator(
        train_paths,
        train_labels,
        preprocessor,
        batch_size=batch_size,
        shuffle=True,
        augmentation=True,
    )
    
    val_generator = DataGenerator(
        val_paths,
        val_labels,
        preprocessor,
        batch_size=batch_size,
        shuffle=False,
        augmentation=False,
    )
    
    # Create model
    if resume_from and os.path.exists(resume_from):
        logger.info(f"Resuming from checkpoint: {resume_from}")
        model_instance = DeepfakeDetectionModel(
            input_shape=tuple(input_size),
            architecture=model_config.get("architecture", "xception"),
            pretrained_weights=None,
        )
        model_instance.load(resume_from)
    else:
        logger.info("Creating new model")
        model_instance = DeepfakeDetectionModel(
            input_shape=tuple(input_size),
            architecture=model_config.get("architecture", "xception"),
            pretrained_weights=model_config.get("pretrained_weights", "imagenet"),
            dropout_rate=model_config.get("dropout_rate", 0.5),
            dense_units=model_config.get("dense_units", 512),
        )
    
    # Compile model
    learning_rate = training_config.get("learning_rate", 0.0001)
    model_instance.compile(learning_rate=learning_rate)
    
    # Print model summary
    model_instance.summary()
    
    # Create callbacks
    callbacks = create_callbacks(
        model_dir=output_dir,
        early_stopping_patience=training_config.get("early_stopping_patience", 10),
        reduce_lr_patience=training_config.get("reduce_lr_patience", 5),
        reduce_lr_factor=training_config.get("reduce_lr_factor", 0.5),
        min_learning_rate=training_config.get("min_learning_rate", 0.000001),
        tensorboard_dir=config.get("paths", {}).get("tensorboard_dir", "tensorboard"),
    )
    
    # Train model
    epochs = training_config.get("epochs", 50)
    logger.info(f"Training for {epochs} epochs")
    
    history = model_instance.model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )
    
    # Save final model
    final_model_path = os.path.join(output_dir, "final_model.h5")
    model_instance.save(final_model_path)
    logger.info(f"Saved final model to {final_model_path}")
    
    return history.history
