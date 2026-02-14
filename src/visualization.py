"""Visualization module for deepfake detection.

This module provides:
- Grad-CAM (Gradient-weighted Class Activation Mapping) heatmaps
- Training history plots
- Confusion matrix visualization
- Prediction result visualization
"""

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow import keras

from src.model import get_grad_cam_model
from src.utils import ensure_dir

logger = logging.getLogger("deepfake_detector.visualization")


def compute_grad_cam(
    model: keras.Model,
    image: np.ndarray,
    layer_name: str = "block14_sepconv2_act",
    epsilon: float = 1e-8,
) -> np.ndarray:
    """Compute Grad-CAM heatmap for an image.
    
    Args:
        model: Trained model
        image: Preprocessed input image (single image, not batch)
        layer_name: Name of convolutional layer for Grad-CAM
        epsilon: Small constant to prevent division by zero
        
    Returns:
        Grad-CAM heatmap as numpy array
    """
    # Add batch dimension
    img_array = np.expand_dims(image, axis=0)
    
    # Get Grad-CAM model
    try:
        grad_model, last_conv_layer = get_grad_cam_model(model, layer_name)
    except ValueError as e:
        logger.warning(f"Could not create Grad-CAM model: {e}")
        # Return empty heatmap
        return np.zeros(image.shape[:2])
    
    # Compute gradient
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]
    
    # Get gradients
    grads = tape.gradient(loss, conv_outputs)
    
    # Pool gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight feature maps
    conv_outputs = conv_outputs[0]
    pooled_grads = pooled_grads.numpy()
    conv_outputs = conv_outputs.numpy()
    
    for i in range(len(pooled_grads)):
        conv_outputs[:, :, i] *= pooled_grads[i]
    
    # Average weighted feature maps
    heatmap = np.mean(conv_outputs, axis=-1)
    
    # Normalize heatmap
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > epsilon:
        heatmap /= heatmap.max()
    
    return heatmap


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Overlay Grad-CAM heatmap on image.
    
    Args:
        image: Original image (RGB, 0-255)
        heatmap: Grad-CAM heatmap (0-1)
        alpha: Transparency of heatmap overlay
        colormap: OpenCV colormap
        
    Returns:
        Image with heatmap overlay
    """
    # Resize heatmap to match image size
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Convert heatmap to color
    heatmap_colored = cv2.applyColorMap(
        (heatmap_resized * 255).astype(np.uint8),
        colormap,
    )
    
    # Convert to RGB
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Ensure image is RGB and uint8
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    # Overlay
    overlayed = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    
    return overlayed


def visualize_grad_cam(
    model: keras.Model,
    image: np.ndarray,
    preprocessed_image: np.ndarray,
    prediction: Dict,
    layer_name: str = "block14_sepconv2_act",
    save_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate and visualize Grad-CAM.
    
    Args:
        model: Trained model
        image: Original image (RGB, 0-255)
        preprocessed_image: Preprocessed image for model input
        prediction: Prediction result dictionary
        layer_name: Name of convolutional layer
        save_path: Optional path to save visualization
        
    Returns:
        Tuple of (heatmap, overlayed_image)
    """
    # Compute Grad-CAM
    heatmap = compute_grad_cam(model, preprocessed_image, layer_name)
    
    # Overlay on original image
    overlayed = overlay_heatmap(image, heatmap)
    
    # Create comparison figure
    if save_path:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis("off")
        
        # Heatmap
        axes[1].imshow(heatmap, cmap="jet")
        axes[1].set_title("Grad-CAM Heatmap")
        axes[1].axis("off")
        
        # Overlay
        axes[2].imshow(overlayed)
        label = prediction.get('label', 'UNKNOWN')
        confidence = prediction.get('confidence', 0)
        axes[2].set_title(f"Overlay\n{label}: {confidence:.1f}%")
        axes[2].axis("off")
        
        plt.tight_layout()
        
        ensure_dir(save_path)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved Grad-CAM visualization to {save_path}")
    
    return heatmap, overlayed


def plot_training_history(
    history: Dict,
    save_path: Optional[str] = None,
    metrics: Optional[List[str]] = None,
) -> None:
    """Plot training history.
    
    Args:
        history: Training history dictionary
        save_path: Optional path to save plot
        metrics: List of metrics to plot (default: loss and accuracy)
    """
    if metrics is None:
        metrics = ['loss', 'accuracy']
    
    # Determine number of subplots
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))
    
    if n_metrics == 1:
        axes = [axes]
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # Plot training metric
        if metric in history:
            ax.plot(history[metric], label=f'Training {metric}')
        
        # Plot validation metric
        val_metric = f'val_{metric}'
        if val_metric in history:
            ax.plot(history[val_metric], label=f'Validation {metric}')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'{metric.capitalize()} over Epochs')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        ensure_dir(save_path)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        logger.info(f"Saved training history plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = None,
    save_path: Optional[str] = None,
    normalize: bool = False,
) -> None:
    """Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Optional path to save plot
        normalize: Whether to normalize confusion matrix
    """
    from sklearn.metrics import confusion_matrix
    
    if class_names is None:
        class_names = ['Real', 'Fake']
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize if requested
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='.2f' if normalize else 'd',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'},
    )
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
    plt.tight_layout()
    
    if save_path:
        ensure_dir(save_path)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        logger.info(f"Saved confusion matrix plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_roc_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    save_path: Optional[str] = None,
) -> None:
    """Plot ROC curve.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        save_path: Optional path to save plot
    """
    from sklearn.metrics import roc_curve, auc
    
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Create plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        ensure_dir(save_path)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        logger.info(f"Saved ROC curve plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_prediction_comparison(
    images: List[np.ndarray],
    predictions: List[Dict],
    titles: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> None:
    """Visualize predictions for multiple images in a grid.
    
    Args:
        images: List of images (RGB)
        predictions: List of prediction dictionaries
        titles: Optional list of titles for each image
        save_path: Optional path to save visualization
    """
    n_images = len(images)
    cols = min(4, n_images)
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    
    if n_images == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, (image, prediction) in enumerate(zip(images, predictions)):
        ax = axes[idx]
        
        # Display image
        ax.imshow(image)
        
        # Create title
        if titles and idx < len(titles):
            title = titles[idx]
        else:
            label = prediction.get('label', 'UNKNOWN')
            confidence = prediction.get('confidence', 0)
            title = f"{label}: {confidence:.1f}%"
        
        # Color code based on prediction
        label = prediction.get('label', 'UNKNOWN')
        if label == 'FAKE':
            title_color = 'red'
        elif label == 'REAL':
            title_color = 'green'
        else:
            title_color = 'gray'
        
        ax.set_title(title, color=title_color, fontsize=10, fontweight='bold')
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(n_images, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        ensure_dir(save_path)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        logger.info(f"Saved prediction comparison to {save_path}")
    else:
        plt.show()
    
    plt.close()
