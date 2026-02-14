#!/usr/bin/env python3
"""CLI script to evaluate trained model on test dataset.

Usage:
    python scripts/evaluate_model.py --model ./models/best_model.h5 --test-dir ./data/test
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.predict import DeepfakePredictor
from src.utils import setup_logging, calculate_metrics, get_confusion_matrix
from src.visualization import plot_confusion_matrix, plot_roc_curve


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate deepfake detection model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model",
    )
    
    parser.add_argument(
        "--test-dir",
        type=str,
        required=True,
        help="Directory containing test data (with 'real' and 'fake' subdirectories)",
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/evaluation",
        help="Directory to save evaluation results",
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Confidence threshold for classification (overrides config)",
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    
    return parser.parse_args()


def load_test_dataset(test_dir: str):
    """Load test dataset.
    
    Args:
        test_dir: Directory containing test data
        
    Returns:
        Tuple of (file_paths, labels)
    """
    file_paths = []
    labels = []
    
    # Load real images (label = 0)
    real_dir = os.path.join(test_dir, "real")
    if os.path.exists(real_dir):
        real_files = list(Path(real_dir).glob("*.[jJ][pP][gG]"))
        real_files.extend(list(Path(real_dir).glob("*.[jJ][pP][eE][gG]")))
        real_files.extend(list(Path(real_dir).glob("*.[pP][nN][gG]")))
        file_paths.extend([str(f) for f in real_files])
        labels.extend([0] * len(real_files))
    
    # Load fake images (label = 1)
    fake_dir = os.path.join(test_dir, "fake")
    if os.path.exists(fake_dir):
        fake_files = list(Path(fake_dir).glob("*.[jJ][pP][gG]"))
        fake_files.extend(list(Path(fake_dir).glob("*.[jJ][pP][eE][gG]")))
        fake_files.extend(list(Path(fake_dir).glob("*.[pP][nN][gG]")))
        file_paths.extend([str(f) for f in fake_files])
        labels.extend([1] * len(fake_files))
    
    return file_paths, labels


def main():
    """Main function."""
    args = parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("DEEPFAKE DETECTION MODEL EVALUATION")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Test directory: {args.test_dir}")
    logger.info(f"Config: {args.config}")
    
    try:
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Create predictor
        logger.info("Loading model...")
        predictor = DeepfakePredictor(
            model_path=args.model,
            config_path=args.config,
        )
        
        # Override threshold if specified
        if args.threshold is not None:
            predictor.confidence_threshold = args.threshold
            logger.info(f"Overriding threshold: {args.threshold}")
        
        # Load test dataset
        logger.info("Loading test dataset...")
        test_paths, test_labels = load_test_dataset(args.test_dir)
        
        if len(test_paths) == 0:
            logger.error(f"No test images found in {args.test_dir}")
            return 1
        
        logger.info(f"Found {len(test_paths)} test images")
        logger.info(f"Real images: {test_labels.count(0)}")
        logger.info(f"Fake images: {test_labels.count(1)}")
        
        # Evaluate model
        logger.info("Evaluating model...")
        predictions = []
        probabilities = []
        
        for image_path in tqdm(test_paths, desc="Processing"):
            result = predictor.predict_image(image_path)
            
            if 'error' in result:
                # Skip images with errors
                logger.warning(f"Error processing {image_path}: {result['error']}")
                predictions.append(-1)  # Invalid prediction
                probabilities.append(0.5)  # Neutral probability
            else:
                # Convert label to binary (0=real, 1=fake)
                pred_label = 1 if result['label'] == 'FAKE' else 0
                predictions.append(pred_label)
                probabilities.append(result['probability'])
        
        # Filter out invalid predictions
        valid_indices = [i for i, p in enumerate(predictions) if p != -1]
        test_labels_valid = [test_labels[i] for i in valid_indices]
        predictions_valid = [predictions[i] for i in valid_indices]
        probabilities_valid = [probabilities[i] for i in valid_indices]
        
        # Convert to numpy arrays
        y_true = np.array(test_labels_valid)
        y_pred = np.array(predictions_valid)
        y_pred_proba = np.array(probabilities_valid)
        
        # Calculate metrics
        metrics = calculate_metrics(y_true, y_pred, y_pred_proba)
        
        # Print metrics
        logger.info("=" * 60)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 60)
        logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall:    {metrics['recall']:.4f}")
        logger.info(f"F1 Score:  {metrics['f1_score']:.4f}")
        logger.info(f"AUC-ROC:   {metrics['auc_roc']:.4f}")
        
        # Confusion matrix
        cm = get_confusion_matrix(y_true, y_pred)
        logger.info("\nConfusion Matrix:")
        logger.info(f"TN: {cm[0, 0]}, FP: {cm[0, 1]}")
        logger.info(f"FN: {cm[1, 0]}, TP: {cm[1, 1]}")
        
        # Save metrics to file
        metrics_file = os.path.join(args.output_dir, "metrics.txt")
        with open(metrics_file, 'w') as f:
            f.write("DEEPFAKE DETECTION MODEL EVALUATION\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Model: {args.model}\n")
            f.write(f"Test Directory: {args.test_dir}\n")
            f.write(f"Total Test Images: {len(test_paths)}\n")
            f.write(f"Valid Predictions: {len(valid_indices)}\n\n")
            f.write("Metrics:\n")
            f.write(f"  Accuracy:  {metrics['accuracy']:.4f}\n")
            f.write(f"  Precision: {metrics['precision']:.4f}\n")
            f.write(f"  Recall:    {metrics['recall']:.4f}\n")
            f.write(f"  F1 Score:  {metrics['f1_score']:.4f}\n")
            f.write(f"  AUC-ROC:   {metrics['auc_roc']:.4f}\n\n")
            f.write("Confusion Matrix:\n")
            f.write(f"  TN: {cm[0, 0]}, FP: {cm[0, 1]}\n")
            f.write(f"  FN: {cm[1, 0]}, TP: {cm[1, 1]}\n")
        
        logger.info(f"Metrics saved to: {metrics_file}")
        
        # Plot confusion matrix
        cm_plot_path = os.path.join(args.output_dir, "confusion_matrix.png")
        plot_confusion_matrix(
            y_true,
            y_pred,
            class_names=['Real', 'Fake'],
            save_path=cm_plot_path,
            normalize=True,
        )
        logger.info(f"Confusion matrix plot saved to: {cm_plot_path}")
        
        # Plot ROC curve
        roc_plot_path = os.path.join(args.output_dir, "roc_curve.png")
        plot_roc_curve(
            y_true,
            y_pred_proba,
            save_path=roc_plot_path,
        )
        logger.info(f"ROC curve plot saved to: {roc_plot_path}")
        
        logger.info("=" * 60)
        logger.info("EVALUATION COMPLETED")
        logger.info("=" * 60)
        
        return 0
    
    except Exception as e:
        logger.error(f"Evaluation failed with error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
