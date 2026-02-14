#!/usr/bin/env python3
"""CLI script to train deepfake detection model.

Usage:
    python scripts/train_model.py --data-dir ./data --output-dir ./models
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.train import train_model
from src.utils import setup_logging


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train deepfake detection model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing training data (with 'real' and 'fake' subdirectories)",
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
        default="models",
        help="Directory to save trained model",
    )
    
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides config)",
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (overrides config)",
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate (overrides config)",
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("DEEPFAKE DETECTION MODEL TRAINING")
    logger.info("=" * 60)
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Config file: {args.config}")
    
    if args.resume_from:
        logger.info(f"Resuming from: {args.resume_from}")
    
    # Load and potentially update config
    from src.utils import load_config, save_config
    import os
    
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
        logger.info(f"Overriding epochs: {args.epochs}")
    
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
        logger.info(f"Overriding batch size: {args.batch_size}")
    
    if args.learning_rate is not None:
        config['training']['learning_rate'] = args.learning_rate
        logger.info(f"Overriding learning rate: {args.learning_rate}")
    
    # Save updated config
    temp_config_path = os.path.join(args.output_dir, "training_config.yaml")
    save_config(config, temp_config_path)
    
    try:
        # Train model
        history = train_model(
            data_dir=args.data_dir,
            config_path=temp_config_path,
            output_dir=args.output_dir,
            resume_from=args.resume_from,
        )
        
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"Final training loss: {history['loss'][-1]:.4f}")
        logger.info(f"Final validation loss: {history['val_loss'][-1]:.4f}")
        logger.info(f"Final training accuracy: {history['accuracy'][-1]:.4f}")
        logger.info(f"Final validation accuracy: {history['val_accuracy'][-1]:.4f}")
        logger.info(f"Model saved to: {args.output_dir}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
