#!/usr/bin/env python3
"""CLI script for real-time deepfake detection using webcam.

Usage:
    python scripts/detect_realtime.py --model ./models/best_model.h5
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.realtime_detector import run_realtime_detection
from src.utils import setup_logging


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Real-time deepfake detection using webcam",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model",
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file",
    )
    
    parser.add_argument(
        "--camera-id",
        type=int,
        default=0,
        help="Camera device ID",
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


def main():
    """Main function."""
    args = parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("DEEPFAKE DETECTION - REAL-TIME MODE")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Camera ID: {args.camera_id}")
    logger.info("Press 'q' to quit")
    logger.info("=" * 60)
    
    try:
        # Override threshold if specified
        if args.threshold is not None:
            from src.utils import load_config, save_config
            import os
            import tempfile
            
            config = load_config(args.config)
            config['detection']['confidence_threshold'] = args.threshold
            
            # Save to temp file
            temp_dir = tempfile.mkdtemp()
            temp_config = os.path.join(temp_dir, "config.yaml")
            save_config(config, temp_config)
            config_path = temp_config
            logger.info(f"Overriding threshold: {args.threshold}")
        else:
            config_path = args.config
        
        # Run real-time detection
        run_realtime_detection(
            model_path=args.model,
            config_path=config_path,
            camera_id=args.camera_id,
        )
        
        logger.info("Real-time detection stopped")
        return 0
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 0
    
    except Exception as e:
        logger.error(f"Real-time detection failed with error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
