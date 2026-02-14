#!/usr/bin/env python3
"""CLI script to detect deepfakes in video files.

Usage:
    python scripts/detect_video.py --video ./test_video.mp4 --model ./models/best_model.h5
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.predict import DeepfakePredictor
from src.video_analyzer import VideoAnalyzer
from src.utils import setup_logging


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Detect deepfakes in video files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to input video file",
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
        "--output-video",
        type=str,
        default=None,
        help="Path to save annotated output video",
    )
    
    parser.add_argument(
        "--output-report",
        type=str,
        default=None,
        help="Path to save analysis report",
    )
    
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=None,
        help="Process every Nth frame (overrides config)",
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Confidence threshold for classification (overrides config)",
    )
    
    parser.add_argument(
        "--aggregation-method",
        type=str,
        default=None,
        choices=["weighted_average", "majority_vote", "max_confidence"],
        help="Method to aggregate frame predictions (overrides config)",
    )
    
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar",
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
    logger.info("DEEPFAKE DETECTION - VIDEO ANALYSIS")
    logger.info("=" * 60)
    logger.info(f"Video: {args.video}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Config: {args.config}")
    
    try:
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
        
        # Determine frame skip
        frame_skip = args.frame_skip
        if frame_skip is None:
            frame_skip = predictor.config.get("detection", {}).get("frame_skip", 5)
        
        # Determine aggregation method
        aggregation_method = args.aggregation_method
        if aggregation_method is None:
            aggregation_method = predictor.config.get("detection", {}).get(
                "aggregation_method", "weighted_average"
            )
        
        # Create analyzer
        analyzer = VideoAnalyzer(
            predictor=predictor,
            frame_skip=frame_skip,
            aggregation_method=aggregation_method,
        )
        
        # Analyze video
        logger.info("Analyzing video...")
        result = analyzer.analyze_video(
            video_path=args.video,
            output_path=args.output_video,
            show_progress=not args.no_progress,
        )
        
        # Print results
        logger.info("=" * 60)
        logger.info("ANALYSIS RESULTS")
        logger.info("=" * 60)
        logger.info(f"Prediction: {result['video_prediction']}")
        logger.info(f"Confidence: {result['video_confidence']:.2f}%")
        logger.info(f"Probability: {result['video_probability']:.4f}")
        logger.info(f"Processed {result['processed_frames']} out of {result['total_frames']} frames")
        
        if args.output_video:
            logger.info(f"Annotated video saved to: {args.output_video}")
        
        # Generate and save report
        report = analyzer.generate_report(
            result,
            output_path=args.output_report,
        )
        
        print("\n" + report)
        
        if args.output_report:
            logger.info(f"Report saved to: {args.output_report}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Video analysis failed with error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
