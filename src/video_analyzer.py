"""Video analysis module for deepfake detection.

This module provides:
- Frame-by-frame video analysis
- Face extraction from video frames
- Prediction aggregation
- Annotated video output
- Progress tracking
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from src.predict import DeepfakePredictor
from src.utils import get_video_info, create_video_writer, ensure_dir

logger = logging.getLogger("deepfake_detector.video_analyzer")


class VideoAnalyzer:
    """Video analysis class for deepfake detection."""
    
    def __init__(
        self,
        predictor: DeepfakePredictor,
        frame_skip: int = 5,
        aggregation_method: str = "weighted_average",
    ):
        """Initialize video analyzer.
        
        Args:
            predictor: Deepfake predictor instance
            frame_skip: Process every Nth frame
            aggregation_method: Method to aggregate frame predictions
                                ('weighted_average', 'majority_vote', 'max_confidence')
        """
        self.predictor = predictor
        self.frame_skip = frame_skip
        self.aggregation_method = aggregation_method
        
        logger.info(f"VideoAnalyzer initialized with frame_skip={frame_skip}")
    
    def analyze_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        show_progress: bool = True,
    ) -> Dict:
        """Analyze video for deepfake detection.
        
        Args:
            video_path: Path to input video
            output_path: Optional path to save annotated video
            show_progress: Whether to show progress bar
            
        Returns:
            Dictionary containing:
                - 'video_prediction': Overall video prediction ('REAL' or 'FAKE')
                - 'video_confidence': Overall confidence score
                - 'video_probability': Aggregated probability
                - 'frame_predictions': List of frame-level predictions
                - 'total_frames': Total frames in video
                - 'processed_frames': Number of frames processed
        """
        logger.info(f"Analyzing video: {video_path}")
        
        # Get video info
        video_info = get_video_info(video_path)
        total_frames = video_info['frame_count']
        fps = video_info['fps']
        width = video_info['width']
        height = video_info['height']
        
        logger.info(f"Video info: {total_frames} frames, {fps:.2f} FPS, {width}x{height}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        
        # Create video writer if output requested
        video_writer = None
        if output_path:
            ensure_dir(os.path.dirname(output_path))
            video_writer = create_video_writer(
                output_path,
                fps,
                (width, height),
            )
        
        # Process frames
        frame_predictions = []
        frame_idx = 0
        
        progress_bar = None
        if show_progress:
            progress_bar = tqdm(total=total_frames, desc="Processing frames")
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Process frame if it's a frame we should analyze
            if frame_idx % self.frame_skip == 0:
                # Predict
                result = self.predictor.predict_image(
                    frame,
                    extract_face=True,
                )
                
                # Store result
                frame_predictions.append({
                    'frame_idx': frame_idx,
                    'timestamp': frame_idx / fps,
                    'prediction': result,
                })
                
                # Annotate frame if output requested
                if video_writer:
                    annotated_frame = self._annotate_frame(frame, result)
                    video_writer.write(annotated_frame)
            else:
                # Write original frame if not processed
                if video_writer:
                    video_writer.write(frame)
            
            frame_idx += 1
            
            if progress_bar:
                progress_bar.update(1)
        
        # Cleanup
        cap.release()
        if video_writer:
            video_writer.release()
        if progress_bar:
            progress_bar.close()
        
        logger.info(f"Processed {len(frame_predictions)} frames out of {total_frames}")
        
        # Aggregate predictions
        video_prediction, video_confidence, video_probability = self._aggregate_predictions(
            frame_predictions
        )
        
        return {
            'video_prediction': video_prediction,
            'video_confidence': video_confidence,
            'video_probability': video_probability,
            'frame_predictions': frame_predictions,
            'total_frames': total_frames,
            'processed_frames': len(frame_predictions),
            'video_info': video_info,
        }
    
    def _aggregate_predictions(
        self,
        frame_predictions: List[Dict],
    ) -> Tuple[str, float, float]:
        """Aggregate frame-level predictions into video-level prediction.
        
        Args:
            frame_predictions: List of frame prediction dictionaries
            
        Returns:
            Tuple of (prediction_label, confidence, probability)
        """
        if not frame_predictions:
            return 'UNKNOWN', 0.0, 0.5
        
        # Extract probabilities (filter out errors)
        valid_predictions = [
            p for p in frame_predictions
            if 'error' not in p['prediction']
        ]
        
        if not valid_predictions:
            return 'UNKNOWN', 0.0, 0.5
        
        probabilities = [p['prediction']['probability'] for p in valid_predictions]
        
        if self.aggregation_method == "weighted_average":
            # Weighted average based on confidence
            confidences = [p['prediction']['confidence'] for p in valid_predictions]
            weights = np.array(confidences) / sum(confidences)
            aggregated_prob = np.average(probabilities, weights=weights)
        
        elif self.aggregation_method == "majority_vote":
            # Majority vote
            labels = [p['prediction']['label'] for p in valid_predictions]
            fake_count = labels.count('FAKE')
            real_count = labels.count('REAL')
            
            if fake_count > real_count:
                aggregated_prob = 0.7  # Default high probability for fake
            else:
                aggregated_prob = 0.3  # Default low probability for real
        
        elif self.aggregation_method == "max_confidence":
            # Use prediction with maximum confidence
            max_confidence_pred = max(valid_predictions, key=lambda p: p['prediction']['confidence'])
            aggregated_prob = max_confidence_pred['prediction']['probability']
        
        else:
            # Default: simple average
            aggregated_prob = np.mean(probabilities)
        
        # Determine label and confidence
        threshold = self.predictor.confidence_threshold
        is_fake = aggregated_prob >= threshold
        label = 'FAKE' if is_fake else 'REAL'
        
        if is_fake:
            confidence = float(aggregated_prob * 100)
        else:
            confidence = float((1 - aggregated_prob) * 100)
        
        return label, confidence, float(aggregated_prob)
    
    def _annotate_frame(
        self,
        frame: np.ndarray,
        result: Dict,
    ) -> np.ndarray:
        """Annotate frame with prediction result.
        
        Args:
            frame: Video frame
            result: Prediction result
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        # Get prediction info
        label = result.get('label', 'UNKNOWN')
        confidence = result.get('confidence', 0.0)
        
        # Choose color
        if label == 'FAKE':
            color = (0, 0, 255)  # Red
        elif label == 'REAL':
            color = (0, 255, 0)  # Green
        else:
            color = (128, 128, 128)  # Gray
        
        # Add text
        text = f"{label}: {confidence:.1f}%"
        
        # Draw background and text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )
        
        cv2.rectangle(
            annotated,
            (10, 10),
            (20 + text_width, 20 + text_height + baseline),
            color,
            -1,
        )
        
        cv2.putText(
            annotated,
            text,
            (15, 15 + text_height),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
        )
        
        return annotated
    
    def generate_report(
        self,
        analysis_result: Dict,
        output_path: Optional[str] = None,
    ) -> str:
        """Generate analysis report.
        
        Args:
            analysis_result: Result from analyze_video
            output_path: Optional path to save report
            
        Returns:
            Report as string
        """
        report_lines = [
            "=" * 60,
            "DEEPFAKE DETECTION VIDEO ANALYSIS REPORT",
            "=" * 60,
            "",
            f"Overall Prediction: {analysis_result['video_prediction']}",
            f"Confidence: {analysis_result['video_confidence']:.2f}%",
            f"Probability: {analysis_result['video_probability']:.4f}",
            "",
            f"Total Frames: {analysis_result['total_frames']}",
            f"Processed Frames: {analysis_result['processed_frames']}",
            "",
            "Frame-by-Frame Analysis:",
            "-" * 60,
        ]
        
        # Add frame details
        for frame_pred in analysis_result['frame_predictions'][:10]:  # Show first 10
            frame_idx = frame_pred['frame_idx']
            timestamp = frame_pred['timestamp']
            pred = frame_pred['prediction']
            
            if 'error' in pred:
                report_lines.append(
                    f"Frame {frame_idx} ({timestamp:.2f}s): ERROR - {pred['error']}"
                )
            else:
                label = pred['label']
                conf = pred['confidence']
                report_lines.append(
                    f"Frame {frame_idx} ({timestamp:.2f}s): {label} ({conf:.1f}%)"
                )
        
        if len(analysis_result['frame_predictions']) > 10:
            report_lines.append(f"... and {len(analysis_result['frame_predictions']) - 10} more frames")
        
        report_lines.append("=" * 60)
        
        report = "\n".join(report_lines)
        
        # Save report if requested
        if output_path:
            ensure_dir(os.path.dirname(output_path))
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to {output_path}")
        
        return report
