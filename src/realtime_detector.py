"""Real-time deepfake detection module using webcam.

This module provides:
- Real-time webcam capture and analysis
- Face detection and prediction in real-time
- Visual feedback with bounding boxes and labels
- FPS counter
"""

import logging
import time
from typing import Optional

import cv2
import numpy as np

from src.predict import DeepfakePredictor
from src.face_extractor import FaceExtractor
from src.utils import load_config

logger = logging.getLogger("deepfake_detector.realtime")


class RealtimeDetector:
    """Real-time deepfake detection class."""
    
    def __init__(
        self,
        predictor: DeepfakePredictor,
        camera_id: int = 0,
        display_fps: bool = True,
        display_confidence: bool = True,
    ):
        """Initialize real-time detector.
        
        Args:
            predictor: Deepfake predictor instance
            camera_id: Camera device ID
            display_fps: Whether to display FPS counter
            display_confidence: Whether to display confidence score
        """
        self.predictor = predictor
        self.camera_id = camera_id
        self.display_fps = display_fps
        self.display_confidence = display_confidence
        
        # Load configuration
        realtime_config = predictor.config.get("realtime", {})
        self.box_color_real = tuple(realtime_config.get("box_color_real", [0, 255, 0]))
        self.box_color_fake = tuple(realtime_config.get("box_color_fake", [0, 0, 255]))
        self.box_thickness = realtime_config.get("box_thickness", 2)
        self.font_scale = realtime_config.get("font_scale", 0.7)
        
        logger.info(f"RealtimeDetector initialized with camera {camera_id}")
    
    def run(
        self,
        window_name: str = "Deepfake Detector - Real-time",
        quit_key: str = "q",
    ) -> None:
        """Run real-time detection.
        
        Args:
            window_name: Name of display window
            quit_key: Key to press to quit
        """
        logger.info("Starting real-time detection")
        
        # Open camera
        cap = cv2.VideoCapture(self.camera_id)
        
        if not cap.isOpened():
            raise ValueError(f"Failed to open camera {self.camera_id}")
        
        # FPS calculation
        fps = 0
        frame_count = 0
        start_time = time.time()
        
        logger.info(f"Press '{quit_key}' to quit")
        
        try:
            while True:
                # Capture frame
                ret, frame = cap.read()
                
                if not ret:
                    logger.warning("Failed to capture frame")
                    break
                
                # Process frame
                processed_frame = self._process_frame(frame)
                
                # Calculate FPS
                frame_count += 1
                elapsed_time = time.time() - start_time
                
                if elapsed_time > 1.0:
                    fps = frame_count / elapsed_time
                    frame_count = 0
                    start_time = time.time()
                
                # Display FPS
                if self.display_fps:
                    cv2.putText(
                        processed_frame,
                        f"FPS: {fps:.1f}",
                        (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )
                
                # Show frame
                cv2.imshow(window_name, processed_frame)
                
                # Check for quit key
                key = cv2.waitKey(1) & 0xFF
                if key == ord(quit_key):
                    logger.info("Quit key pressed")
                    break
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Real-time detection stopped")
    
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process single frame for real-time detection.
        
        Args:
            frame: Input frame
            
        Returns:
            Processed frame with annotations
        """
        annotated = frame.copy()
        
        # Extract faces
        faces = self.predictor.face_extractor.extract_faces(
            frame,
            min_face_size=self.predictor.min_face_size,
        )
        
        # Process each detected face
        for face_data in faces:
            bbox = face_data['bbox']
            face_img = face_data['face']
            
            # Predict
            result = self.predictor.predict_image(
                face_img,
                extract_face=False,  # Already extracted
            )
            
            # Get prediction info
            label = result.get('label', 'UNKNOWN')
            confidence = result.get('confidence', 0.0)
            
            # Choose color based on prediction
            if label == 'FAKE':
                color = self.box_color_fake
            elif label == 'REAL':
                color = self.box_color_real
            else:
                color = (128, 128, 128)  # Gray for unknown
            
            # Draw bounding box
            x, y, w, h = bbox
            cv2.rectangle(
                annotated,
                (x, y),
                (x + w, y + h),
                color,
                self.box_thickness,
            )
            
            # Prepare text
            if self.display_confidence:
                text = f"{label}: {confidence:.1f}%"
            else:
                text = label
            
            # Draw text background
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_width, text_height), baseline = cv2.getTextSize(
                text,
                font,
                self.font_scale,
                2,
            )
            
            # Position text above bounding box
            text_x = x
            text_y = y - 10
            
            # Ensure text is within frame
            if text_y < text_height + baseline:
                text_y = y + h + text_height + 10
            
            # Draw text background
            cv2.rectangle(
                annotated,
                (text_x, text_y - text_height - baseline),
                (text_x + text_width, text_y + baseline),
                color,
                -1,
            )
            
            # Draw text
            cv2.putText(
                annotated,
                text,
                (text_x, text_y),
                font,
                self.font_scale,
                (255, 255, 255),
                2,
            )
        
        # If no faces detected, show message
        if len(faces) == 0:
            cv2.putText(
                annotated,
                "No face detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
        
        return annotated


def run_realtime_detection(
    model_path: str,
    config_path: str = "config/config.yaml",
    camera_id: int = 0,
) -> None:
    """Convenience function to run real-time detection.
    
    Args:
        model_path: Path to trained model
        config_path: Path to configuration file
        camera_id: Camera device ID
    """
    # Create predictor
    predictor = DeepfakePredictor(model_path, config_path)
    
    # Create detector
    detector = RealtimeDetector(
        predictor,
        camera_id=camera_id,
    )
    
    # Run detection
    detector.run()
