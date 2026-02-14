"""Face extraction module using OpenCV and MediaPipe.

This module provides face detection and extraction functionality using:
- MediaPipe Face Detection for fast and accurate detection
- OpenCV DNN module as an alternative method
- Facial landmark detection for alignment
- Support for multiple faces per frame
"""

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

logger = logging.getLogger("deepfake_detector.face_extractor")


class FaceExtractor:
    """Face extraction class using MediaPipe or OpenCV DNN."""
    
    def __init__(
        self,
        method: str = "mediapipe",
        min_detection_confidence: float = 0.5,
        model_selection: int = 0,
        opencv_model_path: Optional[str] = None,
        opencv_config_path: Optional[str] = None,
    ):
        """Initialize face extractor.
        
        Args:
            method: Detection method ('mediapipe' or 'opencv_dnn')
            min_detection_confidence: Minimum confidence for detection
            model_selection: MediaPipe model (0=short-range, 1=full-range)
            opencv_model_path: Path to OpenCV DNN model weights
            opencv_config_path: Path to OpenCV DNN model config
        """
        self.method = method.lower()
        self.min_detection_confidence = min_detection_confidence
        
        if self.method == "mediapipe":
            self.mp_face_detection = mp.solutions.face_detection
            self.face_detection = self.mp_face_detection.FaceDetection(
                min_detection_confidence=min_detection_confidence,
                model_selection=model_selection,
            )
        elif self.method == "opencv_dnn":
            if opencv_model_path is None or opencv_config_path is None:
                # Use default OpenCV face detector
                logger.warning("OpenCV DNN paths not provided, using Haar Cascade instead")
                self.face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                )
                self.method = "opencv_haar"
            else:
                self.net = cv2.dnn.readNetFromCaffe(opencv_config_path, opencv_model_path)
        else:
            raise ValueError(f"Unknown detection method: {method}")
    
    def extract_faces(
        self,
        image: np.ndarray,
        min_face_size: int = 30,
        padding: float = 0.2,
    ) -> List[Dict]:
        """Extract faces from image.
        
        Args:
            image: Input image in BGR format
            min_face_size: Minimum face size in pixels
            padding: Padding around face bounding box (as fraction of box size)
            
        Returns:
            List of dictionaries containing:
                - 'bbox': Bounding box as (x, y, w, h)
                - 'confidence': Detection confidence
                - 'face': Cropped face image
                - 'landmarks': Facial landmarks (if available)
        """
        if self.method == "mediapipe":
            return self._extract_faces_mediapipe(image, min_face_size, padding)
        elif self.method == "opencv_dnn":
            return self._extract_faces_opencv_dnn(image, min_face_size, padding)
        elif self.method == "opencv_haar":
            return self._extract_faces_opencv_haar(image, min_face_size, padding)
        else:
            return []
    
    def _extract_faces_mediapipe(
        self,
        image: np.ndarray,
        min_face_size: int,
        padding: float,
    ) -> List[Dict]:
        """Extract faces using MediaPipe."""
        faces = []
        
        # Convert BGR to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(image_rgb)
        
        if not results.detections:
            return faces
        
        h, w = image.shape[:2]
        
        for detection in results.detections:
            # Get bounding box
            bbox = detection.location_data.relative_bounding_box
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            
            # Check minimum size
            if width < min_face_size or height < min_face_size:
                continue
            
            # Apply padding
            pad_w = int(width * padding)
            pad_h = int(height * padding)
            
            x1 = max(0, x - pad_w)
            y1 = max(0, y - pad_h)
            x2 = min(w, x + width + pad_w)
            y2 = min(h, y + height + pad_h)
            
            # Extract face
            face_img = image[y1:y2, x1:x2]
            
            # Get confidence
            confidence = detection.score[0] if hasattr(detection, 'score') else 1.0
            
            # Get landmarks (if available)
            landmarks = None
            if hasattr(detection.location_data, 'relative_keypoints'):
                landmarks = []
                for keypoint in detection.location_data.relative_keypoints:
                    landmarks.append((int(keypoint.x * w), int(keypoint.y * h)))
            
            faces.append({
                'bbox': (x, y, width, height),
                'bbox_padded': (x1, y1, x2 - x1, y2 - y1),
                'confidence': confidence,
                'face': face_img,
                'landmarks': landmarks,
            })
        
        return faces
    
    def _extract_faces_opencv_dnn(
        self,
        image: np.ndarray,
        min_face_size: int,
        padding: float,
    ) -> List[Dict]:
        """Extract faces using OpenCV DNN."""
        faces = []
        h, w = image.shape[:2]
        
        # Prepare image for DNN
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0)
        )
        
        self.net.setInput(blob)
        detections = self.net.forward()
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence < self.min_detection_confidence:
                continue
            
            # Get bounding box
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x, y, x2, y2 = box.astype(int)
            width = x2 - x
            height = y2 - y
            
            # Check minimum size
            if width < min_face_size or height < min_face_size:
                continue
            
            # Apply padding
            pad_w = int(width * padding)
            pad_h = int(height * padding)
            
            x1 = max(0, x - pad_w)
            y1 = max(0, y - pad_h)
            x2 = min(w, x + width + pad_w)
            y2 = min(h, y + height + pad_h)
            
            # Extract face
            face_img = image[y1:y2, x1:x2]
            
            faces.append({
                'bbox': (x, y, width, height),
                'bbox_padded': (x1, y1, x2 - x1, y2 - y1),
                'confidence': float(confidence),
                'face': face_img,
                'landmarks': None,
            })
        
        return faces
    
    def _extract_faces_opencv_haar(
        self,
        image: np.ndarray,
        min_face_size: int,
        padding: float,
    ) -> List[Dict]:
        """Extract faces using OpenCV Haar Cascade."""
        faces = []
        
        # Convert to grayscale for Haar Cascade
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        detected_faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(min_face_size, min_face_size),
        )
        
        h, w = image.shape[:2]
        
        for (x, y, width, height) in detected_faces:
            # Apply padding
            pad_w = int(width * padding)
            pad_h = int(height * padding)
            
            x1 = max(0, x - pad_w)
            y1 = max(0, y - pad_h)
            x2 = min(w, x + width + pad_w)
            y2 = min(h, y + height + pad_h)
            
            # Extract face
            face_img = image[y1:y2, x1:x2]
            
            faces.append({
                'bbox': (x, y, width, height),
                'bbox_padded': (x1, y1, x2 - x1, y2 - y1),
                'confidence': 1.0,  # Haar Cascade doesn't provide confidence
                'face': face_img,
                'landmarks': None,
            })
        
        return faces
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'face_detection'):
            self.face_detection.close()


def extract_largest_face(
    image: np.ndarray,
    extractor: Optional[FaceExtractor] = None,
    **kwargs
) -> Optional[Dict]:
    """Extract the largest face from image.
    
    Args:
        image: Input image in BGR format
        extractor: FaceExtractor instance (creates new one if None)
        **kwargs: Additional arguments for face extraction
        
    Returns:
        Dictionary with face data or None if no face found
    """
    if extractor is None:
        extractor = FaceExtractor()
    
    faces = extractor.extract_faces(image, **kwargs)
    
    if not faces:
        return None
    
    # Return face with largest area
    largest_face = max(faces, key=lambda f: f['bbox'][2] * f['bbox'][3])
    return largest_face
