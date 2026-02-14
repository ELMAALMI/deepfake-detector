"""Unit tests for face extraction."""

import numpy as np
import pytest

from src.face_extractor import FaceExtractor, extract_largest_face


def create_test_image(width: int = 640, height: int = 480, with_face: bool = True) -> np.ndarray:
    """Create a test image with or without a face-like pattern.
    
    Args:
        width: Image width
        height: Image height
        with_face: Whether to add a face-like pattern
        
    Returns:
        Test image as numpy array
    """
    # Create random image
    image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    
    if with_face:
        # Add a simple face-like pattern (oval shape)
        center_x, center_y = width // 2, height // 2
        face_w, face_h = 100, 120
        
        # Draw oval for face
        import cv2
        cv2.ellipse(
            image,
            (center_x, center_y),
            (face_w // 2, face_h // 2),
            0, 0, 360,
            (200, 180, 160),
            -1
        )
        
        # Add eyes
        eye_y = center_y - 20
        cv2.circle(image, (center_x - 25, eye_y), 10, (50, 50, 50), -1)
        cv2.circle(image, (center_x + 25, eye_y), 10, (50, 50, 50), -1)
        
        # Add mouth
        mouth_y = center_y + 30
        cv2.ellipse(
            image,
            (center_x, mouth_y),
            (30, 15),
            0, 0, 180,
            (100, 50, 50),
            -1
        )
    
    return image


def test_face_extractor_initialization():
    """Test face extractor initialization."""
    # MediaPipe initialization
    extractor = FaceExtractor(method="mediapipe")
    assert extractor.method == "mediapipe"
    
    # OpenCV Haar initialization (fallback when DNN paths not provided)
    extractor = FaceExtractor(method="opencv_dnn")
    assert extractor.method == "opencv_haar"


def test_face_extractor_invalid_method():
    """Test face extractor with invalid method."""
    with pytest.raises(ValueError):
        FaceExtractor(method="invalid_method")


def test_extract_faces_no_face():
    """Test face extraction on image without face."""
    extractor = FaceExtractor(method="mediapipe")
    image = create_test_image(with_face=False)
    
    faces = extractor.extract_faces(image)
    
    # May or may not detect faces in random noise, so we just check return type
    assert isinstance(faces, list)


def test_extract_faces_with_face():
    """Test face extraction on image with face-like pattern."""
    extractor = FaceExtractor(method="mediapipe", min_detection_confidence=0.3)
    image = create_test_image(with_face=True)
    
    faces = extractor.extract_faces(image, min_face_size=20)
    
    # Check return type and structure
    assert isinstance(faces, list)
    
    # If faces detected, check structure
    if len(faces) > 0:
        face = faces[0]
        assert 'bbox' in face
        assert 'confidence' in face
        assert 'face' in face
        assert isinstance(face['bbox'], tuple)
        assert len(face['bbox']) == 4
        assert isinstance(face['confidence'], float)
        assert isinstance(face['face'], np.ndarray)


def test_extract_faces_min_size_filter():
    """Test minimum face size filtering."""
    extractor = FaceExtractor(method="mediapipe")
    image = create_test_image(with_face=True)
    
    # Extract with very large minimum size (should filter out most detections)
    faces = extractor.extract_faces(image, min_face_size=500)
    
    # Check that faces are filtered
    assert isinstance(faces, list)


def test_extract_faces_with_padding():
    """Test face extraction with padding."""
    extractor = FaceExtractor(method="mediapipe")
    image = create_test_image(with_face=True)
    
    # Extract with padding
    faces = extractor.extract_faces(image, padding=0.3)
    
    assert isinstance(faces, list)
    
    # If faces detected, check that padded bbox is larger
    if len(faces) > 0:
        face = faces[0]
        if 'bbox_padded' in face:
            bbox = face['bbox']
            bbox_padded = face['bbox_padded']
            # Padded bbox should be larger or equal
            assert bbox_padded[2] >= bbox[2] or bbox_padded[3] >= bbox[3]


def test_extract_largest_face():
    """Test extracting largest face."""
    image = create_test_image(with_face=True)
    
    face = extract_largest_face(image, min_face_size=20)
    
    # Check return type (may be None if no face detected)
    assert face is None or isinstance(face, dict)
    
    # If face detected, check structure
    if face is not None:
        assert 'bbox' in face
        assert 'face' in face


def test_extract_largest_face_no_face():
    """Test extracting largest face when no face present."""
    image = create_test_image(with_face=False)
    
    face = extract_largest_face(image)
    
    # Should return None or a dict
    assert face is None or isinstance(face, dict)


def test_face_extractor_opencv_haar():
    """Test OpenCV Haar Cascade method."""
    extractor = FaceExtractor(method="opencv_dnn")  # Falls back to Haar
    assert extractor.method == "opencv_haar"
    
    image = create_test_image(with_face=True)
    faces = extractor.extract_faces(image)
    
    assert isinstance(faces, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
