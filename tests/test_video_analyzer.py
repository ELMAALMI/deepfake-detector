"""Unit tests for video analyzer."""

import os
import tempfile

import cv2
import numpy as np
import pytest

# Note: These are integration tests that require a trained model
# They test the structure and error handling rather than actual detection


def create_test_video(output_path: str, num_frames: int = 30, fps: int = 10) -> None:
    """Create a test video file.
    
    Args:
        output_path: Path to save video
        num_frames: Number of frames
        fps: Frames per second
    """
    width, height = 320, 240
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for i in range(num_frames):
        # Create a frame with changing colors
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = (i * 8 % 256, (i * 4) % 256, (i * 2) % 256)
        
        # Add a simple face-like pattern
        center_x, center_y = width // 2, height // 2
        cv2.ellipse(frame, (center_x, center_y), (40, 50), 0, 0, 360, (200, 180, 160), -1)
        cv2.circle(frame, (center_x - 15, center_y - 10), 5, (50, 50, 50), -1)
        cv2.circle(frame, (center_x + 15, center_y - 10), 5, (50, 50, 50), -1)
        
        writer.write(frame)
    
    writer.release()


def test_video_creation():
    """Test video creation utility."""
    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, "test_video.mp4")
        create_test_video(video_path, num_frames=10)
        
        assert os.path.exists(video_path)
        
        # Verify video can be opened
        cap = cv2.VideoCapture(video_path)
        assert cap.isOpened()
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        assert frame_count == 10
        
        cap.release()


def test_video_analyzer_import():
    """Test that video analyzer can be imported."""
    from src.video_analyzer import VideoAnalyzer
    assert VideoAnalyzer is not None


def test_realtime_detector_import():
    """Test that realtime detector can be imported."""
    from src.realtime_detector import RealtimeDetector
    assert RealtimeDetector is not None


# Note: Full integration tests would require:
# 1. A trained model
# 2. Test videos with actual faces
# 3. Longer running time
# These should be run separately as integration tests


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
