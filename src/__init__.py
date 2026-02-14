"""Deepfake Detector - AI-powered deepfake detection tool."""

__version__ = "1.0.0"
__author__ = "ELMAALMI"
__license__ = "MIT"

from src import (
    face_extractor,
    preprocessing,
    model,
    train,
    predict,
    video_analyzer,
    realtime_detector,
    visualization,
    utils,
)

__all__ = [
    "face_extractor",
    "preprocessing",
    "model",
    "train",
    "predict",
    "video_analyzer",
    "realtime_detector",
    "visualization",
    "utils",
]
