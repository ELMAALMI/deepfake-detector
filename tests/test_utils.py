"""Unit tests for utility functions."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.utils import (
    setup_logging,
    load_config,
    save_config,
    calculate_metrics,
    get_confusion_matrix,
    detect_device,
    ensure_dir,
    get_file_paths,
    format_time,
)


def test_setup_logging():
    """Test logging setup."""
    logger = setup_logging(level="INFO")
    assert logger is not None
    assert logger.name == "deepfake_detector"


def test_load_save_config():
    """Test configuration loading and saving."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "test_config.yaml")
        
        # Test config
        test_config = {
            "model": {"input_size": [299, 299, 3]},
            "training": {"epochs": 10},
        }
        
        # Save config
        save_config(test_config, config_path)
        assert os.path.exists(config_path)
        
        # Load config
        loaded_config = load_config(config_path)
        assert loaded_config == test_config


def test_load_config_not_found():
    """Test loading non-existent config file."""
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent_config.yaml")


def test_calculate_metrics():
    """Test metrics calculation."""
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    y_pred_proba = np.array([0.1, 0.2, 0.8, 0.9, 0.4, 0.3, 0.7, 0.6])
    
    metrics = calculate_metrics(y_true, y_pred, y_pred_proba)
    
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert "auc_roc" in metrics
    
    # Check metric values are in valid range
    for key, value in metrics.items():
        assert 0 <= value <= 1, f"{key} should be between 0 and 1"


def test_calculate_metrics_without_proba():
    """Test metrics calculation without probabilities."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    
    metrics = calculate_metrics(y_true, y_pred)
    
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert "auc_roc" not in metrics


def test_get_confusion_matrix():
    """Test confusion matrix calculation."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    
    cm = get_confusion_matrix(y_true, y_pred)
    
    assert cm.shape == (2, 2)
    assert cm[0, 0] == 2  # True negatives
    assert cm[1, 1] == 2  # True positives


def test_detect_device():
    """Test device detection."""
    device = detect_device()
    assert device in ["GPU", "CPU"]


def test_ensure_dir():
    """Test directory creation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = os.path.join(tmpdir, "test", "nested", "dir")
        ensure_dir(test_dir)
        assert os.path.exists(test_dir)
        assert os.path.isdir(test_dir)


def test_get_file_paths():
    """Test file path retrieval."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        Path(tmpdir, "test1.jpg").touch()
        Path(tmpdir, "test2.png").touch()
        Path(tmpdir, "test3.txt").touch()
        
        # Get image files
        image_files = get_file_paths(tmpdir, extensions=[".jpg", ".png"])
        
        assert len(image_files) == 2
        assert any("test1.jpg" in f for f in image_files)
        assert any("test2.png" in f for f in image_files)


def test_get_file_paths_recursive():
    """Test recursive file path retrieval."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create nested structure
        subdir = Path(tmpdir, "subdir")
        subdir.mkdir()
        Path(tmpdir, "test1.jpg").touch()
        Path(subdir, "test2.jpg").touch()
        
        # Get files recursively
        files = get_file_paths(tmpdir, extensions=[".jpg"], recursive=True)
        assert len(files) == 2


def test_format_time():
    """Test time formatting."""
    assert format_time(45) == "45s"
    assert format_time(90) == "1m 30s"
    assert format_time(3665) == "1h 1m 5s"
    assert format_time(7200) == "2h 0s"  # 0 minutes are skipped


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
