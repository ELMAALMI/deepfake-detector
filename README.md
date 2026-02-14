# Deepfake Detector 🔍

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-red.svg)](https://opencv.org/)

AI-powered deepfake detection tool that analyzes videos and images to determine whether they are real or AI-generated/manipulated. Built with OpenCV, MediaPipe, and TensorFlow, featuring XceptionNet-based deep learning architecture.

## ✨ Features

- **🎯 High-Accuracy Detection**: XceptionNet-based CNN with transfer learning from ImageNet
- **👤 Face Detection**: Multi-face detection using MediaPipe and OpenCV DNN
- **🎥 Video Analysis**: Frame-by-frame video processing with aggregated predictions
- **📹 Real-time Detection**: Live webcam-based deepfake detection
- **🔥 Grad-CAM Visualization**: Explainable AI with heatmap overlays showing detection reasoning
- **📊 Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score, AUC-ROC
- **🔧 Configurable Pipeline**: YAML-based configuration for all parameters
- **📈 Training Pipeline**: Complete training infrastructure with callbacks and monitoring

## 🏗️ Architecture

The deepfake detection pipeline consists of:

1. **Face Extraction**: MediaPipe/OpenCV detect and extract faces from images/video frames
2. **Preprocessing**: Resize to 299x299, normalize, and apply data augmentation (training only)
3. **Model**: XceptionNet base + custom classification head (GlobalAvgPool → Dense(512) → Dropout(0.5) → Dense(1, sigmoid))
4. **Prediction**: Binary classification (Real=0, Fake=1) with confidence scores
5. **Aggregation**: Video-level predictions via weighted average, majority vote, or max confidence

```
Input Image/Video
    ↓
Face Detection (MediaPipe/OpenCV)
    ↓
Face Extraction & Preprocessing
    ↓
XceptionNet Feature Extraction
    ↓
Classification Head
    ↓
Prediction (REAL/FAKE + Confidence)
```

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for training

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/ELMAALMI/deepfake-detector.git
cd deepfake-detector
```

2. **Create virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Install package** (optional, for command-line tools):
```bash
pip install -e .
```

## 🚀 Quick Start

### 1. Train a Model

Prepare your dataset in the following structure:
```
data/
├── real/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
└── fake/
    ├── img1.jpg
    ├── img2.jpg
    └── ...
```

Train the model:
```bash
python scripts/train_model.py \
    --data-dir ./data \
    --output-dir ./models \
    --epochs 50 \
    --batch-size 32
```

### 2. Detect Deepfakes in Video

```bash
python scripts/detect_video.py \
    --video ./test_video.mp4 \
    --model ./models/best_model.h5 \
    --output-video ./output_annotated.mp4 \
    --output-report ./report.txt
```

### 3. Real-time Webcam Detection

```bash
python scripts/detect_realtime.py \
    --model ./models/best_model.h5 \
    --camera-id 0
```

Press 'q' to quit real-time detection.

### 4. Evaluate Model

```bash
python scripts/evaluate_model.py \
    --model ./models/best_model.h5 \
    --test-dir ./data/test \
    --output-dir ./outputs/evaluation
```

## 📖 Usage Examples

### Python API

#### Single Image Prediction

```python
from src.predict import DeepfakePredictor

# Create predictor
predictor = DeepfakePredictor(
    model_path="models/best_model.h5",
    config_path="config/config.yaml"
)

# Predict
result = predictor.predict_image("test_image.jpg")

print(f"Prediction: {result['label']}")
print(f"Confidence: {result['confidence']:.2f}%")
```

#### Video Analysis

```python
from src.predict import DeepfakePredictor
from src.video_analyzer import VideoAnalyzer

# Create predictor and analyzer
predictor = DeepfakePredictor(model_path="models/best_model.h5")
analyzer = VideoAnalyzer(predictor, frame_skip=5)

# Analyze video
result = analyzer.analyze_video(
    video_path="test_video.mp4",
    output_path="output_annotated.mp4"
)

print(f"Video Prediction: {result['video_prediction']}")
print(f"Confidence: {result['video_confidence']:.2f}%")
```

#### Training

```python
from src.train import train_model

history = train_model(
    data_dir="./data",
    config_path="config/config.yaml",
    output_dir="./models"
)
```

## 📁 Project Structure

```
deepfake-detector/
├── README.md                     # This file
├── LICENSE                       # MIT License
├── requirements.txt              # Python dependencies
├── setup.py                      # Package setup
├── config/
│   └── config.yaml              # Configuration file
├── src/
│   ├── __init__.py
│   ├── face_extractor.py        # Face detection & extraction
│   ├── preprocessing.py         # Image preprocessing & augmentation
│   ├── model.py                 # XceptionNet-based model architecture
│   ├── train.py                 # Training pipeline
│   ├── predict.py               # Inference/prediction
│   ├── video_analyzer.py        # Video analysis pipeline
│   ├── realtime_detector.py    # Real-time webcam detection
│   ├── visualization.py         # Grad-CAM & visualization
│   └── utils.py                 # Utility functions
├── scripts/
│   ├── train_model.py           # CLI: Train model
│   ├── detect_video.py          # CLI: Analyze video
│   ├── detect_realtime.py       # CLI: Real-time detection
│   └── evaluate_model.py        # CLI: Evaluate model
├── tests/
│   ├── test_face_extractor.py
│   ├── test_preprocessing.py
│   ├── test_model.py
│   ├── test_video_analyzer.py
│   └── test_utils.py
└── notebooks/
    └── exploration.ipynb         # Jupyter notebook demos
```

## ⚙️ Configuration

Edit `config/config.yaml` to customize:

- **Model parameters**: Architecture, input size, dropout rate
- **Training parameters**: Epochs, batch size, learning rate
- **Detection parameters**: Confidence threshold, frame skip
- **Face detection**: Method (MediaPipe/OpenCV), confidence threshold
- **Preprocessing**: Normalization method, augmentation settings

Example configuration:
```yaml
model:
  architecture: "xception"
  input_size: [299, 299, 3]
  dropout_rate: 0.5

training:
  epochs: 50
  batch_size: 32
  learning_rate: 0.0001

detection:
  confidence_threshold: 0.5
  frame_skip: 5
```

## 📊 Datasets

Recommended datasets for training:

1. **FaceForensics++**: Large-scale dataset with multiple manipulation methods
   - [GitHub](https://github.com/ondyari/FaceForensics)
   
2. **Deepfake Detection Challenge (DFDC)**: Facebook/AWS competition dataset
   - [Kaggle](https://www.kaggle.com/c/deepfake-detection-challenge)

3. **Celeb-DF**: High-quality celebrity deepfake dataset
   - [Website](https://github.com/yuezunli/celeb-deepfakeforensics)

## 🧪 Testing

Run unit tests:
```bash
pytest tests/ -v
```

Run specific test file:
```bash
pytest tests/test_model.py -v
```

## 📈 Model Performance

Expected performance on standard datasets (after proper training):

| Metric | Score |
|--------|-------|
| Accuracy | 90-95% |
| Precision | 88-93% |
| Recall | 89-94% |
| F1-Score | 89-93% |
| AUC-ROC | 0.92-0.97 |

*Note: Actual performance depends on dataset quality, training duration, and hyperparameters.*

## 🔬 Advanced Features

### Grad-CAM Visualization

Generate heatmaps showing which facial regions triggered the detection:

```python
from src.visualization import visualize_grad_cam
from tensorflow import keras

model = keras.models.load_model("models/best_model.h5")
heatmap, overlayed = visualize_grad_cam(
    model, image, preprocessed_image, prediction,
    save_path="gradcam_result.png"
)
```

### Custom Model Training

Fine-tune hyperparameters:

```bash
python scripts/train_model.py \
    --data-dir ./data \
    --epochs 100 \
    --batch-size 16 \
    --learning-rate 0.00001 \
    --output-dir ./models/custom
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **XceptionNet**: Chollet, François. "Xception: Deep learning with depthwise separable convolutions."
- **MediaPipe**: Google's ML solution for face detection
- **FaceForensics++**: Rössler et al., 2019
- **OpenCV**: Open Source Computer Vision Library

## 📚 References

1. Rössler, A., Cozzolino, D., Verdoliva, L., Riess, C., Thies, J., & Nießner, M. (2019). FaceForensics++: Learning to detect manipulated facial images.
2. Chollet, F. (2017). Xception: Deep learning with depthwise separable convolutions.
3. Selvaraju, R. R., et al. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization.

## 📧 Contact

Project Link: [https://github.com/ELMAALMI/deepfake-detector](https://github.com/ELMAALMI/deepfake-detector)

## ⚠️ Disclaimer

This tool is for research and educational purposes. Always verify important content through multiple sources and official channels. Deepfake detection technology is an ongoing research area and no system is 100% accurate.

---

**Made with ❤️ for a safer digital world**