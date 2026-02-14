"""Setup configuration for deepfake-detector package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="deepfake-detector",
    version="1.0.0",
    author="ELMAALMI",
    description="AI-powered deepfake detection tool using OpenCV and Deep Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ELMAALMI/deepfake-detector",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "deepfake-train=scripts.train_model:main",
            "deepfake-detect-video=scripts.detect_video:main",
            "deepfake-detect-realtime=scripts.detect_realtime:main",
            "deepfake-evaluate=scripts.evaluate_model:main",
        ],
    },
)
