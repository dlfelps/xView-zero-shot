# DINOv3 Template Matching for xView Object Detection

This project implements a DINOv3-based template matching architecture for image-conditioned one-shot object detection in satellite imagery scenes, specifically targeting the xView dataset.

## Overview

The system performs object detection by comparing a target template image against a full satellite scene using deep visual features extracted by the DINOv3 model. This approach enables detection of specific objects in large satellite images without requiring extensive training data for each object class.

## Template Matching Approach

### Core Methodology
1. **Feature Extraction**: Uses DINOv3 (Self-Supervised Vision Transformer) to extract rich visual representations from both template and scene images
2. **Sliding Window Search**: Systematically scans the scene image using overlapping windows of configurable size
3. **Similarity Matching**: Computes cosine similarity between template features and each scene window
4. **Detection Filtering**: Applies threshold-based filtering and non-maximum suppression to identify final detections

### Key Advantages
- **One-shot Detection**: Requires only a single template image to detect similar objects
- **Self-supervised Features**: DINOv3 provides robust features without task-specific training
- **Scale Flexibility**: Configurable window sizes accommodate objects of different scales
- **Semantic Understanding**: Deep features capture semantic similarity beyond pixel-level matching

## Background

### DINOv3 (Self-Supervised Vision Transformer)
DINOv3 is a state-of-the-art self-supervised learning method that trains Vision Transformers without labeled data. Key characteristics:

- **Architecture**: Vision Transformer (ViT) trained with self-distillation
- **Training**: Self-supervised on large-scale image datasets
- **Features**: Produces semantically meaningful representations that capture object structure and context
- **Robustness**: Excellent performance across diverse visual tasks without fine-tuning
- **Model**: This implementation uses `facebook/dinov2-base` from Hugging Face

### xView Dataset
xView is a large-scale overhead imagery dataset designed for object detection in satellite images:

- **Scale**: Over 1 million object instances across 1,400+ square kilometers
- **Coverage**: Satellite imagery from around the world
- **Objects**: 60+ object classes including vehicles, buildings, infrastructure
- **Resolution**: 30cm ground sample distance
- **Challenges**: Small objects, dense scenes, varied lighting and weather conditions
- **Use Cases**: Disaster response, urban planning, infrastructure monitoring

The sparse nature of objects in satellite imagery makes template matching particularly suitable, as objects of interest are often isolated and well-defined against backgrounds.

## Usage

```bash
# Install dependencies
uv sync

# Run detection
uv run python main.py --scene path/to/scene.jpg --template path/to/template.jpg

# With custom parameters
uv run python main.py --scene scene.jpg --template target.jpg \
    --threshold 0.8 --stride 16 --output results.jpg
```

## Parameters

- `--scene`: Path to the full xView scene image
- `--template`: Path to the target object template
- `--threshold`: Similarity threshold for detection (default: 0.7)
- `--window-size`: Sliding window dimensions (default: template size)
- `--stride`: Step size for sliding window (default: 32)
- `--nms-threshold`: IoU threshold for non-maximum suppression (default: 0.5)
- `--output`: Output visualization path (default: detections.jpg)