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

## Quick Start

### Basic Usage (No Dataset Required)
```bash
# Install dependencies
uv sync

# Run with your own images
uv run python main.py --scene your_scene.jpg --template your_template.jpg
```

## Full Dataset Setup

### 1. Download the xView Dataset

**Manual Download (Required):**
1. Visit [challenge.xviewdataset.org](https://challenge.xviewdataset.org)
2. Register/login to access the dataset
3. Download these files:
   - `train_images.tgz` (5.0 GB) - Training satellite images
   - `val_images.tgz` (1.3 GB) - Validation images
   - `xView_train.geojson` (30 MB) - Training labels

### 2. Setup Dataset
```bash
# Create data directory
mkdir data

# Extract downloaded files
tar -xzf train_images.tgz -C data/
tar -xzf val_images.tgz -C data/
mv xView_train.geojson data/

# Verify structure
ls data/
# Should show: train_images/ val_images/ xView_train.geojson
```

### 3. Process xView Dataset (Optional)

Extract scenes containing specific object classes for targeted experiments:

```bash
# List available object classes and their counts
uv run python xview_utils.py --list-classes

# Extract scenes containing small cars (class 18)
uv run python xview_utils.py --extract --classes 18 --min-objects 2 --output small_cars

# Extract scenes with multiple vehicle types
uv run python xview_utils.py --extract --classes 17 18 19 20 --output vehicles --max-images 50

# Create template-scene pairs for testing
uv run python xview_utils.py --create-pairs --input vehicles --pairs 10
```

### 4. Run Object Detection

```bash
# Basic usage
uv run python main.py --scene path/to/scene.jpg --template path/to/template.jpg

# With custom parameters
uv run python main.py --scene scene.jpg --template target.jpg \
    --threshold 0.8 --stride 16 --output results.jpg

# Using extracted xView scenes
uv run python main.py --scene data/extracted_scenes/vehicles/images/scene.jpg \
    --template data/extracted_scenes/vehicles/templates/template.jpg

# Using template pairs
uv run python main.py --scene data/extracted_scenes/vehicles/template_pairs/pair_001_scene.jpg \
    --template data/extracted_scenes/vehicles/template_pairs/pair_001_template.jpg
```

## Parameters

- `--scene`: Path to the full xView scene image
- `--template`: Path to the target object template
- `--threshold`: Similarity threshold for detection (default: 0.7)
- `--window-size`: Sliding window dimensions (default: template size)
- `--stride`: Step size for sliding window (default: 32)
- `--nms-threshold`: IoU threshold for non-maximum suppression (default: 0.5)
- `--output`: Output visualization path (default: detections.jpg)

## Dataset Organization

The xView utilities help organize the dataset for efficient experimentation:

### Object Classes
The dataset contains 60 object classes including:
- **Vehicles**: Small Car (18), Pickup Truck (20), Bus (19), Cargo Truck (24)
- **Aircraft**: Fixed-wing Aircraft (11), Helicopter (15), Cargo Plane (13)
- **Maritime**: Motorboat (38), Sailboat (40), Container Ship (49)
- **Buildings**: Building (71), Aircraft Hangar (72), Storage Tank (83)
- **Infrastructure**: Crane (52, 56), Construction Site (76), Helipad (79)

### Extracted Dataset Structure
```
data/extracted_scenes/
├── vehicles/
│   ├── images/           # Full scene images
│   ├── templates/        # Cropped object templates
│   ├── annotations/      # Object metadata
│   ├── template_pairs/   # Ready-to-use test pairs
│   └── summary.json      # Extraction summary
```

### Utility Commands
- `--list-classes`: View all 60 object classes with counts
- `--extract`: Extract scenes by class ID with filtering options
- `--create-pairs`: Generate template-scene pairs for evaluation