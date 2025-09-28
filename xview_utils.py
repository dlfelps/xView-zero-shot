#!/usr/bin/env python3
"""
xView Dataset Utilities

This module provides utilities for processing the xView dataset, including:
- Extracting scenes containing specific object classes
- Organizing images by object categories
- Creating template and scene pairs for template matching
- Converting between coordinate systems and formats

The xView dataset contains 60 object classes in satellite imagery with
GeoJSON annotations providing bounding box coordinates.
"""

import json
import os
import shutil
import argparse
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict, Counter
import numpy as np
from PIL import Image
from tqdm import tqdm


class XViewProcessor:
    """Process xView dataset for object detection and template matching."""

    # xView dataset class mapping (class_id: class_name)
    XVIEW_CLASSES = {
        11: "Fixed-wing Aircraft", 12: "Small Aircraft", 13: "Cargo Plane",
        15: "Helicopter", 17: "Passenger Vehicle", 18: "Small Car", 19: "Bus",
        20: "Pickup Truck", 21: "Utility Truck", 23: "Truck", 24: "Cargo Truck",
        25: "Truck w/Box", 26: "Truck w/Flatbed", 27: "Truck w/Liquid",
        28: "Crane Truck", 29: "Railway Vehicle", 32: "Passenger Car",
        33: "Cargo Car", 34: "Flat Car", 35: "Tank car", 36: "Locomotive",
        37: "Maritime Vessel", 38: "Motorboat", 40: "Sailboat", 41: "Tugboat",
        42: "Barge", 44: "Fishing Vessel", 45: "Ferry", 47: "Yacht",
        49: "Container Ship", 50: "Oil Tanker", 51: "Engineering Vehicle",
        52: "Tower crane", 53: "Container Crane", 54: "Reach Stacker",
        55: "Straddle Carrier", 56: "Mobile Crane", 57: "Dump Truck",
        59: "Haul Truck", 60: "Scraper/Tractor", 61: "Front loader/Bulldozer",
        62: "Excavator", 63: "Cement Mixer", 64: "Ground Grader",
        65: "Hut/Tent", 66: "Shed", 71: "Building", 72: "Aircraft Hangar",
        73: "Damaged Building", 74: "Facility", 76: "Construction Site",
        77: "Vehicle Lot", 79: "Helipad", 83: "Storage Tank",
        84: "Shipping Container Lot", 86: "Shipping Container", 89: "Pylon",
        91: "Tower"
    }

    def __init__(self, data_dir: str = "data"):
        """Initialize processor with data directory."""
        self.data_dir = Path(data_dir)
        self.labels_file = self.data_dir / "xView_train.geojson"
        self.train_images_dir = self.data_dir / "train_images"
        self.output_dir = self.data_dir / "extracted_scenes"

        # Load labels if available
        self.labels_data = None
        if self.labels_file.exists():
            self.load_labels()

    def load_labels(self) -> None:
        """Load GeoJSON labels file."""
        print(f"Loading labels from {self.labels_file}")
        with open(self.labels_file, 'r') as f:
            self.labels_data = json.load(f)
        print(f"Loaded {len(self.labels_data['features'])} annotations")

    def get_class_statistics(self) -> Dict[int, int]:
        """Get count of objects per class."""
        if not self.labels_data:
            raise ValueError("Labels not loaded. Call load_labels() first.")

        class_counts = Counter()
        for feature in self.labels_data['features']:
            class_id = feature['properties']['type_id']
            class_counts[class_id] += 1

        return dict(class_counts)

    def list_available_classes(self) -> None:
        """Print available object classes and their counts."""
        class_counts = self.get_class_statistics()

        print("\nAvailable Object Classes:")
        print("-" * 60)
        print(f"{'ID':<4} {'Class Name':<25} {'Count':<8} {'%':<6}")
        print("-" * 60)

        total_objects = sum(class_counts.values())
        for class_id, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            class_name = self.XVIEW_CLASSES.get(class_id, f"Unknown_{class_id}")
            percentage = (count / total_objects) * 100
            print(f"{class_id:<4} {class_name:<25} {count:<8} {percentage:<6.1f}")

        print("-" * 60)
        print(f"Total: {len(class_counts)} classes, {total_objects} objects")

    def get_images_with_classes(self, target_classes: List[int]) -> Dict[str, List[Dict]]:
        """
        Find images containing specific object classes.

        Args:
            target_classes: List of class IDs to search for

        Returns:
            Dictionary mapping image_ids to list of object annotations
        """
        if not self.labels_data:
            raise ValueError("Labels not loaded. Call load_labels() first.")

        images_with_objects = defaultdict(list)

        for feature in self.labels_data['features']:
            class_id = feature['properties']['type_id']
            image_id = feature['properties']['image_id']

            if class_id in target_classes:
                # Extract bounding box coordinates
                coords = feature['geometry']['coordinates'][0]
                x_coords = [c[0] for c in coords]
                y_coords = [c[1] for c in coords]

                bbox = {
                    'class_id': class_id,
                    'class_name': self.XVIEW_CLASSES.get(class_id, f"Unknown_{class_id}"),
                    'bbox': [min(x_coords), min(y_coords), max(x_coords), max(y_coords)],
                    'area': (max(x_coords) - min(x_coords)) * (max(y_coords) - min(y_coords))
                }
                images_with_objects[image_id].append(bbox)

        return dict(images_with_objects)

    def extract_scenes_by_class(self, target_classes: List[int],
                               output_subdir: str = None,
                               min_objects: int = 1,
                               max_images: int = None) -> None:
        """
        Extract scenes containing specific object classes.

        Args:
            target_classes: List of class IDs to extract
            output_subdir: Subdirectory name (auto-generated if None)
            min_objects: Minimum number of target objects required
            max_images: Maximum number of images to extract
        """
        # Find images with target classes
        images_with_objects = self.get_images_with_classes(target_classes)

        # Filter by minimum object count
        filtered_images = {
            img_id: objects for img_id, objects in images_with_objects.items()
            if len(objects) >= min_objects
        }

        if not filtered_images:
            print(f"No images found with {min_objects}+ objects from classes {target_classes}")
            return

        # Limit number of images if specified
        if max_images:
            filtered_images = dict(list(filtered_images.items())[:max_images])

        # Create output directory
        if output_subdir is None:
            class_names = [self.XVIEW_CLASSES.get(c, f"class_{c}") for c in target_classes]
            output_subdir = "_".join(class_names[:3]).replace(" ", "_").replace("/", "_")

        output_path = self.output_dir / output_subdir
        output_path.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (output_path / "images").mkdir(exist_ok=True)
        (output_path / "templates").mkdir(exist_ok=True)
        (output_path / "annotations").mkdir(exist_ok=True)

        print(f"Extracting {len(filtered_images)} scenes to {output_path}")

        # Process each image
        extracted_count = 0
        template_count = 0

        for image_id, objects in tqdm(filtered_images.items(), desc="Extracting scenes"):
            # Copy original image
            source_path = self.train_images_dir / f"{image_id}.tif"
            if not source_path.exists():
                print(f"Warning: Image {image_id}.tif not found")
                continue

            # Convert to common format and copy
            try:
                img = Image.open(source_path)
                target_image_path = output_path / "images" / f"{image_id}.jpg"
                img.convert('RGB').save(target_image_path, 'JPEG', quality=95)
                extracted_count += 1
            except Exception as e:
                print(f"Error processing {image_id}: {e}")
                continue

            # Extract templates for each object
            for i, obj in enumerate(objects):
                try:
                    bbox = obj['bbox']
                    # Add padding around object
                    padding = 20
                    x1 = max(0, int(bbox[0] - padding))
                    y1 = max(0, int(bbox[1] - padding))
                    x2 = min(img.width, int(bbox[2] + padding))
                    y2 = min(img.height, int(bbox[3] + padding))

                    # Extract template
                    template = img.crop((x1, y1, x2, y2))
                    template_name = f"{image_id}_obj_{i}_{obj['class_name'].replace(' ', '_')}.jpg"
                    template_path = output_path / "templates" / template_name

                    template.convert('RGB').save(template_path, 'JPEG', quality=95)
                    template_count += 1

                except Exception as e:
                    print(f"Error extracting template from {image_id}: {e}")

            # Save annotation info
            annotation_data = {
                'image_id': image_id,
                'objects': objects,
                'image_path': f"images/{image_id}.jpg",
                'template_paths': [
                    f"templates/{image_id}_obj_{i}_{obj['class_name'].replace(' ', '_')}.jpg"
                    for i, obj in enumerate(objects)
                ]
            }

            with open(output_path / "annotations" / f"{image_id}.json", 'w') as f:
                json.dump(annotation_data, f, indent=2)

        # Create summary
        summary = {
            'target_classes': target_classes,
            'class_names': [self.XVIEW_CLASSES.get(c, f"Unknown_{c}") for c in target_classes],
            'total_images': extracted_count,
            'total_templates': template_count,
            'min_objects_per_image': min_objects
        }

        with open(output_path / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"✓ Extracted {extracted_count} images and {template_count} templates")
        print(f"✓ Output saved to: {output_path}")

    def create_template_matching_pairs(self, extracted_dir: str,
                                     pairs_per_class: int = 5) -> None:
        """
        Create template-scene pairs for testing template matching.

        Args:
            extracted_dir: Directory containing extracted scenes
            pairs_per_class: Number of template-scene pairs to create per class
        """
        extracted_path = self.output_dir / extracted_dir
        pairs_dir = extracted_path / "template_pairs"
        pairs_dir.mkdir(exist_ok=True)

        # Load summary to understand the dataset
        with open(extracted_path / "summary.json", 'r') as f:
            summary = json.load(f)

        print(f"Creating template matching pairs for {summary['class_names']}")

        # Get all annotation files
        annotation_files = list((extracted_path / "annotations").glob("*.json"))

        pair_count = 0
        for i, ann_file in enumerate(annotation_files[:pairs_per_class]):
            with open(ann_file, 'r') as f:
                annotation = json.load(f)

            image_path = extracted_path / annotation['image_path']
            template_paths = [extracted_path / tp for tp in annotation['template_paths']]

            # Create pairs with first template and scene
            if template_paths:
                template_path = template_paths[0]
                if template_path.exists() and image_path.exists():
                    # Copy to pairs directory
                    scene_dest = pairs_dir / f"pair_{i:03d}_scene.jpg"
                    template_dest = pairs_dir / f"pair_{i:03d}_template.jpg"

                    shutil.copy2(image_path, scene_dest)
                    shutil.copy2(template_path, template_dest)

                    pair_count += 1

        print(f"✓ Created {pair_count} template-scene pairs in {pairs_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="xView Dataset Processing Utilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available object classes
  python xview_utils.py --list-classes

  # Extract scenes with small cars
  python xview_utils.py --extract --classes 18 --min-objects 2

  # Extract scenes with vehicles (multiple classes)
  python xview_utils.py --extract --classes 17 18 19 20 --output vehicles

  # Create template matching pairs
  python xview_utils.py --create-pairs --input vehicles --pairs 10
        """
    )

    parser.add_argument("--data-dir", default="data",
                       help="Data directory containing xView dataset")
    parser.add_argument("--list-classes", action="store_true",
                       help="List available object classes and counts")
    parser.add_argument("--extract", action="store_true",
                       help="Extract scenes by object class")
    parser.add_argument("--classes", type=int, nargs='+',
                       help="Object class IDs to extract")
    parser.add_argument("--min-objects", type=int, default=1,
                       help="Minimum objects per scene (default: 1)")
    parser.add_argument("--max-images", type=int,
                       help="Maximum number of images to extract")
    parser.add_argument("--output", type=str,
                       help="Output subdirectory name")
    parser.add_argument("--create-pairs", action="store_true",
                       help="Create template matching pairs")
    parser.add_argument("--input", type=str,
                       help="Input directory for creating pairs")
    parser.add_argument("--pairs", type=int, default=5,
                       help="Number of pairs to create (default: 5)")

    args = parser.parse_args()

    # Initialize processor
    processor = XViewProcessor(args.data_dir)

    if args.list_classes:
        if not processor.labels_data:
            print("Error: Labels file not found. Run download_xview.py first.")
            return
        processor.list_available_classes()

    elif args.extract:
        if not args.classes:
            print("Error: --classes required for extraction")
            return
        if not processor.labels_data:
            print("Error: Labels file not found. Run download_xview.py first.")
            return

        processor.extract_scenes_by_class(
            target_classes=args.classes,
            output_subdir=args.output,
            min_objects=args.min_objects,
            max_images=args.max_images
        )

    elif args.create_pairs:
        if not args.input:
            print("Error: --input directory required for creating pairs")
            return

        processor.create_template_matching_pairs(
            extracted_dir=args.input,
            pairs_per_class=args.pairs
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()