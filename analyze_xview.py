#!/usr/bin/env python3
"""
xView Dataset Analysis

This module analyzes the xView GeoJSON annotations to provide insights into
the dataset composition, focusing on unique image counts per class rather
than object occurrence frequency.

Key difference: This counts how many unique images contain each object class,
which is more relevant for scene extraction than total object counts.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Set, List, Tuple
from collections import defaultdict, Counter
import numpy as np


class XViewAnalyzer:
    """Analyze xView dataset composition and statistics."""

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

    # Logical groupings for analysis
    CLASS_GROUPS = {
        "Vehicles": [17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28],
        "Aircraft": [11, 12, 13, 15],
        "Maritime": [37, 38, 40, 41, 42, 44, 45, 47, 49, 50],
        "Railway": [29, 32, 33, 34, 35, 36],
        "Heavy Equipment": [51, 52, 53, 54, 55, 56, 57, 59, 60, 61, 62, 63, 64],
        "Buildings": [65, 66, 71, 72, 73, 74, 76, 83, 84],
        "Infrastructure": [77, 79, 86, 89, 91]
    }

    def __init__(self, labels_file: str = "data/xView_train.geojson"):
        """Initialize analyzer with labels file."""
        self.labels_file = Path(labels_file)
        self.labels_data = None
        self.load_labels()

    def load_labels(self) -> None:
        """Load GeoJSON labels file."""
        if not self.labels_file.exists():
            raise FileNotFoundError(f"Labels file not found: {self.labels_file}")

        print(f"Loading labels from {self.labels_file}")
        with open(self.labels_file, 'r') as f:
            self.labels_data = json.load(f)
        print(f"Loaded {len(self.labels_data['features'])} annotations")

    def count_unique_images_per_class(self) -> Dict[int, Set[str]]:
        """
        Count unique images containing each object class.

        Returns:
            Dictionary mapping class_id to set of image_ids containing that class
        """
        class_to_images = defaultdict(set)

        for feature in self.labels_data['features']:
            class_id = feature['properties']['type_id']
            image_id = feature['properties']['image_id']
            class_to_images[class_id].add(image_id)

        return dict(class_to_images)

    def count_objects_per_class(self) -> Dict[int, int]:
        """
        Count total object occurrences per class (traditional frequency analysis).

        Returns:
            Dictionary mapping class_id to total object count
        """
        class_counts = Counter()
        for feature in self.labels_data['features']:
            class_id = feature['properties']['type_id']
            class_counts[class_id] += 1

        return dict(class_counts)

    def analyze_objects_per_image(self) -> Dict[str, Dict[int, int]]:
        """
        Analyze how many objects of each class appear in each image.

        Returns:
            Dictionary mapping image_id to class_id to count
        """
        image_class_counts = defaultdict(lambda: defaultdict(int))

        for feature in self.labels_data['features']:
            class_id = feature['properties']['type_id']
            image_id = feature['properties']['image_id']
            image_class_counts[image_id][class_id] += 1

        return dict(image_class_counts)

    def get_extraction_statistics(self, min_objects: int = 1) -> Dict[int, Dict]:
        """
        Get statistics relevant for scene extraction.

        Args:
            min_objects: Minimum objects per image to be considered for extraction

        Returns:
            Dictionary with extraction statistics per class
        """
        class_to_images = self.count_unique_images_per_class()
        image_class_counts = self.analyze_objects_per_image()
        object_counts = self.count_objects_per_class()

        extraction_stats = {}

        for class_id, image_set in class_to_images.items():
            # Count images with at least min_objects of this class
            extractable_images = []
            for image_id in image_set:
                if image_class_counts[image_id][class_id] >= min_objects:
                    extractable_images.append(image_id)

            # Calculate statistics
            total_images = len(image_set)
            extractable_count = len(extractable_images)
            total_objects = object_counts[class_id]
            avg_objects_per_image = total_objects / total_images if total_images > 0 else 0

            extraction_stats[class_id] = {
                'class_name': self.XVIEW_CLASSES.get(class_id, f"Unknown_{class_id}"),
                'total_images': total_images,
                'extractable_images': extractable_count,
                'extraction_rate': extractable_count / total_images if total_images > 0 else 0,
                'total_objects': total_objects,
                'avg_objects_per_image': avg_objects_per_image,
                'extractable_image_ids': extractable_images[:10]  # Sample for reference
            }

        return extraction_stats

    def print_unique_image_analysis(self, sort_by: str = "unique_images", top_n: int = None) -> None:
        """
        Print analysis comparing unique images vs object frequency.

        Args:
            sort_by: Sort criterion ("unique_images", "total_objects", "ratio")
            top_n: Number of classes to show (None for all)
        """
        class_to_images = self.count_unique_images_per_class()
        object_counts = self.count_objects_per_class()

        # Prepare data for analysis
        analysis_data = []
        for class_id in class_to_images.keys():
            unique_images = len(class_to_images[class_id])
            total_objects = object_counts[class_id]
            ratio = total_objects / unique_images if unique_images > 0 else 0

            analysis_data.append({
                'class_id': class_id,
                'class_name': self.XVIEW_CLASSES.get(class_id, f"Unknown_{class_id}"),
                'unique_images': unique_images,
                'total_objects': total_objects,
                'objects_per_image': ratio
            })

        # Sort data
        if sort_by == "unique_images":
            analysis_data.sort(key=lambda x: x['unique_images'], reverse=True)
        elif sort_by == "total_objects":
            analysis_data.sort(key=lambda x: x['total_objects'], reverse=True)
        elif sort_by == "ratio":
            analysis_data.sort(key=lambda x: x['objects_per_image'], reverse=True)

        # Limit results
        if top_n:
            analysis_data = analysis_data[:top_n]

        # Print results
        print(f"\nUnique Image Analysis (sorted by {sort_by})")
        print("=" * 80)
        print(f"{'ID':<4} {'Class Name':<25} {'Images':<8} {'Objects':<8} {'Obj/Img':<8} {'Density':<8}")
        print("-" * 80)

        total_unique_images = len(set().union(*class_to_images.values()))

        for data in analysis_data:
            density = (data['unique_images'] / total_unique_images) * 100
            print(f"{data['class_id']:<4} {data['class_name']:<25} "
                  f"{data['unique_images']:<8} {data['total_objects']:<8} "
                  f"{data['objects_per_image']:<8.1f} {density:<8.1f}%")

        print("-" * 80)
        print(f"Total unique images in dataset: {total_unique_images}")
        print(f"Total annotations: {sum(object_counts.values())}")

    def print_extraction_potential(self, min_objects_list: List[int] = [1, 2, 5, 10]) -> None:
        """
        Print extraction potential for different minimum object thresholds.

        Args:
            min_objects_list: List of minimum object thresholds to analyze
        """
        print("\nExtraction Potential Analysis")
        print("=" * 60)
        print("Shows how many images would be extracted for each class")
        print("at different minimum object count thresholds.\n")

        for min_objects in min_objects_list:
            stats = self.get_extraction_statistics(min_objects)

            print(f"Minimum {min_objects} object(s) per image:")
            print("-" * 50)

            # Sort by extractable images
            sorted_stats = sorted(stats.items(),
                                key=lambda x: x[1]['extractable_images'],
                                reverse=True)

            for class_id, data in sorted_stats[:15]:  # Top 15
                if data['extractable_images'] > 0:
                    print(f"{data['class_name']:<25} {data['extractable_images']:<6} images "
                          f"({data['extraction_rate']:<5.1%})")
            print()

    def print_group_analysis(self) -> None:
        """Print analysis by logical object groups."""
        class_to_images = self.count_unique_images_per_class()
        object_counts = self.count_objects_per_class()

        print("\nGroup Analysis")
        print("=" * 50)

        for group_name, class_ids in self.CLASS_GROUPS.items():
            total_images = set()
            total_objects = 0

            for class_id in class_ids:
                if class_id in class_to_images:
                    total_images.update(class_to_images[class_id])
                    total_objects += object_counts.get(class_id, 0)

            print(f"{group_name:<20} {len(total_images):<8} images, {total_objects:<8} objects")

    def export_analysis(self, output_file: str = "xview_analysis.json") -> None:
        """Export complete analysis to JSON file."""
        analysis = {
            'unique_images_per_class': {
                str(k): list(v) for k, v in self.count_unique_images_per_class().items()
            },
            'object_counts_per_class': self.count_objects_per_class(),
            'extraction_statistics': {
                str(k): v for k, v in self.get_extraction_statistics().items()
            },
            'class_mapping': self.XVIEW_CLASSES,
            'class_groups': self.CLASS_GROUPS
        }

        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2)

        print(f"Analysis exported to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze xView dataset for unique image counts per class",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic unique image analysis
  python analyze_xview.py

  # Sort by different criteria
  python analyze_xview.py --sort-by total_objects --top 20

  # Focus on extraction potential
  python analyze_xview.py --extraction-analysis

  # Export full analysis
  python analyze_xview.py --export analysis_results.json
        """
    )

    parser.add_argument("--labels-file", default="data/xView_train.geojson",
                       help="Path to xView GeoJSON labels file")
    parser.add_argument("--sort-by", choices=["unique_images", "total_objects", "ratio"],
                       default="unique_images",
                       help="Sort criterion for analysis")
    parser.add_argument("--top", type=int,
                       help="Show only top N classes")
    parser.add_argument("--extraction-analysis", action="store_true",
                       help="Show extraction potential analysis")
    parser.add_argument("--group-analysis", action="store_true",
                       help="Show analysis by object groups")
    parser.add_argument("--export", type=str,
                       help="Export analysis to JSON file")

    args = parser.parse_args()

    try:
        # Initialize analyzer
        analyzer = XViewAnalyzer(args.labels_file)

        # Run analyses
        analyzer.print_unique_image_analysis(args.sort_by, args.top)

        if args.extraction_analysis:
            analyzer.print_extraction_potential()

        if args.group_analysis:
            analyzer.print_group_analysis()

        if args.export:
            analyzer.export_analysis(args.export)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure to download the xView dataset first:")
        print("1. Visit challenge.xviewdataset.org")
        print("2. Download xView_train.geojson")
        print("3. Place it in the data/ directory")


if __name__ == "__main__":
    main()