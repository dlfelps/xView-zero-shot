import argparse
import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModel
from typing import Tuple, List, Dict
import cv2


class DINOv3TemplateMatching:
    """
    DINOv3-based template matching for object detection in satellite imagery.
    Uses Hugging Face Transformers Pipeline for feature extraction.
    """

    def __init__(self, model_name: str = "facebook/dinov2-base"):
        """Initialize the DINOv3 model and processor."""
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def extract_features(self, image: Image.Image) -> torch.Tensor:
        """Extract DINOv3 features from an image."""
        inputs = self.processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use the CLS token representation
            features = outputs.last_hidden_state[:, 0, :]

        return features

    def sliding_window_features(self, scene_image: Image.Image,
                              window_size: Tuple[int, int],
                              stride: int = 32) -> Tuple[List[torch.Tensor], List[Tuple[int, int]]]:
        """
        Extract features from sliding windows across the scene.

        Args:
            scene_image: Full xView scene image
            window_size: Size of sliding window (width, height)
            stride: Stride for sliding window

        Returns:
            List of feature vectors and their corresponding positions
        """
        scene_array = np.array(scene_image)
        h, w = scene_array.shape[:2]
        window_w, window_h = window_size

        features_list = []
        positions = []

        for y in range(0, h - window_h + 1, stride):
            for x in range(0, w - window_w + 1, stride):
                # Extract window
                window = scene_array[y:y+window_h, x:x+window_w]
                window_image = Image.fromarray(window)

                # Extract features
                features = self.extract_features(window_image)
                features_list.append(features)
                positions.append((x, y))

        return features_list, positions

    def compute_similarity(self, template_features: torch.Tensor,
                          scene_features: List[torch.Tensor]) -> List[float]:
        """Compute cosine similarity between template and scene windows."""
        template_norm = torch.nn.functional.normalize(template_features, p=2, dim=1)
        similarities = []

        for scene_feat in scene_features:
            scene_norm = torch.nn.functional.normalize(scene_feat, p=2, dim=1)
            similarity = torch.cosine_similarity(template_norm, scene_norm, dim=1)
            similarities.append(similarity.item())

        return similarities

    def detect_objects(self, scene_image: Image.Image, template_image: Image.Image,
                      threshold: float = 0.7, window_size: Tuple[int, int] = None) -> List[Dict]:
        """
        Perform template matching on the scene using DINOv3 features.

        Args:
            scene_image: Full xView scene
            template_image: Target object template
            threshold: Similarity threshold for detection
            window_size: Sliding window size (defaults to template size)

        Returns:
            List of detection results with positions and confidence scores
        """
        # Use template size as default window size
        if window_size is None:
            window_size = template_image.size

        # Extract template features
        template_features = self.extract_features(template_image)

        # Extract features from sliding windows
        scene_features, positions = self.sliding_window_features(
            scene_image, window_size
        )

        # Compute similarities
        similarities = self.compute_similarity(template_features, scene_features)

        # Filter detections by threshold
        detections = []
        for i, (sim, pos) in enumerate(zip(similarities, positions)):
            if sim >= threshold:
                detections.append({
                    'position': pos,
                    'confidence': sim,
                    'bbox': (pos[0], pos[1], pos[0] + window_size[0], pos[1] + window_size[1])
                })

        # Sort by confidence
        detections.sort(key=lambda x: x['confidence'], reverse=True)

        return detections


def apply_non_max_suppression(detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
    """Apply non-maximum suppression to remove overlapping detections."""
    if not detections:
        return []

    # Convert to format expected by cv2.dnn.NMSBoxes
    boxes = [det['bbox'] for det in detections]
    scores = [det['confidence'] for det in detections]

    # Apply NMS
    indices = cv2.dnn.NMSBoxes(boxes, scores, 0.0, iou_threshold)

    if len(indices) > 0:
        indices = indices.flatten()
        return [detections[i] for i in indices]
    else:
        return []


def visualize_detections(scene_image: Image.Image, detections: List[Dict],
                        output_path: str = "detections.jpg"):
    """Visualize detection results on the scene image."""
    scene_cv = cv2.cvtColor(np.array(scene_image), cv2.COLOR_RGB2BGR)

    for detection in detections:
        bbox = detection['bbox']
        confidence = detection['confidence']

        # Draw bounding box
        cv2.rectangle(scene_cv, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

        # Add confidence score
        label = f"Conf: {confidence:.3f}"
        cv2.putText(scene_cv, label, (bbox[0], bbox[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imwrite(output_path, scene_cv)
    print(f"Detection results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="DINOv3 Template Matching for xView Object Detection")
    parser.add_argument("--scene", required=True, help="Path to xView scene image")
    parser.add_argument("--template", required=True, help="Path to target template image")
    parser.add_argument("--threshold", type=float, default=0.7,
                       help="Similarity threshold for detection (default: 0.7)")
    parser.add_argument("--window-size", type=int, nargs=2, metavar=('WIDTH', 'HEIGHT'),
                       help="Sliding window size (default: template size)")
    parser.add_argument("--stride", type=int, default=32,
                       help="Sliding window stride (default: 32)")
    parser.add_argument("--output", default="detections.jpg",
                       help="Output visualization path (default: detections.jpg)")
    parser.add_argument("--nms-threshold", type=float, default=0.5,
                       help="NMS IoU threshold (default: 0.5)")

    args = parser.parse_args()

    # Load images
    try:
        scene_image = Image.open(args.scene).convert('RGB')
        template_image = Image.open(args.template).convert('RGB')
        print(f"Loaded scene: {scene_image.size}, template: {template_image.size}")
    except Exception as e:
        print(f"Error loading images: {e}")
        return

    # Initialize detector
    print("Initializing DINOv3 model...")
    detector = DINOv3TemplateMatching()

    # Set window size
    window_size = tuple(args.window_size) if args.window_size else template_image.size
    print(f"Using window size: {window_size}")

    # Perform detection
    print("Performing template matching...")
    detections = detector.detect_objects(
        scene_image=scene_image,
        template_image=template_image,
        threshold=args.threshold,
        window_size=window_size
    )

    print(f"Found {len(detections)} initial detections")

    # Apply non-maximum suppression
    detections = apply_non_max_suppression(detections, args.nms_threshold)
    print(f"After NMS: {len(detections)} detections")

    # Print results
    for i, det in enumerate(detections):
        print(f"Detection {i+1}: Position {det['position']}, "
              f"Confidence {det['confidence']:.3f}, BBox {det['bbox']}")

    # Visualize results
    if detections:
        visualize_detections(scene_image, detections, args.output)
    else:
        print("No detections found above threshold")


if __name__ == "__main__":
    main()