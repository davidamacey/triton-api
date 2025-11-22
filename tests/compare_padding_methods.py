#!/usr/bin/env python3
"""
Compare Padding Methods for YOLO Preprocessing

This script compares three different padding approaches:
1. PyTorch Baseline (Ultralytics center padding - Track A)
2. DALI Center Padding (affine transformation - current Track D)
3. DALI Simple Padding (bottom/right padding - NEW)

The goal is to determine if simple bottom/right padding is accurate enough
compared to the more complex center padding with affine transformation.

Usage:
    # Test on sample images
    python /app/tests/compare_padding_methods.py

    # Test on specific directory
    python /app/tests/compare_padding_methods.py --images /path/to/images

    # Test with different IoU threshold
    python /app/tests/compare_padding_methods.py --iou-threshold 0.7

    # Save detailed results
    python /app/tests/compare_padding_methods.py --output-dir /app/results
"""

import sys
import os
import argparse
import time
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import cv2
from PIL import Image
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.detection_comparison_utils import (
    parse_detections,
    calculate_comparison_metrics,
    format_metrics_table,
    ComparisonMetrics
)

# Import inference clients
from src.utils import TritonEnd2EndClient
from ultralytics import YOLO


class PaddingMethodComparator:
    """Compare different padding methods for YOLO preprocessing."""

    def __init__(
        self,
        pytorch_model_path: str = "/app/pytorch_models/yolo11s.pt",
        triton_url: str = "triton-api:8001",
        verbose: bool = True
    ):
        """
        Initialize comparator with all three methods.

        Args:
            pytorch_model_path: Path to PyTorch YOLO model
            triton_url: Triton server URL
            verbose: Print progress messages
        """
        self.verbose = verbose
        self.triton_url = triton_url

        # Load PyTorch model (Track A - Baseline)
        if self.verbose:
            print("Loading PyTorch baseline model...")
        self.pytorch_model = YOLO(pytorch_model_path)

        # Initialize Triton clients for Track D variants
        if self.verbose:
            print("Initializing Triton clients...")

        # NOTE: These models need to be created first
        # Run: docker compose exec yolo-api python /app/dali/create_dali_letterbox_auto_pipeline.py
        # Run: docker compose exec yolo-api python /app/dali/create_dali_simple_padding_pipeline.py
        # Run: docker compose exec yolo-api python /app/dali/create_simple_padding_ensemble.py

        # Center padding with auto-affine (existing)
        self.client_center = TritonEnd2EndClient(
            triton_url=triton_url,
            model_name="yolov11_small_gpu_e2e_auto",
            use_shared_client=False  # Use dedicated client for testing
        )

        # Simple bottom/right padding (NEW)
        self.client_simple = TritonEnd2EndClient(
            triton_url=triton_url,
            model_name="yolov11_small_simple_padding",
            use_shared_client=False  # Use dedicated client for testing
        )

        if self.verbose:
            print("✓ All models loaded successfully\n")

    def run_pytorch_inference(self, image_path: str) -> Tuple[List[Dict], float]:
        """
        Run PyTorch baseline inference (Track A).

        Args:
            image_path: Path to image file

        Returns:
            - List of detections
            - Inference time (ms)
        """
        start = time.perf_counter()
        results = self.pytorch_model(image_path, verbose=False)
        end = time.perf_counter()

        # Extract detections
        detections = []
        if len(results) > 0 and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                cls = int(boxes.cls[i].cpu().numpy())

                detections.append({
                    'x1': float(x1),
                    'y1': float(y1),
                    'x2': float(x2),
                    'y2': float(y2),
                    'confidence': conf,
                    'class': cls
                })

        inference_time = (end - start) * 1000  # Convert to ms
        return detections, inference_time

    def run_dali_center_inference(self, image_path: str) -> Tuple[List[Dict], float]:
        """
        Run DALI center padding inference (auto-affine).

        Args:
            image_path: Path to image file

        Returns:
            - List of detections
            - Inference time (ms)
        """
        # Read image as bytes
        with open(image_path, 'rb') as f:
            image_bytes = f.read()

        start = time.perf_counter()
        result = self.client_center.infer_raw_bytes_auto(image_bytes)
        end = time.perf_counter()

        # Format detections
        detections = []
        for i in range(result['num_dets']):
            x1, y1, x2, y2 = result['boxes'][i]
            detections.append({
                'x1': float(x1),
                'y1': float(y1),
                'x2': float(x2),
                'y2': float(y2),
                'confidence': float(result['scores'][i]),
                'class': int(result['classes'][i])
            })

        inference_time = (end - start) * 1000
        return detections, inference_time

    def run_dali_simple_inference(self, image_path: str) -> Tuple[List[Dict], float]:
        """
        Run DALI simple padding inference (bottom/right padding).

        Args:
            image_path: Path to image file

        Returns:
            - List of detections
            - Inference time (ms)
        """
        # Read image as bytes
        with open(image_path, 'rb') as f:
            image_bytes = f.read()

        start = time.perf_counter()
        result = self.client_simple.infer_raw_bytes_auto(image_bytes)
        end = time.perf_counter()

        # Format detections
        detections = []
        for i in range(result['num_dets']):
            x1, y1, x2, y2 = result['boxes'][i]
            detections.append({
                'x1': float(x1),
                'y1': float(y1),
                'x2': float(x2),
                'y2': float(y2),
                'confidence': float(result['scores'][i]),
                'class': int(result['classes'][i])
            })

        inference_time = (end - start) * 1000
        return detections, inference_time

    def compare_on_image(
        self,
        image_path: str,
        iou_threshold: float = 0.5
    ) -> Dict:
        """
        Compare all three methods on a single image.

        Args:
            image_path: Path to image file
            iou_threshold: IoU threshold for matching detections

        Returns:
            Dictionary with comparison results
        """
        if self.verbose:
            print(f"\nProcessing: {Path(image_path).name}")

        # Run all three methods
        pytorch_dets, pytorch_time = self.run_pytorch_inference(image_path)
        center_dets, center_time = self.run_dali_center_inference(image_path)
        simple_dets, simple_time = self.run_dali_simple_inference(image_path)

        if self.verbose:
            print(f"  PyTorch:       {len(pytorch_dets)} detections in {pytorch_time:.2f}ms")
            print(f"  DALI Center:   {len(center_dets)} detections in {center_time:.2f}ms")
            print(f"  DALI Simple:   {len(simple_dets)} detections in {simple_time:.2f}ms")

        # Parse detections
        pytorch_parsed = parse_detections(pytorch_dets)
        center_parsed = parse_detections(center_dets)
        simple_parsed = parse_detections(simple_dets)

        # Compare against PyTorch baseline
        center_metrics = calculate_comparison_metrics(
            pytorch_parsed, center_parsed, iou_threshold
        )
        simple_metrics = calculate_comparison_metrics(
            pytorch_parsed, simple_parsed, iou_threshold
        )

        if self.verbose:
            print(f"  Center vs Baseline: F1={center_metrics.f1_score:.3f}, IoU={center_metrics.mean_iou:.3f}")
            print(f"  Simple vs Baseline: F1={simple_metrics.f1_score:.3f}, IoU={simple_metrics.mean_iou:.3f}")

        return {
            'image': str(image_path),
            'detections': {
                'pytorch': pytorch_dets,
                'center': center_dets,
                'simple': simple_dets
            },
            'times_ms': {
                'pytorch': pytorch_time,
                'center': center_time,
                'simple': simple_time
            },
            'metrics': {
                'center_vs_baseline': center_metrics,
                'simple_vs_baseline': simple_metrics
            }
        }

    def compare_on_dataset(
        self,
        image_dir: str,
        iou_threshold: float = 0.5,
        max_images: int = None
    ) -> Dict:
        """
        Compare methods on a directory of images.

        Args:
            image_dir: Directory containing test images
            iou_threshold: IoU threshold for matching
            max_images: Maximum number of images to process (None = all)

        Returns:
            Aggregated comparison results
        """
        # Find all images
        image_dir = Path(image_dir)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = [
            f for f in image_dir.iterdir()
            if f.suffix.lower() in image_extensions
        ]

        if max_images:
            image_files = image_files[:max_images]

        print(f"\n{'='*80}")
        print(f"Comparing Padding Methods on {len(image_files)} images")
        print(f"{'='*80}")

        results = []
        for image_path in image_files:
            result = self.compare_on_image(str(image_path), iou_threshold)
            results.append(result)

        # Aggregate metrics
        center_metrics_list = [r['metrics']['center_vs_baseline'] for r in results]
        simple_metrics_list = [r['metrics']['simple_vs_baseline'] for r in results]

        # Calculate average metrics
        def avg_metrics(metrics_list: List[ComparisonMetrics]) -> Dict:
            return {
                'precision': np.mean([m.precision for m in metrics_list]),
                'recall': np.mean([m.recall for m in metrics_list]),
                'f1_score': np.mean([m.f1_score for m in metrics_list]),
                'mean_iou': np.mean([m.mean_iou for m in metrics_list]),
                'mean_conf_diff': np.mean([m.mean_conf_diff for m in metrics_list]),
                'mean_box_diff': np.mean([m.mean_box_diff for m in metrics_list]),
                'total_matches': sum([m.num_matches for m in metrics_list]),
                'total_reference': sum([m.num_reference for m in metrics_list]),
                'total_test': sum([m.num_test for m in metrics_list]),
            }

        avg_center = avg_metrics(center_metrics_list)
        avg_simple = avg_metrics(simple_metrics_list)

        # Calculate average times
        avg_times = {
            'pytorch': np.mean([r['times_ms']['pytorch'] for r in results]),
            'center': np.mean([r['times_ms']['center'] for r in results]),
            'simple': np.mean([r['times_ms']['simple'] for r in results])
        }

        return {
            'num_images': len(results),
            'iou_threshold': iou_threshold,
            'individual_results': results,
            'aggregated_metrics': {
                'center_vs_baseline': avg_center,
                'simple_vs_baseline': avg_simple
            },
            'average_times_ms': avg_times
        }


def print_summary(results: Dict):
    """Print summary of comparison results."""
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}")

    print(f"\nDataset: {results['num_images']} images")
    print(f"IoU Threshold: {results['iou_threshold']}")

    print(f"\n{'Method':<30} {'Prec':<8} {'Recall':<8} {'F1':<8} {'IoU':<8} {'Time(ms)':<10}")
    print("=" * 80)

    center_metrics = results['aggregated_metrics']['center_vs_baseline']
    simple_metrics = results['aggregated_metrics']['simple_vs_baseline']
    times = results['average_times_ms']

    print(f"{'PyTorch Baseline (Track A)':<30} {'1.000':<8} {'1.000':<8} {'1.000':<8} {'1.000':<8} {times['pytorch']:>8.2f}")
    print(
        f"{'DALI Center Padding':<30} "
        f"{center_metrics['precision']:>7.3f} "
        f"{center_metrics['recall']:>7.3f} "
        f"{center_metrics['f1_score']:>7.3f} "
        f"{center_metrics['mean_iou']:>7.3f} "
        f"{times['center']:>8.2f}"
    )
    print(
        f"{'DALI Simple Padding (NEW)':<30} "
        f"{simple_metrics['precision']:>7.3f} "
        f"{simple_metrics['recall']:>7.3f} "
        f"{simple_metrics['f1_score']:>7.3f} "
        f"{simple_metrics['mean_iou']:>7.3f} "
        f"{times['simple']:>8.2f}"
    )

    print(f"\n{'Detailed Metrics':<30} {'Center':<15} {'Simple'}")
    print("=" * 80)
    print(f"{'Average Box Center Diff (px)':<30} {center_metrics['mean_box_diff']:>14.2f} {simple_metrics['mean_box_diff']:>14.2f}")
    print(f"{'Average Conf Diff':<30} {center_metrics['mean_conf_diff']:>14.4f} {simple_metrics['mean_conf_diff']:>14.4f}")
    print(f"{'Total Matches':<30} {center_metrics['total_matches']:>14} {simple_metrics['total_matches']:>14}")

    # Analysis
    print(f"\n{'='*80}")
    print("ANALYSIS")
    print(f"{'='*80}")

    if simple_metrics['f1_score'] >= 0.99:
        print("✓ EXCELLENT: Simple padding achieves ≥99% F1 score vs baseline")
        print("  Recommendation: Use simple padding - faster and simpler with minimal accuracy loss")
    elif simple_metrics['f1_score'] >= 0.95:
        print("✓ GOOD: Simple padding achieves ≥95% F1 score vs baseline")
        print("  Recommendation: Use simple padding for most applications")
    elif simple_metrics['f1_score'] >= 0.90:
        print("⚠ MODERATE: Simple padding achieves 90-95% F1 score vs baseline")
        print("  Recommendation: Use center padding for critical applications")
    else:
        print("✗ POOR: Simple padding achieves <90% F1 score vs baseline")
        print("  Recommendation: Stick with center padding (affine transformation)")

    speedup_center = times['pytorch'] / times['center']
    speedup_simple = times['pytorch'] / times['simple']
    simple_vs_center = times['center'] / times['simple']

    print(f"\nPerformance:")
    print(f"  Center padding: {speedup_center:.2f}x faster than PyTorch")
    print(f"  Simple padding: {speedup_simple:.2f}x faster than PyTorch")
    print(f"  Simple vs Center: {simple_vs_center:.2f}x {'faster' if simple_vs_center > 1 else 'slower'}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare YOLO padding methods",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--images',
        type=str,
        default='/app/benchmarks/images',
        help='Directory containing test images'
    )
    parser.add_argument(
        '--iou-threshold',
        type=float,
        default=0.5,
        help='IoU threshold for matching detections (default: 0.5)'
    )
    parser.add_argument(
        '--max-images',
        type=int,
        default=None,
        help='Maximum number of images to process (default: all)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save detailed results JSON (default: None)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress per-image output'
    )

    args = parser.parse_args()

    # Initialize comparator
    comparator = PaddingMethodComparator(verbose=not args.quiet)

    # Run comparison
    results = comparator.compare_on_dataset(
        args.images,
        iou_threshold=args.iou_threshold,
        max_images=args.max_images
    )

    # Print summary
    print_summary(results)

    # Save results if requested
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"padding_comparison_{int(time.time())}.json"

        # Convert ComparisonMetrics to dicts for JSON serialization
        def metrics_to_dict(m):
            if isinstance(m, ComparisonMetrics):
                return {
                    'num_matches': m.num_matches,
                    'num_reference': m.num_reference,
                    'num_test': m.num_test,
                    'precision': m.precision,
                    'recall': m.recall,
                    'f1_score': m.f1_score,
                    'mean_iou': m.mean_iou,
                    'mean_conf_diff': m.mean_conf_diff,
                    'mean_box_diff': m.mean_box_diff
                }
            return m

        # Convert metrics in results
        results_copy = results.copy()
        for result in results_copy['individual_results']:
            result['metrics'] = {
                k: metrics_to_dict(v)
                for k, v in result['metrics'].items()
            }

        with open(output_file, 'w') as f:
            json.dump(results_copy, f, indent=2)

        print(f"\n✓ Detailed results saved to: {output_file}")


if __name__ == "__main__":
    main()
