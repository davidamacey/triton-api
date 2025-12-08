#!/usr/bin/env python3
"""
Validate Dual DALI Preprocessing (Track E)

Compares DALI preprocessing outputs against PyTorch reference implementations
to ensure correctness. Tests both branches:
- YOLO: Letterbox preprocessing (640x640)
- MobileCLIP: Center crop preprocessing (256x256)

Also benchmarks preprocessing latency to verify performance targets.

Usage:
    # From yolo-api container
    docker compose exec yolo-api python /app/dali/validate_dual_dali_preprocessing.py

    # With verbose output
    docker compose exec yolo-api python /app/dali/validate_dual_dali_preprocessing.py -v
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image


if TYPE_CHECKING:
    from numpy.typing import NDArray

# Import shared configuration
from dali.config import (
    CLIP_SIZE,
    CORRECTNESS_THRESHOLD,
    DEFAULT_BENCHMARK_ITERATIONS,
    DEFAULT_TRITON_URL,
    DEFAULT_WARMUP_ITERATIONS,
    PERFORMANCE_TARGET_MS,
    TRACK_E_DUAL_DALI_MODEL,
    TRACK_E_OUTPUT_CLIP,
    TRACK_E_OUTPUT_ORIGINAL,
    TRACK_E_OUTPUT_YOLO,
    YOLO_PAD_VALUE,
    YOLO_SIZE,
)
from dali.utils import calculate_letterbox_affine, setup_logging


logger = logging.getLogger(__name__)


# =============================================================================
# PyTorch Reference Implementations
# =============================================================================


def pytorch_yolo_preprocess(
    img_path: str | Path,
    target_size: int = YOLO_SIZE,
) -> tuple[NDArray[np.float32], float, tuple[int, int]]:
    """
    PyTorch reference: YOLO letterbox preprocessing.

    Args:
        img_path: Path to input image.
        target_size: Target square size (default: 640).

    Returns:
        Tuple of (preprocessed_image, scale, (top_pad, left_pad)).
    """
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img.shape[:2]

    # Compute scale
    scale = min(target_size / orig_h, target_size / orig_w)
    new_h, new_w = int(orig_h * scale), int(orig_w * scale)

    # Resize
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create canvas with padding
    canvas = np.full((target_size, target_size, 3), YOLO_PAD_VALUE, dtype=np.uint8)
    top = (target_size - new_h) // 2
    left = (target_size - new_w) // 2
    canvas[top : top + new_h, left : left + new_w, :] = img_resized

    # Normalize to [0, 1] and transpose to CHW
    canvas = canvas.astype(np.float32) / 255.0
    canvas = np.transpose(canvas, (2, 0, 1))

    return canvas, scale, (top, left)


def pytorch_mobileclip_preprocess(
    img_path: str | Path,
    target_size: int = CLIP_SIZE,
) -> NDArray[np.float32]:
    """
    PyTorch reference: MobileCLIP center crop preprocessing.

    Args:
        img_path: Path to input image.
        target_size: Target square size (default: 256).

    Returns:
        Preprocessed image array [3, H, W].
    """
    img = Image.open(img_path).convert('RGB')

    # Resize shorter edge to target_size
    w, h = img.size
    if w < h:
        new_w = target_size
        new_h = int(h * target_size / w)
    else:
        new_h = target_size
        new_w = int(w * target_size / h)

    img = img.resize((new_w, new_h), Image.BILINEAR)

    # Center crop
    left = (new_w - target_size) // 2
    top = (new_h - target_size) // 2
    img = img.crop((left, top, left + target_size, top + target_size))

    # Convert to numpy and normalize
    img_array = np.array(img).astype(np.float32) / 255.0
    return np.transpose(img_array, (2, 0, 1))  # HWC -> CHW


# =============================================================================
# DALI Preprocessing via Triton
# =============================================================================


def dali_dual_preprocess(
    img_path: str | Path,
    triton_url: str = DEFAULT_TRITON_URL,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32] | None]:
    """
    DALI dual preprocessing via Triton.

    Args:
        img_path: Path to input image.
        triton_url: Triton server URL.

    Returns:
        Tuple of (yolo_output, clip_output, original_output).
        original_output may be None if the model doesn't have that output.
    """
    try:
        import tritonclient.grpc as grpcclient
    except ImportError as e:
        raise ImportError('tritonclient not installed: pip install tritonclient[grpc]') from e

    # Read image as JPEG bytes
    with open(img_path, 'rb') as f:
        img_bytes = f.read()

    # Get image dimensions for affine matrix calculation
    img = Image.open(img_path)
    orig_w, orig_h = img.size

    # Calculate affine matrix for YOLO letterbox
    affine_matrix, _, _, _ = calculate_letterbox_affine(orig_w, orig_h, YOLO_SIZE)

    # Create Triton client
    client = grpcclient.InferenceServerClient(url=triton_url, verbose=False)

    # Check model availability
    if not client.is_model_ready(TRACK_E_DUAL_DALI_MODEL):
        raise RuntimeError(f"Model '{TRACK_E_DUAL_DALI_MODEL}' is not ready")

    # Prepare inputs
    img_data = np.frombuffer(img_bytes, dtype=np.uint8)
    img_data = np.expand_dims(img_data, axis=0)  # Add batch dimension

    affine_data = np.expand_dims(affine_matrix, axis=0)  # Add batch dimension

    inputs = [
        grpcclient.InferInput('encoded_images', img_data.shape, 'UINT8'),
        grpcclient.InferInput('affine_matrices', affine_data.shape, 'FP32'),
    ]
    inputs[0].set_data_from_numpy(img_data)
    inputs[1].set_data_from_numpy(affine_data)

    # Request outputs - use correct output names from config
    outputs = [
        grpcclient.InferRequestedOutput(TRACK_E_OUTPUT_YOLO),
        grpcclient.InferRequestedOutput(TRACK_E_OUTPUT_CLIP),
    ]

    # Try to get original_images output if available
    try:
        outputs.append(grpcclient.InferRequestedOutput(TRACK_E_OUTPUT_ORIGINAL))
        has_original = True
    except Exception:
        has_original = False

    # Infer
    response = client.infer(
        model_name=TRACK_E_DUAL_DALI_MODEL,
        inputs=inputs,
        outputs=outputs,
    )

    # Get outputs
    yolo_output = response.as_numpy(TRACK_E_OUTPUT_YOLO)[0]  # Remove batch dim
    clip_output = response.as_numpy(TRACK_E_OUTPUT_CLIP)[0]

    original_output = None
    if has_original:
        with contextlib.suppress(Exception):
            original_output = response.as_numpy(TRACK_E_OUTPUT_ORIGINAL)[0]

    return yolo_output, clip_output, original_output


# =============================================================================
# Comparison Functions
# =============================================================================


def compute_difference(
    arr1: NDArray,
    arr2: NDArray,
    name: str,
    threshold: float = CORRECTNESS_THRESHOLD,
) -> bool:
    """
    Compute and display differences between two arrays.

    Args:
        arr1: Reference array (PyTorch output).
        arr2: Comparison array (DALI output).
        name: Name for display.
        threshold: Maximum acceptable mean absolute difference.

    Returns:
        True if arrays match within threshold.
    """
    print(f'\n{name}:')
    print(f'  Shape match: {arr1.shape == arr2.shape}')
    print(f'  PyTorch range: [{arr1.min():.4f}, {arr1.max():.4f}]')
    print(f'  DALI range:    [{arr2.min():.4f}, {arr2.max():.4f}]')

    if arr1.shape != arr2.shape:
        print(f'  [ERROR] Shape mismatch: {arr1.shape} vs {arr2.shape}')
        return False

    # Compute differences
    abs_diff = np.abs(arr1 - arr2)
    rel_diff = abs_diff / (np.abs(arr1) + 1e-8)

    print(f'  Mean absolute diff: {abs_diff.mean():.6f}')
    print(f'  Max absolute diff:  {abs_diff.max():.6f}')
    print(f'  Mean relative diff: {rel_diff.mean():.4f}')
    print(f'  Pixels >1% diff:    {(abs_diff > 0.01).sum()} / {abs_diff.size}')

    # Check if acceptable
    if abs_diff.mean() < threshold:
        print(f'  [PASS] Mean diff < {threshold}')
        return True
    print(f'  [WARNING] Mean diff >= {threshold}')
    return False


def test_image(
    img_path: str | Path,
    triton_url: str = DEFAULT_TRITON_URL,
) -> bool:
    """
    Test single image through both pipelines.

    Args:
        img_path: Path to test image.
        triton_url: Triton server URL.

    Returns:
        True if all tests pass.
    """
    print('\n' + '=' * 80)
    print(f'Testing image: {img_path}')
    print('=' * 80)

    if not Path(img_path).exists():
        print(f'[ERROR] Image not found: {img_path}')
        return False

    # PyTorch reference
    print('\nRunning PyTorch reference preprocessing...')
    yolo_pytorch, _, _ = pytorch_yolo_preprocess(img_path)
    clip_pytorch = pytorch_mobileclip_preprocess(img_path)

    # DALI
    print('Running DALI dual preprocessing...')
    try:
        yolo_dali, clip_dali, original_dali = dali_dual_preprocess(img_path, triton_url)
    except Exception as e:
        print(f'[ERROR] DALI preprocessing failed: {e}')
        logger.exception('DALI error')
        return False

    # Compare
    print('\nComparing outputs...')
    yolo_match = compute_difference(yolo_pytorch, yolo_dali, 'YOLO Branch')
    clip_match = compute_difference(clip_pytorch, clip_dali, 'MobileCLIP Branch')

    # Check original_images if available
    if original_dali is not None:
        print(
            f'\n  Original images output: shape={original_dali.shape}, dtype={original_dali.dtype}'
        )
        print(f'  Original images range: [{original_dali.min():.4f}, {original_dali.max():.4f}]')

    return yolo_match and clip_match


# =============================================================================
# Benchmarking
# =============================================================================


def benchmark_dali(
    img_path: str | Path,
    triton_url: str = DEFAULT_TRITON_URL,
    num_iterations: int = DEFAULT_BENCHMARK_ITERATIONS,
    warmup_iterations: int = DEFAULT_WARMUP_ITERATIONS,
) -> bool:
    """
    Benchmark DALI preprocessing speed.

    Args:
        img_path: Path to test image.
        triton_url: Triton server URL.
        num_iterations: Number of benchmark iterations.
        warmup_iterations: Number of warmup iterations.

    Returns:
        True if performance target is met.
    """
    try:
        import tritonclient.grpc as grpcclient
    except ImportError as e:
        print(f'[ERROR] tritonclient not installed: {e}')
        return False

    print('\n' + '=' * 80)
    print('Benchmarking DALI Dual Preprocessing')
    print('=' * 80)

    # Read image
    with open(img_path, 'rb') as f:
        img_bytes = f.read()

    img = Image.open(img_path)
    orig_w, orig_h = img.size
    affine_matrix, _, _, _ = calculate_letterbox_affine(orig_w, orig_h, YOLO_SIZE)

    # Setup client
    client = grpcclient.InferenceServerClient(url=triton_url, verbose=False)

    img_data = np.frombuffer(img_bytes, dtype=np.uint8)
    img_data = np.expand_dims(img_data, axis=0)
    affine_data = np.expand_dims(affine_matrix, axis=0)

    inputs = [
        grpcclient.InferInput('encoded_images', img_data.shape, 'UINT8'),
        grpcclient.InferInput('affine_matrices', affine_data.shape, 'FP32'),
    ]
    inputs[0].set_data_from_numpy(img_data)
    inputs[1].set_data_from_numpy(affine_data)

    outputs = [
        grpcclient.InferRequestedOutput(TRACK_E_OUTPUT_YOLO),
        grpcclient.InferRequestedOutput(TRACK_E_OUTPUT_CLIP),
    ]

    # Warmup
    print(f'\nWarming up ({warmup_iterations} iterations)...')
    for _ in range(warmup_iterations):
        client.infer(model_name=TRACK_E_DUAL_DALI_MODEL, inputs=inputs, outputs=outputs)

    # Benchmark
    print(f'Running benchmark ({num_iterations} iterations)...')
    latencies = []

    for _ in range(num_iterations):
        start = time.time()
        client.infer(model_name=TRACK_E_DUAL_DALI_MODEL, inputs=inputs, outputs=outputs)
        latency = (time.time() - start) * 1000
        latencies.append(latency)

    # Statistics
    print('\nLatency Statistics:')
    print(f'  Mean:   {np.mean(latencies):.2f}ms')
    print(f'  Median: {np.median(latencies):.2f}ms')
    print(f'  P95:    {np.percentile(latencies, 95):.2f}ms')
    print(f'  P99:    {np.percentile(latencies, 99):.2f}ms')
    print(f'  Min:    {np.min(latencies):.2f}ms')
    print(f'  Max:    {np.max(latencies):.2f}ms')

    # Check target
    mean_latency = np.mean(latencies)
    if mean_latency < PERFORMANCE_TARGET_MS:
        print(
            f'\n[PASS] Performance target met (mean: {mean_latency:.2f}ms < {PERFORMANCE_TARGET_MS}ms)'
        )
        return True
    print(
        f'\n[WARNING] Performance below target (mean: {mean_latency:.2f}ms >= {PERFORMANCE_TARGET_MS}ms)'
    )
    return False


# =============================================================================
# CLI Interface
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Validate DALI dual preprocessing against PyTorch reference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with default settings
    python dali/validate_dual_dali_preprocessing.py

    # Run with custom images
    python dali/validate_dual_dali_preprocessing.py --images /path/to/img1.jpg /path/to/img2.jpg

    # Skip benchmark
    python dali/validate_dual_dali_preprocessing.py --skip-benchmark
""",
    )
    parser.add_argument(
        '--images',
        nargs='+',
        type=Path,
        help='Test image paths (default: search /app/test_images/)',
    )
    parser.add_argument(
        '--triton-url',
        type=str,
        default=DEFAULT_TRITON_URL,
        help=f'Triton server URL (default: {DEFAULT_TRITON_URL})',
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=DEFAULT_BENCHMARK_ITERATIONS,
        help=f'Benchmark iterations (default: {DEFAULT_BENCHMARK_ITERATIONS})',
    )
    parser.add_argument(
        '--skip-benchmark',
        action='store_true',
        help='Skip performance benchmark',
    )
    parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        help='Enable verbose output',
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)

    print('\n' + '=' * 80)
    print('Track E: DALI Dual Preprocessing Validation')
    print('=' * 80)

    # Find test images
    if args.images:
        test_images = [str(p) for p in args.images]
    else:
        test_images = [
            '/app/test_images/bus.jpg',
            '/app/test_images/zidane.jpg',
        ]

    # Filter to available images
    available_images = [img for img in test_images if Path(img).exists()]

    if not available_images:
        print('\n[WARNING] No test images found, searching for any available...')
        test_dir = Path('/app/test_images')
        if test_dir.exists():
            all_images = [
                str(f) for f in test_dir.iterdir() if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}
            ]
            if all_images:
                available_images = [all_images[0]]

    if not available_images:
        print('[ERROR] No test images available!')
        return 1

    print(f'\nFound {len(available_images)} test image(s)')

    # Test correctness
    results = []
    for img_path in available_images:
        result = test_image(img_path, args.triton_url)
        results.append(result)

    # Benchmark
    benchmark_result = True
    if not args.skip_benchmark:
        benchmark_result = benchmark_dali(
            available_images[0],
            args.triton_url,
            args.iterations,
        )

    # Summary
    print('\n' + '=' * 80)
    print('VALIDATION SUMMARY')
    print('=' * 80)

    correctness_passed = all(results)
    print(f'Correctness: {"[PASS]" if correctness_passed else "[FAIL]"}')
    print(f'Performance: {"[PASS]" if benchmark_result else "[WARNING] Below target"}')

    if correctness_passed:
        print('\n[SUCCESS] Dual DALI preprocessing validated!')
        return 0
    print('\n[ERROR] Validation failed!')
    return 1


if __name__ == '__main__':
    sys.exit(main())
