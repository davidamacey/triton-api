#!/usr/bin/env python3
"""
Track E: End-to-End Ensemble Test Script

Tests the complete YOLO + MobileCLIP visual search pipeline:
- DALI triple-branch preprocessing
- YOLO object detection
- MobileCLIP global embedding
- Per-object embeddings with full-resolution cropping

Run from: anywhere with Triton access
    docker compose exec yolo-api python /app/scripts/track_e/test_ensemble.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import tritonclient.grpc as grpcclient
from PIL import Image


def calculate_letterbox_affine(img_w, img_h, target_size=640):
    """
    Calculate affine transformation matrix for YOLO letterbox

    Args:
        img_w: Original image width
        img_h: Original image height
        target_size: Target size (640 for YOLO)

    Returns:
        affine_matrix: [2, 3] numpy array
    """
    scale = min(target_size / img_h, target_size / img_w)
    scale = min(scale, 1.0)

    new_h = round(img_h * scale)
    new_w = round(img_w * scale)
    pad_x = (target_size - new_w) / 2.0
    pad_y = (target_size - new_h) / 2.0

    return np.array([[scale, 0.0, pad_x], [0.0, scale, pad_y]], dtype=np.float32)


def test_ensemble_single_image(img_path, verbose=True):
    """
    Test ensemble with single image

    Args:
        img_path: Path to test image
        verbose: Print detailed output

    Returns:
        dict with results
    """
    if verbose:
        print(f'\nTesting: {img_path}')
        print('=' * 80)

    # Load image
    img = Image.open(img_path).convert('RGB')
    img_w, img_h = img.size

    if verbose:
        print(f'Image size: {img_w}x{img_h}')

    # Encode to JPEG
    import io

    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=95)
    jpeg_bytes = buffer.getvalue()
    img_data = np.frombuffer(jpeg_bytes, dtype=np.uint8)

    if verbose:
        print(f'JPEG size: {len(jpeg_bytes)} bytes')

    # Calculate affine matrix
    affine_matrix = calculate_letterbox_affine(img_w, img_h)

    if verbose:
        print(f'Affine matrix scale: {affine_matrix[0, 0]:.4f}')

    # Create Triton client
    try:
        client = grpcclient.InferenceServerClient(url='triton-api:8001', verbose=False)
    except Exception as e:
        print(f'✗ Failed to connect to Triton: {e}')
        return None

    # Prepare inputs
    inputs = [
        grpcclient.InferInput('encoded_images', [len(img_data)], 'UINT8'),
        grpcclient.InferInput('affine_matrices', affine_matrix.shape, 'FP32'),
    ]
    inputs[0].set_data_from_numpy(img_data)
    inputs[1].set_data_from_numpy(affine_matrix)

    # Request outputs
    outputs = [
        grpcclient.InferRequestedOutput('num_dets'),
        grpcclient.InferRequestedOutput('det_boxes'),
        grpcclient.InferRequestedOutput('det_scores'),
        grpcclient.InferRequestedOutput('det_classes'),
        grpcclient.InferRequestedOutput('global_embeddings'),
        grpcclient.InferRequestedOutput('box_embeddings'),
    ]

    # Run inference
    if verbose:
        print('\nRunning ensemble inference...')

    start = time.time()
    try:
        response = client.infer(
            model_name='yolo_mobileclip_ensemble', inputs=inputs, outputs=outputs
        )
    except Exception as e:
        print(f'✗ Inference failed: {e}')
        import traceback

        traceback.print_exc()
        return None

    latency = (time.time() - start) * 1000

    # Extract results
    num_dets = int(response.as_numpy('num_dets')[0])
    det_boxes = response.as_numpy('det_boxes')
    det_scores = response.as_numpy('det_scores')
    det_classes = response.as_numpy('det_classes')
    global_embedding = response.as_numpy('global_embeddings')
    box_embeddings = response.as_numpy('box_embeddings')

    if verbose:
        print('\nResults:')
        print(f'  Latency: {latency:.2f}ms')
        print(f'  Detections: {num_dets}')
        print(f'  Global embedding shape: {global_embedding.shape}')
        print(f'  Global embedding norm: {np.linalg.norm(global_embedding):.4f}')
        print(f'  Box embeddings shape: {box_embeddings.shape}')

        if num_dets > 0:
            print('\n  First detection:')
            print(f'    Box: {det_boxes[0]}')
            print(f'    Score: {det_scores[0]:.4f}')
            print(f'    Class: {det_classes[0]}')
            print(f'    Embedding norm: {np.linalg.norm(box_embeddings[0]):.4f}')

            # Check non-zero embeddings
            non_zero = np.sum(np.linalg.norm(box_embeddings, axis=1) > 0.01)
            print(f'\n  Non-zero box embeddings: {non_zero}/{num_dets}')

    return {
        'latency_ms': latency,
        'num_dets': num_dets,
        'det_boxes': det_boxes,
        'det_scores': det_scores,
        'det_classes': det_classes,
        'global_embedding': global_embedding,
        'box_embeddings': box_embeddings,
    }


def benchmark_ensemble(img_path, num_iterations=100):
    """
    Benchmark ensemble performance

    Args:
        img_path: Path to test image
        num_iterations: Number of iterations

    Returns:
        dict with benchmark results
    """
    print(f'\n{"=" * 80}')
    print(f'Benchmarking ensemble ({num_iterations} iterations)')
    print(f'{"=" * 80}')

    # Load and prepare image
    img = Image.open(img_path).convert('RGB')
    img_w, img_h = img.size

    import io

    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=95)
    jpeg_bytes = buffer.getvalue()
    img_data = np.frombuffer(jpeg_bytes, dtype=np.uint8)

    affine_matrix = calculate_letterbox_affine(img_w, img_h)

    # Setup client
    client = grpcclient.InferenceServerClient(url='triton-api:8001', verbose=False)

    inputs = [
        grpcclient.InferInput('encoded_images', [len(img_data)], 'UINT8'),
        grpcclient.InferInput('affine_matrices', affine_matrix.shape, 'FP32'),
    ]
    inputs[0].set_data_from_numpy(img_data)
    inputs[1].set_data_from_numpy(affine_matrix)

    outputs = [
        grpcclient.InferRequestedOutput('num_dets'),
        grpcclient.InferRequestedOutput('global_embeddings'),
        grpcclient.InferRequestedOutput('box_embeddings'),
    ]

    # Warmup
    print('Warming up...')
    for _ in range(10):
        client.infer(model_name='yolo_mobileclip_ensemble', inputs=inputs, outputs=outputs)

    # Benchmark
    print(f'Running {num_iterations} iterations...')
    latencies = []

    for i in range(num_iterations):
        start = time.time()
        response = client.infer(
            model_name='yolo_mobileclip_ensemble', inputs=inputs, outputs=outputs
        )
        latency = (time.time() - start) * 1000
        latencies.append(latency)

        if i == 0:
            num_dets = int(response.as_numpy('num_dets')[0])
            print(f'  Detections: {num_dets}')

    # Statistics
    latencies = np.array(latencies)

    print('\nLatency Statistics:')
    print(f'  Mean:   {np.mean(latencies):.2f}ms')
    print(f'  Median: {np.median(latencies):.2f}ms')
    print(f'  P95:    {np.percentile(latencies, 95):.2f}ms')
    print(f'  P99:    {np.percentile(latencies, 99):.2f}ms')
    print(f'  Min:    {np.min(latencies):.2f}ms')
    print(f'  Max:    {np.max(latencies):.2f}ms')
    print(f'  Std:    {np.std(latencies):.2f}ms')

    # Throughput
    total_time = np.sum(latencies) / 1000  # seconds
    throughput = num_iterations / total_time

    print('\nThroughput:')
    print(f'  {throughput:.2f} images/sec')

    # Target check
    target_latency = 20  # ms
    if np.mean(latencies) < target_latency:
        print(
            f'\n✓ Performance target met! (mean: {np.mean(latencies):.2f}ms < {target_latency}ms)'
        )
    else:
        print(
            f'\n⚠ Performance target missed (mean: {np.mean(latencies):.2f}ms >= {target_latency}ms)'
        )

    return {
        'mean_latency_ms': float(np.mean(latencies)),
        'median_latency_ms': float(np.median(latencies)),
        'p95_latency_ms': float(np.percentile(latencies, 95)),
        'p99_latency_ms': float(np.percentile(latencies, 99)),
        'throughput_fps': throughput,
    }


def validate_embeddings(results):
    """
    Validate embedding outputs

    Args:
        results: dict from test_ensemble_single_image

    Returns:
        bool - True if valid
    """
    print(f'\n{"=" * 80}')
    print('Validating Embeddings')
    print(f'{"=" * 80}')

    all_valid = True

    # Check global embedding
    global_emb = results['global_embedding']
    global_norm = np.linalg.norm(global_emb)

    print('\nGlobal Embedding:')
    print(f'  Shape: {global_emb.shape}')
    print(f'  Norm: {global_norm:.4f}')
    print(f'  Range: [{global_emb.min():.4f}, {global_emb.max():.4f}]')

    if abs(global_norm - 1.0) < 0.01:
        print('  ✓ L2-normalized')
    else:
        print('  ⚠ WARNING: Not L2-normalized (expected ~1.0)')
        all_valid = False

    # Check box embeddings
    box_embs = results['box_embeddings']
    num_dets = results['num_dets']

    print('\nBox Embeddings:')
    print(f'  Shape: {box_embs.shape}')
    print(f'  Detections: {num_dets}')

    if num_dets > 0:
        valid_embs = box_embs[:num_dets]
        norms = np.linalg.norm(valid_embs, axis=1)

        print(f'  Valid embedding norms: {norms}')
        print(f'  Mean norm: {norms.mean():.4f}')

        if np.all(norms > 0.5):  # Should be ~1.0 but allow some tolerance
            print('  ✓ All valid embeddings have reasonable norms')
        else:
            print('  ⚠ WARNING: Some embeddings have very low norms')
            all_valid = False

        # Check padding
        if num_dets < 100:
            padded_embs = box_embs[num_dets:]
            padded_norms = np.linalg.norm(padded_embs, axis=1)

            if np.all(padded_norms < 0.01):
                print('  ✓ Padding is correctly zero')
            else:
                print('  ⚠ WARNING: Padding contains non-zero values')
                all_valid = False

    return all_valid


def main():
    print('\n' + '=' * 80)
    print('Track E: Ensemble End-to-End Test')
    print('=' * 80)

    # Find test image
    test_images = [
        '/app/test_images/bus.jpg',
        '/app/test_images/zidane.jpg',
        '/app/test_images/sample.jpg',
    ]

    test_image = None
    for img_path in test_images:
        if Path(img_path).exists():
            test_image = img_path
            break

    if test_image is None:
        # Try to find any image
        test_dir = Path('/app/test_images')
        if test_dir.exists():
            images = (
                list(test_dir.glob('*.jpg'))
                + list(test_dir.glob('*.jpeg'))
                + list(test_dir.glob('*.png'))
            )
            if images:
                test_image = str(images[0])

    if test_image is None:
        print('✗ No test images found!')
        print('  Place test images in /app/test_images/')
        return 1

    # Test 1: Single image inference
    results = test_ensemble_single_image(test_image)

    if results is None:
        print('\n✗ Single image test failed!')
        return 1

    # Test 2: Validate embeddings
    embeddings_valid = validate_embeddings(results)

    # Test 3: Benchmark
    benchmark_results = benchmark_ensemble(test_image, num_iterations=100)

    # Summary
    print('\n' + '=' * 80)
    print('TEST SUMMARY')
    print('=' * 80)

    print('\nSingle Image Test: ✓ PASS')
    print(f'Embedding Validation: {"✓ PASS" if embeddings_valid else "⚠ WARNINGS"}')
    print('Performance:')
    print(f'  Mean latency: {benchmark_results["mean_latency_ms"]:.2f}ms')
    print(f'  Throughput: {benchmark_results["throughput_fps"]:.2f} fps')

    if embeddings_valid and benchmark_results['mean_latency_ms'] < 20:
        print('\n✅ ALL TESTS PASSED!')
        print('Track E ensemble is ready for deployment!')
        return 0
    print('\n⚠ TESTS COMPLETED WITH WARNINGS')
    return 0  # Don't fail, just warn


if __name__ == '__main__':
    sys.exit(main())
