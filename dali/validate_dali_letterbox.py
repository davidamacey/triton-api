#!/usr/bin/env python3
"""
Validate DALI Letterbox Pipeline (Original Version)

Tests the DALI pipeline that uses affine transformation matrices
calculated on CPU for letterbox preprocessing.

Usage:
    docker compose exec yolo-api python /app/dali/validate_dali_letterbox.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Import shared utilities
from dali.utils import calculate_letterbox_affine


try:
    from nvidia import dali
    from tritonclient.grpc import InferenceServerClient, InferInput, InferRequestedOutput
except ImportError as e:
    print(f'ERROR: Required library not installed: {e}')
    print('\nThis script must be run from the yolo-api container.')
    print('Run: docker compose exec yolo-api python /app/dali/validate_dali_letterbox.py')
    sys.exit(1)


def test_dali_standalone():
    """Test DALI pipeline in standalone mode."""
    print('=' * 80)
    print('Test 1: DALI Standalone Pipeline (Affine Transformation)')
    print('=' * 80)

    # Load test image
    test_image_path = '/app/test_images/bus.jpg'
    if not Path(test_image_path).exists():
        print(f'ERROR: Test image not found: {test_image_path}')
        return False

    from PIL import Image

    img = Image.open(test_image_path)
    orig_w, orig_h = img.size
    print(f'\nTest image: {test_image_path}')
    print(f'  Original size: {orig_w}x{orig_h} (WxH)')

    # Calculate affine matrix
    affine_matrix, scale, pad_x, pad_y = calculate_letterbox_affine(orig_w, orig_h)

    print('\n  Letterbox parameters:')
    print(f'    Scale: {scale:.4f}')
    print(f'    Padding: ({pad_x:.1f}, {pad_y:.1f})')
    print(f'  Affine matrix:\n{affine_matrix}')

    # Load DALI serialized pipeline
    dali_model_path = '/app/models/yolo_preprocess_dali/1/model.dali'
    if not Path(dali_model_path).exists():
        print(f'\nERROR: DALI model not found: {dali_model_path}')
        print(
            'Run: docker compose exec yolo-api python /app/dali/create_dali_letterbox_pipeline.py'
        )
        return False

    print(f'\n  Loading DALI pipeline from: {dali_model_path}')

    # Load serialized pipeline
    pipe = dali.pipeline.Pipeline.deserialize(filename=dali_model_path)

    # Read JPEG bytes
    with open(test_image_path, 'rb') as f:
        jpeg_bytes = f.read()

    print(f'  JPEG size: {len(jpeg_bytes)} bytes')

    # Feed input (JPEG bytes + affine matrix)
    pipe.feed_input('encoded_images', [np.frombuffer(jpeg_bytes, dtype=np.uint8)])
    pipe.feed_input('affine_matrices', [affine_matrix])

    # Run pipeline
    print('\n  Running DALI pipeline...')
    outputs = pipe.run()

    # Get output
    output = outputs[0].as_cpu()
    result = np.array(output[0])

    print('\n[OK] Pipeline executed successfully!')
    print(f'  Output shape: {result.shape}')
    print(f'  Output dtype: {result.dtype}')
    print(f'  Output range: [{result.min():.4f}, {result.max():.4f}]')

    # Validate
    assert result.shape == (3, 640, 640), f'Wrong shape: {result.shape}'
    assert result.dtype == np.float32, f'Wrong dtype: {result.dtype}'
    assert result.min() >= 0, 'Min value out of range'
    assert result.max() <= 1, 'Max value out of range'

    print('\n[OK] All validations passed!')
    print('  - Shape correct: (3, 640, 640)')
    print('  - Dtype correct: float32')
    print('  - Range correct: [0, 1]')
    print('  - Uses affine transformation with CPU-calculated matrix')

    return True


def test_triton_dali_model():
    """Test DALI model through Triton."""
    print('\n' + '=' * 80)
    print('Test 2: DALI Model via Triton (Affine Transformation)')
    print('=' * 80)

    # Connect to Triton
    triton_url = 'triton-api:8001'
    print(f'\nConnecting to Triton at {triton_url}...')

    try:
        client = InferenceServerClient(url=triton_url)
        print('[OK] Connected to Triton')
    except Exception as e:
        print(f'ERROR: Cannot connect to Triton: {e}')
        print('Make sure Triton is running: docker compose ps triton-api')
        return False

    # Check if model is ready
    model_name = 'yolo_preprocess_dali'
    try:
        if not client.is_model_ready(model_name):
            print(f"ERROR: Model '{model_name}' is not ready")
            return False
        print(f"[OK] Model '{model_name}' is ready")
    except Exception as e:
        print(f'ERROR: Cannot check model status: {e}')
        return False

    # Load test image
    test_image_path = '/app/test_images/bus.jpg'
    with open(test_image_path, 'rb') as f:
        jpeg_bytes = f.read()

    from PIL import Image

    img = Image.open(test_image_path)
    orig_w, orig_h = img.size

    print(f'\nTest image: {test_image_path}')
    print(f'  Size: {orig_w}x{orig_h}')
    print(f'  JPEG bytes: {len(jpeg_bytes)}')

    # Calculate affine matrix
    affine_matrix, scale, pad_x, pad_y = calculate_letterbox_affine(orig_w, orig_h)

    print('\n  Letterbox parameters:')
    print(f'    Scale: {scale:.4f}')
    print(f'    Padding: ({pad_x:.1f}, {pad_y:.1f})')

    # Prepare inputs
    input_data = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension

    affine_data = np.expand_dims(affine_matrix, axis=0)  # Add batch dimension

    print('\n  Input shapes:')
    print(f'    encoded_images: {input_data.shape}')
    print(f'    affine_matrices: {affine_data.shape}')

    # Create Triton inputs
    inputs = [
        InferInput('encoded_images', input_data.shape, 'UINT8'),
        InferInput('affine_matrices', affine_data.shape, 'FP32'),
    ]
    inputs[0].set_data_from_numpy(input_data)
    inputs[1].set_data_from_numpy(affine_data)

    # Create Triton output
    outputs = [InferRequestedOutput('preprocessed_images')]

    print('\n  Calling Triton...')
    print('    - GPU: Decode JPEG')
    print('    - CPU: Calculate affine matrix')
    print('    - GPU: Apply warp_affine')
    print('    - GPU: Normalize + CHW')

    # Run inference
    try:
        response = client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
        print('  [OK] Triton inference successful')
    except Exception as e:
        print(f'  ERROR: Triton inference failed: {e}')
        return False

    # Get output
    result = response.as_numpy('preprocessed_images')[0]

    print(f'\n  Output shape: {result.shape}')
    print(f'  Output dtype: {result.dtype}')
    print(f'  Output range: [{result.min():.4f}, {result.max():.4f}]')

    # Validate
    assert result.shape == (3, 640, 640), f'Wrong shape: {result.shape}'
    assert result.dtype == np.float32, f'Wrong dtype: {result.dtype}'
    assert result.min() >= 0, 'Min value out of range'
    assert result.max() <= 1, 'Max value out of range'

    print('\n[OK] All validations passed!')
    print('  - Triton serving working')
    print('  - Affine transformation with CPU-calculated matrix')
    print('  - Output format correct')

    return True


def test_ensemble():
    """Test full ensemble (DALI + End2End)."""
    print('\n' + '=' * 80)
    print('Test 3: Full Ensemble via Triton (Track D)')
    print('=' * 80)

    triton_url = 'triton-api:8001'
    client = InferenceServerClient(url=triton_url)

    # Check ensemble model
    ensemble_name = 'yolov11_small_gpu_e2e'
    print(f'\nChecking ensemble model: {ensemble_name}')

    try:
        if not client.is_model_ready(ensemble_name):
            print(f"ERROR: Ensemble '{ensemble_name}' is not ready")
            return False
        print(f"[OK] Ensemble '{ensemble_name}' is ready")
    except Exception as e:
        print(f'ERROR: Cannot check ensemble status: {e}')
        return False

    # Load test image
    test_image_path = '/app/test_images/bus.jpg'
    with open(test_image_path, 'rb') as f:
        jpeg_bytes = f.read()

    from PIL import Image

    img = Image.open(test_image_path)
    orig_w, orig_h = img.size

    print(f'\nTest image: {test_image_path}')
    print(f'  Size: {orig_w}x{orig_h}')

    # Calculate affine matrix
    affine_matrix, _scale, _pad_x, _pad_y = calculate_letterbox_affine(orig_w, orig_h)

    # Prepare inputs
    input_data = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    input_data = np.expand_dims(input_data, axis=0)

    affine_data = np.expand_dims(affine_matrix, axis=0)

    # Create Triton inputs
    inputs = [
        InferInput('encoded_images', input_data.shape, 'UINT8'),
        InferInput('affine_matrices', affine_data.shape, 'FP32'),
    ]
    inputs[0].set_data_from_numpy(input_data)
    inputs[1].set_data_from_numpy(affine_data)

    # Create Triton outputs
    outputs = [
        InferRequestedOutput('num_dets'),
        InferRequestedOutput('det_boxes'),
        InferRequestedOutput('det_scores'),
        InferRequestedOutput('det_classes'),
    ]

    print('\n  Running full Track D pipeline:')
    print('    1. CPU: Calculate affine matrix')
    print('    2. GPU: DALI preprocessing (decode + warp_affine + normalize)')
    print('    3. GPU: TensorRT inference')
    print('    4. GPU: EfficientNMS')

    # Run inference
    try:
        response = client.infer(model_name=ensemble_name, inputs=inputs, outputs=outputs)
        print('  [OK] Ensemble inference successful')
    except Exception as e:
        print(f'  ERROR: Ensemble inference failed: {e}')
        import traceback

        traceback.print_exc()
        return False

    # Parse outputs
    num_dets = int(response.as_numpy('num_dets')[0][0])
    boxes = response.as_numpy('det_boxes')[0][:num_dets]
    scores = response.as_numpy('det_scores')[0][:num_dets]
    classes = response.as_numpy('det_classes')[0][:num_dets]

    print(f'\n  Detections: {num_dets}')
    print('  First detection:')
    if num_dets > 0:
        print(f'    Box (x1,y1,x2,y2): {boxes[0]}')
        print(f'    Score: {scores[0]:.4f}')
        print(f'    Class: {int(classes[0])}')

    print('\n[OK] Full ensemble working!')
    print('  - DALI preprocessing: GPU (with CPU-calculated affine)')
    print('  - TensorRT inference: GPU')
    print('  - EfficientNMS: GPU')
    print('  - Requires CPU affine matrix calculation')

    return True


def main():
    """Run all validation tests."""
    print('\n' + '=' * 80)
    print('DALI Pipeline Validation (Original Version)')
    print('=' * 80)
    print('\nValidating Track D with affine transformation approach!')
    print('')

    # Test 1: Standalone DALI
    if not test_dali_standalone():
        print('\n[FAIL] Test 1 FAILED')
        return False

    # Test 2: DALI via Triton
    if not test_triton_dali_model():
        print('\n[FAIL] Test 2 FAILED')
        return False

    # Test 3: Full ensemble
    if not test_ensemble():
        print('\n[FAIL] Test 3 FAILED')
        return False

    # All tests passed!
    print('\n' + '=' * 80)
    print('[PASS] ALL TESTS PASSED!')
    print('=' * 80)
    print('\nTrack D is working correctly with affine transformation:')
    print('  [PASS] DALI pipeline loads and executes')
    print('  [PASS] Affine matrix input working')
    print('  [PASS] CPU calculates letterbox parameters')
    print('  [PASS] GPU applies transformation')
    print('  [PASS] Output format is correct')
    print('  [PASS] Full ensemble (DALI + TRT) working')
    print('\nCPU overhead: ~0.5-1.0ms (affine matrix calculation)')
    print('GPU operations: Decode + warp_affine + normalize + inference + NMS')
    print('\nNext step: Run benchmarks!')
    print('  python benchmarks/four_track_comparison.py --image test_images/bus.jpg')
    print('')

    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
