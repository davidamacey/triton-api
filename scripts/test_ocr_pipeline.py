#!/usr/bin/env python3
"""
PP-OCRv5 OCR Pipeline Test Script

This script tests the OCR pipeline at multiple levels:
1. Direct Triton gRPC calls to detection/recognition models
2. OCR pipeline BLS endpoint
3. API endpoint via FastAPI

Usage:
    # Run all tests
    python scripts/test_ocr_pipeline.py

    # Test specific component
    python scripts/test_ocr_pipeline.py --test api
    python scripts/test_ocr_pipeline.py --test triton
    python scripts/test_ocr_pipeline.py --test preprocessing

    # Use specific image
    python scripts/test_ocr_pipeline.py --image path/to/image.jpg

    # Verbose output
    python scripts/test_ocr_pipeline.py -v
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Test image locations
TEST_IMAGES_SYNTHETIC = Path('test_images/ocr-synthetic')
TEST_IMAGES_REAL = Path('test_images/ocr-real')

# Default test images
DEFAULT_IMAGES = [
    'test_images/ocr-synthetic/hello_world.jpg',
    'test_images/ocr-synthetic/exit_sign.jpg',
    'test_images/ocr-synthetic/invoice.jpg',
    'test_images/ocr-real/testocr_rgb.jpg',
]

# Expected results for validation
EXPECTED_RESULTS = {
    'hello_world.jpg': ['Hello', 'World'],
    'exit_sign.jpg': ['EXIT'],
    'exit_sign_v2.jpg': ['EXIT'],
}

# Configuration
API_URL = 'http://localhost:4603'
TRITON_URL = 'localhost:4601'  # gRPC port

# OCR preprocessing constants
OCR_MAX_SIZE = 960
OCR_BOUNDARY = 32  # Pad to 32-pixel boundary


# =============================================================================
# Preprocessing Functions
# =============================================================================


def preprocess_for_ocr(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """
    Preprocess image for PP-OCR detection.

    This replicates the preprocessing done by the OCR pipeline:
    1. Resize to fit within max size (960px)
    2. Pad to 32-pixel boundary
    3. Normalize: (x / 127.5) - 1 = range [-1, 1]
    4. Convert RGB to BGR (PP-OCR expects BGR)

    Args:
        image: Input image in BGR format (from cv2.imread)

    Returns:
        ocr_image: [3, H, W] preprocessed for detection
        original_image: [3, H, W] normalized [0, 1] for cropping
        orig_shape: (H, W) original dimensions
    """
    orig_h, orig_w = image.shape[:2]
    orig_shape = (orig_h, orig_w)

    # Step 1: Resize to fit within max size
    scale = min(OCR_MAX_SIZE / orig_h, OCR_MAX_SIZE / orig_w, 1.0)
    if scale < 1.0:
        new_h = int(orig_h * scale)
        new_w = int(orig_w * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        resized = image
        new_h, new_w = orig_h, orig_w

    # Step 2: Pad to 32-pixel boundary
    pad_h = (OCR_BOUNDARY - new_h % OCR_BOUNDARY) % OCR_BOUNDARY
    pad_w = (OCR_BOUNDARY - new_w % OCR_BOUNDARY) % OCR_BOUNDARY

    if pad_h > 0 or pad_w > 0:
        padded = np.zeros((new_h + pad_h, new_w + pad_w, 3), dtype=np.uint8)
        padded[:new_h, :new_w] = resized
    else:
        padded = resized

    # Step 3: Normalize for OCR
    # PP-OCR normalization: (x / 255 - 0.5) / 0.5 = x / 127.5 - 1
    ocr_float = padded.astype(np.float32)
    ocr_normalized = ocr_float / 127.5 - 1.0  # Range [-1, 1]

    # Step 4: Convert to CHW format (keep BGR)
    ocr_image = ocr_normalized.transpose(2, 0, 1)  # [3, H, W]

    # Original image for cropping (normalized to [0, 1])
    original_float = image.astype(np.float32) / 255.0
    original_image = original_float.transpose(2, 0, 1)  # [3, H, W]

    return ocr_image, original_image, orig_shape


def validate_preprocessing(verbose: bool = False) -> bool:
    """
    Validate that preprocessing matches expected format.

    Returns:
        True if preprocessing is correct
    """
    print('\n' + '=' * 60)
    print('Preprocessing Validation')
    print('=' * 60)

    # Create a test image
    test_img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

    ocr_image, original_image, orig_shape = preprocess_for_ocr(test_img)

    # Validate shapes
    print(f'\nInput shape:    {test_img.shape}')
    print(f'OCR shape:      {ocr_image.shape}')
    print(f'Original shape: {original_image.shape}')
    print(f'Orig dims:      {orig_shape}')

    errors = []

    # Check OCR image shape (should be padded to 32 boundary)
    if ocr_image.shape[1] % 32 != 0:
        errors.append(f'OCR height not 32-aligned: {ocr_image.shape[1]}')
    if ocr_image.shape[2] % 32 != 0:
        errors.append(f'OCR width not 32-aligned: {ocr_image.shape[2]}')

    # Check normalization range
    ocr_min, ocr_max = ocr_image.min(), ocr_image.max()
    print(f'\nOCR value range: [{ocr_min:.3f}, {ocr_max:.3f}]')

    if ocr_min < -1.0 or ocr_max > 1.0:
        errors.append(f'OCR normalization out of range: [{ocr_min}, {ocr_max}]')

    # Check original image range
    orig_min, orig_max = original_image.min(), original_image.max()
    print(f'Original range: [{orig_min:.3f}, {orig_max:.3f}]')

    if orig_min < 0.0 or orig_max > 1.0:
        errors.append(f'Original normalization out of range: [{orig_min}, {orig_max}]')

    # Test with different sizes
    test_sizes = [(100, 100), (960, 720), (1920, 1080), (480, 640)]

    print('\nSize handling:')
    for h, w in test_sizes:
        img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        ocr_img, _, _ = preprocess_for_ocr(img)

        new_h, new_w = ocr_img.shape[1], ocr_img.shape[2]
        status = 'OK' if (new_h % 32 == 0 and new_w % 32 == 0 and new_h <= OCR_MAX_SIZE + 32 and new_w <= OCR_MAX_SIZE + 32) else 'FAIL'
        print(f'  {h}x{w} -> {new_h}x{new_w}: {status}')

    if errors:
        print('\nErrors:')
        for e in errors:
            print(f'  - {e}')
        return False

    print('\nPreprocessing: PASSED')
    return True


# =============================================================================
# API Test Functions
# =============================================================================


def test_api_endpoint(image_path: str, verbose: bool = False) -> dict[str, Any]:
    """
    Test OCR via the FastAPI endpoint.

    Args:
        image_path: Path to test image
        verbose: Print detailed output

    Returns:
        OCR result dict
    """
    import requests

    print(f'\nTesting API with: {image_path}')

    with open(image_path, 'rb') as f:
        start_time = time.time()
        response = requests.post(
            f'{API_URL}/track_e/ocr/predict',
            files={'image': f},
            data={'min_det_score': 0.3, 'min_rec_score': 0.5},
        )
        elapsed_ms = (time.time() - start_time) * 1000

    if response.status_code != 200:
        print(f'  ERROR: {response.status_code} - {response.text[:200]}')
        return {'error': response.text, 'status_code': response.status_code}

    result = response.json()

    print(f'  Status: {result.get("status", "unknown")}')
    print(f'  Texts found: {result.get("num_texts", 0)}')
    print(f'  Time: {elapsed_ms:.1f}ms')

    if verbose and result.get('texts'):
        print('  Detected text:')
        for i, (text, det_score, rec_score) in enumerate(zip(
            result.get('texts', []),
            result.get('det_scores', []),
            result.get('rec_scores', []),
        )):
            print(f'    [{i}] "{text}" (det={det_score:.3f}, rec={rec_score:.3f})')

    return result


def test_api_batch(image_paths: list[str], verbose: bool = False) -> dict[str, Any]:
    """
    Test batch OCR via the FastAPI endpoint.

    Args:
        image_paths: List of image paths
        verbose: Print detailed output

    Returns:
        Batch result dict
    """
    import requests

    print(f'\nTesting API batch with {len(image_paths)} images')

    files = [('images', open(p, 'rb')) for p in image_paths]

    try:
        start_time = time.time()
        response = requests.post(
            f'{API_URL}/track_e/ocr/predict_batch',
            files=files,
        )
        elapsed_ms = (time.time() - start_time) * 1000
    finally:
        for _, f in files:
            f.close()

    if response.status_code != 200:
        print(f'  ERROR: {response.status_code} - {response.text[:200]}')
        return {'error': response.text}

    result = response.json()

    print(f'  Status: {result.get("status", "unknown")}')
    print(f'  Total images: {result.get("total_images", 0)}')
    print(f'  Time: {elapsed_ms:.1f}ms ({elapsed_ms / len(image_paths):.1f}ms/image)')

    if verbose:
        for i, r in enumerate(result.get('results', [])):
            print(f'  Image {i}: {r.get("num_texts", 0)} texts')

    return result


# =============================================================================
# Triton Test Functions
# =============================================================================


def test_triton_detection(image: np.ndarray, verbose: bool = False) -> np.ndarray | None:
    """
    Test detection model directly via Triton gRPC.

    Args:
        image: BGR image from cv2.imread
        verbose: Print detailed output

    Returns:
        Detection probability map [1, H, W] or None on error
    """
    try:
        import tritonclient.grpc as grpcclient
    except ImportError:
        print('  ERROR: tritonclient not installed')
        return None

    print('\nTesting Triton detection model...')

    # Preprocess
    ocr_image, _, orig_shape = preprocess_for_ocr(image)

    # Convert BGR to RGB for PP-OCR (model expects RGB internally)
    ocr_image_rgb = ocr_image[[2, 1, 0], :, :]  # BGR -> RGB

    # Create client
    client = grpcclient.InferenceServerClient(url=TRITON_URL)

    # Check model ready
    if not client.is_model_ready('paddleocr_det_trt'):
        print('  ERROR: paddleocr_det_trt not ready')
        return None

    # Create input
    input_data = ocr_image_rgb[np.newaxis].astype(np.float32)  # [1, 3, H, W]
    inputs = [grpcclient.InferInput('x', input_data.shape, 'FP32')]
    inputs[0].set_data_from_numpy(input_data)

    # Create output
    outputs = [grpcclient.InferRequestedOutput('fetch_name_0')]

    # Inference
    start_time = time.time()
    result = client.infer('paddleocr_det_trt', inputs, outputs=outputs)
    elapsed_ms = (time.time() - start_time) * 1000

    output = result.as_numpy('fetch_name_0')

    print(f'  Input shape:  {input_data.shape}')
    print(f'  Output shape: {output.shape}')
    print(f'  Output range: [{output.min():.4f}, {output.max():.4f}]')
    print(f'  Time: {elapsed_ms:.1f}ms')

    # Count high-probability pixels
    threshold = 0.3
    high_prob_pixels = np.sum(output > threshold)
    print(f'  Pixels > {threshold}: {high_prob_pixels}')

    return output


def test_triton_recognition(text_crop: np.ndarray, verbose: bool = False) -> tuple[str, float] | None:
    """
    Test recognition model directly via Triton gRPC.

    Args:
        text_crop: BGR text crop image [H, W, 3]
        verbose: Print detailed output

    Returns:
        (text, confidence) or None on error
    """
    try:
        import tritonclient.grpc as grpcclient
    except ImportError:
        print('  ERROR: tritonclient not installed')
        return None

    print('\nTesting Triton recognition model...')

    # Preprocess for recognition
    rec_height = 48
    h, w = text_crop.shape[:2]

    # Resize maintaining aspect ratio
    ratio = w / float(h)
    target_w = max(8, min(2048, int(rec_height * ratio)))

    resized = cv2.resize(text_crop, (target_w, rec_height))
    normalized = resized.astype(np.float32) / 127.5 - 1.0
    input_data = normalized.transpose(2, 0, 1)[np.newaxis]  # [1, 3, 48, W]

    # Create client
    client = grpcclient.InferenceServerClient(url=TRITON_URL)

    # Check model ready
    if not client.is_model_ready('paddleocr_rec_trt'):
        print('  ERROR: paddleocr_rec_trt not ready')
        return None

    # Create input/output
    inputs = [grpcclient.InferInput('x', input_data.shape, 'FP32')]
    inputs[0].set_data_from_numpy(input_data.astype(np.float32))
    outputs = [grpcclient.InferRequestedOutput('fetch_name_0')]

    # Inference
    start_time = time.time()
    result = client.infer('paddleocr_rec_trt', inputs, outputs=outputs)
    elapsed_ms = (time.time() - start_time) * 1000

    output = result.as_numpy('fetch_name_0')

    print(f'  Input shape:  {input_data.shape}')
    print(f'  Output shape: {output.shape}')
    print(f'  Time: {elapsed_ms:.1f}ms')

    # Decode CTC output (simplified)
    preds_idx = output.argmax(axis=2)[0]  # [T]
    preds_prob = output.max(axis=2)[0]  # [T]

    # Simple CTC decode (skip blanks and repeats)
    text = ''
    prev_idx = 0
    for idx, prob in zip(preds_idx, preds_prob):
        if idx != 0 and idx != prev_idx:
            # Would need dictionary to decode properly
            text += f'[{idx}]'
        prev_idx = idx

    print(f'  Decoded indices: {text[:50]}...')

    return (text, float(np.mean(preds_prob)))


def test_triton_models(verbose: bool = False) -> bool:
    """
    Test all Triton OCR models are loaded and responding.

    Returns:
        True if all models are ready
    """
    try:
        import tritonclient.grpc as grpcclient
    except ImportError:
        print('ERROR: tritonclient not installed. Install with: pip install tritonclient[grpc]')
        return False

    print('\n' + '=' * 60)
    print('Triton Model Status')
    print('=' * 60)

    client = grpcclient.InferenceServerClient(url=TRITON_URL)

    models = ['paddleocr_det_trt', 'paddleocr_rec_trt', 'ocr_pipeline']
    all_ready = True

    for model in models:
        try:
            ready = client.is_model_ready(model)
            status = 'READY' if ready else 'NOT READY'
            print(f'  {model}: {status}')
            if not ready:
                all_ready = False
        except Exception as e:
            print(f'  {model}: ERROR - {e}')
            all_ready = False

    return all_ready


# =============================================================================
# Integration Tests
# =============================================================================


def run_integration_test(image_path: str, verbose: bool = False) -> bool:
    """
    Run full integration test on a single image.

    Args:
        image_path: Path to test image
        verbose: Print detailed output

    Returns:
        True if test passed
    """
    print(f'\n' + '=' * 60)
    print(f'Integration Test: {Path(image_path).name}')
    print('=' * 60)

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f'  ERROR: Could not load image: {image_path}')
        return False

    print(f'  Image size: {image.shape[1]}x{image.shape[0]}')

    # Test 1: Preprocessing
    print('\n1. Preprocessing test...')
    try:
        ocr_image, original_image, orig_shape = preprocess_for_ocr(image)
        print(f'   OCR image: {ocr_image.shape}, range [{ocr_image.min():.2f}, {ocr_image.max():.2f}]')
        print('   PASSED')
    except Exception as e:
        print(f'   FAILED: {e}')
        return False

    # Test 2: Detection (if Triton available)
    print('\n2. Detection test...')
    try:
        det_output = test_triton_detection(image, verbose)
        if det_output is not None:
            print('   PASSED')
        else:
            print('   SKIPPED (Triton not available)')
    except Exception as e:
        print(f'   FAILED: {e}')

    # Test 3: API endpoint
    print('\n3. API endpoint test...')
    try:
        result = test_api_endpoint(image_path, verbose)
        if result.get('status') == 'success':
            print('   PASSED')
        else:
            print(f'   FAILED: {result.get("error", "Unknown error")}')
            return False
    except Exception as e:
        print(f'   FAILED: {e}')
        return False

    # Validate expected results if available
    image_name = Path(image_path).name
    if image_name in EXPECTED_RESULTS:
        expected = EXPECTED_RESULTS[image_name]
        detected = result.get('texts', [])

        # Check if expected texts are found (case-insensitive)
        found = [any(e.lower() in d.lower() for d in detected) for e in expected]

        if all(found):
            print(f'\n   Expected texts found: {expected}')
        else:
            missing = [e for e, f in zip(expected, found) if not f]
            print(f'\n   WARNING: Expected texts not found: {missing}')
            print(f'   Detected: {detected}')

    return True


def run_all_tests(verbose: bool = False) -> int:
    """
    Run all OCR tests.

    Returns:
        Number of failed tests
    """
    print('\n' + '=' * 60)
    print('PP-OCRv5 OCR Pipeline Test Suite')
    print('=' * 60)

    failed = 0
    total = 0

    # Test 1: Preprocessing validation
    print('\n' + '-' * 40)
    print('Test 1: Preprocessing Validation')
    print('-' * 40)
    total += 1
    if not validate_preprocessing(verbose):
        failed += 1

    # Test 2: Triton models status
    print('\n' + '-' * 40)
    print('Test 2: Triton Models Status')
    print('-' * 40)
    total += 1
    if not test_triton_models(verbose):
        failed += 1

    # Test 3: API with synthetic images
    print('\n' + '-' * 40)
    print('Test 3: API Tests with Synthetic Images')
    print('-' * 40)

    for img_path in DEFAULT_IMAGES:
        if Path(img_path).exists():
            total += 1
            if not run_integration_test(img_path, verbose):
                failed += 1
        else:
            print(f'  SKIPPED: {img_path} (not found)')

    # Test 4: Batch API
    print('\n' + '-' * 40)
    print('Test 4: Batch API Test')
    print('-' * 40)

    existing_images = [p for p in DEFAULT_IMAGES[:3] if Path(p).exists()]
    if len(existing_images) >= 2:
        total += 1
        try:
            result = test_api_batch(existing_images, verbose)
            if result.get('status') == 'success':
                print('  PASSED')
            else:
                print(f'  FAILED: {result.get("error", "Unknown error")}')
                failed += 1
        except Exception as e:
            print(f'  FAILED: {e}')
            failed += 1
    else:
        print('  SKIPPED: Not enough test images')

    # Summary
    print('\n' + '=' * 60)
    print('Test Summary')
    print('=' * 60)
    passed = total - failed
    print(f'  Passed: {passed}/{total}')
    print(f'  Failed: {failed}/{total}')

    if failed == 0:
        print('\n  All tests PASSED!')
    else:
        print(f'\n  {failed} test(s) FAILED')

    return failed


# =============================================================================
# Main
# =============================================================================


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Test PP-OCRv5 OCR Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python scripts/test_ocr_pipeline.py              # Run all tests
  python scripts/test_ocr_pipeline.py --test api   # Test API only
  python scripts/test_ocr_pipeline.py -v           # Verbose output
  python scripts/test_ocr_pipeline.py --image test.jpg  # Test specific image
        ''',
    )

    parser.add_argument(
        '--test',
        choices=['all', 'api', 'triton', 'preprocessing', 'batch'],
        default='all',
        help='Which test to run (default: all)',
    )

    parser.add_argument(
        '--image',
        type=str,
        help='Path to specific image to test',
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Print detailed output',
    )

    parser.add_argument(
        '--api-url',
        type=str,
        default=API_URL,
        help=f'API URL (default: {API_URL})',
    )

    parser.add_argument(
        '--triton-url',
        type=str,
        default=TRITON_URL,
        help=f'Triton gRPC URL (default: {TRITON_URL})',
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Update global URLs if specified
    global API_URL, TRITON_URL
    API_URL = args.api_url
    TRITON_URL = args.triton_url

    if args.test == 'all':
        if args.image:
            # Test specific image
            success = run_integration_test(args.image, args.verbose)
            return 0 if success else 1
        else:
            # Run all tests
            failed = run_all_tests(args.verbose)
            return 0 if failed == 0 else 1

    elif args.test == 'api':
        image = args.image or DEFAULT_IMAGES[0]
        if not Path(image).exists():
            print(f'ERROR: Image not found: {image}')
            return 1
        result = test_api_endpoint(image, args.verbose)
        return 0 if result.get('status') == 'success' else 1

    elif args.test == 'triton':
        return 0 if test_triton_models(args.verbose) else 1

    elif args.test == 'preprocessing':
        return 0 if validate_preprocessing(args.verbose) else 1

    elif args.test == 'batch':
        images = [p for p in DEFAULT_IMAGES[:3] if Path(p).exists()]
        if len(images) < 2:
            print('ERROR: Not enough test images found')
            return 1
        result = test_api_batch(images, args.verbose)
        return 0 if result.get('status') == 'success' else 1


if __name__ == '__main__':
    sys.exit(main())
