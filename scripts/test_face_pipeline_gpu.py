#!/usr/bin/env python3
"""
Test Face Detection Pipeline (GPU Version)

Verifies that the GPU-accelerated face pipeline produces correct detections
by testing on images with known faces.

Usage:
    docker compose exec yolo-api python /app/scripts/test_face_pipeline_gpu.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import requests


# Test configuration
API_BASE = 'http://localhost:8000'
TEST_IMAGE_DIR = Path('/app/test_images/faces')


def find_test_images(directory: Path, max_images: int = 10) -> list[Path]:
    """Find test images in directory."""
    extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    images = []

    for ext in extensions:
        images.extend(directory.rglob(f'*{ext}'))
        images.extend(directory.rglob(f'*{ext.upper()}'))

    return sorted(images)[:max_images]


def test_single_image(image_path: Path) -> dict:
    """Test face detection on a single image."""
    with open(image_path, 'rb') as f:
        image_bytes = f.read()

    # Test /track_e/faces/detect endpoint
    start = time.time()
    response = requests.post(
        f'{API_BASE}/track_e/faces/detect',
        files={'image': (image_path.name, image_bytes, 'image/jpeg')},
        timeout=30,
    )
    latency_ms = (time.time() - start) * 1000

    if response.status_code != 200:
        return {'success': False, 'error': response.text, 'latency_ms': latency_ms}

    data = response.json()
    return {
        'success': True,
        'num_faces': data.get('num_faces', 0),
        'faces': data.get('faces', []),
        'latency_ms': latency_ms,
    }


def test_face_embeddings(image_path: Path) -> dict:
    """Test face recognition (detection + embeddings)."""
    with open(image_path, 'rb') as f:
        image_bytes = f.read()

    # Test /track_e/faces/recognize endpoint
    start = time.time()
    response = requests.post(
        f'{API_BASE}/track_e/faces/recognize',
        files={'image': (image_path.name, image_bytes, 'image/jpeg')},
        timeout=30,
    )
    latency_ms = (time.time() - start) * 1000

    if response.status_code != 200:
        return {'success': False, 'error': response.text, 'latency_ms': latency_ms}

    data = response.json()

    # Verify embedding dimensions (embeddings are in separate array)
    num_faces = data.get('num_faces', 0)
    embeddings = data.get('embeddings', [])
    embedding_valid = len(embeddings) == num_faces and all(len(emb) == 512 for emb in embeddings)

    # Verify embedding normalization (should be ~1.0)
    norm_valid = all(0.99 < np.linalg.norm(emb) < 1.01 for emb in embeddings)

    return {
        'success': True,
        'num_faces': data.get('num_faces', 0),
        'embedding_dim_valid': embedding_valid,
        'embedding_norm_valid': norm_valid,
        'latency_ms': latency_ms,
    }


def validate_face_detection(result: dict, _image_path: Path) -> list[str]:
    """Validate face detection results."""
    issues = []

    if not result.get('success'):
        issues.append(f'Request failed: {result.get("error")}')
        return issues

    num_faces = result.get('num_faces', 0)
    faces = result.get('faces', [])

    # Check consistency
    if num_faces != len(faces):
        issues.append(f'num_faces ({num_faces}) != len(faces) ({len(faces)})')

    # Validate each face
    for i, face in enumerate(faces):
        # Check box format [x1, y1, x2, y2]
        box = face.get('box', [])
        if len(box) != 4:
            issues.append(f'Face {i}: Invalid box format')
            continue

        x1, y1, x2, y2 = box

        # Check box validity (normalized 0-1)
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            issues.append(f'Face {i}: Invalid box coordinates {box}')

        # Check score
        score = face.get('score', 0)
        if not (0.5 <= score <= 1.0):
            issues.append(f'Face {i}: Score {score} out of expected range')

        # Check landmarks (5 points = 10 values)
        landmarks = face.get('landmarks', [])
        if len(landmarks) != 10:
            issues.append(f'Face {i}: Invalid landmark count {len(landmarks)}')

    return issues


def main():
    print('=' * 60)
    print('Face Detection Pipeline Test (GPU Version)')
    print('=' * 60)

    # Find test images
    if not TEST_IMAGE_DIR.exists():
        print(f'Error: Test directory not found: {TEST_IMAGE_DIR}')
        print('Creating test directory...')
        TEST_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
        print(f'Please add face images to: {TEST_IMAGE_DIR}')
        return 1

    test_images = find_test_images(TEST_IMAGE_DIR, max_images=20)

    if not test_images:
        print(f'No test images found in {TEST_IMAGE_DIR}')
        return 1

    print(f'\nFound {len(test_images)} test images')

    # Test face detection
    print('\n' + '-' * 60)
    print('FACE DETECTION TEST')
    print('-' * 60)

    detection_results = []
    total_faces = 0
    total_latency = 0

    for image_path in test_images:
        result = test_single_image(image_path)
        detection_results.append((image_path, result))

        if result['success']:
            total_faces += result['num_faces']
            total_latency += result['latency_ms']

            issues = validate_face_detection(result, image_path)
            status = '✓' if not issues else '⚠'

            print(
                f'  {status} {image_path.name}: {result["num_faces"]} faces, {result["latency_ms"]:.1f}ms'
            )
            for issue in issues:
                print(f'      ⚠ {issue}')
        else:
            print(f'  ✗ {image_path.name}: FAILED - {result.get("error", "Unknown error")[:50]}')

    success_count = sum(1 for _, r in detection_results if r['success'])

    print('\nDetection Summary:')
    print(f'  Images tested: {len(test_images)}')
    print(f'  Successful: {success_count}/{len(test_images)}')
    print(f'  Total faces detected: {total_faces}')
    if success_count > 0:
        print(f'  Avg latency: {total_latency / success_count:.1f}ms')

    # Test face recognition (embeddings)
    print('\n' + '-' * 60)
    print('FACE RECOGNITION TEST (Embeddings)')
    print('-' * 60)

    embedding_results = []

    for image_path in test_images[:5]:  # Test fewer for embeddings
        result = test_face_embeddings(image_path)
        embedding_results.append((image_path, result))

        if result['success']:
            dim_ok = '✓' if result['embedding_dim_valid'] else '✗'
            norm_ok = '✓' if result['embedding_norm_valid'] else '✗'

            print(f'  {image_path.name}:')
            print(f'    Faces: {result["num_faces"]}, Latency: {result["latency_ms"]:.1f}ms')
            print(f'    Embedding dim (512): {dim_ok}')
            print(f'    Embedding norm (~1.0): {norm_ok}')
        else:
            print(f'  ✗ {image_path.name}: FAILED - {result.get("error", "Unknown")[:50]}')

    # Summary
    print('\n' + '=' * 60)
    print('TEST SUMMARY')
    print('=' * 60)

    detection_pass = all(r['success'] for _, r in detection_results)
    embedding_pass = all(
        r['success'] and r['embedding_dim_valid'] and r['embedding_norm_valid']
        for _, r in embedding_results
    )

    if detection_pass and embedding_pass:
        print('✓ All tests PASSED')
        return 0
    print('✗ Some tests FAILED')
    if not detection_pass:
        print('  - Face detection failures')
    if not embedding_pass:
        print('  - Face embedding failures')
    return 1


if __name__ == '__main__':
    sys.exit(main())
