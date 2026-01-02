#!/usr/bin/env python3
"""
Client-side Multi-Index Ingestion

Standalone script that ingests images via HTTP API calls.
No container dependencies - runs from host machine.

Usage:
    # Normal ingestion (skips duplicates)
    python scripts/client_ingest.py \
        --images-dir /mnt/nvm/KILLBOY_SAMPLE_PICTURES \
        --max-images 500

    # Benchmark mode (processes same images repeatedly for speed testing)
    python scripts/client_ingest.py \
        --images-dir /mnt/nvm/KILLBOY_SAMPLE_PICTURES \
        --max-images 100 \
        --benchmark

    python scripts/client_ingest.py \
        --images-dir /mnt/nvm/repos/triton-api/test_images/faces/lfw-deepfunneled \
        --max-images 200
"""

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
API_BASE = 'http://localhost:4603'
MAX_WORKERS = 8


def find_images(directory: Path, max_images: int = 1000) -> list[Path]:
    """Recursively find image files."""
    extensions = {'.jpg', '.jpeg', '.png', '.webp', '.gif'}
    images = []

    for root, _, files in os.walk(directory):
        for f in files:
            if Path(f).suffix.lower() in extensions:
                images.append(Path(root) / f)
                if len(images) >= max_images:
                    return sorted(images)

    return sorted(images)


def process_image(image_path: Path, skip_duplicates: bool = True) -> dict:
    """Process single image through all pipelines."""
    result = {
        'path': str(image_path),
        'success': False,
        'duplicate': False,
        'yolo_detections': 0,
        'faces': 0,
        'global_ingested': False,
        'faces_ingested': 0,
        'error': None,
    }

    try:
        with open(image_path, 'rb') as f:
            image_bytes = f.read()

        # 1. Ingest for YOLO + global embedding (visual_search_global, vehicles, people)
        ingest_resp = requests.post(
            f'{API_BASE}/track_e/ingest',
            files={'file': (image_path.name, image_bytes, 'image/jpeg')},
            params={
                'image_path': str(image_path),
                'skip_duplicates': str(skip_duplicates).lower(),
            },
            timeout=60,
        )

        if ingest_resp.status_code == 200:
            ingest_data = ingest_resp.json()
            if ingest_data.get('status') == 'duplicate':
                result['duplicate'] = True
                result['success'] = True  # Still counts as "processed"
            else:
                result['yolo_detections'] = ingest_data.get('num_detections', 0)
                result['global_ingested'] = True
        else:
            result['error'] = f'Ingest failed: {ingest_resp.status_code}'

        # 2. Face detection + embeddings ingestion
        face_ingest_resp = requests.post(
            f'{API_BASE}/track_e/faces/ingest',
            files={'image': (image_path.name, image_bytes, 'image/jpeg')},
            params={'image_path': str(image_path)},
            timeout=60,
        )

        if face_ingest_resp.status_code == 200:
            face_data = face_ingest_resp.json()
            result['faces'] = face_data.get('num_faces', 0)
            result['faces_ingested'] = face_data.get('indexed', 0)

        result['success'] = result['global_ingested']

    except requests.exceptions.Timeout:
        result['error'] = 'Request timeout'
    except Exception as e:
        result['error'] = str(e)

    return result


def main(args, skip_duplicates: bool = True):
    print('=' * 60)
    print('Multi-Index Image Ingestion (Unified Pipeline)')
    print('=' * 60)
    print('Face detection: Person-crop only (optimized)')
    if not skip_duplicates:
        print('⚠️  Benchmark mode: duplicate detection DISABLED')
    print('=' * 60)

    # Check API health
    try:
        health = requests.get(f'{API_BASE}/health', timeout=5)
        if health.status_code != 200:
            print(f'ERROR: API not healthy at {API_BASE}')
            return 1
        print(f'✓ API healthy at {API_BASE}')
    except Exception as e:
        print(f'ERROR: Cannot connect to API: {e}')
        return 1

    # Find images
    images_dir = Path(args.images_dir)
    if not images_dir.exists():
        print(f'ERROR: Directory not found: {images_dir}')
        return 1

    print(f'\nScanning {images_dir}...')
    images = find_images(images_dir, args.max_images)
    print(f'Found {len(images)} images')

    if not images:
        print('No images to process')
        return 1

    # Optionally recreate indexes
    if args.recreate_indexes:
        print('\nRecreating indexes...')
        resp = requests.post(f'{API_BASE}/track_e/index/create', params={'force_recreate': True})
        if resp.status_code == 200:
            print('✓ Indexes recreated')
        else:
            print(f'⚠ Index creation returned: {resp.status_code}')

    # Process images
    print(f'\nIngesting {len(images)} images with {args.workers} workers...')
    start_time = time.time()

    results = {
        'total': len(images),
        'success': 0,
        'failed': 0,
        'duplicates': 0,
        'total_detections': 0,
        'total_faces': 0,
        'faces_indexed': 0,
    }

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_image, img, skip_duplicates): img for img in images}

        for i, future in enumerate(as_completed(futures)):
            result = future.result()

            if result['duplicate']:
                results['duplicates'] += 1
            elif result['success']:
                results['success'] += 1
                results['total_detections'] += result['yolo_detections']
                results['total_faces'] += result['faces']
                results['faces_indexed'] += result['faces_ingested']
            else:
                results['failed'] += 1
                if result['error']:
                    logger.debug(f'Failed: {result["path"]} - {result["error"]}')

            # Progress update every 50 images
            if (i + 1) % 50 == 0 or (i + 1) == len(images):
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                print(
                    f'  Progress: {i + 1}/{len(images)} ({rate:.1f} img/s) - '
                    f'Detections: {results["total_detections"]}, '
                    f'Faces: {results["total_faces"]}'
                )

    elapsed = time.time() - start_time

    # Summary
    print('\n' + '=' * 60)
    print('INGESTION COMPLETE')
    print('=' * 60)
    print(f'Images processed: {results["total"]}')
    print(f'  New images indexed: {results["success"]}')
    print(f'  Duplicates skipped: {results["duplicates"]}')
    print(f'  Failed: {results["failed"]}')
    print(f'Total YOLO detections: {results["total_detections"]}')
    print(f'Total faces detected: {results["total_faces"]}')
    print(f'Faces indexed: {results["faces_indexed"]}')
    print(f'Time: {elapsed:.1f}s ({results["total"] / elapsed:.1f} img/s)')

    # Get index stats
    print('\nIndex Statistics:')
    try:
        stats = requests.get(f'{API_BASE}/track_e/index/stats', timeout=10)
        if stats.status_code == 200:
            data = stats.json()
            for idx in data.get('indexes', []):
                print(f'  {idx["name"]}: {idx.get("doc_count", 0)} documents')
    except Exception:
        pass

    return 0


def run():
    parser = argparse.ArgumentParser(description='Client-side image ingestion')
    parser.add_argument('--images-dir', type=str, required=True, help='Directory containing images')
    parser.add_argument('--max-images', type=int, default=500, help='Maximum images to process')
    parser.add_argument('--workers', type=int, default=8, help='Number of parallel workers')
    parser.add_argument(
        '--recreate-indexes', action='store_true', help='Recreate indexes before ingestion'
    )
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Benchmark mode: disable duplicate detection to allow repeated processing of same images',
    )

    args = parser.parse_args()

    # In benchmark mode, skip_duplicates=False allows processing same images repeatedly
    skip_duplicates = not args.benchmark

    return main(args, skip_duplicates=skip_duplicates)


if __name__ == '__main__':
    sys.exit(run())
