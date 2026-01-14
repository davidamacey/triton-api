#!/usr/bin/env python3
"""
Benchmark script for batch ingestion endpoint.

Compares performance of:
1. Single image /track_e/ingest (sequential)
2. Batch /track_e/ingest_batch (parallel)

Usage:
    # Default: 100 images from test_images/
    python scripts/benchmark_batch.py

    # Custom settings
    python scripts/benchmark_batch.py \
        --images-dir /path/to/images \
        --total-images 500 \
        --batch-size 32 \
        --url http://localhost:4603

Expected results (RTX A6000):
    - Single endpoint: ~100-130 images/sec (8-10ms per image)
    - Batch endpoint:  ~300-500 images/sec (2-3ms per image)
"""

import argparse
import asyncio
import glob
import os
import sys
import time
from pathlib import Path

import httpx


async def benchmark_single(
    client: httpx.AsyncClient,
    image_paths: list[Path],
    url: str,
) -> dict:
    """Benchmark single-image ingestion (sequential)."""
    print(f'\n=== Benchmarking SINGLE image endpoint ===')
    print(f'Total images: {len(image_paths)}')

    start_time = time.time()
    processed = 0
    errors = 0

    for i, path in enumerate(image_paths):
        try:
            with open(path, 'rb') as f:
                files = {'file': (path.name, f, 'image/jpeg')}
                response = await client.post(
                    f'{url}/track_e/ingest',
                    files=files,
                    params={'skip_duplicates': 'false'},
                )
                if response.status_code == 200:
                    processed += 1
                else:
                    errors += 1
        except Exception as e:
            errors += 1
            print(f'  Error on image {i}: {e}')

        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            print(f'  Progress: {i + 1}/{len(image_paths)} ({rate:.1f} images/sec)')

    elapsed = time.time() - start_time
    rate = processed / elapsed if elapsed > 0 else 0

    return {
        'method': 'single',
        'total': len(image_paths),
        'processed': processed,
        'errors': errors,
        'elapsed_sec': round(elapsed, 2),
        'images_per_sec': round(rate, 1),
        'ms_per_image': round(1000 / rate, 2) if rate > 0 else 0,
    }


async def benchmark_batch(
    client: httpx.AsyncClient,
    image_paths: list[Path],
    url: str,
    batch_size: int = 32,
) -> dict:
    """Benchmark batch ingestion endpoint."""
    print(f'\n=== Benchmarking BATCH endpoint (batch_size={batch_size}) ===')
    print(f'Total images: {len(image_paths)}')

    start_time = time.time()
    processed = 0
    errors = 0
    batches_sent = 0

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]

        try:
            files = []
            for path in batch_paths:
                f = open(path, 'rb')
                files.append(('files', (path.name, f, 'image/jpeg')))

            response = await client.post(
                f'{url}/track_e/ingest_batch',
                files=files,
                params={'skip_duplicates': 'false'},
                timeout=120.0,  # Longer timeout for batches
            )

            # Close file handles
            for _, (_, f, _) in files:
                f.close()

            if response.status_code == 200:
                result = response.json()
                processed += result.get('processed', 0)
                errors += result.get('errors', 0)
            else:
                errors += len(batch_paths)
                print(f'  Batch error: {response.status_code} - {response.text[:200]}')

        except Exception as e:
            errors += len(batch_paths)
            print(f'  Batch exception: {e}')

        batches_sent += 1

        # Progress update
        current_total = min((batches_sent * batch_size), len(image_paths))
        elapsed = time.time() - start_time
        rate = processed / elapsed if elapsed > 0 else 0
        print(f'  Progress: {current_total}/{len(image_paths)} ({rate:.1f} images/sec)')

    elapsed = time.time() - start_time
    rate = processed / elapsed if elapsed > 0 else 0

    return {
        'method': 'batch',
        'batch_size': batch_size,
        'total': len(image_paths),
        'batches': batches_sent,
        'processed': processed,
        'errors': errors,
        'elapsed_sec': round(elapsed, 2),
        'images_per_sec': round(rate, 1),
        'ms_per_image': round(1000 / rate, 2) if rate > 0 else 0,
    }


async def main():
    parser = argparse.ArgumentParser(description='Benchmark batch ingestion')
    parser.add_argument(
        '--images-dir',
        type=str,
        default='test_images/faces/lfw-deepfunneled',
        help='Directory with test images',
    )
    parser.add_argument(
        '--total-images',
        type=int,
        default=100,
        help='Total images to process',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for batch endpoint',
    )
    parser.add_argument(
        '--url',
        type=str,
        default='http://localhost:4603',
        help='API base URL',
    )
    parser.add_argument(
        '--skip-single',
        action='store_true',
        help='Skip single-image benchmark',
    )
    parser.add_argument(
        '--skip-batch',
        action='store_true',
        help='Skip batch benchmark',
    )

    args = parser.parse_args()

    # Find images
    image_dir = Path(args.images_dir)
    if not image_dir.exists():
        print(f'Error: Directory not found: {image_dir}')
        sys.exit(1)

    # Find all image files (recursively)
    patterns = ['**/*.jpg', '**/*.jpeg', '**/*.png', '**/*.webp']
    image_paths = []
    for pattern in patterns:
        image_paths.extend(image_dir.glob(pattern))

    if not image_paths:
        print(f'Error: No images found in {image_dir}')
        sys.exit(1)

    # Limit to requested count
    image_paths = image_paths[: args.total_images]
    print(f'Found {len(image_paths)} images in {image_dir}')

    results = []

    async with httpx.AsyncClient(timeout=60.0) as client:
        # Check server health
        try:
            response = await client.get(f'{args.url}/health')
            if response.status_code != 200:
                print(f'Warning: Server health check failed: {response.status_code}')
        except Exception as e:
            print(f'Warning: Could not connect to server: {e}')

        # Run benchmarks
        if not args.skip_single:
            result = await benchmark_single(client, image_paths, args.url)
            results.append(result)
            print(f'\nSINGLE results: {result["images_per_sec"]} images/sec')

        if not args.skip_batch:
            result = await benchmark_batch(
                client, image_paths, args.url, args.batch_size
            )
            results.append(result)
            print(f'\nBATCH results: {result["images_per_sec"]} images/sec')

    # Summary
    print('\n' + '=' * 60)
    print('BENCHMARK SUMMARY')
    print('=' * 60)

    for r in results:
        print(f"\n{r['method'].upper()} method:")
        print(f"  Total images: {r['total']}")
        print(f"  Processed:    {r['processed']}")
        print(f"  Errors:       {r['errors']}")
        print(f"  Time:         {r['elapsed_sec']}s")
        print(f"  Throughput:   {r['images_per_sec']} images/sec")
        print(f"  Latency:      {r['ms_per_image']} ms/image")

    if len(results) == 2:
        single = results[0]
        batch = results[1]
        if single['images_per_sec'] > 0:
            speedup = batch['images_per_sec'] / single['images_per_sec']
            print(f'\n  SPEEDUP: {speedup:.1f}x (batch vs single)')


if __name__ == '__main__':
    asyncio.run(main())
