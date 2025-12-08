#!/usr/bin/env python3
"""
Resize Test Images for Performance Comparison (Multiprocessing)

Creates a resized copy of test images at specified resolution (default 640px).
Maintains aspect ratio and uses 100% JPEG quality for fair comparison testing.
Uses multiprocessing for fast processing of thousands of images.

Usage:
    python scripts/resize_images.py /path/to/images --size 640
    python scripts/resize_images.py /mnt/nvm/KILLBOY_SAMPLE_PICTURES --size 1024
    python scripts/resize_images.py /path/to/images --size 640 --workers 16
"""

import argparse
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path

from PIL import Image
from tqdm import tqdm


def resize_single_image(args_tuple):
    """
    Resize a single image (worker function for multiprocessing).

    Args:
        args_tuple: (input_path, output_path, max_size)

    Returns:
        tuple: (orig_width, orig_height, new_width, new_height, input_size, output_size)
    """
    input_path, output_path, max_size = args_tuple

    try:
        with Image.open(input_path) as orig_img:
            # Convert to RGB if needed (handles PNG with alpha, etc.)
            img = orig_img.convert('RGB') if orig_img.mode != 'RGB' else orig_img

            # Get original dimensions and file size
            width, height = img.size
            input_size = input_path.stat().st_size
            max_dim = max(width, height)

            # Skip resizing if already smaller or equal
            if max_dim <= max_size:
                img.save(output_path, format='JPEG', quality=100, optimize=True)
                output_size = output_path.stat().st_size
                return width, height, width, height, input_size, output_size

            # Calculate new dimensions maintaining aspect ratio
            if width > height:
                new_width = max_size
                new_height = int(height * (max_size / width))
            else:
                new_height = max_size
                new_width = int(width * (max_size / height))

            # Resize using high-quality Lanczos filter
            img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Save with 100% quality (no compression artifacts)
            img_resized.save(output_path, format='JPEG', quality=100, optimize=True)

            output_size = output_path.stat().st_size
            return width, height, new_width, new_height, input_size, output_size

    except Exception as e:
        print(f'\nError processing {input_path.name}: {e}', file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Resize test images for performance comparison (multiprocessing)'
    )
    parser.add_argument('input_dir', type=str, help='Input directory containing images')
    parser.add_argument(
        '--size', type=int, default=640, help='Maximum dimension in pixels (default: 640)'
    )
    parser.add_argument(
        '--output', type=str, default=None, help='Output directory (default: <input_dir>_<size>px)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help=f'Number of worker processes (default: {cpu_count()})',
    )

    args = parser.parse_args()

    # Set worker count
    num_workers = args.workers if args.workers else cpu_count()

    # Validate input directory
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist", file=sys.stderr)
        sys.exit(1)

    # Create output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(f'{input_dir}_{args.size}px')

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = [
        f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() in image_extensions
    ]

    if not image_files:
        print(f"Error: No images found in '{input_dir}'", file=sys.stderr)
        sys.exit(1)

    print(f'Resizing {len(image_files)} images to max dimension {args.size}px')
    print(f'Input:   {input_dir}')
    print(f'Output:  {output_dir}')
    print('Quality: 100 (maximum, no compression artifacts)')
    print(f'Workers: {num_workers} processes')
    print()

    # Prepare arguments for multiprocessing
    task_args = [
        (img_file, output_dir / f'{img_file.stem}.jpg', args.size) for img_file in image_files
    ]

    # Process images with multiprocessing pool and progress bar
    total_input_size = 0
    total_output_size = 0
    resized_count = 0
    skipped_count = 0

    with Pool(processes=num_workers) as pool:
        # Use imap_unordered for better performance with progress bar
        results = list(
            tqdm(
                pool.imap_unordered(resize_single_image, task_args),
                total=len(task_args),
                desc='Processing',
                unit='img',
            )
        )

    # Analyze results
    for result in results:
        if result:
            orig_w, orig_h, new_w, new_h, input_size, output_size = result
            total_input_size += input_size
            total_output_size += output_size

            if orig_w == new_w and orig_h == new_h:
                skipped_count += 1
            else:
                resized_count += 1

    # Print summary
    print()
    print('=' * 80)
    print('RESIZE SUMMARY')
    print('=' * 80)
    print(f'Total images:    {len(image_files)}')
    print(f'Resized:         {resized_count}')
    print(f'Already optimal: {skipped_count}')
    print()
    print(f'Total input size:  {total_input_size / 1024 / 1024:.1f} MB')
    print(f'Total output size: {total_output_size / 1024 / 1024:.1f} MB')
    if total_input_size > 0:
        print(f'Size reduction:    {(1 - total_output_size / total_input_size) * 100:.1f}%')
    print()
    print(f'Output directory: {output_dir}')
    print('=' * 80)


if __name__ == '__main__':
    main()
