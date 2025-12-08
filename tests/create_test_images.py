#!/usr/bin/env python3
"""
Generic Test Image Generator

Creates resized versions of an input image in various sizes for testing.
Useful for DALI letterbox validation and aspect ratio testing.

Usage:
    # Generate standard test suite
    python tests/create_test_images.py --source /path/to/image.jpg

    # Custom output directory
    python tests/create_test_images.py --source image.jpg --output ./my_tests

    # Generate specific sizes
    python tests/create_test_images.py --source image.jpg --sizes 640x640 1920x1080

    # Generate standard + custom sizes
    python tests/create_test_images.py --source image.jpg --sizes standard 800x600
"""

import argparse
import sys
from pathlib import Path

from PIL import Image


# Standard test sizes (name, width, height)
STANDARD_SIZES = [
    ('square_640x640', 640, 640),
    ('portrait_2-3_400x600', 400, 600),
    ('portrait_1-2_320x640', 320, 640),
    ('landscape_3-2_600x400', 600, 400),
    ('landscape_2-1_640x320', 640, 320),
    ('wide_16-9_1920x1080', 1920, 1080),
    ('tall_9-16_1080x1920', 1080, 1920),
    ('small_128x128', 128, 128),
]


def parse_size(size_str):
    """Parse size string like '640x480' into (width, height)."""
    try:
        w, h = size_str.lower().split('x')
        return int(w), int(h)
    except (ValueError, AttributeError) as e:
        raise ValueError(
            f'Invalid size format: {size_str}. Use WIDTHxHEIGHT (e.g., 640x480)'
        ) from e


def generate_test_image(source_img, width, height, output_path, quality=95):
    """
    Generate a test image by cropping and resizing source image.

    Args:
        source_img: PIL Image object
        width: Target width
        height: Target height
        output_path: Path to save output
        quality: JPEG quality (default 95)
    """
    # Calculate aspect ratios
    source_aspect = source_img.size[0] / source_img.size[1]
    target_aspect = width / height

    # Crop to match target aspect ratio
    if source_aspect > target_aspect:
        # Source is wider - crop width
        crop_height = source_img.size[1]
        crop_width = int(crop_height * target_aspect)
    else:
        # Source is taller - crop height
        crop_width = source_img.size[0]
        crop_height = int(crop_width / target_aspect)

    # Center crop
    left = (source_img.size[0] - crop_width) // 2
    top = (source_img.size[1] - crop_height) // 2
    right = left + crop_width
    bottom = top + crop_height

    # Crop and resize
    cropped = source_img.crop((left, top, right, bottom))
    resized = cropped.resize((width, height), Image.Resampling.LANCZOS)

    # Save
    resized.save(output_path, 'JPEG', quality=quality)


def main():
    parser = argparse.ArgumentParser(
        description='Generate test images in various sizes from a source image',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate standard test suite
  %(prog)s --source /path/to/image.jpg

  # Custom output directory
  %(prog)s --source image.jpg --output ./test_output

  # Generate specific custom sizes
  %(prog)s --source image.jpg --sizes 640x640 1920x1080 800x600

  # Mix standard + custom sizes
  %(prog)s --source image.jpg --sizes standard 800x600
        """,
    )

    parser.add_argument('--source', '-s', type=str, required=True, help='Source image file path')

    parser.add_argument(
        '--output',
        '-o',
        type=str,
        default='./test_images/generated',
        help='Output directory (default: ./test_images/generated)',
    )

    parser.add_argument(
        '--sizes',
        nargs='+',
        default=['standard'],
        help='Sizes to generate. Use "standard" for default set, or specify custom sizes like "640x480 1920x1080"',
    )

    parser.add_argument(
        '--quality', '-q', type=int, default=95, help='JPEG quality 1-100 (default: 95)'
    )

    parser.add_argument('--prefix', type=str, default='', help='Prefix for output filenames')

    args = parser.parse_args()

    # Validate source image
    source_path = Path(args.source)
    if not source_path.exists():
        print(f'Error: Source image not found: {source_path}')
        sys.exit(1)

    # Load source image
    print('=' * 60)
    print('Test Image Generator')
    print('=' * 60)
    print(f'\nLoading source image: {source_path}')
    try:
        source_img = Image.open(source_path)
    except Exception as e:
        print(f'Error loading image: {e}')
        sys.exit(1)

    print(f'  Source size: {source_img.size[0]}x{source_img.size[1]} (W×H)')

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f'  Output directory: {output_dir}')

    # Determine which sizes to generate
    test_cases = []

    if 'standard' in args.sizes:
        test_cases.extend(STANDARD_SIZES)
        print(f'\n  Using standard test suite ({len(STANDARD_SIZES)} sizes)')

    # Add custom sizes
    custom_sizes = [s for s in args.sizes if s != 'standard']
    for size_str in custom_sizes:
        try:
            w, h = parse_size(size_str)
            name = f'custom_{w}x{h}'
            test_cases.append((name, w, h))
        except ValueError as e:
            print(f'Warning: {e}, skipping')

    if not test_cases:
        print('Error: No valid sizes specified')
        sys.exit(1)

    # Generate test images
    print(f'\nGenerating {len(test_cases)} test images...')
    print('-' * 60)

    success_count = 0
    for name, width, height in test_cases:
        # Add prefix if specified
        if args.prefix:
            filename = f'{args.prefix}_{name}.jpg'
        else:
            filename = f'{name}.jpg'

        output_path = output_dir / filename

        try:
            generate_test_image(source_img, width, height, output_path, args.quality)
            print(f'  ✓ {filename} ({width}×{height})')
            success_count += 1
        except Exception as e:
            print(f'  ✗ {filename}: {e}')

    # Summary
    print('-' * 60)
    print(f'\n✓ Generated {success_count}/{len(test_cases)} images')
    print(f'  Location: {output_dir.absolute()}')
    print('\n' + '=' * 60)

    if success_count < len(test_cases):
        sys.exit(1)


if __name__ == '__main__':
    main()
