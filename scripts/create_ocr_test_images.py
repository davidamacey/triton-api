#!/usr/bin/env python3
"""
Create sample OCR test images with text for testing the OCR pipeline.

Generates synthetic images with various text styles and layouts.

Usage:
    python scripts/create_ocr_test_images.py
"""

import sys
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print('PIL not found. Install with: pip install pillow')
    sys.exit(1)

# Output directory
OUTPUT_DIR = Path('test_images/ocr')


def create_simple_text_image(
    text: str,
    filename: str,
    size: tuple[int, int] = (800, 200),
    bg_color: str = 'white',
    text_color: str = 'black',
    font_size: int = 48,
) -> Path:
    """Create a simple image with text."""
    img = Image.new('RGB', size, color=bg_color)
    draw = ImageDraw.Draw(img)

    # Try to use a system font, fall back to default
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', font_size)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype('/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf', font_size)
        except (OSError, IOError):
            font = ImageFont.load_default()

    # Center the text
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (size[0] - text_width) // 2
    y = (size[1] - text_height) // 2

    draw.text((x, y), text, fill=text_color, font=font)

    output_path = OUTPUT_DIR / filename
    img.save(output_path, 'JPEG', quality=95)
    return output_path


def create_multiline_text_image(
    lines: list[str],
    filename: str,
    size: tuple[int, int] = (800, 400),
    bg_color: str = 'white',
    text_color: str = 'black',
    font_size: int = 36,
) -> Path:
    """Create an image with multiple lines of text."""
    img = Image.new('RGB', size, color=bg_color)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', font_size)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype('/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf', font_size)
        except (OSError, IOError):
            font = ImageFont.load_default()

    # Calculate total height
    line_height = font_size + 10
    total_height = len(lines) * line_height
    start_y = (size[1] - total_height) // 2

    for i, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        x = (size[0] - text_width) // 2
        y = start_y + i * line_height
        draw.text((x, y), line, fill=text_color, font=font)

    output_path = OUTPUT_DIR / filename
    img.save(output_path, 'JPEG', quality=95)
    return output_path


def create_sign_image(text: str, filename: str) -> Path:
    """Create a sign-like image."""
    return create_simple_text_image(
        text=text,
        filename=filename,
        size=(600, 200),
        bg_color='#1a5f2a',  # Green background
        text_color='white',
        font_size=64,
    )


def create_document_image(lines: list[str], filename: str) -> Path:
    """Create a document-like image."""
    return create_multiline_text_image(
        lines=lines,
        filename=filename,
        size=(800, 600),
        bg_color='#fffff0',  # Ivory
        text_color='#333333',
        font_size=28,
    )


def main():
    """Create all sample OCR images."""
    print('=' * 60)
    print('OCR Test Image Creator')
    print('=' * 60)
    print(f'Output: {OUTPUT_DIR.absolute()}')
    print()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    created = []

    # Simple text images
    print('Creating simple text images...')
    created.append(create_simple_text_image(
        'Hello World',
        'hello_world.jpg',
    ))

    created.append(create_simple_text_image(
        'STOP',
        'stop_sign.jpg',
        size=(300, 300),
        bg_color='red',
        text_color='white',
        font_size=72,
    ))

    created.append(create_simple_text_image(
        'EXIT',
        'exit_sign.jpg',
        size=(400, 150),
        bg_color='green',
        text_color='white',
        font_size=64,
    ))

    created.append(create_simple_text_image(
        'CAUTION: WET FLOOR',
        'caution_sign.jpg',
        size=(500, 150),
        bg_color='yellow',
        text_color='black',
        font_size=36,
    ))

    # Sign images
    print('Creating sign images...')
    created.append(create_sign_image('HIGHWAY 101 NORTH', 'highway_sign.jpg'))
    created.append(create_sign_image('San Francisco 25 mi', 'distance_sign.jpg'))

    # Document images
    print('Creating document images...')
    created.append(create_document_image([
        'INVOICE',
        '',
        'Item: Widget Pro',
        'Quantity: 100',
        'Price: $25.00 each',
        'Total: $2,500.00',
    ], 'invoice.jpg'))

    created.append(create_document_image([
        'MEETING NOTES',
        '',
        'Date: January 2, 2026',
        'Attendees: Alice, Bob, Charlie',
        'Topics: Q1 Planning, Budget Review',
        'Next meeting: January 9, 2026',
    ], 'meeting_notes.jpg'))

    # Multi-language (if font supports)
    print('Creating multi-language images...')
    created.append(create_simple_text_image(
        'Welcome / Bienvenue / Willkommen',
        'multilang.jpg',
        size=(800, 150),
        font_size=36,
    ))

    # Numbers and special characters
    created.append(create_simple_text_image(
        'License: ABC-1234',
        'license_plate.jpg',
        size=(400, 120),
        bg_color='#f0f0f0',
        font_size=48,
    ))

    created.append(create_simple_text_image(
        'Phone: (555) 123-4567',
        'phone_number.jpg',
        size=(500, 100),
        font_size=36,
    ))

    created.append(create_simple_text_image(
        'Email: test@example.com',
        'email.jpg',
        size=(500, 100),
        font_size=36,
    ))

    print()
    print('=' * 60)
    print(f'Created: {len(created)} images')
    print(f'Output directory: {OUTPUT_DIR.absolute()}')
    print()

    # List created files
    print('Files:')
    for f in sorted(OUTPUT_DIR.iterdir()):
        if f.is_file():
            size_kb = f.stat().st_size / 1024
            print(f'  - {f.name} ({size_kb:.1f} KB)')

    return 0


if __name__ == '__main__':
    sys.exit(main())
