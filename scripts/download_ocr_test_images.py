#!/usr/bin/env python3
"""
Download sample OCR test images with text for testing the OCR pipeline.

Downloads images from various sources containing text:
- Street signs
- Documents
- Posters/banners
- License plates

Usage:
    python scripts/download_ocr_test_images.py
"""

import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve, urlopen, Request
import shutil

# Output directory
OUTPUT_DIR = Path('test_images/ocr')

# Sample images with text from public datasets and sources
# These are creative commons or public domain images with visible text
SAMPLE_IMAGES = {
    # Street signs and outdoor text
    'street_sign_1.jpg': 'https://upload.wikimedia.org/wikipedia/commons/thumb/9/9a/Big_Buck_Bunny_Poster.jpg/640px-Big_Buck_Bunny_Poster.jpg',
    'stop_sign.jpg': 'https://upload.wikimedia.org/wikipedia/commons/thumb/e/e8/Stop_sign_in_Winter.JPG/640px-Stop_sign_in_Winter.JPG',
    'road_sign.jpg': 'https://upload.wikimedia.org/wikipedia/commons/thumb/7/71/I-80_Eastshore_Freeway.jpg/640px-I-80_Eastshore_Freeway.jpg',
    'storefront.jpg': 'https://upload.wikimedia.org/wikipedia/commons/thumb/5/57/Duane_Reade_253_W_42_jeh.jpg/640px-Duane_Reade_253_W_42_jeh.jpg',

    # License plates
    'license_plate_1.jpg': 'https://upload.wikimedia.org/wikipedia/commons/thumb/2/24/New_york_license_plate_-_blue.JPG/640px-New_york_license_plate_-_blue.JPG',

    # Posters and banners
    'poster_1.jpg': 'https://upload.wikimedia.org/wikipedia/commons/thumb/1/19/WPA_Posters_-_Reading.jpg/480px-WPA_Posters_-_Reading.jpg',
    'movie_poster.jpg': 'https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Casablanca%2C_bande-annonce_fran%C3%A7aise.jpg/480px-Casablanca%2C_bande-annonce_fran%C3%A7aise.jpg',

    # Books and documents
    'book_cover.jpg': 'https://upload.wikimedia.org/wikipedia/commons/thumb/c/cd/RWEmerson.jpg/480px-RWEmerson.jpg',
    'newspaper.jpg': 'https://upload.wikimedia.org/wikipedia/commons/thumb/6/60/NYTimes-Page1-11-11-1918.jpg/480px-NYTimes-Page1-11-11-1918.jpg',

    # Digital/screenshots
    'menu.jpg': 'https://upload.wikimedia.org/wikipedia/commons/thumb/9/9d/The_Aster_Restaurant_Menu.jpg/480px-The_Aster_Restaurant_Menu.jpg',

    # Mixed text images
    'billboard.jpg': 'https://upload.wikimedia.org/wikipedia/commons/thumb/5/5f/Greyhound_bus_station_billboard.jpg/640px-Greyhound_bus_station_billboard.jpg',
}


def download_image(url: str, output_path: Path) -> bool:
    """Download a single image with proper User-Agent."""
    try:
        print(f'  Downloading {output_path.name}...', end='', flush=True)
        # Use User-Agent to avoid 403 errors
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        req = Request(url, headers=headers)
        with urlopen(req, timeout=30) as response:
            with open(output_path, 'wb') as f:
                shutil.copyfileobj(response, f)
        size_kb = output_path.stat().st_size / 1024
        print(f' OK ({size_kb:.1f} KB)')
        return True
    except (HTTPError, URLError) as e:
        print(f' FAILED: {e}')
        return False
    except Exception as e:
        print(f' ERROR: {e}')
        return False


def main():
    """Download all sample OCR images."""
    print('=' * 60)
    print('OCR Test Image Downloader')
    print('=' * 60)
    print(f'Output: {OUTPUT_DIR.absolute()}')
    print()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    success = 0
    failed = 0

    for filename, url in SAMPLE_IMAGES.items():
        output_path = OUTPUT_DIR / filename

        # Skip if already exists
        if output_path.exists():
            print(f'  {filename}: Already exists, skipping')
            success += 1
            continue

        if download_image(url, output_path):
            success += 1
        else:
            failed += 1

    print()
    print('=' * 60)
    print(f'Downloaded: {success}/{len(SAMPLE_IMAGES)} images')
    if failed > 0:
        print(f'Failed: {failed} images')
    print(f'Output directory: {OUTPUT_DIR.absolute()}')
    print()

    # List downloaded files
    print('Files:')
    for f in sorted(OUTPUT_DIR.iterdir()):
        if f.is_file():
            size_kb = f.stat().st_size / 1024
            print(f'  - {f.name} ({size_kb:.1f} KB)')

    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
