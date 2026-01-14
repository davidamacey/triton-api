#!/usr/bin/env python3
"""
Download YOLO11-face models from YapaLab/yolo-face.

YOLO11-face provides face detection with 5-point landmark output (keypoints).
Based on Ultralytics YOLO11 architecture with pose/keypoint head for landmarks.

Source: https://github.com/YapaLab/yolo-face

Models are downloaded to pytorch_models/yolo11_face/

Usage:
    # From yolo-api container:
    docker compose exec yolo-api python /app/scripts/download_yolo11_face.py --models small

    # Or from host with venv:
    python scripts/download_yolo11_face.py --models small medium
"""

import argparse
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve


# =============================================================================
# Model Configuration
# =============================================================================

# YOLO11-face model URLs from YapaLab/yolo-face GitHub releases
# Source: https://github.com/YapaLab/yolo-face/releases/tag/1.0.0
# Note: These use the pose/keypoint architecture for landmark output
MODEL_URLS = {
    'nano': {
        'url': 'https://github.com/YapaLab/yolo-face/releases/download/1.0.0/yolov11n-face.pt',
        'filename': 'yolov11n-face.pt',
        'size_mb': 5.2,
        'description': 'YOLO11-face nano (fastest, ~5MB)',
    },
    'small': {
        'url': 'https://github.com/YapaLab/yolo-face/releases/download/1.0.0/yolov11s-face.pt',
        'filename': 'yolov11s-face.pt',
        'size_mb': 18.3,
        'description': 'YOLO11-face small (balanced, ~18MB)',
    },
    'medium': {
        'url': 'https://github.com/YapaLab/yolo-face/releases/download/1.0.0/yolov11m-face.pt',
        'filename': 'yolov11m-face.pt',
        'size_mb': 38.6,
        'description': 'YOLO11-face medium (best accuracy, ~39MB)',
    },
    'large': {
        'url': 'https://github.com/YapaLab/yolo-face/releases/download/1.0.0/yolov11l-face.pt',
        'filename': 'yolov11l-face.pt',
        'size_mb': 48.8,
        'description': 'YOLO11-face large (highest accuracy, ~49MB)',
    },
}

# Output directory
OUTPUT_DIR = Path('/app/pytorch_models/yolo11_face') if Path('/app').exists() else Path('pytorch_models/yolo11_face')


# =============================================================================
# Download Functions
# =============================================================================


def get_file_size_mb(filepath: Path) -> float:
    """Get file size in MB."""
    if filepath.exists():
        return filepath.stat().st_size / (1024 * 1024)
    return 0


def download_with_progress(url: str, output_path: Path, desc: str = '') -> bool:
    """Download file with progress indicator."""
    print(f'  Downloading from: {url}')

    try:
        start_time = time.time()

        def progress_hook(block_count, block_size, total_size):
            if total_size > 0:
                downloaded = block_count * block_size
                percent = min(100, downloaded * 100 / total_size)
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                elapsed = time.time() - start_time
                speed = mb_downloaded / elapsed if elapsed > 0 else 0
                print(
                    f'\r  Progress: {percent:5.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB) '
                    f'@ {speed:.1f} MB/s',
                    end='',
                    flush=True,
                )

        urlretrieve(url, output_path, reporthook=progress_hook)
        print()  # New line after progress
        return True

    except (URLError, HTTPError) as e:
        print(f'\n  Failed: {e}')
        return False
    except Exception as e:
        print(f'\n  Error: {e}')
        return False


def verify_pytorch_model(filepath: Path) -> bool:
    """Verify PyTorch model is valid and can be loaded."""
    try:
        import torch

        checkpoint = torch.load(str(filepath), map_location='cpu', weights_only=False)

        # YOLO models store model info in checkpoint
        if 'model' in checkpoint:
            print('  Model checkpoint valid')
            return True
        elif isinstance(checkpoint, dict) and any(k.endswith('.weight') for k in checkpoint.keys()):
            print('  State dict valid')
            return True
        else:
            print(f'  Unknown checkpoint format: {list(checkpoint.keys())[:5]}...')
            return True  # Still might be valid

    except ImportError:
        print('  Warning: torch not installed, skipping validation')
        return True
    except Exception as e:
        print(f'  Validation failed: {e}')
        return False


def download_model(model_id: str, config: dict, force: bool = False) -> bool:
    """Download a single model."""
    output_path = OUTPUT_DIR / config['filename']

    print(f'\n{"=" * 60}')
    print(f'Model: yolo11_{model_id}_face')
    print(f'{"=" * 60}')
    print(f'Description: {config["description"]}')
    print(f'Expected size: ~{config["size_mb"]} MB')
    print(f'Output path: {output_path}')

    # Check if already exists
    if output_path.exists() and not force:
        actual_size = get_file_size_mb(output_path)
        expected_size = config['size_mb']

        # Allow 50% tolerance for size (PyTorch models can vary)
        if abs(actual_size - expected_size) / max(expected_size, 1) < 0.5:
            print(f'  Already exists: {actual_size:.1f} MB')
            if verify_pytorch_model(output_path):
                print('  Skipping download (use --force to re-download)')
                return True
            print('  Existing file invalid, re-downloading...')
        else:
            print(
                f'  Existing file size mismatch ({actual_size:.1f} vs ~{expected_size} MB), re-downloading...'
            )

    # Download
    if download_with_progress(config['url'], output_path, config['filename']):
        actual_size = get_file_size_mb(output_path)
        print(f'  Downloaded: {actual_size:.1f} MB')

        if verify_pytorch_model(output_path):
            return True
        print('  Model verification failed')
        output_path.unlink(missing_ok=True)
        return False

    print(f'\n  Failed to download {model_id}')
    return False


def download_models(models: list[str] | None = None, force: bool = False) -> dict:
    """Download all specified models."""
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Select models to download
    if models is None:
        models = ['small']  # Default to small model

    results = {}
    for model_id in models:
        if model_id not in MODEL_URLS:
            print(f'\nUnknown model: {model_id}')
            print(f'Available: {list(MODEL_URLS.keys())}')
            results[model_id] = False
            continue

        results[model_id] = download_model(model_id, MODEL_URLS[model_id], force)

    return results


# =============================================================================
# Main
# =============================================================================


def print_summary(results: dict):
    """Print download summary."""
    print('\n' + '=' * 60)
    print('Download Summary')
    print('=' * 60)

    for model_id, success in results.items():
        status = 'OK' if success else 'FAILED'
        filepath = OUTPUT_DIR / MODEL_URLS[model_id]['filename']
        size = get_file_size_mb(filepath) if filepath.exists() else 0
        print(f'  yolo11_{model_id}_face: {status} ({size:.1f} MB)')

    all_success = all(results.values())
    print()

    if all_success:
        print('All models downloaded successfully!')
        print(f'Output directory: {OUTPUT_DIR}')
        print('\nNext steps:')
        print('  1. Export to TensorRT: python export/export_yolo11_face.py')
        print('  2. Restart Triton: make restart-triton')
        print('  3. Test: curl -X POST http://localhost:4603/track_e/faces/yolo11/detect -F "file=@test.jpg"')
    else:
        print('Some downloads failed. Check network and try again.')
        return 1

    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Download YOLO11-face models')
    parser.add_argument(
        '--models',
        nargs='+',
        choices=[*list(MODEL_URLS.keys()), 'all'],
        default=['small'],
        help='Models to download (default: small)',
    )
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory')
    parser.add_argument('--force', action='store_true', help='Re-download even if exists')
    parser.add_argument('--list', action='store_true', help='List available models')
    args = parser.parse_args()

    if args.list:
        print('Available YOLO11-face models:')
        print()
        for model_id, config in MODEL_URLS.items():
            print(f'  {model_id}:')
            print(f'    {config["description"]}')
            print(f'    Size: ~{config["size_mb"]} MB')
            print(f'    URL: {config["url"]}')
            print()
        return 0

    # Override output directory if specified
    global OUTPUT_DIR
    if args.output_dir:
        OUTPUT_DIR = args.output_dir

    # Select models
    models = args.models
    if 'all' in models:
        models = list(MODEL_URLS.keys())

    print('=' * 60)
    print('YOLO11-Face Model Downloader')
    print('=' * 60)
    print(f'Output directory: {OUTPUT_DIR}')
    print(f'Models: {models}')

    results = download_models(models, args.force)
    return print_summary(results)


if __name__ == '__main__':
    sys.exit(main())
