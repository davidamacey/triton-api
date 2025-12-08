#!/usr/bin/env python3
"""
Download PyTorch YOLO Models Script

Uses Ultralytics' built-in download functionality for reliable model retrieval
with automatic retry, validation, and disk space checking.
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path


# Try to import ultralytics download function at module level
try:
    from ultralytics.utils.downloads import attempt_download_asset

    ULTRALYTICS_AVAILABLE = True
except ImportError:
    attempt_download_asset = None
    ULTRALYTICS_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

# Model configurations
# Maps model size to filename and release version
# Full YOLO11 model lineup from Ultralytics
MODELS = {
    'nano': {
        'filename': 'yolo11n.pt',
        'release': 'v8.3.0',
    },
    'small': {
        'filename': 'yolo11s.pt',
        'release': 'v8.3.0',
    },
    'medium': {
        'filename': 'yolo11m.pt',
        'release': 'v8.3.0',
    },
    'large': {
        'filename': 'yolo11l.pt',
        'release': 'v8.3.0',
    },
    'xlarge': {
        'filename': 'yolo11x.pt',
        'release': 'v8.3.0',
    },
}

# Default output directory (relative to project root)
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / 'pytorch_models'


def download_model(model_size: str, output_dir: Path) -> bool:
    """
    Download a YOLO model using Ultralytics built-in download functionality.

    Args:
        model_size: Model size key (small, nano, medium, large, xlarge)
        output_dir: Directory to save the model

    Returns:
        True if download successful, False otherwise
    """
    if not ULTRALYTICS_AVAILABLE:
        logger.error('Ultralytics not installed. Install with: pip install ultralytics')
        return False

    if model_size not in MODELS:
        logger.error(f'Unknown model size: {model_size}. Available: {list(MODELS.keys())}')
        return False

    model_config = MODELS[model_size]
    filename = model_config['filename']
    target_path = output_dir / filename

    # Check if already exists
    if target_path.exists():
        file_size = target_path.stat().st_size / (1024 * 1024)  # MB
        logger.info(f'Model already exists: {filename} ({file_size:.1f} MB)')
        return True

    logger.info(f'Downloading {filename}...')

    try:
        # attempt_download_asset returns the path to the downloaded file
        # It handles retry logic, disk space validation, and curl fallback
        downloaded_path = attempt_download_asset(
            filename,
            repo='ultralytics/assets',
            release=model_config['release'],
        )

        # Move to target directory if downloaded elsewhere
        downloaded_path = Path(downloaded_path)
        if downloaded_path.exists() and downloaded_path != target_path:
            # If downloaded to different location, move it
            output_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(downloaded_path), str(target_path))
            logger.info(f'Moved model to: {target_path}')

        if target_path.exists():
            file_size = target_path.stat().st_size / (1024 * 1024)  # MB
            logger.info(f'Downloaded: {filename} ({file_size:.1f} MB)')
            return True
        logger.error(f'Failed to download {filename}')
        return False

    except Exception as e:
        logger.error(f'Error downloading {filename}: {e}')
        return False


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Download PyTorch YOLO models using Ultralytics built-in methods',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_pytorch_models.py                          # Download small model (default)
  python download_pytorch_models.py --models small medium    # Download specific models
  python download_pytorch_models.py --models all             # Download all models (nano-xlarge)
  python download_pytorch_models.py --output ./models        # Custom output directory
  python download_pytorch_models.py --list                   # List available models
        """,
    )

    parser.add_argument(
        '--models',
        nargs='+',
        choices=['nano', 'small', 'medium', 'large', 'xlarge', 'all'],
        default=['small'],
        help='Model sizes to download (default: small)',
    )

    parser.add_argument(
        '--output',
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f'Output directory (default: {DEFAULT_OUTPUT_DIR})',
    )

    parser.add_argument(
        '--list',
        action='store_true',
        help='List available models and exit',
    )

    args = parser.parse_args()

    # List mode
    if args.list:
        print('\nAvailable YOLO models:')
        print('-' * 40)
        for size, config in MODELS.items():
            print(f'  {size:8} -> {config["filename"]} (release {config["release"]})')
        print()
        return 0

    # Check ultralytics availability early
    if not ULTRALYTICS_AVAILABLE:
        logger.error('Ultralytics not installed. Install with: pip install ultralytics')
        return 1

    # Determine which models to download
    models_to_download = list(MODELS.keys()) if 'all' in args.models else args.models

    # Ensure output directory exists
    args.output.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print('PyTorch YOLO Model Downloader')
    print('=' * 60)
    print(f'\nTarget directory: {args.output}')
    print(f'Models to download: {", ".join(models_to_download)}\n')

    # Download each model
    success_count = 0
    for model_size in models_to_download:
        if download_model(model_size, args.output):
            success_count += 1

    # Summary
    print()
    print('=' * 60)
    if success_count == len(models_to_download):
        logger.info(f'All {success_count} model(s) downloaded successfully')
        print('=' * 60)
        print(f'\nModels location: {args.output}')
        print('\nDownloaded files:')
        for f in sorted(args.output.glob('*.pt')):
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f'  {f.name}: {size_mb:.1f} MB')
        return 0
    logger.error(
        f'Downloaded {success_count}/{len(models_to_download)} model(s). Some downloads failed.'
    )
    print('=' * 60)
    return 1


if __name__ == '__main__':
    sys.exit(main())
