#!/usr/bin/env python3
"""
Download pre-trained face detection and recognition models.

Models:
- SCRFD-10G-GNKPS: Face detection with 5-point landmarks (~17MB)
- ArcFace w600k_r50: Face recognition ResNet-50 (~166MB)

Sources:
- InsightFace GitHub releases
- InsightFace HuggingFace model hub

Usage:
    # From yolo-api container:
    docker compose exec yolo-api python /app/export/download_face_models.py

    # Or from host with venv:
    python export/download_face_models.py
"""

import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve


# =============================================================================
# Model Configuration
# =============================================================================

MODELS = {
    # SCRFD-10G with Batch Normalization and Keypoints
    # Best accuracy/speed trade-off for GPU deployment
    # Note: bnkps = Batch Norm + KeyPoints, gnkps = Group Norm + KeyPoints
    # Using bnkps as it's more widely available
    'scrfd_10g_bnkps': {
        'urls': [
            # Primary: HuggingFace LPDoctor mirror
            'https://huggingface.co/LPDoctor/insightface/resolve/main/scrfd_10g_bnkps.onnx',
            # Backup: HuggingFace Aitrepreneur mirror (antelopev2)
            'https://huggingface.co/Aitrepreneur/insightface/resolve/main/models/antelopev2/scrfd_10g_bnkps.onnx',
        ],
        'filename': 'scrfd_10g_bnkps.onnx',
        'size_mb': 17,
        'description': 'SCRFD face detector with 5-point landmarks (95.2%/93.9%/83.1% WiderFace)',
        'input_shape': '[B, 3, 640, 640]',
        'output': 'boxes, landmarks, scores',
    },
    # ArcFace w600k ResNet-50 (WebFace600K trained)
    # Production-grade face recognition
    'arcface_w600k_r50': {
        'urls': [
            # Primary: InsightFace buffalo_l model pack (need to extract w600k_r50.onnx)
            'https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l/w600k_r50.onnx',
            # Backup
            'https://github.com/deepinsight/insightface/releases/download/v0.7/w600k_r50.onnx',
        ],
        'filename': 'arcface_w600k_r50.onnx',
        'size_mb': 166,
        'description': 'ArcFace recognition model (99.8% LFW, 512-dim embeddings)',
        'input_shape': '[B, 3, 112, 112]',
        'output': '512-dim L2-normalized embeddings',
    },
    # MobileFaceNet (lightweight alternative)
    'arcface_w600k_mbf': {
        'urls': [
            'https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_s/w600k_mbf.onnx',
        ],
        'filename': 'arcface_w600k_mbf.onnx',
        'size_mb': 13,
        'description': 'MobileFaceNet recognition (99.7% LFW, faster, smaller)',
        'input_shape': '[B, 3, 112, 112]',
        'output': '512-dim L2-normalized embeddings',
    },
}

# Output directory
OUTPUT_DIR = Path('/app/pytorch_models') if Path('/app').exists() else Path('pytorch_models')


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
    print(f'  Downloading from: {url[:80]}...')

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


def verify_onnx_model(filepath: Path) -> bool:
    """Verify ONNX model is valid."""
    try:
        import onnx

        model = onnx.load(str(filepath))
        onnx.checker.check_model(model)
        print(f'  Model valid: {len(model.graph.node)} nodes')

        # Print input/output info
        inputs = [
            f'{i.name}: {[d.dim_value for d in i.type.tensor_type.shape.dim]}'
            for i in model.graph.input
        ]
        outputs = [
            f'{o.name}: {[d.dim_value for d in o.type.tensor_type.shape.dim]}'
            for o in model.graph.output
        ]
        print(f'  Inputs: {inputs}')
        print(f'  Outputs: {outputs}')

        return True
    except ImportError:
        print('  Warning: onnx not installed, skipping validation')
        return True
    except Exception as e:
        print(f'  Validation failed: {e}')
        return False


def download_model(model_id: str, config: dict, force: bool = False) -> bool:
    """Download a single model."""
    output_path = OUTPUT_DIR / config['filename']

    print(f'\n{"=" * 60}')
    print(f'Model: {model_id}')
    print(f'{"=" * 60}')
    print(f'Description: {config["description"]}')
    print(f'Expected size: ~{config["size_mb"]} MB')
    print(f'Input shape: {config["input_shape"]}')
    print(f'Output: {config["output"]}')
    print(f'Output path: {output_path}')

    # Check if already exists
    if output_path.exists() and not force:
        actual_size = get_file_size_mb(output_path)
        expected_size = config['size_mb']

        # Allow 20% tolerance for size
        if abs(actual_size - expected_size) / expected_size < 0.2:
            print(f'  Already exists: {actual_size:.1f} MB')
            if verify_onnx_model(output_path):
                print('  Skipping download (use --force to re-download)')
                return True
            print('  Existing file invalid, re-downloading...')
        else:
            print(
                f'  Existing file wrong size ({actual_size:.1f} vs {expected_size} MB), re-downloading...'
            )

    # Try each URL
    for i, url in enumerate(config['urls']):
        print(f'\n  Trying source {i + 1}/{len(config["urls"])}...')
        if download_with_progress(url, output_path, config['filename']):
            actual_size = get_file_size_mb(output_path)
            print(f'  Downloaded: {actual_size:.1f} MB')

            if verify_onnx_model(output_path):
                return True
            print('  Model verification failed, trying next source...')
            output_path.unlink(missing_ok=True)

    print(f'\n  Failed to download {model_id} from all sources')
    return False


def download_all_models(models: list[str] | None = None, force: bool = False) -> dict:
    """Download all specified models."""
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Select models to download
    if models is None:
        # Default: SCRFD + ArcFace R50 (not MBF)
        models = ['scrfd_10g_bnkps', 'arcface_w600k_r50']

    results = {}
    for model_id in models:
        if model_id not in MODELS:
            print(f'\nUnknown model: {model_id}')
            print(f'Available: {list(MODELS.keys())}')
            results[model_id] = False
            continue

        results[model_id] = download_model(model_id, MODELS[model_id], force)

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
        filepath = OUTPUT_DIR / MODELS[model_id]['filename']
        size = get_file_size_mb(filepath) if filepath.exists() else 0
        print(f'  {model_id}: {status} ({size:.1f} MB)')

    all_success = all(results.values())
    print()

    if all_success:
        print('All models downloaded successfully!')
        print(f'Output directory: {OUTPUT_DIR}')
        print('\nNext steps:')
        print('  1. Export to TensorRT: python export/export_face_detection.py')
        print('  2. Export to TensorRT: python export/export_face_recognition.py')
    else:
        print('Some downloads failed. Check network and try again.')
        return 1

    return 0


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Download face detection and recognition models')
    parser.add_argument(
        '--models',
        nargs='+',
        choices=[*list(MODELS.keys()), 'all'],
        default=None,
        help='Models to download (default: scrfd_10g_bnkps, arcface_w600k_r50)',
    )
    parser.add_argument('--force', action='store_true', help='Re-download even if exists')
    parser.add_argument('--list', action='store_true', help='List available models')
    args = parser.parse_args()

    if args.list:
        print('Available face models:')
        print()
        for model_id, config in MODELS.items():
            print(f'  {model_id}:')
            print(f'    {config["description"]}')
            print(f'    Size: ~{config["size_mb"]} MB')
            print(f'    Input: {config["input_shape"]}')
            print()
        return 0

    # Select models
    models = args.models
    if models and 'all' in models:
        models = list(MODELS.keys())

    print('=' * 60)
    print('Face Model Downloader')
    print('=' * 60)
    print(f'Output directory: {OUTPUT_DIR}')
    print(f'Models: {models or ["scrfd_10g_bnkps", "arcface_w600k_r50"]}')

    results = download_all_models(models, args.force)
    return print_summary(results)


if __name__ == '__main__':
    sys.exit(main())
