#!/usr/bin/env python3
"""
Download PP-OCRv5 models for text detection and recognition.

Models:
- PP-OCRv5 Detection (Mobile): Text detection ~4MB
- PP-OCRv5 Recognition (Mobile): Text recognition ~11MB
- Character dictionaries for Chinese/English

Sources:
- PaddlePaddle/PaddleOCR GitHub releases
- RapidOCR model repository

Usage:
    # From yolo-api container:
    docker compose exec yolo-api python /app/export/download_paddleocr.py

    # Or from host with venv:
    python export/download_paddleocr.py
"""

import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve


# =============================================================================
# Model Configuration
# =============================================================================

# PP-OCRv5 Mobile Models (best speed/accuracy tradeoff)
# Source: https://github.com/MeKo-Christian/paddleocr-onnx/releases
MODELS = {
    # PP-OCRv5 Detection (DB++)
    # Input: [B, 3, H, W] where H,W are multiples of 32
    # Output: Text probability map
    'ppocr_det_v5_mobile': {
        'urls': [
            # Primary: MeKo-Christian/paddleocr-onnx releases (pre-converted ONNX)
            'https://github.com/MeKo-Christian/paddleocr-onnx/releases/download/v1.0.0/PP-OCRv5_mobile_det.onnx',
            # Backup: Hugging Face (server version, larger)
            'https://huggingface.co/marsena/paddleocr-onnx-models/resolve/main/PP-OCRv5_server_det_infer.onnx',
        ],
        'filename': 'ppocr_det_v5_mobile.onnx',
        'size_mb': 5,  # Mobile is ~4.6MB
        'description': 'PP-OCRv5 text detector (DB++ architecture)',
        'input_shape': '[B, 3, H, W] (H,W multiples of 32, max 960)',
        'output': 'probability map for text regions',
        'preprocessing': '(x / 255 - 0.5) / 0.5 = x / 127.5 - 1, BGR',
    },
    # PP-OCRv5 Recognition (SVTR-LCNet)
    # Input: [B, 3, 48, 320] text crops
    # Output: text sequence logits
    'ppocr_rec_v5_mobile': {
        'urls': [
            # Primary: MeKo-Christian/paddleocr-onnx releases (pre-converted ONNX)
            'https://github.com/MeKo-Christian/paddleocr-onnx/releases/download/v1.0.0/PP-OCRv5_mobile_rec.onnx',
            # Backup: Hugging Face (server version, larger)
            'https://huggingface.co/marsena/paddleocr-onnx-models/resolve/main/PP-OCRv5_server_rec_infer.onnx',
        ],
        'filename': 'ppocr_rec_v5_mobile.onnx',
        'size_mb': 16,  # Mobile is ~16MB
        'description': 'PP-OCRv5 text recognizer (SVTR-LCNet architecture)',
        'input_shape': '[B, 3, 48, 320]',
        'output': 'character sequence logits',
        'preprocessing': '(x / 255 - 0.5) / 0.5 = x / 127.5 - 1, BGR',
    },
}

# Character dictionaries
DICTIONARIES = {
    'ppocr_keys_v1': {
        'urls': [
            'https://github.com/RapidAI/RapidOCR/raw/main/python/rapidocr/inference_engine/resources/rec/ch/keys.txt',
            'https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/release/2.7/ppocr/utils/ppocr_keys_v1.txt',
        ],
        'filename': 'ppocr_keys_v1.txt',
        'description': 'Chinese + English + symbols character dictionary (6623 chars)',
    },
    'en_dict': {
        'urls': [
            'https://github.com/RapidAI/RapidOCR/raw/main/python/rapidocr/inference_engine/resources/rec/en/keys.txt',
            'https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/release/2.7/ppocr/utils/en_dict.txt',
        ],
        'filename': 'en_dict.txt',
        'description': 'English-only character dictionary (95 chars)',
    },
}

# Output directory
OUTPUT_DIR = Path('/app/pytorch_models/paddleocr') if Path('/app').exists() else Path('pytorch_models/paddleocr')


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
        inputs = []
        for i in model.graph.input:
            dims = [d.dim_value if d.dim_value else d.dim_param for d in i.type.tensor_type.shape.dim]
            inputs.append(f'{i.name}: {dims}')

        outputs = []
        for o in model.graph.output:
            dims = [d.dim_value if d.dim_value else d.dim_param for d in o.type.tensor_type.shape.dim]
            outputs.append(f'{o.name}: {dims}')

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
    if 'size_mb' in config:
        print(f'Expected size: ~{config["size_mb"]} MB')
    if 'input_shape' in config:
        print(f'Input shape: {config["input_shape"]}')
    if 'output' in config:
        print(f'Output: {config["output"]}')
    print(f'Output path: {output_path}')

    # Check if already exists
    if output_path.exists() and not force:
        actual_size = get_file_size_mb(output_path)

        if 'size_mb' in config:
            expected_size = config['size_mb']
            # Allow 50% tolerance for size (ONNX sizes vary)
            if abs(actual_size - expected_size) / expected_size < 0.5 or actual_size > 0.1:
                print(f'  Already exists: {actual_size:.2f} MB')
                if output_path.suffix == '.onnx':
                    if verify_onnx_model(output_path):
                        print('  Skipping download (use --force to re-download)')
                        return True
                else:
                    # For dictionaries, just check file exists
                    print('  Skipping download (use --force to re-download)')
                    return True
        else:
            # Dictionary file
            if actual_size > 0:
                print(f'  Already exists: {actual_size:.2f} MB')
                print('  Skipping download (use --force to re-download)')
                return True

    # Try each URL
    for i, url in enumerate(config['urls']):
        print(f'\n  Trying source {i + 1}/{len(config["urls"])}...')

        # Handle tar files (need extraction)
        if url.endswith('.tar'):
            print('  Note: .tar format not supported, trying next source...')
            continue

        if download_with_progress(url, output_path, config.get('filename', '')):
            actual_size = get_file_size_mb(output_path)
            print(f'  Downloaded: {actual_size:.2f} MB')

            if output_path.suffix == '.onnx':
                if verify_onnx_model(output_path):
                    return True
                print('  Model verification failed, trying next source...')
                output_path.unlink(missing_ok=True)
            else:
                # Dictionary file - just check it's not empty
                if actual_size > 0:
                    # Count lines for dictionary
                    with open(output_path, 'r', encoding='utf-8') as f:
                        lines = len(f.readlines())
                    print(f'  Dictionary: {lines} characters')
                    return True

    print(f'\n  Failed to download {model_id} from all sources')
    return False


def download_all(force: bool = False) -> dict:
    """Download all models and dictionaries."""
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {}

    # Download models
    for model_id, config in MODELS.items():
        results[model_id] = download_model(model_id, config, force)

    # Download dictionaries
    for dict_id, config in DICTIONARIES.items():
        results[dict_id] = download_model(dict_id, config, force)

    return results


# =============================================================================
# Main
# =============================================================================


def print_summary(results: dict):
    """Print download summary."""
    print('\n' + '=' * 60)
    print('Download Summary')
    print('=' * 60)

    for item_id, success in results.items():
        status = 'OK' if success else 'FAILED'

        # Get config from either MODELS or DICTIONARIES
        config = MODELS.get(item_id) or DICTIONARIES.get(item_id)
        if config:
            filepath = OUTPUT_DIR / config['filename']
            size = get_file_size_mb(filepath) if filepath.exists() else 0
            print(f'  {item_id}: {status} ({size:.2f} MB)')
        else:
            print(f'  {item_id}: {status}')

    all_success = all(results.values())
    print()

    if all_success:
        print('All downloads completed successfully!')
        print(f'Output directory: {OUTPUT_DIR}')
        print('\nFiles:')
        for f in OUTPUT_DIR.iterdir():
            print(f'  - {f.name} ({get_file_size_mb(f):.2f} MB)')
        print('\nNext steps:')
        print('  1. Export detection to TRT: python export/export_paddleocr_det.py')
        print('  2. Export recognition to TRT: python export/export_paddleocr_rec.py')
        return 0
    else:
        print('Some downloads failed. Check network and try again.')
        return 1


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Download PP-OCRv5 models')
    parser.add_argument('--force', action='store_true', help='Re-download even if exists')
    parser.add_argument('--list', action='store_true', help='List available models')
    args = parser.parse_args()

    if args.list:
        print('Available PP-OCRv5 models:')
        print()
        for model_id, config in MODELS.items():
            print(f'  {model_id}:')
            print(f'    {config["description"]}')
            print(f'    Size: ~{config["size_mb"]} MB')
            print(f'    Input: {config["input_shape"]}')
            print()
        print('Character dictionaries:')
        print()
        for dict_id, config in DICTIONARIES.items():
            print(f'  {dict_id}:')
            print(f'    {config["description"]}')
            print()
        return 0

    print('=' * 60)
    print('PP-OCRv5 Model Downloader')
    print('=' * 60)
    print(f'Output directory: {OUTPUT_DIR}')
    print()
    print('Models to download:')
    print('  - PP-OCRv5 Detection (text region detection)')
    print('  - PP-OCRv5 Recognition (text reading)')
    print('  - Character dictionaries')

    results = download_all(args.force)
    return print_summary(results)


if __name__ == '__main__':
    sys.exit(main())
