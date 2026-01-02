#!/usr/bin/env python3
"""
Generate equalized benchmark configurations for all tracks.

Creates config.pbtxt files with standardized instance counts for fair comparison:
- DALI preprocessing: 6 instances (I/O-bound, feeds GPU)
- TRT inference: 4 instances (compute-bound)
- MobileCLIP encoder: 4 instances (TRT)
- Python backend: 4 instances

All models use GPU 0 for isolation testing.
"""

import re
import sys
from pathlib import Path


# Project paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODELS_DIR = PROJECT_ROOT / 'models'
BENCHMARK_CONFIGS = SCRIPT_DIR / 'configs'

# Standard instance counts for fair comparison
STANDARD_INSTANCES = {
    'dali': 6,  # I/O-bound, needs more to feed GPU
    'tensorrt': 4,  # Compute-bound TRT inference
    'python': 4,  # Python backend (BLS)
}

# GPU assignment for benchmarks (all on GPU 0 for isolation)
BENCHMARK_GPU = 0

# Track definitions: track_name -> list of (model_name, backend_type)
TRACKS = {
    'track_b': [
        ('yolov11_small_trt', 'tensorrt'),
    ],
    'track_c': [
        ('yolov11_small_trt_end2end', 'tensorrt'),
    ],
    'track_d_streaming': [
        ('yolo_preprocess_dali_streaming', 'dali'),
        ('yolov11_small_trt_end2end_streaming', 'tensorrt'),
    ],
    'track_d_balanced': [
        ('yolo_preprocess_dali', 'dali'),
        ('yolov11_small_trt_end2end', 'tensorrt'),
    ],
    'track_d_batch': [
        ('yolo_preprocess_dali_batch', 'dali'),
        ('yolov11_small_trt_end2end_batch', 'tensorrt'),
    ],
    'track_e_simple': [
        ('yolo_clip_preprocess_dali', 'dali'),
        ('yolov11_small_trt_end2end', 'tensorrt'),
        ('mobileclip2_s2_image_encoder', 'tensorrt'),
    ],
    'track_e_full': [
        ('dual_preprocess_dali', 'dali'),
        ('yolov11_small_trt_end2end', 'tensorrt'),
        ('mobileclip2_s2_image_encoder', 'tensorrt'),
        ('box_embedding_extractor', 'python'),
    ],
}


def read_config(model_name: str) -> str:
    """Read existing config.pbtxt for a model."""
    config_path = MODELS_DIR / model_name / 'config.pbtxt'
    if not config_path.exists():
        raise FileNotFoundError(f'Config not found: {config_path}')
    return config_path.read_text()


def modify_instance_group(config: str, count: int, gpu: int) -> str:
    """Modify instance_group in config to use specified count and GPU."""
    # Pattern to match instance_group block (handles multi-line)
    pattern = r'instance_group\s*\[\s*\{[^}]*\}\s*\]'

    # New instance_group
    new_instance_group = f"""instance_group [
  {{
    count: {count}
    kind: KIND_GPU
    gpus: [ {gpu} ]
  }}
]"""

    # Replace existing instance_group
    modified = re.sub(pattern, new_instance_group, config, flags=re.DOTALL)

    # If no instance_group found, append it
    if 'instance_group' not in modified:
        modified += f'\n{new_instance_group}\n'

    return modified


def create_benchmark_config(model_name: str, backend_type: str) -> str:
    """Create benchmark config with equalized instances."""
    # Read original config
    config = read_config(model_name)

    # Get standard instance count for this backend type
    count = STANDARD_INSTANCES.get(backend_type, 4)

    # Modify instance group
    modified = modify_instance_group(config, count, BENCHMARK_GPU)

    # Add benchmark header comment
    header = f"""# ============================================================================
# BENCHMARK CONFIGURATION - EQUALIZED INSTANCES
# ============================================================================
# Model: {model_name}
# Backend: {backend_type}
# Instance count: {count} (standardized for fair comparison)
# GPU: {BENCHMARK_GPU} (isolated benchmark mode)
# ============================================================================
#
# Standard instance counts:
#   - DALI preprocessing: 6 (I/O-bound, feeds GPU)
#   - TRT inference: 4 (compute-bound)
#   - Python backend: 4
#
# This config is used by isolated_benchmark.sh for fair testing.
# Production configs remain in models/{model_name}/config.pbtxt
# ============================================================================

"""

    return header + modified


def main():
    print('=' * 80)
    print('Creating Equalized Benchmark Configurations')
    print('=' * 80)
    print()
    print('Standard instance counts:')
    print(f'  - DALI preprocessing: {STANDARD_INSTANCES["dali"]} instances')
    print(f'  - TRT inference:      {STANDARD_INSTANCES["tensorrt"]} instances')
    print(f'  - Python backend:     {STANDARD_INSTANCES["python"]} instances')
    print(f'  - All models on GPU:  {BENCHMARK_GPU}')
    print()

    # Create output directory
    BENCHMARK_CONFIGS.mkdir(parents=True, exist_ok=True)

    created_count = 0
    skipped_count = 0

    for track_name, models in TRACKS.items():
        print(f'\nTrack: {track_name}')
        print('-' * 40)

        # Create track subdirectory
        track_dir = BENCHMARK_CONFIGS / track_name
        track_dir.mkdir(parents=True, exist_ok=True)

        for model_name, backend_type in models:
            try:
                config = create_benchmark_config(model_name, backend_type)

                # Write config file
                output_path = track_dir / f'{model_name}.config.pbtxt'
                output_path.write_text(config)

                count = STANDARD_INSTANCES.get(backend_type, 4)
                print(f'  {model_name}: count={count} gpu={BENCHMARK_GPU}')
                created_count += 1

            except FileNotFoundError:
                print(f'  SKIP: {model_name} (model not found)')
                skipped_count += 1
            except Exception as e:
                print(f'  ERROR: {model_name}: {e}')
                skipped_count += 1

    print()
    print('=' * 80)
    print(f'Created {created_count} benchmark configuration files')
    if skipped_count > 0:
        print(f'Skipped {skipped_count} models (not found or error)')
    print(f'Location: {BENCHMARK_CONFIGS}')
    print('=' * 80)

    return 0 if created_count > 0 else 1


if __name__ == '__main__':
    sys.exit(main())
