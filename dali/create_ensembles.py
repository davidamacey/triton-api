#!/usr/bin/env python3
"""
Create Three-Tier Ensemble Configs for Track D

Generates ensemble models that chain DALI preprocessing with YOLO TRT End2End models.
Each ensemble combines:
1. DALI preprocessing (GPU decode + letterbox + normalize)
2. YOLO TRT End2End (GPU inference + NMS)

Three tiers optimize for different workloads:
- Streaming: Real-time video (0.1ms batching, low latency)
- Balanced: General purpose (0.5ms batching)
- Batch: Offline processing (5ms batching, high throughput)

Usage:
    # Create ensembles for specific model sizes
    python dali/create_ensembles.py --models small

    # Create ensembles for multiple sizes
    python dali/create_ensembles.py --models nano small medium

    # Create ensembles for all model sizes
    python dali/create_ensembles.py --models all

Output:
    models/yolov11_{size}_gpu_e2e_streaming/config.pbtxt
    models/yolov11_{size}_gpu_e2e/config.pbtxt
    models/yolov11_{size}_gpu_e2e_batch/config.pbtxt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Import shared configuration
from dali.config import (
    DEFAULT_MODEL_DIR,
    ENSEMBLE_TIERS,
    MODEL_SIZES,
    TRACK_D_DALI_MODELS,
    ModelConfig,
    TierConfig,
)


# =============================================================================
# Config Generation
# =============================================================================


def generate_ensemble_config(
    model_name: str,
    model_config: ModelConfig,
    tier_name: str,
    tier_config: TierConfig,
) -> str:
    """
    Generate ensemble config.pbtxt content.

    Args:
        model_name: Model name (nano, small, medium).
        model_config: Model configuration object.
        tier_name: Tier name (streaming, balanced, batch).
        tier_config: Tier configuration object.

    Returns:
        config.pbtxt content as string.
    """
    triton_name = model_config.triton_name
    topk = model_config.topk
    suffix = tier_config.suffix
    description = tier_config.description
    # Use tier-specific DALI model
    dali_model = TRACK_D_DALI_MODELS.get(tier_name, 'yolo_preprocess_dali')
    # Ensemble batch size must not exceed child model's max batch size
    max_batch = min(tier_config.max_batch, model_config.max_batch)
    # Filter preferred batch sizes to not exceed max_batch
    preferred_batches = [b for b in tier_config.preferred_batch_sizes if b <= max_batch]
    if not preferred_batches:
        preferred_batches = [max_batch]
    queue_delay = tier_config.max_queue_delay_us
    preserve_order = str(tier_config.preserve_ordering).lower()
    instance_count = tier_config.instance_count

    return f"""# {triton_name.upper()}{suffix.upper()} - Track D Ensemble
# {'=' * 70}
# {description}
#
# Pipeline:
#   1. DALI preprocessing (GPU: decode + letterbox + normalize)
#   2. YOLO TRT End2End (GPU: inference + NMS)
#
# Configuration tier: {tier_name}
# - Max batch: {max_batch}
# - Batching window: {queue_delay / 1000:.1f}ms
# - Preserve ordering: {preserve_order}
# - Instance count: {instance_count}

name: "{triton_name}{suffix}"
platform: "ensemble"
max_batch_size: {max_batch}

# Input: Raw JPEG/PNG bytes + affine transformation matrices
input [
  {{
    name: "encoded_images"
    data_type: TYPE_UINT8
    dims: [ -1 ]  # Variable-length byte array
  }},
  {{
    name: "affine_matrices"
    data_type: TYPE_FP32
    dims: [ 2, 3 ]  # 2x3 affine transformation matrix per image
  }}
]

# Output: Final detections (after GPU NMS)
output [
  {{
    name: "num_dets"
    data_type: TYPE_INT32
    dims: [ 1 ]
  }},
  {{
    name: "det_boxes"
    data_type: TYPE_FP32
    dims: [ {topk}, 4 ]  # XYXY format
  }},
  {{
    name: "det_scores"
    data_type: TYPE_FP32
    dims: [ {topk} ]
  }},
  {{
    name: "det_classes"
    data_type: TYPE_INT32
    dims: [ {topk} ]
  }}
]

# Ensemble: DALI -> YOLO TRT End2End
# Note: Ensemble models cannot have dynamic_batching alongside ensemble_scheduling.
# Batching is configured in the child models (DALI and TRT End2End).
ensemble_scheduling {{
  step [
    {{
      # Step 1: DALI preprocessing (GPU decode + letterbox + normalize)
      model_name: "{dali_model}"
      model_version: -1
      input_map {{
        key: "encoded_images"
        value: "encoded_images"
      }}
      input_map {{
        key: "affine_matrices"
        value: "affine_matrices"
      }}
      output_map {{
        key: "preprocessed_images"
        value: "preprocessed_images"
      }}
    }},
    {{
      # Step 2: YOLO TRT End2End (GPU inference + NMS)
      model_name: "{triton_name}_trt_end2end"
      model_version: -1
      input_map {{
        key: "images"
        value: "preprocessed_images"
      }}
      output_map {{
        key: "num_dets"
        value: "num_dets"
      }}
      output_map {{
        key: "det_boxes"
        value: "det_boxes"
      }}
      output_map {{
        key: "det_scores"
        value: "det_scores"
      }}
      output_map {{
        key: "det_classes"
        value: "det_classes"
      }}
    }}
  ]
}}
"""


def create_ensemble(
    model_name: str,
    tier_name: str,
    model_dir: Path = DEFAULT_MODEL_DIR,
) -> None:
    """
    Create a single ensemble model directory and config.

    Args:
        model_name: Model name (nano, small, medium).
        tier_name: Tier name (streaming, balanced, batch).
        model_dir: Base model directory.
    """
    model_config = MODEL_SIZES[model_name]
    tier_config = ENSEMBLE_TIERS[tier_name]

    # Ensemble name
    triton_name = model_config.triton_name
    suffix = tier_config.suffix
    ensemble_name = f'{triton_name}{suffix}'

    # Create directory structure
    ensemble_dir = model_dir / ensemble_name
    ensemble_dir.mkdir(parents=True, exist_ok=True)
    (ensemble_dir / '1').mkdir(exist_ok=True)  # Empty version directory

    # Generate and write config
    config_content = generate_ensemble_config(model_name, model_config, tier_name, tier_config)

    config_path = ensemble_dir / 'config.pbtxt'
    with open(config_path, 'w') as f:
        f.write(config_content)

    print(f'  [OK] Created: {ensemble_name}')
    print(f'       Path: {config_path}')
    print(f'       Tier: {tier_name} ({tier_config.description})')
    print(f'       Max batch: {tier_config.max_batch}')
    print(f'       Batching window: {tier_config.max_queue_delay_us / 1000:.1f}ms')


# =============================================================================
# CLI Interface
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Create Track D ensemble models (DALI + TRT End2End)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Create ensembles for small model only (recommended for clean benchmarking)
    python dali/create_ensembles.py --models small

    # Create ensembles for multiple specific sizes
    python dali/create_ensembles.py --models nano small medium

    # Create ensembles for all sizes
    python dali/create_ensembles.py --models all
""",
    )
    parser.add_argument(
        '--models',
        nargs='+',
        choices=['nano', 'small', 'medium', 'all'],
        default=['small'],
        help='Model sizes to create ensembles for (default: small)',
    )
    parser.add_argument(
        '--model-dir',
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help=f'Model directory (default: {DEFAULT_MODEL_DIR})',
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Determine which models to process
    if 'all' in args.models:
        model_names = list(MODEL_SIZES.keys())
    else:
        model_names = args.models

    # Validate model names
    for model_name in model_names:
        if model_name not in MODEL_SIZES:
            print(f"[ERROR] Invalid model name: '{model_name}'")
            print(f'Available models: {", ".join(MODEL_SIZES.keys())}')
            return 1

    print('=' * 80)
    print('Track D Ensemble Model Generator')
    print('=' * 80)
    print(f'\nCreating {len(model_names)} x 3 = {len(model_names) * 3} ensemble models\n')
    print(f'Model sizes: {", ".join(model_names)}')
    print(f'Tiers: {", ".join(ENSEMBLE_TIERS.keys())}\n')

    total = 0

    for tier_name in ENSEMBLE_TIERS:
        print(f'\n{tier_name.upper()} Tier')
        print('-' * 80)

        for model_name in model_names:
            create_ensemble(model_name, tier_name, args.model_dir)
            total += 1

    print('\n' + '=' * 80)
    print(f'[OK] Created {total} ensemble models')
    print('=' * 80)

    # Print naming convention
    print('\nModel naming convention:')
    for model_name in model_names:
        triton_name = MODEL_SIZES[model_name].triton_name
        print(f'  {triton_name}_gpu_e2e_streaming  -> Real-time streaming (0.1ms)')
        print(f'  {triton_name}_gpu_e2e            -> General purpose (0.5ms)')
        print(f'  {triton_name}_gpu_e2e_batch      -> Offline batch (5ms)')
        if model_name != model_names[-1]:
            print()

    # Print directory structure
    print('\nDirectory structure:')
    print('  models/')
    for tier_config in ENSEMBLE_TIERS.values():
        suffix = tier_config.suffix
        for model_name in model_names:
            triton_name = MODEL_SIZES[model_name].triton_name
            print(f'    {triton_name}{suffix}/')
            print('      1/  (empty for ensemble)')
            print('      config.pbtxt')

    # Next steps
    print('\nNext steps:')
    print('  1. Restart Triton: docker compose restart triton-api')
    print('  2. Test endpoints:')
    for model_name in model_names:
        print(f"     curl http://localhost:4603/predict/{model_name}_gpu_e2e -F 'image=@test.jpg'")
    print('  3. Run benchmarks: make bench-track-d')

    return 0


if __name__ == '__main__':
    sys.exit(main())
