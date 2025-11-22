#!/usr/bin/env python3
"""
Create Three-Tier Ensemble Configs for Track D

Generates ensemble models that chain DALI preprocessing → YOLO TRT End2End models.

Three tiers optimize for different workloads:
- Streaming: Real-time video (0.1ms batching, low latency)
- Balanced: General purpose (0.5ms batching, balanced)
- Batch: Offline processing (5ms batching, high throughput)

Usage:
    # Create ensembles for specific model sizes
    python dali/create_ensembles.py --models small
    python dali/create_ensembles.py --models nano small

    # Create ensembles for all model sizes
    python dali/create_ensembles.py --models all
"""

import argparse
from pathlib import Path
from typing import Dict, List


# Model configurations
MODELS = {
    "nano": {
        "triton_name": "yolov11_nano",
        "max_batch": 128,
        "topk": 100,
    },
    "small": {
        "triton_name": "yolov11_small",
        "max_batch": 64,
        "topk": 100,
    },
    "medium": {
        "triton_name": "yolov11_medium",
        "max_batch": 32,
        "topk": 100,
    },
}

# Ensemble tier configurations
ENSEMBLE_TIERS = {
    "streaming": {
        "suffix": "_gpu_e2e_streaming",
        "description": "Real-time video streaming (optimized for low latency)",
        "max_batch": 8,
        "preferred_batch_sizes": [1, 2, 4],
        "max_queue_delay_us": 100,       # 0.1ms
        "preserve_ordering": True,        # Essential for video frames
        "timeout_us": 5000,               # 5ms (strict)
        "max_queue_size": 16,
        "instance_count": 3,              # Handle multiple concurrent streams
    },
    "balanced": {
        "suffix": "_gpu_e2e",
        "description": "General purpose (balanced latency/throughput)",
        "max_batch": 64,
        "preferred_batch_sizes": [4, 8, 16, 32],
        "max_queue_delay_us": 500,       # 0.5ms
        "preserve_ordering": False,
        "timeout_us": 50000,              # 50ms
        "max_queue_size": 64,
        "instance_count": 2,
    },
    "batch": {
        "suffix": "_gpu_e2e_batch",
        "description": "Offline batch processing (optimized for throughput)",
        "max_batch": 128,
        "preferred_batch_sizes": [32, 64, 128],
        "max_queue_delay_us": 5000,      # 5ms
        "preserve_ordering": False,
        "timeout_us": 100000,             # 100ms (very lenient)
        "max_queue_size": 256,
        "instance_count": 1,              # Single instance with large batches
    },
}


def generate_ensemble_config(
    model_name: str,
    model_config: Dict,
    tier_name: str,
    tier_config: Dict
) -> str:
    """
    Generate ensemble config.pbtxt content.

    Args:
        model_name: Model name (nano, small, medium)
        model_config: Model configuration
        tier_name: Tier name (streaming, balanced, batch)
        tier_config: Tier configuration

    Returns:
        config.pbtxt content as string
    """
    triton_name = model_config["triton_name"]
    topk = model_config["topk"]
    suffix = tier_config["suffix"]
    description = tier_config["description"]
    max_batch = tier_config["max_batch"]
    preferred_batches = tier_config["preferred_batch_sizes"]
    queue_delay = tier_config["max_queue_delay_us"]
    preserve_order = str(tier_config["preserve_ordering"]).lower()
    timeout = tier_config["timeout_us"]
    queue_size = tier_config["max_queue_size"]
    instance_count = tier_config["instance_count"]

    # Format preferred batch sizes
    batch_list = ", ".join(str(b) for b in preferred_batches)

    config = f"""# {triton_name.upper()}{suffix.upper()} - Track D Ensemble
# {'='*70}
# {description}
#
# Pipeline:
#   1. DALI preprocessing (GPU: decode + letterbox + normalize)
#   2. YOLO TRT End2End (GPU: inference + NMS)
#
# Configuration tier: {tier_name}
# - Max batch: {max_batch}
# - Batching window: {queue_delay/1000:.1f}ms
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
    dims: [ {topk}, 4 ]  # [x, y, w, h] format
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

# Ensemble: DALI → YOLO TRT End2End
ensemble_scheduling {{
  step [
    {{
      # Step 1: DALI preprocessing (GPU decode + letterbox + normalize)
      model_name: "yolo_preprocess_dali"
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

# Dynamic batching configuration (ensemble-level)
dynamic_batching {{
  preferred_batch_size: [ {batch_list} ]
  max_queue_delay_microseconds: {queue_delay}
  preserve_ordering: {preserve_order}

  default_queue_policy {{
    timeout_action: REJECT
    default_timeout_microseconds: {timeout}
    allow_timeout_override: false
    max_queue_size: {queue_size}
  }}
}}

# Instance group (inherited from child models, but specified for clarity)
instance_group [
  {{
    count: {instance_count}
    kind: KIND_GPU
    gpus: [ 0 ]
  }}
]
"""

    return config


def create_ensemble(model_name: str, tier_name: str):
    """
    Create a single ensemble model directory and config.

    Args:
        model_name: Model name (nano, small, medium)
        tier_name: Tier name (streaming, balanced, batch)
    """
    model_config = MODELS[model_name]
    tier_config = ENSEMBLE_TIERS[tier_name]

    # Ensemble name
    triton_name = model_config["triton_name"]
    suffix = tier_config["suffix"]
    ensemble_name = f"{triton_name}{suffix}"

    # Create directory structure
    ensemble_dir = Path(f"/app/models/{ensemble_name}")
    ensemble_dir.mkdir(parents=True, exist_ok=True)
    (ensemble_dir / "1").mkdir(exist_ok=True)  # Empty version directory for ensembles

    # Generate config
    config_content = generate_ensemble_config(model_name, model_config, tier_name, tier_config)

    # Write config
    config_path = ensemble_dir / "config.pbtxt"
    with open(config_path, 'w') as f:
        f.write(config_content)

    print(f"  ✓ Created: {ensemble_name}")
    print(f"    Path: {config_path}")
    print(f"    Tier: {tier_name} ({tier_config['description']})")
    print(f"    Max batch: {tier_config['max_batch']}")
    print(f"    Batching window: {tier_config['max_queue_delay_us']/1000:.1f}ms")


def main():
    """
    Create ensemble models based on command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Create Track D ensemble models (DALI + TRT End2End)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create ensembles for small model only (recommended for clean benchmarking)
  python scripts/create_ensembles.py --models small

  # Create ensembles for multiple specific sizes
  python scripts/create_ensembles.py --models nano small medium

  # Create ensembles for all sizes
  python scripts/create_ensembles.py --models all
        """
    )

    parser.add_argument(
        '--models',
        nargs='+',
        choices=['nano', 'small', 'medium', 'all'],
        default=['small'],
        help='Model sizes to create ensembles for (default: small)'
    )

    args = parser.parse_args()

    # Determine which models to process
    if 'all' in args.models:
        model_names = list(MODELS.keys())
    else:
        model_names = args.models

    # Validate model names
    for model_name in model_names:
        if model_name not in MODELS:
            print(f"Error: Invalid model name '{model_name}'")
            print(f"Available models: {', '.join(MODELS.keys())}")
            return 1

    print("="*80)
    print("Track D Ensemble Model Generator")
    print("="*80)
    print(f"\nCreating {len(model_names)} × 3 = {len(model_names) * 3} ensemble models\n")
    print(f"Model sizes: {', '.join(model_names)}")
    print(f"Tiers: {', '.join(ENSEMBLE_TIERS.keys())}\n")

    total = 0

    for tier_name in ENSEMBLE_TIERS.keys():
        print(f"\n{tier_name.upper()} Tier")
        print("-" * 80)

        for model_name in model_names:
            create_ensemble(model_name, tier_name)
            total += 1

    print("\n" + "="*80)
    print(f"✓ Created {total} ensemble models")
    print("="*80)

    print("\nModel naming convention:")
    for model_name in model_names:
        triton_name = MODELS[model_name]["triton_name"]
        print(f"  {triton_name}_gpu_e2e_streaming  → Real-time streaming (0.1ms batching)")
        print(f"  {triton_name}_gpu_e2e            → General purpose (0.5ms batching)")
        print(f"  {triton_name}_gpu_e2e_batch      → Offline batch (5ms batching)")
        if model_name != model_names[-1]:
            print()

    print("\nDirectory structure:")
    print("  models/")
    for tier_name, tier_config in ENSEMBLE_TIERS.items():
        suffix = tier_config["suffix"]
        for model_name in model_names:
            triton_name = MODELS[model_name]["triton_name"]
            print(f"    ├── {triton_name}{suffix}/")
            print(f"    │   ├── 1/  (empty for ensemble)")
            print(f"    │   └── config.pbtxt")

    print("\nNext steps:")
    print("  1. Update docker-compose.yml to load these models")
    print("  2. Restart Triton: docker compose restart triton-api")
    print("  3. Test endpoints:")
    for model_name in model_names:
        triton_name = MODELS[model_name]["triton_name"]
        print(f"     curl http://localhost:9600/predict/{model_name}_gpu_e2e_streaming -F 'image=@test.jpg'")
    print("  4. Run benchmarks: python benchmarks/compare_all_tracks.py")

    return 0


if __name__ == "__main__":
    exit(main() or 0)
