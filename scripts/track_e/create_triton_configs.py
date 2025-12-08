#!/usr/bin/env python3
"""
Track E: Create Triton Model Configurations

This script creates all config.pbtxt files for Track E models:
1. mobileclip2_s2_image_encoder - TensorRT plan
2. mobileclip2_s2_text_encoder - TensorRT plan
3. box_embedding_extractor - Python backend
4. yolo_mobileclip_ensemble - Ensemble pipeline

Run from: yolo-api container
    docker compose exec yolo-api python /app/scripts/track_e/create_triton_configs.py
"""

from pathlib import Path


# Configuration
MODEL_VARIANT = 'mobileclip2_s2'  # Change to mobileclip2_b for B variant
IMAGE_SIZE = 256
EMBEDDING_DIM = 768
CONTEXT_LENGTH = 77
MAX_BATCH_SIZE = 128


def create_image_encoder_config():
    """Create config for MobileCLIP2 image encoder (TensorRT)."""

    model_dir = Path(f'/app/models/{MODEL_VARIANT}_image_encoder')
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / '1').mkdir(exist_ok=True)

    config_path = model_dir / 'config.pbtxt'

    config = f"""# MobileCLIP2 Image Encoder - TensorRT
# Converts preprocessed images to 768-dim L2-normalized embeddings
#
# Input: [B, 3, 256, 256] FP32, normalized [0, 1]
# Output: [B, 768] FP32, L2-normalized

name: "{MODEL_VARIANT}_image_encoder"
platform: "tensorrt_plan"
max_batch_size: {MAX_BATCH_SIZE}

input [
  {{
    name: "images"
    data_type: TYPE_FP32
    dims: [ 3, {IMAGE_SIZE}, {IMAGE_SIZE} ]
  }}
]

output [
  {{
    name: "image_embeddings"
    data_type: TYPE_FP32
    dims: [ {EMBEDDING_DIM} ]
  }}
]

# Dynamic batching for maximum throughput
dynamic_batching {{
  preferred_batch_size: [ 8, 16, 32, 64 ]
  max_queue_delay_microseconds: 5000
}}

# Multiple instances for parallel processing
instance_group [
  {{
    count: 2
    kind: KIND_GPU
    gpus: [ 0 ]
  }}
]

# TensorRT optimization
optimization {{
  cuda {{
    graphs: true
    busy_wait_events: true
  }}
}}
"""

    with open(config_path, 'w') as f:
        f.write(config)

    print(f'✓ Image encoder config: {config_path}')
    return config_path


def create_text_encoder_config():
    """Create config for MobileCLIP2 text encoder (TensorRT)."""

    model_dir = Path(f'/app/models/{MODEL_VARIANT}_text_encoder')
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / '1').mkdir(exist_ok=True)

    config_path = model_dir / 'config.pbtxt'

    config = f"""# MobileCLIP2 Text Encoder - TensorRT
# Converts tokenized text to 768-dim L2-normalized embeddings
#
# Input: [B, 77] INT64 token IDs
# Output: [B, 768] FP32, L2-normalized

name: "{MODEL_VARIANT}_text_encoder"
platform: "tensorrt_plan"
max_batch_size: 64

input [
  {{
    name: "text_tokens"
    data_type: TYPE_INT64
    dims: [ {CONTEXT_LENGTH} ]
  }}
]

output [
  {{
    name: "text_embeddings"
    data_type: TYPE_FP32
    dims: [ {EMBEDDING_DIM} ]
  }}
]

# Dynamic batching
dynamic_batching {{
  preferred_batch_size: [ 4, 8, 16 ]
  max_queue_delay_microseconds: 5000
}}

instance_group [
  {{
    count: 1
    kind: KIND_GPU
    gpus: [ 0 ]
  }}
]
"""

    with open(config_path, 'w') as f:
        f.write(config)

    print(f'✓ Text encoder config: {config_path}')
    return config_path


def create_box_embedding_extractor_config():
    """Create config for box embedding extractor (Python backend)."""

    # NOTE: This config is already created in models/box_embedding_extractor/config.pbtxt
    # This function is kept for completeness but the actual config file already exists

    print(
        '✓ Box embedding extractor config: /app/models/box_embedding_extractor/config.pbtxt (already exists)'
    )
    return Path('/app/models/box_embedding_extractor/config.pbtxt')


def create_ensemble_config():
    """Create ensemble config combining DALI + YOLO + MobileCLIP."""

    model_dir = Path('/app/models/yolo_mobileclip_ensemble')
    model_dir.mkdir(parents=True, exist_ok=True)

    config_path = model_dir / 'config.pbtxt'

    MAX_DETS = 100

    config = f"""# Track E: YOLO + MobileCLIP Visual Search Ensemble
#
# Full GPU pipeline for visual search with native-resolution cropping:
# 1. DALI: Decode + triple preprocessing (YOLO 640x640, CLIP 256x256, Original NATIVE-res)
# 2. YOLO: Object detection with GPU NMS (parallel with step 3)
# 3. MobileCLIP: Global image embedding (parallel with step 2)
# 4. Box Extractor: Per-object embeddings from NATIVE-RES crops (via BLS + ROI align)
#
# Key Features:
# - Native resolution: No upscaling of small images, no arbitrary downscaling
# - Normalized outputs: Bounding boxes in [0, 1] range for any image size
# - MobileCLIP2 compliant: Follows official preprocessing guidelines
# - High quality: Crops from original resolution, not from 256x256
#
# Input: Raw JPEG bytes + affine matrix
# Outputs: Detections + embeddings + normalized boxes [0, 1]

name: "yolo_mobileclip_ensemble"
platform: "ensemble"
max_batch_size: 64

input [
  {{
    name: "encoded_images"
    data_type: TYPE_UINT8
    dims: [ -1 ]
  }},
  {{
    name: "affine_matrices"
    data_type: TYPE_FP32
    dims: [ 2, 3 ]
  }}
]

output [
  # YOLO detections
  {{
    name: "num_dets"
    data_type: TYPE_INT32
    dims: [ 1 ]
  }},
  {{
    name: "det_boxes"
    data_type: TYPE_FP32
    dims: [ {MAX_DETS}, 4 ]
  }},
  {{
    name: "det_scores"
    data_type: TYPE_FP32
    dims: [ {MAX_DETS} ]
  }},
  {{
    name: "det_classes"
    data_type: TYPE_INT32
    dims: [ {MAX_DETS} ]
  }},
  # MobileCLIP embeddings
  {{
    name: "global_embeddings"
    data_type: TYPE_FP32
    dims: [ {EMBEDDING_DIM} ]
  }},
  {{
    name: "box_embeddings"
    data_type: TYPE_FP32
    dims: [ {MAX_DETS}, {EMBEDDING_DIM} ]
  }},
  {{
    name: "normalized_boxes"
    data_type: TYPE_FP32
    dims: [ {MAX_DETS}, 4 ]  # Normalized [0, 1] boxes for any image size
  }}
]

ensemble_scheduling {{
  step [
    # Step 1: Triple-branch DALI preprocessing
    # Outputs: yolo_images (640x640) + clip_images (256x256) + original_images (full-res)
    {{
      model_name: "dual_preprocess_dali"
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
        key: "yolo_images"
        value: "_yolo_images"
      }}
      output_map {{
        key: "clip_images"
        value: "_clip_images"
      }}
      output_map {{
        key: "original_images"
        value: "_original_images"
      }}
    }},

    # Step 2: YOLO detection (runs in parallel with Step 3)
    {{
      model_name: "yolov11_small_trt_end2end"
      model_version: -1
      input_map {{
        key: "images"
        value: "_yolo_images"
      }}
      output_map {{
        key: "num_dets"
        value: "num_dets"
      }}
      output_map {{
        key: "det_boxes"
        value: "_det_boxes"
      }}
      output_map {{
        key: "det_scores"
        value: "det_scores"
      }}
      output_map {{
        key: "det_classes"
        value: "det_classes"
      }}
    }},

    # Step 3: Global image embedding (runs in parallel with Step 2)
    {{
      model_name: "{MODEL_VARIANT}_image_encoder"
      model_version: -1
      input_map {{
        key: "images"
        value: "_clip_images"
      }}
      output_map {{
        key: "image_embeddings"
        value: "global_embeddings"
      }}
    }},

    # Step 4: Per-box embeddings (depends on Steps 1 & 2)
    # Uses native-resolution original image for high-quality crops
    # Outputs normalized boxes [0, 1] for any image size
    {{
      model_name: "box_embedding_extractor"
      model_version: -1
      input_map {{
        key: "original_image"
        value: "_original_images"
      }}
      input_map {{
        key: "det_boxes"
        value: "_det_boxes"
      }}
      input_map {{
        key: "num_dets"
        value: "num_dets"
      }}
      input_map {{
        key: "affine_matrix"
        value: "affine_matrices"
      }}
      output_map {{
        key: "box_embeddings"
        value: "box_embeddings"
      }}
      output_map {{
        key: "normalized_boxes"
        value: "normalized_boxes"
      }}
    }}
  ]
}}
"""

    with open(config_path, 'w') as f:
        f.write(config)

    print(f'✓ Ensemble config: {config_path}')
    return config_path


def main():
    print('=' * 80)
    print('Track E: Creating Triton Model Configurations')
    print('=' * 80)
    print(f'\nModel variant: {MODEL_VARIANT}')

    # Create all configs
    create_image_encoder_config()
    create_text_encoder_config()
    create_box_embedding_extractor_config()
    create_ensemble_config()

    print('\n' + '=' * 80)
    print('✅ All configurations created!')
    print('=' * 80)
    print('\nModel directories:')
    print(f'  - models/{MODEL_VARIANT}_image_encoder/')
    print(f'  - models/{MODEL_VARIANT}_text_encoder/')
    print('  - models/box_embedding_extractor/')
    print('  - models/yolo_mobileclip_ensemble/')
    print('\nFiles created/updated:')
    print('  ✓ MobileCLIP image encoder config')
    print('  ✓ MobileCLIP text encoder config')
    print('  ✓ Box embedding extractor config (already exists)')
    print('  ✓ Ensemble config (triple-branch DALI + full-res cropping)')
    print('\nNext steps:')
    print('  1. Setup MobileCLIP environment: bash scripts/track_e/setup_mobileclip_env.sh')
    print('  2. Export MobileCLIP models to ONNX/TensorRT')
    print('  3. Create DALI pipeline: python dali/create_dual_dali_pipeline.py')
    print('  4. Restart Triton: docker compose restart triton-api')
    print('  5. Test ensemble: python scripts/track_e/test_ensemble.py')


if __name__ == '__main__':
    main()
