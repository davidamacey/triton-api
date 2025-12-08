#!/usr/bin/env python3
"""
Track E Simple: Create YOLO + CLIP DALI Pipeline (NO original_images)

This pipeline handles preprocessing for YOLO + MobileCLIP ONLY:
- Decode JPEG once (shared)
- Branch 1: YOLO letterbox (640x640 with padding)
- Branch 2: MobileCLIP center crop (256x256)

NO original_images output = Much faster due to fixed-size outputs only.

Run from: yolo-api container
    docker compose exec yolo-api python /app/dali/create_yolo_clip_dali_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Import shared configuration
from dali.config import CLIP_SIZE, MAX_BATCH_SIZE, YOLO_PAD_VALUE, YOLO_SIZE


try:
    from nvidia import dali
    from nvidia.dali import fn, types
except ImportError as e:
    print(f'ERROR: NVIDIA DALI not installed: {e}')
    print('Run from yolo-api container:')
    print('  docker compose exec yolo-api python /app/dali/create_yolo_clip_dali_pipeline.py')
    sys.exit(1)


@dali.pipeline_def(batch_size=MAX_BATCH_SIZE, num_threads=4, device_id=0)
def yolo_clip_preprocess_pipeline():
    """
    Two-branch preprocessing pipeline for YOLO + MobileCLIP (fast, fixed-size outputs).

    Inputs:
        encoded_images: Raw JPEG/PNG bytes [variable length]
        affine_matrices: Pre-calculated YOLO letterbox matrices [2, 3]

    Outputs:
        yolo_images: [N, 3, 640, 640] FP32, normalized [0, 1]
        clip_images: [N, 3, 256, 256] FP32, normalized [0, 1]
    """

    # External inputs
    encoded = fn.external_source(name='encoded_images', dtype=types.UINT8, ndim=1)

    affine_matrix = fn.external_source(
        name='affine_matrices',
        dtype=types.FLOAT,
        ndim=2,  # [2, 3]
    )

    # Shared JPEG decode (GPU - nvJPEG)
    images = fn.decoders.image(encoded, device='mixed', output_type=types.RGB, hw_decoder_load=0.65)

    # Branch 1: YOLO preprocessing (640x640 letterbox)
    yolo_images = fn.warp_affine(
        images,
        matrix=affine_matrix,
        size=[YOLO_SIZE, YOLO_SIZE],
        fill_value=YOLO_PAD_VALUE,
        interp_type=types.INTERP_LINEAR,
        inverse_map=False,
        device='gpu',
    )

    yolo_images = fn.crop_mirror_normalize(
        yolo_images,
        mean=[0.0, 0.0, 0.0],
        std=[255.0, 255.0, 255.0],
        output_layout='CHW',
        output_dtype=types.FLOAT,
        device='gpu',
    )

    # Branch 2: MobileCLIP preprocessing (256x256 center crop)
    clip_images_resized = fn.resize(
        images, resize_shorter=CLIP_SIZE, interp_type=types.INTERP_LINEAR, device='gpu'
    )

    clip_images = fn.crop(
        clip_images_resized,
        crop_h=CLIP_SIZE,
        crop_w=CLIP_SIZE,
        crop_pos_x=0.5,
        crop_pos_y=0.5,
        device='gpu',
    )

    clip_images = fn.crop_mirror_normalize(
        clip_images,
        mean=[0.0, 0.0, 0.0],
        std=[255.0, 255.0, 255.0],
        output_layout='CHW',
        output_dtype=types.FLOAT,
        device='gpu',
    )

    return yolo_images, clip_images


def serialize_pipeline():
    """Build and serialize the YOLO+CLIP DALI pipeline for Triton."""
    print('=' * 80)
    print('Track E Simple: YOLO + CLIP DALI Pipeline (Fast)')
    print('=' * 80)

    output_dir = Path('/app/models/yolo_clip_preprocess_dali/1')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'model.dali'

    print(f'\nTarget: {output_path}')

    print('\nBuilding pipeline...')
    print(f'  Batch size: {MAX_BATCH_SIZE}')
    print(f'  YOLO output: {YOLO_SIZE}x{YOLO_SIZE}')
    print(f'  CLIP output: {CLIP_SIZE}x{CLIP_SIZE}')
    print('  NO original_images (fast, fixed-size outputs only)')

    try:
        pipe = yolo_clip_preprocess_pipeline(batch_size=MAX_BATCH_SIZE, num_threads=4, device_id=0)
        pipe.build()
        print('[OK] Pipeline built successfully')

    except Exception as e:
        print(f'[ERROR] Build failed: {e}')
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Serialize
    print(f'\nSerializing to: {output_path}')
    pipe.serialize(filename=str(output_path))
    print(f'[OK] Serialized ({output_path.stat().st_size / 1024:.1f} KB)')

    # Create config.pbtxt
    create_triton_config(output_dir.parent)

    return True


def create_triton_config(model_dir):
    """Create Triton config.pbtxt for YOLO+CLIP DALI pipeline."""

    config_path = model_dir / 'config.pbtxt'

    config_content = f"""# Track E Simple: YOLO + CLIP DALI Preprocessing (Fast)
# GPU-accelerated preprocessing for YOLO (640x640) + MobileCLIP (256x256)
# NO original_images output = fixed-size outputs only = fast!
#
# Inputs:
#   - encoded_images: JPEG/PNG bytes
#   - affine_matrices: YOLO letterbox transformation [2, 3]
#
# Outputs:
#   - yolo_images: [N, 3, 640, 640] FP32
#   - clip_images: [N, 3, 256, 256] FP32

name: "yolo_clip_preprocess_dali"
backend: "dali"
max_batch_size: {MAX_BATCH_SIZE}

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
  {{
    name: "yolo_images"
    data_type: TYPE_FP32
    dims: [ 3, {YOLO_SIZE}, {YOLO_SIZE} ]
  }},
  {{
    name: "clip_images"
    data_type: TYPE_FP32
    dims: [ 3, {CLIP_SIZE}, {CLIP_SIZE} ]
  }}
]

# High throughput configuration - multiple instances for pipeline parallelism
instance_group [
  {{
    count: 6
    kind: KIND_GPU
    gpus: [ 0 ]
  }}
]

dynamic_batching {{
  preferred_batch_size: [ 4, 8, 16, 32 ]
  max_queue_delay_microseconds: 10000
}}

parameters: {{
  key: "num_threads"
  value: {{ string_value: "4" }}
}}
"""

    with open(config_path, 'w') as f:
        f.write(config_content)

    print(f'[OK] Config created: {config_path}')


def test_pipeline():
    """Test the pipeline with dummy data."""
    print('\n' + '=' * 80)
    print('Testing YOLO+CLIP pipeline...')
    print('=' * 80)

    import io

    from PIL import Image

    # Create test image
    print('\nCreating test image (1080x810)...')
    test_img = np.random.randint(0, 255, (1080, 810, 3), dtype=np.uint8)
    pil_img = Image.fromarray(test_img)

    buffer = io.BytesIO()
    pil_img.save(buffer, format='JPEG', quality=90)
    jpeg_bytes = buffer.getvalue()
    print(f'  JPEG size: {len(jpeg_bytes)} bytes')

    # Calculate YOLO letterbox affine matrix
    orig_h, orig_w = 1080, 810
    scale = min(YOLO_SIZE / orig_h, YOLO_SIZE / orig_w)
    scale = min(scale, 1.0)

    new_h = round(orig_h * scale)
    new_w = round(orig_w * scale)
    pad_x = (YOLO_SIZE - new_w) / 2.0
    pad_y = (YOLO_SIZE - new_h) / 2.0

    affine_matrix = np.array([[scale, 0.0, pad_x], [0.0, scale, pad_y]], dtype=np.float32)

    print(f'  Affine matrix: scale={scale:.4f}, pad=({pad_x:.1f}, {pad_y:.1f})')

    # Build pipeline
    pipe = yolo_clip_preprocess_pipeline(batch_size=1, num_threads=2, device_id=0)
    pipe.build()

    # Feed data
    print('\nRunning pipeline...')
    pipe.feed_input('encoded_images', [np.frombuffer(jpeg_bytes, dtype=np.uint8)])
    pipe.feed_input('affine_matrices', [affine_matrix])

    outputs = pipe.run()

    # Check outputs
    yolo_out = np.array(outputs[0].as_cpu()[0])
    clip_out = np.array(outputs[1].as_cpu()[0])

    print('\n[OK] YOLO output:')
    print(f'    Shape: {yolo_out.shape} (expected: [3, {YOLO_SIZE}, {YOLO_SIZE}])')
    print(f'    Range: [{yolo_out.min():.4f}, {yolo_out.max():.4f}]')

    print('\n[OK] MobileCLIP output:')
    print(f'    Shape: {clip_out.shape} (expected: [3, {CLIP_SIZE}, {CLIP_SIZE}])')
    print(f'    Range: [{clip_out.min():.4f}, {clip_out.max():.4f}]')

    # Validate
    assert yolo_out.shape == (3, YOLO_SIZE, YOLO_SIZE), f'Wrong YOLO shape: {yolo_out.shape}'
    assert clip_out.shape == (3, CLIP_SIZE, CLIP_SIZE), f'Wrong CLIP shape: {clip_out.shape}'
    assert yolo_out.dtype == np.float32
    assert clip_out.dtype == np.float32
    assert yolo_out.min() >= 0
    assert yolo_out.max() <= 1
    assert clip_out.min() >= 0
    assert clip_out.max() <= 1

    print('\n[COMPLETE] All assertions passed!')
    return True


def main():
    print('\n' + '=' * 80)
    print('Track E Simple: YOLO + CLIP DALI Pipeline Creator')
    print('=' * 80)

    print(f'\n[OK] NVIDIA DALI version: {dali.__version__}')

    if not test_pipeline():
        print('\n[ERROR] Test failed')
        sys.exit(1)

    if not serialize_pipeline():
        print('\n[ERROR] Serialization failed')
        sys.exit(1)

    print('\n' + '=' * 80)
    print('[COMPLETE] YOLO + CLIP DALI pipeline ready!')
    print('=' * 80)
    print('\nModel files created:')
    print('  - models/yolo_clip_preprocess_dali/1/model.dali')
    print('  - models/yolo_clip_preprocess_dali/config.pbtxt')
    print('\nOutputs (FIXED SIZE = FAST):')
    print('  1. YOLO preprocessed: [3, 640, 640]')
    print('  2. MobileCLIP preprocessed: [3, 256, 256]')
    print('\nNext steps:')
    print('  1. Update yolo_clip_ensemble to use yolo_clip_preprocess_dali')
    print('  2. Restart Triton: docker compose restart triton-api')


if __name__ == '__main__':
    main()
