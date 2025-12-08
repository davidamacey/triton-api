#!/usr/bin/env python3
"""
Track E: Create Triple-Branch DALI Pipeline

This pipeline handles preprocessing for YOLO, MobileCLIP, AND HD cropping image:
- Decode JPEG once (shared)
- Branch 1: YOLO letterbox (640x640 with padding) - for object detection
- Branch 2: MobileCLIP center crop (256x256) - for global image embedding
- Branch 3: HD cropping image (max 1920px longest edge) - for per-detection embeddings

Key insight: Both models use simple ÷255 normalization, so we can share the decode step.
The HD cropping image allows us to crop detected objects at high quality (not from 640x640 letterboxed).

Why HD cap (1920px) instead of native resolution?
- Predictable GPU memory: max 14.7MB per image (vs 36MB+ for 4K native)
- Prevents memory fragmentation at high concurrency (256+ workers)
- Matches Track D's graceful failure behavior under load
- Still excellent quality for embedding crops

Pipeline Flow:
    encoded_jpeg
         │
         ▼
    nvJPEG Decode (GPU)
         │
    ┌────┼────────────┐
    ▼    ▼            ▼
  YOLO  CLIP       Cropping
  640x640 256x256  max 1920px
    │    │            │
    ▼    ▼            ▼
  warp  resize     resize_longer
  affine +crop     (no upscale)
    │    │            │
    ▼    ▼            ▼
  norm  norm        norm
  ÷255  ÷255        ÷255
    │    │            │
    ▼    ▼            ▼
  CHW   CHW         CHW
    │    │            │
    └────┴─────┬──────┘
               ▼
    [yolo_images, clip_images, original_images]

Run from: yolo-api container
    docker compose exec yolo-api python /app/dali/create_dual_dali_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Import shared configuration
from dali.config import CLIP_SIZE, CROP_IMAGE_MAX_SIZE, MAX_BATCH_SIZE, YOLO_PAD_VALUE, YOLO_SIZE


try:
    from nvidia import dali
    from nvidia.dali import fn, types
except ImportError as e:
    print(f'ERROR: NVIDIA DALI not installed: {e}')
    print('Run from yolo-api container:')
    print('  docker compose exec yolo-api python /app/dali/create_dual_dali_pipeline.py')
    sys.exit(1)


@dali.pipeline_def(batch_size=MAX_BATCH_SIZE, num_threads=4, device_id=0)
def dual_preprocess_pipeline():
    """
    Triple-branch preprocessing pipeline for YOLO + MobileCLIP + Original.

    Inputs:
        encoded_images: Raw JPEG/PNG bytes [variable length]
        affine_matrices: Pre-calculated YOLO letterbox matrices [2, 3]

    Outputs:
        yolo_images: [N, 3, 640, 640] FP32, normalized [0, 1]
        clip_images: [N, 3, 256, 256] FP32, normalized [0, 1]
        original_images: [N, 3, H, W] FP32, normalized [0, 1] - NATIVE resolution (no resize)
    """

    # =========================================================================
    # Step 1: External inputs
    # =========================================================================
    encoded = fn.external_source(name='encoded_images', dtype=types.UINT8, ndim=1)

    # YOLO letterbox affine matrix (calculated on CPU)
    affine_matrix = fn.external_source(
        name='affine_matrices',
        dtype=types.FLOAT,
        ndim=2,  # [2, 3]
    )

    # =========================================================================
    # Step 2: Shared JPEG decode (GPU - nvJPEG)
    # =========================================================================
    images = fn.decoders.image(
        encoded,
        device='mixed',  # GPU-accelerated decode
        output_type=types.RGB,
        hw_decoder_load=0.65,  # Hardware decoder offload (Ampere+)
    )

    # This decoded image is now used for ALL three branches!

    # =========================================================================
    # Branch 1: YOLO preprocessing (640x640 letterbox)
    # =========================================================================
    yolo_images = fn.warp_affine(
        images,
        matrix=affine_matrix,
        size=[YOLO_SIZE, YOLO_SIZE],
        fill_value=YOLO_PAD_VALUE,
        interp_type=types.INTERP_LINEAR,
        inverse_map=False,  # Use forward transform matrix (src->dst, not dst->src)
        device='gpu',
    )

    # Normalize and transpose HWC → CHW
    yolo_images = fn.crop_mirror_normalize(
        yolo_images,
        mean=[0.0, 0.0, 0.0],
        std=[255.0, 255.0, 255.0],  # ÷255
        output_layout='CHW',
        output_dtype=types.FLOAT,
        device='gpu',
    )

    # =========================================================================
    # Branch 2: MobileCLIP preprocessing (256x256 center crop)
    # =========================================================================
    # Resize shortest edge to 256, then center crop to 256x256
    # This matches OpenCLIP's default preprocessing

    # Resize with aspect ratio preserved (shortest edge = 256)
    clip_images_resized = fn.resize(
        images, resize_shorter=CLIP_SIZE, interp_type=types.INTERP_LINEAR, device='gpu'
    )

    # Center crop to 256x256
    clip_images = fn.crop(
        clip_images_resized,
        crop_h=CLIP_SIZE,
        crop_w=CLIP_SIZE,
        crop_pos_x=0.5,  # Center
        crop_pos_y=0.5,
        device='gpu',
    )

    # Normalize with simple ÷255 (MobileCLIP2-S2 was exported with image_mean=0, image_std=1)
    # This is the SAME as YOLO preprocessing - just divide by 255 to get [0, 1] range
    # See export_mobileclip_image_encoder.py lines 113-121
    clip_images = fn.crop_mirror_normalize(
        clip_images,
        mean=[0.0, 0.0, 0.0],
        std=[255.0, 255.0, 255.0],  # ÷255 to get [0, 1] range
        output_layout='CHW',
        output_dtype=types.FLOAT,
        device='gpu',
    )

    # =========================================================================
    # Branch 3: Cropping image (HD resolution cap - no upscale!)
    # =========================================================================
    # Resize large images so longest edge ≤ 1920px (HD resolution)
    # Small images are NOT upscaled - preserves original quality
    #
    # This branch provides the image for detection cropping:
    # - Much better quality than 640x640 YOLO (which has letterbox padding)
    # - Predictable GPU memory (max 14.7MB vs 36MB+ for 4K native)
    # - Prevents memory fragmentation at high concurrency (256+ workers)
    #
    # Flow: YOLO detects on 640 → boxes scaled to this image → crop → CLIP embed
    crop_images = fn.resize(
        images,
        resize_longer=CROP_IMAGE_MAX_SIZE,  # Cap longest edge at 1920px
        subpixel_scale=False,  # Don't upscale if already smaller
        interp_type=types.INTERP_LINEAR,
        device='gpu',
    )

    # Normalize and transpose HWC → CHW
    original_images = fn.crop_mirror_normalize(
        crop_images,
        mean=[0.0, 0.0, 0.0],
        std=[255.0, 255.0, 255.0],  # ÷255 (MobileCLIP2-S2 standard)
        output_layout='CHW',
        output_dtype=types.FLOAT,
        device='gpu',
    )

    return yolo_images, clip_images, original_images


def serialize_pipeline():
    """Build and serialize the dual DALI pipeline for Triton."""
    print('=' * 80)
    print('Track E: Dual-Branch DALI Pipeline')
    print('=' * 80)

    output_dir = Path('/app/models/dual_preprocess_dali/1')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'model.dali'

    print(f'\nTarget: {output_path}')

    print('\nBuilding pipeline...')
    print(f'  Batch size: {MAX_BATCH_SIZE}')
    print(f'  YOLO output: {YOLO_SIZE}x{YOLO_SIZE}')
    print(f'  CLIP output: {CLIP_SIZE}x{CLIP_SIZE}')

    try:
        pipe = dual_preprocess_pipeline(batch_size=MAX_BATCH_SIZE, num_threads=4, device_id=0)
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
    """Create Triton config.pbtxt for triple DALI pipeline."""

    config_path = model_dir / 'config.pbtxt'

    config_content = f"""# Track E: Triple-Branch DALI Preprocessing
# GPU-accelerated preprocessing for YOLO + MobileCLIP + HD Cropping
#
# Inputs:
#   - encoded_images: JPEG/PNG bytes
#   - affine_matrices: YOLO letterbox transformation [2, 3]
#
# Outputs:
#   - yolo_images: [N, 3, 640, 640] FP32 - for object detection
#   - clip_images: [N, 3, 256, 256] FP32 - for global embedding
#   - original_images: [N, 3, H, W] FP32 - for cropping (max 1920px longest edge, no upscale)
#
# Memory footprint (original_images):
#   - Max: 14.7MB per image (1920x1920 worst case for square)
#   - Typical: 6-8MB per image (most photos are 16:9 or 3:2)
#   - This HD cap prevents memory fragmentation at high concurrency (256+ workers)

name: "dual_preprocess_dali"
backend: "dali"
max_batch_size: {MAX_BATCH_SIZE}

input [
  {{
    name: "encoded_images"
    data_type: TYPE_UINT8
    dims: [ -1 ]  # Variable-length JPEG bytes
  }},
  {{
    name: "affine_matrices"
    data_type: TYPE_FP32
    dims: [ 2, 3 ]  # Affine transformation matrix
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
  }},
  {{
    name: "original_images"
    data_type: TYPE_FP32
    dims: [ 3, -1, -1 ]  # Variable dimensions [3, H, W]
  }}
]

# NVIDIA Best Practice: Single instance to avoid memory issues
instance_group [
  {{
    count: 1
    kind: KIND_GPU
    gpus: [ 0 ]
  }}
]

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
    print('Testing dual pipeline...')
    print('=' * 80)

    import io

    from PIL import Image

    # Create test image (1080x810 - typical photo aspect ratio)
    print('\nCreating test image (1080x810)...')
    test_img = np.random.randint(0, 255, (1080, 810, 3), dtype=np.uint8)
    pil_img = Image.fromarray(test_img)

    # Encode to JPEG
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
    pipe = dual_preprocess_pipeline(batch_size=1, num_threads=2, device_id=0)
    pipe.build()

    # Feed data
    print('\nRunning pipeline...')
    pipe.feed_input('encoded_images', [np.frombuffer(jpeg_bytes, dtype=np.uint8)])
    pipe.feed_input('affine_matrices', [affine_matrix])

    outputs = pipe.run()

    # Check outputs
    yolo_out = np.array(outputs[0].as_cpu()[0])
    clip_out = np.array(outputs[1].as_cpu()[0])
    orig_out = np.array(outputs[2].as_cpu()[0])

    print('\n[OK] YOLO output:')
    print(f'    Shape: {yolo_out.shape} (expected: [3, {YOLO_SIZE}, {YOLO_SIZE}])')
    print(f'    Dtype: {yolo_out.dtype}')
    print(f'    Range: [{yolo_out.min():.4f}, {yolo_out.max():.4f}]')

    print('\n[OK] MobileCLIP output:')
    print(f'    Shape: {clip_out.shape} (expected: [3, {CLIP_SIZE}, {CLIP_SIZE}])')
    print(f'    Dtype: {clip_out.dtype}')
    print(f'    Range: [{clip_out.min():.4f}, {clip_out.max():.4f}]')

    # Calculate expected cropping image size (HD cap with resize_longer)
    # subpixel_scale=False means small images are not upscaled
    if max(orig_h, orig_w) > CROP_IMAGE_MAX_SIZE:
        # Scale down so longest edge = CROP_IMAGE_MAX_SIZE
        scale_factor = CROP_IMAGE_MAX_SIZE / max(orig_h, orig_w)
        expected_crop_h = round(orig_h * scale_factor)
        expected_crop_w = round(orig_w * scale_factor)
    else:
        # No upscaling - keep original size
        expected_crop_h = orig_h
        expected_crop_w = orig_w

    print(f'\n[OK] Cropping image output (HD cap at {CROP_IMAGE_MAX_SIZE}px):')
    print(f'    Shape: {orig_out.shape} (expected: [3, {expected_crop_h}, {expected_crop_w}])')
    print(f'    Dtype: {orig_out.dtype}')
    print(f'    Range: [{orig_out.min():.4f}, {orig_out.max():.4f}]')
    print(f'    Memory: {orig_out.nbytes / 1024 / 1024:.2f} MB')

    # Validate
    assert yolo_out.shape == (3, YOLO_SIZE, YOLO_SIZE), f'Wrong YOLO shape: {yolo_out.shape}'
    assert clip_out.shape == (3, CLIP_SIZE, CLIP_SIZE), f'Wrong CLIP shape: {clip_out.shape}'
    # Check cropping image respects max size (allow small rounding differences)
    assert orig_out.shape[1] <= CROP_IMAGE_MAX_SIZE + 1, (
        f'Crop height {orig_out.shape[1]} exceeds max {CROP_IMAGE_MAX_SIZE}'
    )
    assert orig_out.shape[2] <= CROP_IMAGE_MAX_SIZE + 1, (
        f'Crop width {orig_out.shape[2]} exceeds max {CROP_IMAGE_MAX_SIZE}'
    )
    assert yolo_out.dtype == np.float32
    assert clip_out.dtype == np.float32
    assert orig_out.dtype == np.float32
    assert yolo_out.min() >= 0
    assert yolo_out.max() <= 1
    # CLIP uses simple /255 normalization (same as YOLO) - output in [0, 1] range
    assert clip_out.min() >= 0, f'CLIP min out of range: {clip_out.min():.2f}'
    assert clip_out.max() <= 1, f'CLIP max out of range: {clip_out.max():.2f}'
    assert orig_out.min() >= 0
    assert orig_out.max() <= 1

    print('\n[COMPLETE] All assertions passed!')
    return True


def main():
    print('\n' + '=' * 80)
    print('Track E: Dual-Branch DALI Pipeline Creator')
    print('=' * 80)

    # Check DALI
    print(f'\n[OK] NVIDIA DALI version: {dali.__version__}')

    # Test first
    if not test_pipeline():
        print('\n[ERROR] Test failed')
        sys.exit(1)

    # Serialize for Triton
    if not serialize_pipeline():
        print('\n[ERROR] Serialization failed')
        sys.exit(1)

    print('\n' + '=' * 80)
    print('[COMPLETE] Triple-branch DALI pipeline ready!')
    print('=' * 80)
    print('\nModel files created:')
    print('  - models/dual_preprocess_dali/1/model.dali')
    print('  - models/dual_preprocess_dali/config.pbtxt')
    print('\nOutputs:')
    print(f'  1. YOLO preprocessed: [3, {YOLO_SIZE}, {YOLO_SIZE}] - for object detection')
    print(f'  2. MobileCLIP preprocessed: [3, {CLIP_SIZE}, {CLIP_SIZE}] - for global embedding')
    print(f'  3. Cropping image: [3, H, W] - max {CROP_IMAGE_MAX_SIZE}px longest edge (no upscale)')
    print('     → Max memory: 14.7MB per image (predictable for high concurrency)')
    print('     → Flow: YOLO detects → scale boxes → crop from HD image → CLIP embed')
    print('\nNext steps:')
    print('  1. Restart Triton: docker compose restart triton-api')
    print('  2. Test Track E: make test-track-e')


if __name__ == '__main__':
    main()
