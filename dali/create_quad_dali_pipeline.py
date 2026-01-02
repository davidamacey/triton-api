#!/usr/bin/env python3
"""
Track E: Create Quad-Branch DALI Pipeline

This pipeline handles preprocessing for YOLO, MobileCLIP, SCRFD, AND HD cropping:
- Decode JPEG once (shared)
- Branch 1: YOLO letterbox (640x640 with padding) - for object detection
- Branch 2: MobileCLIP center crop (256x256) - for global image embedding
- Branch 3: SCRFD resize (640x640 no padding) - for face detection
- Branch 4: HD cropping image (max 1920px longest edge) - for per-detection embeddings

Key insight: All models use simple /255 normalization, so we can share the decode step.
The quad-branch enables parallel object detection AND face detection in a single pass.

Pipeline Flow:
    encoded_jpeg
         |
         v
    nvJPEG Decode (GPU)
         |
    +----+----+--------+--------+
    v    v    v        v        v
  YOLO  CLIP SCRFD  Cropping
  640   256  640    max 1920px
    |    |    |        |
    v    v    v        v
  warp  crop resize  resize_longer
  affine             (no upscale)
    |    |    |        |
    v    v    v        v
  norm  norm norm    norm
  /255  /255 /255    /255
    |    |    |        |
    v    v    v        v
  CHW   CHW  CHW     CHW
    |    |    |        |
    +----+----+---+----+
                  v
    [yolo_images, clip_images, face_images, original_images]

Run from: yolo-api container
    docker compose exec yolo-api python /app/dali/create_quad_dali_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Import shared configuration
from dali.config import (
    CLIP_SIZE,
    CROP_IMAGE_MAX_SIZE,
    MAX_BATCH_SIZE,
    SCRFD_SIZE,
    YOLO_PAD_VALUE,
    YOLO_SIZE,
)


try:
    from nvidia import dali
    from nvidia.dali import fn, types
except ImportError as e:
    print(f'ERROR: NVIDIA DALI not installed: {e}')
    print('Run from yolo-api container:')
    print('  docker compose exec yolo-api python /app/dali/create_quad_dali_pipeline.py')
    sys.exit(1)


@dali.pipeline_def(batch_size=MAX_BATCH_SIZE, num_threads=4, device_id=0)
def quad_preprocess_pipeline():
    """
    Quad-branch preprocessing pipeline for YOLO + MobileCLIP + SCRFD + Original.

    Inputs:
        encoded_images: Raw JPEG/PNG bytes [variable length]
        affine_matrices: Pre-calculated YOLO letterbox matrices [2, 3]

    Outputs:
        yolo_images: [N, 3, 640, 640] FP32, normalized [0, 1] - for YOLO detection
        clip_images: [N, 3, 256, 256] FP32, normalized [0, 1] - for global embedding
        face_images: [N, 3, 640, 640] FP32, normalized [0, 1] - for SCRFD face detection
        original_images: [N, 3, H, W] FP32, normalized [0, 1] - for cropping
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

    # This decoded image is now used for ALL four branches!

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

    # Normalize and transpose HWC -> CHW
    yolo_images = fn.crop_mirror_normalize(
        yolo_images,
        mean=[0.0, 0.0, 0.0],
        std=[255.0, 255.0, 255.0],  # /255
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

    # Normalize with simple /255 (MobileCLIP2-S2 was exported with image_mean=0, image_std=1)
    clip_images = fn.crop_mirror_normalize(
        clip_images,
        mean=[0.0, 0.0, 0.0],
        std=[255.0, 255.0, 255.0],  # /255 to get [0, 1] range
        output_layout='CHW',
        output_dtype=types.FLOAT,
        device='gpu',
    )

    # =========================================================================
    # Branch 3: SCRFD face detection preprocessing (640x640 resize - NO letterbox)
    # =========================================================================
    # SCRFD uses simple resize without letterbox padding
    # This is different from YOLO which uses letterbox to preserve aspect ratio
    face_images = fn.resize(
        images,
        size=[SCRFD_SIZE, SCRFD_SIZE],
        interp_type=types.INTERP_LINEAR,
        device='gpu',
    )

    # Normalize to [0, 1] and transpose HWC -> CHW
    face_images = fn.crop_mirror_normalize(
        face_images,
        mean=[0.0, 0.0, 0.0],
        std=[255.0, 255.0, 255.0],  # /255
        output_layout='CHW',
        output_dtype=types.FLOAT,
        device='gpu',
    )

    # =========================================================================
    # Branch 4: Cropping image (HD resolution cap - no upscale!)
    # =========================================================================
    # Resize large images so longest edge <= 1920px (HD resolution)
    # Small images are NOT upscaled - preserves original quality
    crop_images = fn.resize(
        images,
        resize_longer=CROP_IMAGE_MAX_SIZE,  # Cap longest edge at 1920px
        subpixel_scale=False,  # Don't upscale if already smaller
        interp_type=types.INTERP_LINEAR,
        device='gpu',
    )

    # Normalize and transpose HWC -> CHW
    original_images = fn.crop_mirror_normalize(
        crop_images,
        mean=[0.0, 0.0, 0.0],
        std=[255.0, 255.0, 255.0],  # /255
        output_layout='CHW',
        output_dtype=types.FLOAT,
        device='gpu',
    )

    return yolo_images, clip_images, face_images, original_images


def serialize_pipeline():
    """Build and serialize the quad DALI pipeline for Triton."""
    print('=' * 80)
    print('Track E: Quad-Branch DALI Pipeline')
    print('=' * 80)

    output_dir = Path('/app/models/quad_preprocess_dali/1')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'model.dali'

    print(f'\nTarget: {output_path}')

    print('\nBuilding pipeline...')
    print(f'  Batch size: {MAX_BATCH_SIZE}')
    print(f'  YOLO output: {YOLO_SIZE}x{YOLO_SIZE}')
    print(f'  CLIP output: {CLIP_SIZE}x{CLIP_SIZE}')
    print(f'  SCRFD output: {SCRFD_SIZE}x{SCRFD_SIZE}')
    print(f'  Cropping max: {CROP_IMAGE_MAX_SIZE}px')

    try:
        pipe = quad_preprocess_pipeline(batch_size=MAX_BATCH_SIZE, num_threads=4, device_id=0)
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
    """Create Triton config.pbtxt for quad DALI pipeline."""

    config_path = model_dir / 'config.pbtxt'

    config_content = f"""# Track E: Quad-Branch DALI Preprocessing
# GPU-accelerated preprocessing for YOLO + MobileCLIP + SCRFD + HD Cropping
#
# Enables parallel object detection AND face detection in a single pass!
#
# Inputs:
#   - encoded_images: JPEG/PNG bytes
#   - affine_matrices: YOLO letterbox transformation [2, 3]
#
# Outputs:
#   - yolo_images: [N, 3, 640, 640] FP32 - for YOLO object detection
#   - clip_images: [N, 3, 256, 256] FP32 - for global MobileCLIP embedding
#   - face_images: [N, 3, 640, 640] FP32 - for SCRFD face detection
#   - original_images: [N, 3, H, W] FP32 - for cropping (max 1920px, no upscale)

name: "quad_preprocess_dali"
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
    name: "face_images"
    data_type: TYPE_FP32
    dims: [ 3, {SCRFD_SIZE}, {SCRFD_SIZE} ]
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
    print('Testing quad pipeline...')
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
    pipe = quad_preprocess_pipeline(batch_size=1, num_threads=2, device_id=0)
    pipe.build()

    # Feed data
    print('\nRunning pipeline...')
    pipe.feed_input('encoded_images', [np.frombuffer(jpeg_bytes, dtype=np.uint8)])
    pipe.feed_input('affine_matrices', [affine_matrix])

    outputs = pipe.run()

    # Check outputs
    yolo_out = np.array(outputs[0].as_cpu()[0])
    clip_out = np.array(outputs[1].as_cpu()[0])
    face_out = np.array(outputs[2].as_cpu()[0])
    orig_out = np.array(outputs[3].as_cpu()[0])

    print('\n[OK] YOLO output:')
    print(f'    Shape: {yolo_out.shape} (expected: [3, {YOLO_SIZE}, {YOLO_SIZE}])')
    print(f'    Dtype: {yolo_out.dtype}')
    print(f'    Range: [{yolo_out.min():.4f}, {yolo_out.max():.4f}]')

    print('\n[OK] MobileCLIP output:')
    print(f'    Shape: {clip_out.shape} (expected: [3, {CLIP_SIZE}, {CLIP_SIZE}])')
    print(f'    Dtype: {clip_out.dtype}')
    print(f'    Range: [{clip_out.min():.4f}, {clip_out.max():.4f}]')

    print('\n[OK] SCRFD face output:')
    print(f'    Shape: {face_out.shape} (expected: [3, {SCRFD_SIZE}, {SCRFD_SIZE}])')
    print(f'    Dtype: {face_out.dtype}')
    print(f'    Range: [{face_out.min():.4f}, {face_out.max():.4f}]')

    # Calculate expected cropping image size (HD cap with resize_longer)
    if max(orig_h, orig_w) > CROP_IMAGE_MAX_SIZE:
        scale_factor = CROP_IMAGE_MAX_SIZE / max(orig_h, orig_w)
        expected_crop_h = round(orig_h * scale_factor)
        expected_crop_w = round(orig_w * scale_factor)
    else:
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
    assert face_out.shape == (3, SCRFD_SIZE, SCRFD_SIZE), f'Wrong SCRFD shape: {face_out.shape}'
    assert orig_out.shape[1] <= CROP_IMAGE_MAX_SIZE + 1, (
        f'Crop height {orig_out.shape[1]} exceeds max {CROP_IMAGE_MAX_SIZE}'
    )
    assert orig_out.shape[2] <= CROP_IMAGE_MAX_SIZE + 1, (
        f'Crop width {orig_out.shape[2]} exceeds max {CROP_IMAGE_MAX_SIZE}'
    )
    assert yolo_out.dtype == np.float32
    assert clip_out.dtype == np.float32
    assert face_out.dtype == np.float32
    assert orig_out.dtype == np.float32
    assert yolo_out.min() >= 0
    assert yolo_out.max() <= 1
    assert clip_out.min() >= 0
    assert clip_out.max() <= 1
    assert face_out.min() >= 0
    assert face_out.max() <= 1
    assert orig_out.min() >= 0
    assert orig_out.max() <= 1

    print('\n[COMPLETE] All assertions passed!')
    return True


def main():
    print('\n' + '=' * 80)
    print('Track E: Quad-Branch DALI Pipeline Creator')
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
    print('[COMPLETE] Quad-branch DALI pipeline ready!')
    print('=' * 80)
    print('\nModel files created:')
    print('  - models/quad_preprocess_dali/1/model.dali')
    print('  - models/quad_preprocess_dali/config.pbtxt')
    print('\nOutputs:')
    print(f'  1. YOLO preprocessed: [3, {YOLO_SIZE}, {YOLO_SIZE}] - for object detection')
    print(f'  2. MobileCLIP preprocessed: [3, {CLIP_SIZE}, {CLIP_SIZE}] - for global embedding')
    print(f'  3. SCRFD preprocessed: [3, {SCRFD_SIZE}, {SCRFD_SIZE}] - for face detection')
    print(f'  4. Cropping image: [3, H, W] - max {CROP_IMAGE_MAX_SIZE}px (no upscale)')
    print('\nNext steps:')
    print('  1. Restart Triton: docker compose restart triton-api')
    print('  2. Test Track E faces: make test-track-e-faces')


if __name__ == '__main__':
    main()
