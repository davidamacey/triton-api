#!/usr/bin/env python3
"""
Track E: Create Penta-Branch DALI Pipeline (with OCR)

This pipeline extends quad-branch to handle preprocessing for YOLO, MobileCLIP,
SCRFD, HD cropping, AND PP-OCRv5 text detection:

- Decode JPEG once (shared)
- Branch 1: YOLO letterbox (640x640 with padding) - for object detection
- Branch 2: MobileCLIP center crop (256x256) - for global image embedding
- Branch 3: SCRFD resize (640x640 no padding) - for face detection
- Branch 4: HD cropping image (max 1920px longest edge) - for per-detection embeddings
- Branch 5: OCR preprocessing (max 960px, pad to 32px, BGR) - for text detection

Key insight: OCR uses different normalization: (x / 127.5) - 1 instead of x / 255.
The penta-branch enables parallel processing of all modalities in a single pass.

Pipeline Flow:
    encoded_jpeg
         |
         v
    nvJPEG Decode (GPU)
         |
    +----+----+--------+--------+--------+
    v    v    v        v        v        v
  YOLO  CLIP SCRFD  Cropping   OCR
  640   256  640    max 1920   max 960
    |    |    |        |          |
    v    v    v        v          v
  warp  crop resize  resize    resize
  affine             longer    longer
    |    |    |        |          |
    v    v    v        v          v
  norm  norm norm    norm      norm_ocr
  /255  /255 /255    /255      /127.5-1
    |    |    |        |          |
    v    v    v        v          v
  CHW   CHW  CHW     CHW        CHW
    |    |    |        |          |
    +----+----+---+----+----+-----+
                  v
    [yolo_images, clip_images, face_images, original_images, ocr_images]

Run from: yolo-api container
    docker compose exec yolo-api python /app/dali/create_penta_dali_pipeline.py
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
    OCR_MAX_SIZE,
    OCR_NORM_MEAN,
    OCR_NORM_STD,
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
    print('  docker compose exec yolo-api python /app/dali/create_penta_dali_pipeline.py')
    sys.exit(1)


@dali.pipeline_def(batch_size=MAX_BATCH_SIZE, num_threads=4, device_id=0)
def penta_preprocess_pipeline():
    """
    Penta-branch preprocessing pipeline for YOLO + MobileCLIP + SCRFD + Original + OCR.

    Inputs:
        encoded_images: Raw JPEG/PNG bytes [variable length]
        affine_matrices: Pre-calculated YOLO letterbox matrices [2, 3]

    Outputs:
        yolo_images: [N, 3, 640, 640] FP32, normalized [0, 1] - for YOLO detection
        clip_images: [N, 3, 256, 256] FP32, normalized [0, 1] - for global embedding
        face_images: [N, 3, 640, 640] FP32, normalized [0, 1] - for SCRFD face detection
        original_images: [N, 3, H, W] FP32, normalized [0, 1] - for cropping
        ocr_images: [N, 3, H, W] FP32, normalized [-1, 1] - for PP-OCR text detection
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

    # This decoded image is now used for ALL five branches!

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

    # =========================================================================
    # Branch 5: OCR preprocessing (max 960px, PP-OCR normalization)
    # =========================================================================
    # PP-OCRv5 uses different preprocessing:
    # - Max resolution 960px on longest edge
    # - Normalization: (x / 127.5) - 1 = [-1, 1] range
    # - Note: PP-OCR expects BGR but DALI outputs RGB
    #   We handle RGB->BGR conversion in the Python BLS model

    ocr_resized = fn.resize(
        images,
        resize_longer=OCR_MAX_SIZE,  # Cap at 960px
        subpixel_scale=False,  # Don't upscale small images
        interp_type=types.INTERP_LINEAR,
        device='gpu',
    )

    # PP-OCR normalization: (x - 127.5) / 127.5 = x / 127.5 - 1
    # This gives range [-1, 1] instead of [0, 1]
    ocr_images = fn.crop_mirror_normalize(
        ocr_resized,
        mean=OCR_NORM_MEAN,  # [127.5, 127.5, 127.5]
        std=OCR_NORM_STD,    # [127.5, 127.5, 127.5]
        output_layout='CHW',
        output_dtype=types.FLOAT,
        device='gpu',
    )

    return yolo_images, clip_images, face_images, original_images, ocr_images


def serialize_pipeline():
    """Build and serialize the penta DALI pipeline for Triton."""
    print('=' * 80)
    print('Track E: Penta-Branch DALI Pipeline (with OCR)')
    print('=' * 80)

    output_dir = Path('/app/models/penta_preprocess_dali/1')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'model.dali'

    print(f'\nTarget: {output_path}')

    print('\nBuilding pipeline...')
    print(f'  Batch size: {MAX_BATCH_SIZE}')
    print(f'  YOLO output: {YOLO_SIZE}x{YOLO_SIZE}')
    print(f'  CLIP output: {CLIP_SIZE}x{CLIP_SIZE}')
    print(f'  SCRFD output: {SCRFD_SIZE}x{SCRFD_SIZE}')
    print(f'  Cropping max: {CROP_IMAGE_MAX_SIZE}px')
    print(f'  OCR max: {OCR_MAX_SIZE}px')

    try:
        pipe = penta_preprocess_pipeline(batch_size=MAX_BATCH_SIZE, num_threads=4, device_id=0)
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
    """Create Triton config.pbtxt for penta DALI pipeline."""

    config_path = model_dir / 'config.pbtxt'

    config_content = f"""# Track E: Penta-Branch DALI Preprocessing (with OCR)
# GPU-accelerated preprocessing for YOLO + MobileCLIP + SCRFD + HD Cropping + OCR
#
# Enables parallel object detection, face detection, AND text detection in one pass!
#
# Inputs:
#   - encoded_images: JPEG/PNG bytes
#   - affine_matrices: YOLO letterbox transformation [2, 3]
#
# Outputs:
#   - yolo_images: [N, 3, 640, 640] FP32 [0,1] - for YOLO object detection
#   - clip_images: [N, 3, 256, 256] FP32 [0,1] - for global MobileCLIP embedding
#   - face_images: [N, 3, 640, 640] FP32 [0,1] - for SCRFD face detection
#   - original_images: [N, 3, H, W] FP32 [0,1] - for cropping (max 1920px)
#   - ocr_images: [N, 3, H, W] FP32 [-1,1] - for PP-OCR text detection (max 960px)

name: "penta_preprocess_dali"
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
  }},
  {{
    name: "ocr_images"
    data_type: TYPE_FP32
    dims: [ 3, -1, -1 ]  # Variable dimensions [3, H, W], max {OCR_MAX_SIZE}px
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
    print('Testing penta pipeline...')
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
    pipe = penta_preprocess_pipeline(batch_size=1, num_threads=2, device_id=0)
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
    ocr_out = np.array(outputs[4].as_cpu()[0])

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

    # Calculate expected OCR image size
    if max(orig_h, orig_w) > OCR_MAX_SIZE:
        ocr_scale = OCR_MAX_SIZE / max(orig_h, orig_w)
        expected_ocr_h = round(orig_h * ocr_scale)
        expected_ocr_w = round(orig_w * ocr_scale)
    else:
        expected_ocr_h = orig_h
        expected_ocr_w = orig_w

    print(f'\n[OK] OCR output (max {OCR_MAX_SIZE}px, norm [-1,1]):')
    print(f'    Shape: {ocr_out.shape} (expected: [3, ~{expected_ocr_h}, ~{expected_ocr_w}])')
    print(f'    Dtype: {ocr_out.dtype}')
    print(f'    Range: [{ocr_out.min():.4f}, {ocr_out.max():.4f}] (should be ~[-1, 1])')

    # Validate
    assert yolo_out.shape == (3, YOLO_SIZE, YOLO_SIZE), f'Wrong YOLO shape: {yolo_out.shape}'
    assert clip_out.shape == (3, CLIP_SIZE, CLIP_SIZE), f'Wrong CLIP shape: {clip_out.shape}'
    assert face_out.shape == (3, SCRFD_SIZE, SCRFD_SIZE), f'Wrong SCRFD shape: {face_out.shape}'
    assert orig_out.shape[1] <= CROP_IMAGE_MAX_SIZE + 1, (
        f'Crop height {orig_out.shape[1]} exceeds max {CROP_IMAGE_MAX_SIZE}'
    )
    assert ocr_out.shape[1] <= OCR_MAX_SIZE + 1, (
        f'OCR height {ocr_out.shape[1]} exceeds max {OCR_MAX_SIZE}'
    )

    # Check dtypes
    assert yolo_out.dtype == np.float32
    assert clip_out.dtype == np.float32
    assert face_out.dtype == np.float32
    assert orig_out.dtype == np.float32
    assert ocr_out.dtype == np.float32

    # Check ranges for [0, 1] normalized outputs
    assert yolo_out.min() >= 0 and yolo_out.max() <= 1
    assert clip_out.min() >= 0 and clip_out.max() <= 1
    assert face_out.min() >= 0 and face_out.max() <= 1
    assert orig_out.min() >= 0 and orig_out.max() <= 1

    # Check OCR range is [-1, 1] (PP-OCR normalization)
    assert ocr_out.min() >= -1.1 and ocr_out.max() <= 1.1, (
        f'OCR range [{ocr_out.min():.2f}, {ocr_out.max():.2f}] should be ~[-1, 1]'
    )

    print('\n[COMPLETE] All assertions passed!')
    return True


def main():
    print('\n' + '=' * 80)
    print('Track E: Penta-Branch DALI Pipeline Creator (with OCR)')
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
    print('[COMPLETE] Penta-branch DALI pipeline ready!')
    print('=' * 80)
    print('\nModel files created:')
    print('  - models/penta_preprocess_dali/1/model.dali')
    print('  - models/penta_preprocess_dali/config.pbtxt')
    print('\nOutputs:')
    print(f'  1. YOLO preprocessed: [3, {YOLO_SIZE}, {YOLO_SIZE}] [0,1] - object detection')
    print(f'  2. MobileCLIP preprocessed: [3, {CLIP_SIZE}, {CLIP_SIZE}] [0,1] - global embedding')
    print(f'  3. SCRFD preprocessed: [3, {SCRFD_SIZE}, {SCRFD_SIZE}] [0,1] - face detection')
    print(f'  4. Cropping image: [3, H, W] [0,1] - max {CROP_IMAGE_MAX_SIZE}px')
    print(f'  5. OCR preprocessed: [3, H, W] [-1,1] - max {OCR_MAX_SIZE}px')
    print('\nNext steps:')
    print('  1. Create OCR pipeline: models/ocr_pipeline/')
    print('  2. Restart Triton: docker compose restart triton-api')
    print('  3. Test OCR: make test-ocr')


if __name__ == '__main__':
    main()
