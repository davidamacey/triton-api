#!/usr/bin/env python3
"""
Create DALI Letterbox Pipeline for Track D

This script creates a GPU-accelerated YOLO preprocessing pipeline using NVIDIA DALI.
The pipeline performs:
1. JPEG decode (nvJPEG - GPU)
2. Affine transformation for letterbox (warp_affine - GPU)
3. Normalize to [0, 1] and convert HWC to CHW (GPU)

The affine transformation matrices must be calculated on the CPU by the client,
providing the scale and padding parameters for letterbox preprocessing.

Usage:
    # From yolo-api container (has DALI installed)
    docker compose exec yolo-api python /app/dali/create_dali_letterbox_pipeline.py

    # Or via Makefile
    make create-dali

Output:
    - models/yolo_preprocess_dali/1/model.dali
    - models/yolo_preprocess_dali/config.pbtxt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# Import shared configuration
from dali.config import (
    DEFAULT_DEVICE_ID,
    DEFAULT_MODEL_DIR,
    HW_DECODER_LOAD,
    MAX_BATCH_SIZE,
    NUM_THREADS,
    TRACK_D_DALI_MODEL,
    TRACK_D_OUTPUT_PREPROCESSED,
    YOLO_PAD_VALUE,
    YOLO_SIZE,
)
from dali.utils import calculate_letterbox_affine, create_test_image_jpeg, setup_logging


# Try to import NVIDIA DALI
try:
    from nvidia import dali
    from nvidia.dali import fn, types

    DALI_AVAILABLE = True
except ImportError as e:
    DALI_AVAILABLE = False
    DALI_IMPORT_ERROR = str(e)

logger = logging.getLogger(__name__)


# =============================================================================
# DALI Pipeline Definition
# =============================================================================


def create_yolo_letterbox_pipeline(
    batch_size: int = MAX_BATCH_SIZE,
    num_threads: int = NUM_THREADS,
    device_id: int = DEFAULT_DEVICE_ID,
) -> object:
    """
    Create YOLO letterbox preprocessing pipeline.

    This pipeline performs GPU-accelerated preprocessing using affine transformation
    matrices calculated on the CPU. The flow is:
    1. GPU: Decode JPEG using nvJPEG
    2. GPU: Apply affine transformation (warp_affine) with CPU-calculated matrix
    3. GPU: Normalize to [0, 1] and transpose HWC to CHW

    Args:
        batch_size: Maximum batch size for the pipeline.
        num_threads: Number of CPU threads for pipeline orchestration.
        device_id: GPU device ID.

    Returns:
        DALI pipeline object (not yet built).

    Inputs (external, fed at runtime):
        encoded_images: Raw JPEG/PNG bytes [variable length]
        affine_matrices: Pre-calculated affine matrices [N, 2, 3] FP32

    Output:
        preprocessed_images: [N, 3, 640, 640] FP32, normalized [0, 1]
    """

    @dali.pipeline_def(batch_size=batch_size, num_threads=num_threads, device_id=device_id)
    def pipeline():
        # External input: encoded image bytes
        encoded = fn.external_source(
            name='encoded_images',
            dtype=types.UINT8,
            ndim=1,
        )

        # External input: affine transformation matrix from CPU
        affine_matrix = fn.external_source(
            name='affine_matrices',
            dtype=types.FLOAT,
            ndim=2,  # [2, 3]
        )

        # Step 1: Decode JPEG on GPU using nvJPEG
        images = fn.decoders.image(
            encoded,
            device='mixed',  # GPU-accelerated decode
            output_type=types.RGB,
            hw_decoder_load=HW_DECODER_LOAD,
        )

        # Step 2: Apply affine transformation on GPU
        images_transformed = fn.warp_affine(
            images,
            matrix=affine_matrix,
            size=[YOLO_SIZE, YOLO_SIZE],
            fill_value=YOLO_PAD_VALUE,
            interp_type=types.INTERP_LINEAR,
            device='gpu',
        )

        # Step 3: Normalize and transpose on GPU
        return fn.crop_mirror_normalize(
            images_transformed,
            mean=[0.0, 0.0, 0.0],
            std=[255.0, 255.0, 255.0],  # Divide by 255
            output_layout='CHW',
            output_dtype=types.FLOAT,
            device='gpu',
        )

    return pipeline()


# =============================================================================
# Pipeline Serialization
# =============================================================================


def serialize_pipeline(
    model_dir: Path = DEFAULT_MODEL_DIR,
    batch_size: int = MAX_BATCH_SIZE,
    device_id: int = DEFAULT_DEVICE_ID,
) -> bool:
    """
    Build and serialize DALI pipeline for Triton deployment.

    Args:
        model_dir: Base directory for Triton models.
        batch_size: Maximum batch size.
        device_id: GPU device ID.

    Returns:
        True if successful, False otherwise.
    """
    output_dir = model_dir / TRACK_D_DALI_MODEL / '1'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'model.dali'

    print('=' * 80)
    print('Track D: DALI Letterbox Pipeline Creation')
    print('=' * 80)
    print(f'\nTarget directory: {output_dir}')
    print('\nConfiguration:')
    print(f'  - Batch size: {batch_size}')
    print(f'  - Device ID: {device_id}')
    print(f'  - Target size: {YOLO_SIZE}x{YOLO_SIZE}')
    print(f'  - Pad value: {YOLO_PAD_VALUE}')
    print('\nPipeline operations:')
    print('  1. [GPU] nvJPEG decode')
    print('  2. [GPU] Affine transformation (warp_affine)')
    print('  3. [GPU] Normalize /255 + HWC->CHW')
    print('\n  Note: Requires CPU calculation of affine matrices')

    # Build pipeline
    try:
        pipe = create_yolo_letterbox_pipeline(
            batch_size=batch_size,
            num_threads=NUM_THREADS,
            device_id=device_id,
        )
        pipe.build()
        print('\n[OK] Pipeline built successfully')
    except Exception as e:
        print(f'\n[ERROR] Pipeline build failed: {e}')
        logger.exception('Pipeline build error')
        return False

    # Serialize
    try:
        pipe.serialize(filename=str(output_path))
        print(f'\n[OK] Serialized to: {output_path}')
        print(f'     File size: {output_path.stat().st_size / 1024:.2f} KB')
    except Exception as e:
        print(f'\n[ERROR] Serialization failed: {e}')
        logger.exception('Serialization error')
        return False

    # Create config.pbtxt
    config_path = output_dir.parent / 'config.pbtxt'
    config_content = generate_config(batch_size)

    with open(config_path, 'w') as f:
        f.write(config_content)

    print(f'\n[OK] Config created: {config_path}')

    # Summary
    print('\n' + '=' * 80)
    print('[SUCCESS] DALI model ready for Triton')
    print('=' * 80)
    print('\nModel files created:')
    print(f'  - Pipeline: {output_path}')
    print(f'  - Config: {config_path}')
    print('\nNext steps:')
    print('  1. Create ensemble configs: make create-ensembles')
    print('  2. Restart Triton: docker compose restart triton-api')
    print('  3. Validate: docker compose exec yolo-api python /app/dali/validate_dali_letterbox.py')

    return True


def generate_config(batch_size: int = MAX_BATCH_SIZE) -> str:
    """
    Generate Triton config.pbtxt content.

    Args:
        batch_size: Maximum batch size for the model.

    Returns:
        config.pbtxt content as string.
    """
    return f"""# Track D: YOLO Preprocessing - DALI Backend
# GPU-accelerated preprocessing pipeline using affine transformation
#
# Pipeline: nvJPEG decode -> warp_affine -> Normalize -> CHW
#
# Note: Requires CPU calculation of affine matrices by client.
# Client must compute letterbox parameters (scale, padding) and pass
# as affine_matrices input [N, 2, 3] FP32.
#
# NVIDIA Best Practices:
# - device="mixed" for image decoder (uses nvJPEG GPU acceleration)
# - instance count=1 (count>1 causes unnaturally high memory usage)
# - hw_decoder_load=0.65 (optimal for Ampere+ hardware decoder offload)

name: "{TRACK_D_DALI_MODEL}"
backend: "dali"
max_batch_size: {batch_size}

input [
  {{
    name: "encoded_images"
    data_type: TYPE_UINT8
    dims: [ -1 ]  # Variable-length JPEG bytes
  }},
  {{
    name: "affine_matrices"
    data_type: TYPE_FP32
    dims: [ 2, 3 ]  # Affine transformation matrix [2x3]
  }}
]

output [
  {{
    name: "{TRACK_D_OUTPUT_PREPROCESSED}"
    data_type: TYPE_FP32
    dims: [ 3, {YOLO_SIZE}, {YOLO_SIZE} ]  # CHW format, normalized [0, 1]
  }}
]

# NVIDIA Best Practice: Use count=1 to avoid high memory consumption
instance_group [
  {{
    count: 1
    kind: KIND_GPU
    gpus: [ 0 ]
  }}
]

parameters: {{
  key: "num_threads"
  value: {{ string_value: "{NUM_THREADS}" }}
}}
"""


# =============================================================================
# Pipeline Testing
# =============================================================================


def test_pipeline(device_id: int = DEFAULT_DEVICE_ID) -> bool:
    """
    Test pipeline with synthetic data to verify correctness.

    Args:
        device_id: GPU device ID.

    Returns:
        True if test passes, False otherwise.
    """
    print('\n' + '=' * 80)
    print('Testing Pipeline with Synthetic Data')
    print('=' * 80)

    # Create test image
    print('\nCreating test image (1080x810)...')
    jpeg_bytes, orig_w, orig_h = create_test_image_jpeg(1080, 810)
    print(f'  JPEG size: {len(jpeg_bytes)} bytes')

    # Calculate affine matrix
    affine_matrix, scale, pad_x, pad_y = calculate_letterbox_affine(orig_w, orig_h)
    print('\nLetterbox parameters:')
    print(f'  - Scale: {scale:.4f}')
    print(f'  - Padding: ({pad_x:.1f}, {pad_y:.1f})')

    # Build test pipeline
    try:
        pipe = create_yolo_letterbox_pipeline(
            batch_size=1,
            num_threads=2,
            device_id=device_id,
        )
        pipe.build()
    except Exception as e:
        print(f'\n[ERROR] Pipeline build failed: {e}')
        return False

    # Run pipeline
    print('\nRunning pipeline...')
    try:
        pipe.feed_input('encoded_images', [np.frombuffer(jpeg_bytes, dtype=np.uint8)])
        pipe.feed_input('affine_matrices', [affine_matrix])
        outputs = pipe.run()
    except Exception as e:
        print(f'\n[ERROR] Pipeline execution failed: {e}')
        return False

    # Validate output
    result = np.array(outputs[0].as_cpu()[0])

    print('\nOutput:')
    print(f'  - Shape: {result.shape} (expected: [3, {YOLO_SIZE}, {YOLO_SIZE}])')
    print(f'  - Dtype: {result.dtype} (expected: float32)')
    print(f'  - Range: [{result.min():.4f}, {result.max():.4f}] (expected: [0, 1])')

    # Assertions
    try:
        assert result.shape == (3, YOLO_SIZE, YOLO_SIZE), f'Wrong shape: {result.shape}'
        assert result.dtype == np.float32, f'Wrong dtype: {result.dtype}'
        assert result.min() >= 0, f'Min value out of range: {result.min()}'
        assert result.max() <= 1, f'Max value out of range: {result.max()}'
    except AssertionError as e:
        print(f'\n[ERROR] Validation failed: {e}')
        return False

    print('\n[OK] All validations passed!')
    return True


# =============================================================================
# CLI Interface
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Create DALI letterbox preprocessing pipeline for Track D',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Create pipeline with default settings
    python dali/create_dali_letterbox_pipeline.py

    # Create with custom batch size
    python dali/create_dali_letterbox_pipeline.py --batch-size 64

    # Skip test and only serialize
    python dali/create_dali_letterbox_pipeline.py --skip-test

    # Verbose output
    python dali/create_dali_letterbox_pipeline.py -v
""",
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=MAX_BATCH_SIZE,
        help=f'Maximum batch size (default: {MAX_BATCH_SIZE})',
    )
    parser.add_argument(
        '--device-id',
        type=int,
        default=DEFAULT_DEVICE_ID,
        help=f'GPU device ID (default: {DEFAULT_DEVICE_ID})',
    )
    parser.add_argument(
        '--model-dir',
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help=f'Model directory (default: {DEFAULT_MODEL_DIR})',
    )
    parser.add_argument(
        '--skip-test',
        action='store_true',
        help='Skip pipeline test before serialization',
    )
    parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        help='Enable verbose output',
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)

    print('\n' + '=' * 80)
    print('Track D: DALI Letterbox Pipeline Creator')
    print('=' * 80)

    # Check DALI availability
    if not DALI_AVAILABLE:
        print(f'\n[ERROR] NVIDIA DALI not installed: {DALI_IMPORT_ERROR}')
        print('\nThis script must be run from the yolo-api container.')
        print(
            'Run: docker compose exec yolo-api python /app/dali/create_dali_letterbox_pipeline.py'
        )
        return 1

    print(f'\n[OK] NVIDIA DALI version: {dali.__version__}')

    # Test pipeline
    if not args.skip_test and not test_pipeline(args.device_id):
        print('\n[ERROR] Pipeline test failed')
        return 1

    # Serialize pipeline
    if not serialize_pipeline(args.model_dir, args.batch_size, args.device_id):
        print('\n[ERROR] Serialization failed')
        return 1

    print('\n' + '=' * 80)
    print('[COMPLETE] Track D DALI pipeline ready!')
    print('=' * 80 + '\n')

    return 0


if __name__ == '__main__':
    sys.exit(main())
