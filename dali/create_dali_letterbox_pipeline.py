#!/usr/bin/env python3
"""
Create DALI Letterbox Pipeline for YOLO11 (Original Version)

This script creates a DALI pipeline that performs GPU-accelerated preprocessing
using affine transformation matrices calculated on CPU:

1. JPEG decode (nvJPEG - GPU)
2. Apply affine transformation (warp_affine - GPU)
3. Normalize to [0, 1] and CHW FP32 (GPU)

Requires CPU calculation of letterbox affine matrices (calculated by client).

Usage:
    # From yolo-api container (has DALI installed)
    docker compose exec yolo-api python /app/dali/create_dali_letterbox_pipeline.py
"""

import sys
import os
from pathlib import Path
import numpy as np

try:
    import nvidia.dali as dali
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
except ImportError as e:
    print(f"ERROR: NVIDIA DALI not installed: {e}")
    print("\nThis script must be run from the yolo-api container which has DALI.")
    print("Run: docker compose exec yolo-api python /app/dali/create_dali_letterbox_pipeline.py")
    sys.exit(1)


@dali.pipeline_def(batch_size=128, num_threads=4, device_id=0)
def yolo_letterbox_pipeline():
    """
    YOLO letterbox preprocessing pipeline using affine transformation.

    This is the original implementation that requires CPU-calculated affine matrices.
    The client (FastAPI/Python) calculates the letterbox transformation parameters
    and passes them as affine_matrices input.

    Flow:
    1. GPU: Decode JPEG (nvJPEG)
    2. GPU: Apply affine transformation (warp_affine with CPU-calculated matrix)
    3. GPU: Normalize to [0, 1] and convert HWC → CHW

    Inputs:
        encoded_images: Raw JPEG/PNG bytes (variable length)
        affine_matrices: Pre-calculated affine transformation matrices [N, 2, 3] FP32
                        Format: [[scale_x, 0, offset_x], [0, scale_y, offset_y]]

    Output:
        preprocessed_images: [N, 3, 640, 640] FP32 tensor, normalized [0, 1]
    """
    # Constants
    TARGET_SIZE = 640
    PAD_VALUE = 114  # Gray padding (YOLO standard)

    # Step 1: Read encoded image bytes from external source (fed by Triton)
    encoded = fn.external_source(
        name="encoded_images",
        dtype=types.UINT8,
        ndim=1  # 1D byte array
    )

    # Step 2: Read affine transformation matrices from external source
    # Client calculates these matrices with letterbox parameters
    affine_matrix = fn.external_source(
        name="affine_matrices",
        dtype=types.FLOAT,
        ndim=2  # 2D matrix [2, 3] per image
    )

    # Step 3: Decode JPEG on GPU using nvJPEG
    # device="mixed" = CPU decode → GPU output (uses nvJPEG hardware acceleration)
    images = fn.decoders.image(
        encoded,
        device="mixed",              # GPU-accelerated decode
        output_type=types.RGB,       # RGB output (YOLO expects RGB)
        hw_decoder_load=0.65         # Use hardware decoder (Ampere+)
    )

    # Step 4: Apply affine transformation on GPU
    # warp_affine applies the letterbox transformation:
    # - Resize with scale factor (embedded in matrix)
    # - Translate with padding offset (embedded in matrix)
    # - Fill border with gray color
    images_transformed = fn.warp_affine(
        images,
        matrix=affine_matrix,
        size=[TARGET_SIZE, TARGET_SIZE],      # Output size
        fill_value=PAD_VALUE,                 # Gray padding for letterbox
        interp_type=types.INTERP_LINEAR,      # Linear interpolation (matches cv2.INTER_LINEAR)
        device="gpu"
    )

    # Step 5: Normalize and transpose ON GPU
    # - Divide by 255 to get [0, 1] range
    # - Transpose HWC → CHW (YOLO/PyTorch convention)
    # - Convert uint8 → FP32
    images_final = fn.crop_mirror_normalize(
        images_transformed,
        mean=[0.0, 0.0, 0.0],        # No mean subtraction
        std=[255.0, 255.0, 255.0],   # Divide by 255 (uint8 → [0, 1])
        output_layout="CHW",          # HWC → CHW transpose
        output_dtype=types.FLOAT,     # uint8 → FP32
        device="gpu"
    )

    return images_final


def serialize_pipeline():
    """
    Build and serialize DALI pipeline for Triton deployment.
    """
    print("="*80)
    print("DALI Letterbox Pipeline Creation (Original Version)")
    print("="*80)
    print("\nGPU preprocessing with CPU-calculated affine matrices")

    # Output directory
    output_dir = Path("/app/models/yolo_preprocess_dali/1")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "model.dali"

    print(f"\nTarget directory: {output_dir}")

    # Build pipeline
    print("\nBuilding DALI pipeline...")
    print("  - Batch size: 128")
    print("  - Device: GPU 0")
    print("  - Threads: 4")
    print("\nPipeline operations:")
    print("  1. ✅ nvJPEG decode (GPU)")
    print("  2. ✅ Affine transformation with warp_affine (GPU)")
    print("  3. ✅ Normalize /255 + HWC→CHW (GPU)")
    print("\n  ⚠️  Requires CPU calculation of affine matrices")
    print("  ⚠️  Client must calculate letterbox parameters")

    try:
        pipe = yolo_letterbox_pipeline(
            batch_size=128,      # Max batch size (matches Triton config)
            num_threads=4,       # CPU threads for pipeline orchestration
            device_id=0          # GPU 0
        )
        pipe.build()

        print("\n✓ Pipeline built successfully")

    except Exception as e:
        print(f"\n✗ Pipeline build failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Serialize for Triton
    print(f"\nSerializing to Triton format...")
    print(f"  Output: {output_path}")

    try:
        pipe.serialize(filename=str(output_path))
        print(f"\n✓ Pipeline serialized successfully")
        print(f"  File size: {output_path.stat().st_size / 1024:.2f} KB")

    except Exception as e:
        print(f"\n✗ Serialization failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Create config.pbtxt with affine_matrices input
    print(f"\nCreating config.pbtxt (with affine_matrices input)...")
    config_path = output_dir.parent / "config.pbtxt"

    config_content = """# YOLO Preprocessing - DALI Backend
# GPU-accelerated preprocessing pipeline for YOLO11 using affine transformation
# Pipeline: nvJPEG decode → warp_affine (with CPU-calculated matrix) → Normalize → CHW
#
# ⚠️  Requires CPU calculation of affine matrices
# - Client must calculate letterbox parameters (scale, padding)
# - Pass as affine_matrices input [N, 2, 3] FP32
#
# NVIDIA Best Practices:
# - device="mixed" for image decoder (uses nvJPEG GPU acceleration)
# - instance count=1 (NVIDIA warning: count>1 causes unnaturally high memory usage)
# - hw_decoder_load=0.65 (optimal for Ampere+ hardware decoder offload)

name: "yolo_preprocess_dali"
backend: "dali"
max_batch_size: 128

# Inputs: Raw JPEG bytes + CPU-calculated affine matrices
input [
  {
    name: "encoded_images"
    data_type: TYPE_UINT8
    dims: [ -1 ]  # Variable-length JPEG bytes
  },
  {
    name: "affine_matrices"
    data_type: TYPE_FP32
    dims: [ 2, 3 ]  # Affine transformation matrix [2x3]
  }
]

# Output: Preprocessed images ready for YOLO inference
output [
  {
    name: "preprocessed_images"
    data_type: TYPE_FP32
    dims: [ 3, 640, 640 ]  # CHW format, normalized [0, 1]
  }
]

# NVIDIA Best Practice: Use count=1 to avoid unnaturally increased memory consumption
instance_group [
  {
    count: 1         # Single DALI instance (NVIDIA recommended)
    kind: KIND_GPU
    gpus: [ 0 ]
  }
]

parameters: {
  key: "num_threads"
  value: { string_value: "4" }
}
"""

    with open(config_path, 'w') as f:
        f.write(config_content)

    print(f"  ✓ Config created: {config_path}")

    # Verify files exist
    if output_path.exists() and config_path.exists():
        print(f"\n{'='*80}")
        print("✅ SUCCESS: DALI model ready for Triton")
        print(f"{'='*80}")
        print(f"\nModel files created:")
        print(f"  - Pipeline: {output_path}")
        print(f"  - Config: {config_path}")
        print(f"\nHost paths:")
        print(f"  - ./models/yolo_preprocess_dali/1/model.dali")
        print(f"  - ./models/yolo_preprocess_dali/config.pbtxt")
        print("\nKey characteristics:")
        print("  ✅ GPU decode, transform, normalize")
        print("  ⚠️  CPU calculates affine matrices")
        print("  ⚠️  Client must compute letterbox parameters")
        print("\nNext steps:")
        print("  1. Update ensemble config (ensure affine_matrices input)")
        print("  2. Update client code to calculate affine matrices")
        print("  3. Restart Triton: docker compose restart triton-api")
        return True
    else:
        print(f"\n✗ ERROR: Files not created properly")
        return False


def test_pipeline():
    """
    Test pipeline with dummy data to verify affine transformation.
    """
    print("\n" + "="*80)
    print("Testing pipeline with dummy data...")
    print("="*80)

    import numpy as np
    from PIL import Image
    import io

    # Create test image (810x1080 to match our benchmark image)
    print("\nCreating test image (1080x810 HxW)...")
    test_img = np.random.randint(0, 255, (1080, 810, 3), dtype=np.uint8)
    pil_img = Image.fromarray(test_img)

    # Encode to JPEG
    buffer = io.BytesIO()
    pil_img.save(buffer, format='JPEG', quality=90)
    jpeg_bytes = buffer.getvalue()

    # Calculate letterbox affine matrix
    TARGET_SIZE = 640
    orig_h, orig_w = 1080, 810

    # Calculate scale (same as YOLO letterbox)
    scale = min(TARGET_SIZE / orig_h, TARGET_SIZE / orig_w)
    if scale > 1.0:
        scale = 1.0

    # New dimensions after scaling
    new_h = int(round(orig_h * scale))
    new_w = int(round(orig_w * scale))

    # Calculate padding to center
    pad_x = (TARGET_SIZE - new_w) / 2.0
    pad_y = (TARGET_SIZE - new_h) / 2.0

    # Create affine matrix [2, 3]
    # Format: [[scale_x, 0, offset_x], [0, scale_y, offset_y]]
    affine_matrix = np.array([
        [scale, 0.0, pad_x],
        [0.0, scale, pad_y]
    ], dtype=np.float32)

    print(f"  Original image: {orig_h}×{orig_w} (H×W)")
    print(f"  JPEG size: {len(jpeg_bytes)} bytes")
    print(f"\n  Letterbox parameters:")
    print(f"    - Scale: {scale:.4f}")
    print(f"    - New size: {new_h}×{new_w} (H×W)")
    print(f"    - Padding: ({pad_x:.1f}, {pad_y:.1f})")
    print(f"  Affine matrix:\n{affine_matrix}")

    # Build pipeline
    pipe = yolo_letterbox_pipeline(batch_size=1, num_threads=2, device_id=0)
    pipe.build()

    # Feed data (JPEG bytes + affine matrix)
    print("\nRunning DALI pipeline...")
    pipe.feed_input("encoded_images", [np.frombuffer(jpeg_bytes, dtype=np.uint8)])
    pipe.feed_input("affine_matrices", [affine_matrix])
    outputs = pipe.run()

    # Get output
    output = outputs[0].as_cpu()
    result = np.array(output[0])

    print(f"\n✓ Pipeline executed successfully")
    print(f"  Output shape: {result.shape} (expected: [3, 640, 640])")
    print(f"  Output dtype: {result.dtype} (expected: float32)")
    print(f"  Output range: [{result.min():.4f}, {result.max():.4f}] (expected: [0, 1])")

    # Validate output
    assert result.shape == (3, 640, 640), f"Wrong shape: {result.shape}"
    assert result.dtype == np.float32, f"Wrong dtype: {result.dtype}"
    assert result.min() >= 0 and result.max() <= 1, f"Values out of range: [{result.min()}, {result.max()}]"

    print("\n✓ All assertions passed!")
    print("\n🎉 DALI letterbox with affine transformation working!")
    print("   - GPU decode, transform, normalize")
    print("   - CPU calculates affine matrices")
    print("   - Output matches expected format")

    return True


def main():
    """
    Main entry point.
    """
    print("\n" + "="*80)
    print("DALI Letterbox Pipeline Creator (Original Version)")
    print("="*80)
    print("\nThis creates a GPU-accelerated preprocessing pipeline")
    print("using affine transformation with CPU-calculated matrices.\n")

    # Check DALI availability
    try:
        dali_version = dali.__version__
        print(f"✓ NVIDIA DALI version: {dali_version}")
    except:
        print("✗ NVIDIA DALI not available")
        sys.exit(1)

    # Test pipeline first
    if not test_pipeline():
        print("\n✗ Pipeline test failed")
        sys.exit(1)

    # Serialize for Triton
    if not serialize_pipeline():
        print("\n✗ Serialization failed")
        sys.exit(1)

    print("\n" + "="*80)
    print("✅ COMPLETE: DALI model ready!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
