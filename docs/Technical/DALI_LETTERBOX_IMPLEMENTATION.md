# DALI Letterbox Preprocessing for YOLO11

## Requirement

**CRITICAL**: Must maintain aspect ratio using YOLO letterbox preprocessing to ensure accuracy parity with Track C.

YOLO letterbox preprocessing:
1. Compute scale factor to fit image into 640x640 while preserving aspect ratio
2. Resize image using scale factor
3. Pad to 640x640 with gray color (114, 114, 114)
4. Normalize to [0, 1] and convert to CHW FP32

## YOLO Letterbox Algorithm

**Reference from Ultralytics**:
```python
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    """
    Resize image with unchanged aspect ratio using padding.

    Args:
        im: Input image (HWC, BGR)
        new_shape: Target size (height, width)
        color: Padding color (BGR)

    Returns:
        Letterboxed image, scale ratio, padding (dw, dh)
    """
    # Current shape [height, width]
    shape = im.shape[:2]

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute new unpadded shape
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))  # (width, height)

    # Compute padding
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    # Divide padding into 2 sides
    dw /= 2
    dh /= 2

    # Resize
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    # Add border (padding)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return im, r, (dw, dh)
```

## DALI Implementation

DALI doesn't have a built-in letterbox operator, so we need to implement it using DALI primitives.

### Approach 1: DALI Python Function (Simplest)

Use DALI's `fn.python_function` to call custom letterbox code:

**Pros**: Easy to implement, matches Ultralytics exactly
**Cons**: CPU-based preprocessing (defeats the purpose!)

❌ **Not suitable for Track D** - we need GPU preprocessing.

---

### Approach 2: DALI GPU Operators (Recommended)

Implement letterbox using DALI's GPU operators:

```python
import nvidia.dali as dali
import nvidia.dali.fn as fn
import nvidia.dali.types as types


@dali.pipeline_def(batch_size=8, num_threads=4, device_id=0)
def yolo_letterbox_pipeline():
    """
    YOLO letterbox preprocessing pipeline using DALI GPU operators.

    Input: Encoded JPEG bytes
    Output: [N, 3, 640, 640] FP32 letterboxed images
    """
    # Step 1: Read encoded JPEG bytes (from external source or file)
    encoded = fn.external_source(name="encoded_images", dtype=types.UINT8)

    # Step 2: Decode JPEG on GPU (nvJPEG)
    images = fn.decoders.image(
        encoded,
        device="mixed",           # CPU decode → GPU output (nvJPEG acceleration)
        output_type=types.RGB,
        hw_decoder_load=0.65      # Use hardware decoder when available
    )

    # Step 3: Get original image dimensions
    # DALI will broadcast these as scalars per image in batch
    shapes = fn.shapes(images, dtype=types.INT32)  # Returns [H, W, C]

    # Step 4: Compute letterbox parameters
    # Target size: 640x640
    target_h = 640
    target_w = 640

    # Original dimensions (per image in batch)
    orig_h = fn.slice(shapes, 0, 1, axes=[0])  # Extract H
    orig_w = fn.slice(shapes, 1, 1, axes=[0])  # Extract W

    # Compute scale factor: min(640/H, 640/W)
    scale_h = target_h / fn.cast(orig_h, dtype=types.FLOAT)
    scale_w = target_w / fn.cast(orig_w, dtype=types.FLOAT)
    scale = fn.min(scale_h, scale_w)

    # Compute new unpadded dimensions
    new_h = fn.cast(fn.cast(orig_h, dtype=types.FLOAT) * scale, dtype=types.INT32)
    new_w = fn.cast(fn.cast(orig_w, dtype=types.FLOAT) * scale, dtype=types.INT32)

    # Step 5: Resize to new unpadded dimensions
    # DALI's fn.resize with per-sample sizes
    images_resized = fn.resize(
        images,
        size=fn.cat(new_h, new_w),  # Per-sample target size
        interp_type=types.INTERP_LINEAR,
        device="gpu"
    )

    # Step 6: Pad to 640x640 with gray (114, 114, 114)
    # Compute padding (centered)
    pad_h = target_h - new_h
    pad_w = target_w - new_w

    # Split padding into top/bottom, left/right
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    # Apply padding (constant value = 114 for all channels)
    images_padded = fn.pad(
        images_resized,
        axes=[0, 1],                          # Pad H and W dimensions
        shape=fn.cat(target_h, target_w),     # Target shape
        fill_value=114,                       # Gray padding (YOLO standard)
        align=[pad_top, pad_left],            # Align top-left
        device="gpu"
    )

    # Step 7: Normalize and transpose (HWC → CHW)
    images_normalized = fn.crop_mirror_normalize(
        images_padded,
        mean=[0.0, 0.0, 0.0],
        std=[255.0, 255.0, 255.0],   # Divide by 255
        output_layout="CHW",          # HWC → CHW
        output_dtype=types.FLOAT,     # uint8 → FP32
        device="gpu"
    )

    return images_normalized
```

**Issues with this approach**:
- ❌ `fn.pad` doesn't support per-sample `align` parameter (padding must be symmetric or pre-computed)
- ❌ `fn.resize` with per-sample sizes requires complex tensor manipulation
- ❌ DALI's expression system doesn't handle conditionals well (rounding, min/max per pixel)

**Complexity**: High - requires DALI expertise and may hit limitations.

---

### Approach 3: DALI + Custom CUDA Kernel (Most Flexible)

Write a custom CUDA operator for letterbox and integrate with DALI.

**Pros**: Full control, GPU-accelerated, can match Ultralytics exactly
**Cons**: Requires CUDA programming, build system integration

**Example custom operator**:
```cuda
// letterbox_kernel.cu
__global__ void letterbox_kernel(
    const uint8_t* input,
    float* output,
    int orig_h, int orig_w,
    int target_h, int target_w
) {
    // Compute scale, resize, pad, normalize in single kernel
    // ...
}
```

**Complexity**: Very high - requires C++/CUDA expertise.

---

### Approach 4: DALI Resize with "Fit" Mode (RECOMMENDED)

DALI has a **built-in letterbox mode** via `fn.resize` with `mode="not_larger"`:

```python
@dali.pipeline_def(batch_size=8, num_threads=4, device_id=0)
def yolo_letterbox_pipeline_simple():
    """
    YOLO letterbox using DALI's built-in resize modes.

    Uses fn.resize with mode="not_larger" to fit image into 640x640
    while preserving aspect ratio, then pads to square.
    """
    # Step 1: Decode JPEG on GPU
    encoded = fn.external_source(name="encoded_images", dtype=types.UINT8)
    images = fn.decoders.image(
        encoded,
        device="mixed",
        output_type=types.RGB
    )

    # Step 2: Resize with aspect-preserving mode
    # mode="not_larger" ensures max(new_h, new_w) <= 640
    images_resized = fn.resize(
        images,
        size=[640, 640],
        mode="not_larger",            # Preserve aspect, fit inside 640x640
        interp_type=types.INTERP_LINEAR,
        device="gpu"
    )

    # Step 3: Pad to exact 640x640 (centered, gray padding)
    images_padded = fn.pad(
        images_resized,
        fill_value=114,               # YOLO standard gray
        align=0.5,                    # Center the image (0.5 = centered)
        shape=[640, 640, 3],          # Target HWC shape
        device="gpu"
    )

    # Step 4: Normalize and transpose
    images_final = fn.crop_mirror_normalize(
        images_padded,
        mean=[0.0, 0.0, 0.0],
        std=[255.0, 255.0, 255.0],
        output_layout="CHW",
        output_dtype=types.FLOAT,
        device="gpu"
    )

    return images_final
```

**Verification**:
- ✅ Preserves aspect ratio (`mode="not_larger"`)
- ✅ Pads to 640x640 with gray (114)
- ✅ Centered padding (`align=0.5`)
- ✅ GPU-accelerated (nvJPEG + CUDA kernels)
- ✅ Matches YOLO preprocessing semantics

**THIS IS THE RECOMMENDED APPROACH** ✅

---

## Testing Letterbox Accuracy

To verify DALI letterbox matches Ultralytics:

```python
# Test script: scripts/verify_letterbox.py

import numpy as np
import cv2
from ultralytics.utils.ops import letterbox
import nvidia.dali as dali

def test_letterbox_equivalence():
    """
    Compare DALI letterbox output vs Ultralytics letterbox.
    """
    # Load test image
    img = cv2.imread("test_images/sample.jpg")

    # Ultralytics letterbox
    img_ultra, ratio, (dw, dh) = letterbox(img, new_shape=(640, 640))
    img_ultra_norm = img_ultra.astype(np.float32) / 255.0
    img_ultra_chw = np.transpose(img_ultra_norm, (2, 0, 1))

    # DALI letterbox
    pipe = yolo_letterbox_pipeline_simple(batch_size=1)
    pipe.build()
    # ... feed image ...
    img_dali = pipe.run()[0].as_cpu().as_array()[0]

    # Compare
    diff = np.abs(img_ultra_chw - img_dali)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    print(f"Max pixel diff: {max_diff:.6f}")
    print(f"Mean pixel diff: {mean_diff:.6f}")

    # Acceptable tolerance (due to floating point precision)
    assert max_diff < 0.01, "Letterbox mismatch!"
    assert mean_diff < 0.001, "Letterbox accuracy issue!"

    print("✓ DALI letterbox matches Ultralytics!")
```

---

## DALI Pipeline Serialization

Once pipeline is verified, serialize to `.dali` format for Triton:

```python
# scripts/create_dali_model.py

from nvidia.dali.plugin.triton import autoserialize

# Build pipeline
pipe = yolo_letterbox_pipeline_simple(
    batch_size=128,      # Match Triton max_batch_size
    num_threads=4,
    device_id=0
)

# Serialize for Triton
autoserialize(
    pipe,
    "/app/models/yolo_preprocess_dali/1/model.dali",
    device_id=0
)

print("✓ DALI model serialized for Triton!")
```

---

## Triton Config for DALI Model

```protobuf
# models/yolo_preprocess_dali/config.pbtxt

name: "yolo_preprocess_dali"
backend: "dali"
max_batch_size: 128

input [
  {
    name: "encoded_images"
    data_type: TYPE_UINT8
    dims: [ -1 ]  # Variable-length JPEG bytes
  }
]

output [
  {
    name: "preprocessed_images"
    data_type: TYPE_FP32
    dims: [ 3, 640, 640 ]
  }
]

# GPU instances for decode
instance_group [
  {
    count: 2               # 2 parallel DALI instances
    kind: KIND_GPU
    gpus: [ 0 ]
  }
]

# DALI-specific parameters
parameters: {
  key: "num_threads"
  value: { string_value: "4" }
}

# No dynamic batching at DALI level (handled by ensemble)
```

---

## Performance Considerations

**DALI Letterbox Performance** (estimated):

| Operation | Time (batch=1) | Time (batch=8) | Notes |
|-----------|----------------|----------------|-------|
| nvJPEG decode | 0.5-1.5ms | 0.3-0.8ms/img | Hardware-accelerated |
| Resize (GPU) | 0.2-0.5ms | 0.1-0.2ms/img | CUDA kernel |
| Pad (GPU) | 0.1-0.2ms | <0.1ms/img | Memory copy |
| Normalize (GPU) | 0.1-0.2ms | <0.1ms/img | Fused with transpose |
| **Total** | **1-2.5ms** | **0.5-1.2ms/img** | **2-3x faster than CPU** |

**CPU Letterbox (Track C baseline)**:

| Operation | Time (batch=1) | Time (batch=8) | Notes |
|-----------|----------------|----------------|-------|
| cv2.imdecode | 1-3ms | 1-3ms/img | CPU-bound |
| cv2.resize | 1-2ms | 1-2ms/img | CPU-bound |
| Pad + normalize | 0.5-1ms | 0.5-1ms/img | CPU-bound |
| **Total** | **2.5-6ms** | **2.5-6ms/img** | **Serial, no batching** |

**Expected Track D Speedup**:
- Single image: 30-40% faster preprocessing
- Batch-8: 3-5x faster preprocessing (amortized decode + parallel GPU ops)
- Batch-16+: 5-8x faster preprocessing

---

## Validation Checklist

Before deploying DALI letterbox:

- [ ] Verify pixel-level accuracy vs Ultralytics (max diff < 0.01)
- [ ] Test with various aspect ratios (portrait, landscape, square)
- [ ] Test with edge cases (very small images, very large images)
- [ ] Benchmark preprocessing time vs CPU baseline
- [ ] Verify JPEG decode works with all common formats (JPEG, PNG, etc.)
- [ ] Test dynamic batching (1, 8, 16, 32, 64, 128)
- [ ] Profile GPU memory usage (should be <2GB for batch-128)

---

## References

- [DALI fn.resize modes](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/operations/nvidia.dali.fn.resize.html)
- [DALI fn.pad](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/operations/nvidia.dali.fn.pad.html)
- [Ultralytics letterbox implementation](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py#L127)
- [NVIDIA DALI letterbox example](https://github.com/NVIDIA/DALI/blob/main/docs/examples/use_cases/paddle/yolo/yolo_preprocessing.py)

---

**Status**: Ready for implementation. Use Approach 4 (DALI resize with "not_larger" mode + pad).

**Next Step**: Create `scripts/create_dali_letterbox_pipeline.py` and serialize for Triton.
