"""
Affine matrix calculation for YOLO letterbox transformation.

This is the ONLY CPU preprocessing needed for the GPU pipeline.
Everything else (decode, resize, normalize) happens on GPU via DALI.

Performance optimizations:
- @lru_cache for affine matrix calculation (same dimensions = same matrix)
- Fast JPEG header parsing (no full decode)
"""

from functools import lru_cache
from io import BytesIO

import numpy as np
from PIL import Image


@lru_cache(maxsize=1000)
def calculate_affine_matrix(
    orig_width: int, orig_height: int, target_size: int = 640
) -> tuple[np.ndarray, float, tuple[float, float]]:
    """
    Calculate YOLO letterbox affine transformation matrix.

    Cached by dimensions - same image dimensions = same matrix.
    This is ~1000x faster for repeated image sizes.

    Args:
        orig_width: Original image width
        orig_height: Original image height
        target_size: Target size for YOLO (default 640)

    Returns:
        Tuple of:
        - affine_matrix: np.ndarray [2, 3] for DALI warp_affine
        - scale: float for inverse transformation
        - padding: tuple (pad_x, pad_y) for inverse transformation
    """
    scale = target_size / max(orig_height, orig_width)
    scale = min(scale, 1.0)  # Don't upscale (matches YOLO scaleup=False)

    new_w = round(orig_width * scale)
    new_h = round(orig_height * scale)

    pad_x = (target_size - new_w) / 2.0
    pad_y = (target_size - new_h) / 2.0

    # DALI warp_affine format: [[scale_x, 0, offset_x], [0, scale_y, offset_y]]
    affine_matrix = np.array([[scale, 0.0, pad_x], [0.0, scale, pad_y]], dtype=np.float32)

    return affine_matrix, scale, (pad_x, pad_y)


def get_jpeg_dimensions_fast(image_bytes: bytes) -> tuple[int, int]:
    """
    Get JPEG dimensions from header without full decode.

    This is ~100x faster than PIL.Image.open().size for large images.
    Falls back to PIL for non-JPEG or corrupted headers.

    Args:
        image_bytes: Raw JPEG/PNG bytes

    Returns:
        (width, height) tuple
    """
    # Try fast JPEG header parse first
    if len(image_bytes) > 2 and image_bytes[0:2] == b'\xff\xd8':
        try:
            # JPEG format - parse SOF markers
            pos = 2
            while pos < len(image_bytes) - 8:
                if image_bytes[pos] != 0xFF:
                    break
                marker = image_bytes[pos + 1]

                # SOF markers (Start Of Frame) contain dimensions
                if marker in (0xC0, 0xC1, 0xC2):  # SOF0, SOF1, SOF2
                    height = (image_bytes[pos + 5] << 8) | image_bytes[pos + 6]
                    width = (image_bytes[pos + 7] << 8) | image_bytes[pos + 8]
                    return width, height

                # Skip to next marker
                if marker in {0xD8, 0xD9}:  # SOI, EOI
                    pos += 2
                elif marker == 0xFF:
                    pos += 1
                else:
                    length = (image_bytes[pos + 2] << 8) | image_bytes[pos + 3]
                    pos += 2 + length
        except (IndexError, ValueError):
            pass  # Fall back to PIL

    # Fallback: Use PIL (handles all formats)
    img = Image.open(BytesIO(image_bytes))
    return img.size  # (width, height)


def prepare_triton_inputs(image_bytes: bytes, target_size: int = 640) -> dict:
    """
    Prepare inputs for Triton GPU pipeline.

    CPU work is minimal:
    - Fast JPEG header parse for dimensions (~0.1ms)
    - Cached affine matrix lookup (~0.001ms for cache hit)

    Everything else (decode, resize, normalize) happens on GPU via DALI.

    Args:
        image_bytes: Raw JPEG/PNG file bytes
        target_size: YOLO input size (default 640)

    Returns:
        Dictionary with:
        - encoded_images: np.ndarray (uint8 bytes)
        - affine_matrix: np.ndarray [1, 2, 3] (batched)
        - orig_shape: (height, width) tuple
        - scale: float for inverse transformation
        - padding: (pad_x, pad_y) tuple for inverse transformation
    """
    # Get dimensions (fast header parse)
    orig_w, orig_h = get_jpeg_dimensions_fast(image_bytes)

    # Get cached affine matrix
    affine_matrix, scale, padding = calculate_affine_matrix(orig_w, orig_h, target_size)

    # Prepare numpy arrays for Triton
    encoded_images = np.frombuffer(image_bytes, dtype=np.uint8)
    affine_matrix_batched = np.expand_dims(affine_matrix, axis=0)  # Add batch dim

    return {
        'encoded_images': encoded_images,
        'affine_matrix': affine_matrix_batched,
        'orig_shape': (orig_h, orig_w),
        'scale': scale,
        'padding': padding,
    }


def clear_affine_cache():
    """Clear the affine matrix cache (for testing/memory management)."""
    calculate_affine_matrix.cache_clear()


def get_affine_cache_stats() -> dict:
    """Get affine matrix cache statistics."""
    info = calculate_affine_matrix.cache_info()
    return {
        'hits': info.hits,
        'misses': info.misses,
        'size': info.currsize,
        'maxsize': info.maxsize,
        'hit_rate': info.hits / (info.hits + info.misses) if (info.hits + info.misses) > 0 else 0.0,
    }


def inverse_letterbox_coords(
    boxes: np.ndarray,
    orig_shape: tuple[int, int],
    scale: float,
    padding: tuple[float, float],
    input_size: int = 640,
) -> np.ndarray:
    """
    Convert bounding boxes from letterboxed space to original image coordinates.

    EfficientNMS with normalize_boxes=True outputs coords normalized to the 640x640
    letterboxed input [0,1]. This function applies inverse letterbox transformation
    to convert them to [0,1] normalized coordinates relative to original image dimensions.

    This matches Track A (PyTorch) which uses Ultralytics boxes.xyxyn for
    original-image-normalized output.

    Args:
        boxes: np.ndarray [N, 4] in XYXY format, normalized to letterbox [0,1]
        orig_shape: Tuple (height, width) of original image
        scale: Letterbox scale factor
        padding: Tuple (pad_x, pad_y) letterbox padding
        input_size: Letterbox target size (default 640)

    Returns:
        np.ndarray [N, 4] in XYXY format, normalized to original image [0,1]
    """
    if len(boxes) == 0:
        return boxes

    orig_h, orig_w = orig_shape
    pad_x, pad_y = padding

    # Work on copy to avoid modifying input
    boxes_out = boxes.copy().astype(np.float64)

    # Auto-detect if boxes are normalized [0,1] or pixel coordinates
    # If max value > 1.0, boxes are already in pixel coords (640x640 space)
    is_pixel_coords = float(boxes_out.max()) > 1.0

    # 1. Scale letterbox-normalized [0,1] to letterbox pixels (skip if already pixels)
    if not is_pixel_coords:
        boxes_out *= input_size

    # 2. Remove padding and scale to get original pixels
    boxes_out[:, 0] = (boxes_out[:, 0] - pad_x) / scale  # x1
    boxes_out[:, 1] = (boxes_out[:, 1] - pad_y) / scale  # y1
    boxes_out[:, 2] = (boxes_out[:, 2] - pad_x) / scale  # x2
    boxes_out[:, 3] = (boxes_out[:, 3] - pad_y) / scale  # y2

    # 3. Normalize to original image dimensions
    boxes_out[:, 0] /= orig_w
    boxes_out[:, 2] /= orig_w
    boxes_out[:, 1] /= orig_h
    boxes_out[:, 3] /= orig_h

    # 4. Clip to [0, 1]
    boxes_out = np.clip(boxes_out, 0.0, 1.0)

    return boxes_out.astype(np.float32)


def format_detections_from_triton(result: dict, input_size: int = 640) -> list:
    """
    Format Triton EfficientNMS detections with original-image-normalized coordinates.

    Shared utility for Tracks C, D, E that use TensorRT End2End models with EfficientNMS.
    Applies inverse letterbox transformation to match Track A (PyTorch boxes.xyxyn).

    Args:
        result: Inference result dict with:
            - boxes: np.ndarray [N, 4] letterbox-normalized
            - scores: np.ndarray [N]
            - classes: np.ndarray [N]
            - orig_shape: (height, width) optional
            - scale: float optional
            - padding: (pad_x, pad_y) optional

    Returns:
        List of detection dicts with x1, y1, x2, y2 normalized to original image
    """
    boxes = result['boxes']
    scores = result['scores']
    classes = result['classes']

    if len(boxes) == 0:
        return []

    # Apply inverse letterbox if transformation params available
    orig_shape = result.get('orig_shape')
    scale = result.get('scale')
    padding = result.get('padding')

    if orig_shape is not None and scale is not None and padding is not None:
        boxes = inverse_letterbox_coords(boxes, orig_shape, scale, padding, input_size)

    return [
        {
            'x1': float(boxes[i, 0]),
            'y1': float(boxes[i, 1]),
            'x2': float(boxes[i, 2]),
            'y2': float(boxes[i, 3]),
            'confidence': float(scores[i]),
            'class': int(classes[i]),
        }
        for i in range(len(boxes))
    ]
