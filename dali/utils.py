#!/usr/bin/env python3
"""
DALI Pipeline Utilities

Shared utility functions for DALI preprocessing pipelines.
Reduces code duplication and ensures consistent behavior across:
- Pipeline creation scripts
- Validation scripts
- Test utilities

Usage:
    from dali.utils import calculate_letterbox_affine, create_test_image
"""

from __future__ import annotations

import io
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from numpy.typing import NDArray

from dali.config import (
    DEFAULT_DEVICE_ID,
    DEFAULT_MODEL_DIR,
    DEFAULT_TEST_IMAGE,
    DEFAULT_TRITON_URL,
    MAX_BATCH_SIZE,
    NUM_THREADS,
    YOLO_SIZE,
)


# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# Affine Matrix Calculation
# =============================================================================


def calculate_letterbox_affine(
    orig_w: int,
    orig_h: int,
    target_size: int = YOLO_SIZE,
    no_upscale: bool = True,
) -> tuple[NDArray[np.float32], float, float, float]:
    """
    Calculate affine transformation matrix for YOLO letterbox preprocessing.

    The affine matrix transforms the original image to a letterboxed square image:
    - Scales the image to fit within target_size while preserving aspect ratio
    - Centers the scaled image with gray padding (value=114)
    - Optionally prevents upscaling of small images

    Args:
        orig_w: Original image width in pixels.
        orig_h: Original image height in pixels.
        target_size: Target square size (default: 640).
        no_upscale: If True, don't upscale small images (default: True).

    Returns:
        Tuple of (affine_matrix, scale, pad_x, pad_y) where:
        - affine_matrix: [2, 3] float32 array [[scale_x, 0, offset_x], [0, scale_y, offset_y]]
        - scale: Scale factor applied to the image
        - pad_x: Horizontal padding (left side) in pixels
        - pad_y: Vertical padding (top side) in pixels

    Example:
        >>> affine, scale, pad_x, pad_y = calculate_letterbox_affine(1920, 1080)
        >>> print(f'Scale: {scale:.4f}, Padding: ({pad_x:.1f}, {pad_y:.1f})')
        Scale: 0.3333, Padding: (0.0, 80.0)
    """
    # Calculate scale to fit within target_size
    scale = min(target_size / orig_h, target_size / orig_w)

    # Prevent upscaling if requested
    if no_upscale:
        scale = min(scale, 1.0)

    # Calculate new dimensions after scaling
    new_h = round(orig_h * scale)
    new_w = round(orig_w * scale)

    # Calculate padding to center the image
    pad_x = (target_size - new_w) / 2.0
    pad_y = (target_size - new_h) / 2.0

    # Create affine matrix [2, 3]
    # Format: [[scale_x, 0, offset_x], [0, scale_y, offset_y]]
    affine_matrix = np.array(
        [[scale, 0.0, pad_x], [0.0, scale, pad_y]],
        dtype=np.float32,
    )

    return affine_matrix, scale, pad_x, pad_y


# =============================================================================
# Test Image Creation
# =============================================================================


def create_test_image_jpeg(
    height: int = 1080,
    width: int = 810,
    quality: int = 90,
) -> tuple[bytes, int, int]:
    """
    Create a test JPEG image with random noise.

    Args:
        height: Image height in pixels (default: 1080).
        width: Image width in pixels (default: 810).
        quality: JPEG quality 1-100 (default: 90).

    Returns:
        Tuple of (jpeg_bytes, width, height) matching PIL convention.
    """
    try:
        from PIL import Image
    except ImportError as e:
        logger.error('PIL not installed: %s', e)
        raise

    # Create random noise image
    img_array = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    pil_img = Image.fromarray(img_array)

    # Encode to JPEG
    buffer = io.BytesIO()
    pil_img.save(buffer, format='JPEG', quality=quality)
    jpeg_bytes = buffer.getvalue()

    return jpeg_bytes, width, height


def load_test_image(
    image_path: Path | str = DEFAULT_TEST_IMAGE,
) -> tuple[bytes, int, int]:
    """
    Load a test image and return JPEG bytes with dimensions.

    Args:
        image_path: Path to the test image file.

    Returns:
        Tuple of (jpeg_bytes, width, height).

    Raises:
        FileNotFoundError: If image file doesn't exist.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f'Test image not found: {image_path}')

    try:
        from PIL import Image
    except ImportError as e:
        logger.error('PIL not installed: %s', e)
        raise

    with open(image_path, 'rb') as f:
        jpeg_bytes = f.read()

    img = Image.open(image_path)
    width, height = img.size

    return jpeg_bytes, width, height


# =============================================================================
# DALI Pipeline Helpers
# =============================================================================


def check_dali_available() -> bool:
    """
    Check if NVIDIA DALI is available.

    Returns:
        True if DALI is available, False otherwise.
    """
    try:
        from nvidia import dali

        logger.info('NVIDIA DALI version: %s', dali.__version__)
        return True
    except ImportError as e:
        logger.error('NVIDIA DALI not installed: %s', e)
        return False


def get_dali_version() -> str | None:
    """
    Get the installed NVIDIA DALI version.

    Returns:
        Version string or None if not installed.
    """
    try:
        from nvidia import dali

        return dali.__version__
    except ImportError:
        return None


def build_dali_pipeline(
    pipeline_func: callable,
    batch_size: int = MAX_BATCH_SIZE,
    num_threads: int = NUM_THREADS,
    device_id: int = DEFAULT_DEVICE_ID,
) -> object:
    """
    Build a DALI pipeline with standard configuration.

    Args:
        pipeline_func: DALI pipeline function decorated with @dali.pipeline_def.
        batch_size: Maximum batch size.
        num_threads: Number of CPU threads for pipeline orchestration.
        device_id: GPU device ID.

    Returns:
        Built DALI pipeline object.

    Raises:
        RuntimeError: If pipeline build fails.
    """
    try:
        pipe = pipeline_func(
            batch_size=batch_size,
            num_threads=num_threads,
            device_id=device_id,
        )
        pipe.build()
        return pipe
    except Exception as e:
        logger.error('Failed to build DALI pipeline: %s', e)
        raise RuntimeError(f'DALI pipeline build failed: {e}') from e


# =============================================================================
# Triton Client Helpers
# =============================================================================


def create_triton_client(
    url: str = DEFAULT_TRITON_URL,
    verbose: bool = False,
) -> object:
    """
    Create a Triton gRPC inference client.

    Args:
        url: Triton server URL (default: triton-api:8001).
        verbose: Enable verbose logging.

    Returns:
        Triton InferenceServerClient object.

    Raises:
        ConnectionError: If connection to Triton fails.
    """
    try:
        from tritonclient.grpc import InferenceServerClient
    except ImportError as e:
        logger.error('tritonclient not installed: %s', e)
        raise ImportError('Install tritonclient: pip install tritonclient[grpc]') from e

    try:
        client = InferenceServerClient(url=url, verbose=verbose)
        logger.info('Connected to Triton at %s', url)
        return client
    except Exception as e:
        logger.error('Failed to connect to Triton at %s: %s', url, e)
        raise ConnectionError(f'Cannot connect to Triton at {url}: {e}') from e


def check_model_ready(client: object, model_name: str) -> bool:
    """
    Check if a Triton model is ready for inference.

    Args:
        client: Triton InferenceServerClient.
        model_name: Name of the model to check.

    Returns:
        True if model is ready, False otherwise.
    """
    try:
        if client.is_model_ready(model_name):
            logger.info("Model '%s' is ready", model_name)
            return True
        logger.warning("Model '%s' is not ready", model_name)
        return False
    except Exception as e:
        logger.error("Failed to check model '%s': %s", model_name, e)
        return False


# =============================================================================
# Output Validation
# =============================================================================


def validate_preprocessing_output(
    output: NDArray,
    expected_shape: tuple[int, ...],
    name: str = 'output',
) -> bool:
    """
    Validate preprocessing output against expected specifications.

    Args:
        output: NumPy array output from preprocessing.
        expected_shape: Expected shape tuple (e.g., (3, 640, 640)).
        name: Name for logging (default: "output").

    Returns:
        True if validation passes, False otherwise.
    """
    is_valid = True

    # Check shape
    if output.shape != expected_shape:
        logger.error('%s shape mismatch: got %s, expected %s', name, output.shape, expected_shape)
        is_valid = False

    # Check dtype
    if output.dtype != np.float32:
        logger.error('%s dtype mismatch: got %s, expected float32', name, output.dtype)
        is_valid = False

    # Check range [0, 1]
    if output.min() < 0:
        logger.error('%s min value out of range: %.4f < 0', name, output.min())
        is_valid = False

    if output.max() > 1:
        logger.error('%s max value out of range: %.4f > 1', name, output.max())
        is_valid = False

    if is_valid:
        logger.info(
            '%s validation passed: shape=%s, dtype=%s, range=[%.4f, %.4f]',
            name,
            output.shape,
            output.dtype,
            output.min(),
            output.max(),
        )

    return is_valid


# =============================================================================
# Path Utilities
# =============================================================================


def ensure_model_directory(model_name: str, model_dir: Path = DEFAULT_MODEL_DIR) -> Path:
    """
    Ensure model directory structure exists.

    Creates: {model_dir}/{model_name}/1/

    Args:
        model_name: Name of the model.
        model_dir: Base model directory.

    Returns:
        Path to the version directory (e.g., models/model_name/1/).
    """
    version_dir = model_dir / model_name / '1'
    version_dir.mkdir(parents=True, exist_ok=True)
    return version_dir


def get_config_path(model_name: str, model_dir: Path = DEFAULT_MODEL_DIR) -> Path:
    """
    Get the config.pbtxt path for a model.

    Args:
        model_name: Name of the model.
        model_dir: Base model directory.

    Returns:
        Path to config.pbtxt.
    """
    return model_dir / model_name / 'config.pbtxt'


# =============================================================================
# CLI Utilities
# =============================================================================


def setup_logging(verbose: bool = False) -> None:
    """
    Configure logging for DALI scripts.

    Args:
        verbose: Enable debug-level logging.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(levelname)s: %(message)s',
        stream=sys.stdout,
    )


def require_dali() -> None:
    """
    Exit with error if DALI is not available.

    Call this at the start of scripts that require DALI.
    """
    if not check_dali_available():
        print('ERROR: NVIDIA DALI not installed')
        print('This script must be run from the yolo-api container.')
        print('Run: docker compose exec yolo-api python /app/dali/<script>.py')
        sys.exit(1)
