"""
DALI Preprocessing Pipelines for Triton Inference Server

This package contains GPU-accelerated preprocessing pipelines using NVIDIA DALI:

Modules:
    config: Shared configuration constants and dataclasses
    utils: Utility functions for DALI operations

Scripts:
    create_dali_letterbox_pipeline.py: Track D YOLO-only preprocessing
    create_dual_dali_pipeline.py: Track E triple-branch (YOLO + CLIP + HD)
    create_yolo_clip_dali_pipeline.py: Track E dual-branch (YOLO + CLIP)
    create_ensembles.py: Generate ensemble model configurations
    validate_dali_letterbox.py: Validate Track D pipeline
    validate_dual_dali_preprocessing.py: Validate Track E pipeline

Usage:
    # Import configuration
    from dali.config import YOLO_SIZE, CLIP_SIZE, DALIConfig

    # Import utilities
    from dali.utils import calculate_letterbox_affine, load_test_image
"""

from dali.config import (
    CLIP_SIZE,
    CROP_IMAGE_MAX_SIZE,
    DEFAULT_MODEL_DIR,
    DEFAULT_TRITON_URL,
    ENSEMBLE_TIERS,
    MAX_BATCH_SIZE,
    MODEL_SIZES,
    NUM_THREADS,
    YOLO_PAD_VALUE,
    YOLO_SIZE,
    DALIConfig,
)
from dali.utils import (
    calculate_letterbox_affine,
    check_dali_available,
    create_test_image_jpeg,
    create_triton_client,
    load_test_image,
    validate_preprocessing_output,
)


__all__ = [
    # Config
    'YOLO_SIZE',
    'CLIP_SIZE',
    'YOLO_PAD_VALUE',
    'CROP_IMAGE_MAX_SIZE',
    'MAX_BATCH_SIZE',
    'NUM_THREADS',
    'DEFAULT_MODEL_DIR',
    'DEFAULT_TRITON_URL',
    'DALIConfig',
    'ENSEMBLE_TIERS',
    'MODEL_SIZES',
    # Utils
    'calculate_letterbox_affine',
    'check_dali_available',
    'create_test_image_jpeg',
    'create_triton_client',
    'load_test_image',
    'validate_preprocessing_output',
]
