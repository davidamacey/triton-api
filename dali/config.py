#!/usr/bin/env python3
"""
DALI Pipeline Configuration

Shared configuration constants for all DALI preprocessing pipelines.
This module centralizes configuration to ensure consistency across:
- Track D: YOLO-only preprocessing (create_dali_letterbox_pipeline.py)
- Track E Full: YOLO + CLIP + HD cropping (create_dual_dali_pipeline.py)
- Track E Simple: YOLO + CLIP only (create_yolo_clip_dali_pipeline.py)

Usage:
    from dali.config import DALIConfig, YOLO_SIZE, CLIP_SIZE
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar


# =============================================================================
# Core Constants
# =============================================================================

# YOLO preprocessing
YOLO_SIZE: int = 640
YOLO_PAD_VALUE: int = 114  # Gray padding (YOLO standard)

# MobileCLIP preprocessing
CLIP_SIZE: int = 256

# SCRFD face detection preprocessing
SCRFD_SIZE: int = 640  # SCRFD uses 640x640 (resized, not letterboxed)
ARCFACE_SIZE: int = 112  # ArcFace aligned face size

# HD cropping (Track E full only)
CROP_IMAGE_MAX_SIZE: int = 1920  # Max pixels on longest edge

# Pipeline settings
MAX_BATCH_SIZE: int = 128
NUM_THREADS: int = 4
HW_DECODER_LOAD: float = 0.65  # Hardware decoder offload (Ampere+)

# Triton settings
DEFAULT_TRITON_URL: str = os.getenv('TRITON_URL', 'triton-api:8001')
DEFAULT_DEVICE_ID: int = int(os.getenv('DALI_DEVICE_ID', '0'))


# =============================================================================
# Model Names and Paths
# =============================================================================

# Track D: YOLO-only preprocessing
TRACK_D_DALI_MODEL: str = 'yolo_preprocess_dali'
TRACK_D_TRT_MODEL: str = 'yolov11_{size}_trt_end2end'
TRACK_D_ENSEMBLE_SUFFIX: dict[str, str] = {
    'streaming': '_gpu_e2e_streaming',
    'balanced': '_gpu_e2e',
    'batch': '_gpu_e2e_batch',
}
# Tier-specific DALI model names (for ensembles that need tier-matched DALI)
TRACK_D_DALI_MODELS: dict[str, str] = {
    'streaming': 'yolo_preprocess_dali_streaming',
    'balanced': 'yolo_preprocess_dali',
    'batch': 'yolo_preprocess_dali_batch',
}

# Track E: YOLO + CLIP preprocessing
TRACK_E_DUAL_DALI_MODEL: str = 'dual_preprocess_dali'  # Triple-branch (YOLO + CLIP + HD)
TRACK_E_SIMPLE_DALI_MODEL: str = 'yolo_clip_preprocess_dali'  # Dual-branch (YOLO + CLIP)
TRACK_E_QUAD_DALI_MODEL: str = 'quad_preprocess_dali'  # Quad-branch (YOLO + CLIP + SCRFD + HD)

# Default paths
DEFAULT_MODEL_DIR: Path = Path(os.getenv('TRITON_MODEL_REPO', '/app/models'))
DEFAULT_TEST_IMAGE: Path = Path('/app/test_images/bus.jpg')


# =============================================================================
# Output Names (must match Triton config.pbtxt)
# =============================================================================

# Track D outputs
TRACK_D_OUTPUT_PREPROCESSED: str = 'preprocessed_images'

# Track E outputs (dual/triple/quad-branch)
TRACK_E_OUTPUT_YOLO: str = 'yolo_images'
TRACK_E_OUTPUT_CLIP: str = 'clip_images'
TRACK_E_OUTPUT_ORIGINAL: str = 'original_images'  # Only in triple/quad-branch
TRACK_E_OUTPUT_FACE: str = 'face_images'  # Only in quad-branch (SCRFD)

# Detection outputs (from TRT End2End)
OUTPUT_NUM_DETS: str = 'num_dets'
OUTPUT_DET_BOXES: str = 'det_boxes'
OUTPUT_DET_SCORES: str = 'det_scores'
OUTPUT_DET_CLASSES: str = 'det_classes'


# =============================================================================
# Ensemble Tier Configuration
# =============================================================================


@dataclass
class TierConfig:
    """Configuration for a single ensemble tier."""

    suffix: str
    description: str
    max_batch: int
    preferred_batch_sizes: list[int]
    max_queue_delay_us: int
    preserve_ordering: bool
    timeout_us: int
    max_queue_size: int
    instance_count: int


ENSEMBLE_TIERS: dict[str, TierConfig] = {
    'streaming': TierConfig(
        suffix='_gpu_e2e_streaming',
        description='Real-time video streaming (optimized for low latency)',
        max_batch=8,
        preferred_batch_sizes=[1, 2, 4],
        max_queue_delay_us=100,  # 0.1ms
        preserve_ordering=True,  # Essential for video frames
        timeout_us=5000,  # 5ms (strict)
        max_queue_size=16,
        instance_count=3,
    ),
    'balanced': TierConfig(
        suffix='_gpu_e2e',
        description='General purpose (balanced latency/throughput)',
        max_batch=64,
        preferred_batch_sizes=[4, 8, 16, 32],
        max_queue_delay_us=500,  # 0.5ms
        preserve_ordering=False,
        timeout_us=50000,  # 50ms
        max_queue_size=64,
        instance_count=2,
    ),
    'batch': TierConfig(
        suffix='_gpu_e2e_batch',
        description='Offline batch processing (optimized for throughput)',
        max_batch=128,
        preferred_batch_sizes=[32, 64, 128],
        max_queue_delay_us=5000,  # 5ms
        preserve_ordering=False,
        timeout_us=100000,  # 100ms (very lenient)
        max_queue_size=256,
        instance_count=1,
    ),
}


# =============================================================================
# Model Size Configuration
# =============================================================================


@dataclass
class ModelConfig:
    """Configuration for a YOLO model size."""

    triton_name: str
    max_batch: int
    topk: int = 300  # Max detections per image


MODEL_SIZES: dict[str, ModelConfig] = {
    'nano': ModelConfig(triton_name='yolov11_nano', max_batch=128, topk=300),
    'small': ModelConfig(triton_name='yolov11_small', max_batch=64, topk=300),
    'medium': ModelConfig(triton_name='yolov11_medium', max_batch=32, topk=300),
}


# =============================================================================
# Validation Thresholds
# =============================================================================

CORRECTNESS_THRESHOLD: float = 0.01  # 1% mean absolute difference
PERFORMANCE_TARGET_MS: float = 5.0  # Target latency for preprocessing
DEFAULT_BENCHMARK_ITERATIONS: int = 100
DEFAULT_WARMUP_ITERATIONS: int = 10


# =============================================================================
# DALIConfig Dataclass (for backwards compatibility)
# =============================================================================


@dataclass
class DALIConfig:
    """
    Complete DALI pipeline configuration.

    Use this for programmatic configuration of DALI pipelines.
    All values have sensible defaults matching production settings.
    """

    # Image sizes
    yolo_size: int = YOLO_SIZE
    clip_size: int = CLIP_SIZE
    pad_value: int = YOLO_PAD_VALUE
    crop_image_max_size: int = CROP_IMAGE_MAX_SIZE

    # Pipeline settings
    max_batch_size: int = MAX_BATCH_SIZE
    num_threads: int = NUM_THREADS
    device_id: int = DEFAULT_DEVICE_ID
    hw_decoder_load: float = HW_DECODER_LOAD

    # Paths
    model_dir: Path = field(default_factory=lambda: DEFAULT_MODEL_DIR)
    test_image: Path = field(default_factory=lambda: DEFAULT_TEST_IMAGE)

    # Triton
    triton_url: str = DEFAULT_TRITON_URL

    # Class variables for normalization (not instance fields)
    NORM_MEAN: ClassVar[list[float]] = [0.0, 0.0, 0.0]
    NORM_STD: ClassVar[list[float]] = [255.0, 255.0, 255.0]  # Divide by 255

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.yolo_size <= 0:
            raise ValueError(f'yolo_size must be positive, got {self.yolo_size}')
        if self.clip_size <= 0:
            raise ValueError(f'clip_size must be positive, got {self.clip_size}')
        if self.max_batch_size <= 0:
            raise ValueError(f'max_batch_size must be positive, got {self.max_batch_size}')
        if self.device_id < 0:
            raise ValueError(f'device_id must be non-negative, got {self.device_id}')

    def get_track_d_model_path(self) -> Path:
        """Get the path for Track D DALI model."""
        return self.model_dir / TRACK_D_DALI_MODEL / '1' / 'model.dali'

    def get_track_e_dual_model_path(self) -> Path:
        """Get the path for Track E dual DALI model (triple-branch)."""
        return self.model_dir / TRACK_E_DUAL_DALI_MODEL / '1' / 'model.dali'

    def get_track_e_simple_model_path(self) -> Path:
        """Get the path for Track E simple DALI model (dual-branch)."""
        return self.model_dir / TRACK_E_SIMPLE_DALI_MODEL / '1' / 'model.dali'
