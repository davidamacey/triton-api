"""
Shared utilities for YOLO inference APIs.

This module contains common code shared across different API implementations
to maintain consistency and reduce code duplication.
"""

from .image_processing import decode_image, validate_image
from .models import InferenceResult
from .pytorch_utils import (
    thread_safe_predict,
    thread_safe_predict_batch,
    load_pytorch_models,
    cleanup_pytorch_models,
    pytorch_lifespan,
    format_detections,
    log_gpu_info,
    log_gpu_memory,
)
from .triton_end2end_client import TritonEnd2EndClient
from .triton_shared_client import get_triton_client, close_all_clients, get_client_pool_stats  # NEW!

__all__ = [
    # Common utilities
    "decode_image",
    "validate_image",
    "InferenceResult",
    # PyTorch-specific utilities
    "thread_safe_predict",
    "thread_safe_predict_batch",
    "load_pytorch_models",
    "cleanup_pytorch_models",
    "pytorch_lifespan",
    "format_detections",
    "log_gpu_info",
    "log_gpu_memory",
    # Triton-specific utilities
    "TritonEnd2EndClient",
    "get_triton_client",  # NEW - shared client for batching
    "close_all_clients",  # NEW - cleanup on shutdown
    "get_client_pool_stats",  # NEW - monitoring
]
