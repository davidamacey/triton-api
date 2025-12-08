"""
Shared utilities for YOLO inference APIs.

This module contains common code shared across different API implementations
to maintain consistency and reduce code duplication.

Key modules:
- affine: Cached affine matrix calculation for YOLO letterbox transformation
- cache: Embedding caching with LRU and TTL
- image_processing: Image decoding and validation
- pytorch_utils: PyTorch model loading and inference

For Triton clients, use src.clients:
- src.clients.triton_client.TritonClient (unified client for Tracks C/D/E)
- src.clients.triton_pool.TritonClientManager (connection pool manager)
"""

# Image processing
# Triton clients moved to src.clients - import directly from src.clients to avoid circular imports
# from src.clients.triton_client import TritonClient, get_triton_client as get_unified_triton_client
# from src.clients.triton_pool import (...)

# DO NOT import TritonClient here - it creates circular dependency
# triton_client.py imports from utils/affine.py which would trigger this __init__.py
# Always import from src.clients.triton_client or src.clients.triton_pool directly

# =============================================================================
# Backward Compatibility Imports
# =============================================================================
# InferenceResult moved to schemas
from src.schemas.detection import InferenceResult

# Affine transformation (Track D)
from .affine import (
    calculate_affine_matrix,
    clear_affine_cache,
    get_affine_cache_stats,
    get_jpeg_dimensions_fast,
    prepare_triton_inputs,
)

# Caching (Track E)
from .cache import (
    EmbeddingCache,
    clear_all_caches,
    get_all_cache_stats,
    get_clip_tokenizer,
    get_image_cache,
    get_text_cache,
)
from .image_processing import decode_image, validate_image

# PyTorch utilities (Track A)
from .pytorch_utils import (
    cleanup_pytorch_models,
    format_detections,
    load_pytorch_models,
    log_gpu_info,
    log_gpu_memory,
    pytorch_lifespan,
    thread_safe_predict,
    thread_safe_predict_batch,
)


__all__ = [
    # Cache utilities
    'EmbeddingCache',
    # Backward compat
    'InferenceResult',
    # Affine utilities
    'calculate_affine_matrix',
    # PyTorch utilities
    'cleanup_pytorch_models',
    'clear_affine_cache',
    # Cache management
    'clear_all_caches',
    # Image processing
    'decode_image',
    'format_detections',
    'get_affine_cache_stats',
    'get_all_cache_stats',
    'get_clip_tokenizer',
    'get_image_cache',
    'get_jpeg_dimensions_fast',
    'get_text_cache',
    'load_pytorch_models',
    'log_gpu_info',
    'log_gpu_memory',
    'prepare_triton_inputs',
    'pytorch_lifespan',
    'thread_safe_predict',
    'thread_safe_predict_batch',
    'validate_image',
]

# Note: TritonClient and pool functions removed from __all__ to prevent circular imports
# Import these directly from src.clients:
#   from src.clients.triton_client import TritonClient, get_triton_client
#   from src.clients.triton_pool import get_triton_client, close_all_clients, etc.
