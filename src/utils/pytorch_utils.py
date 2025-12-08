"""
PyTorch-specific utilities for YOLO inference.

Thread-safe wrappers, model loading, and lifecycle management for PyTorch models.
"""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils import ThreadingLocked


logger = logging.getLogger(__name__)


# ============================================================================
# Thread-Safe Inference Wrappers
# ============================================================================
# PyTorch models are NOT thread-safe - multiple threads accessing the same
# model instance can cause CUDA race conditions and crashes.
#
# Solution: Use @ThreadingLocked() decorator from ultralytics
# https://docs.ultralytics.com/guides/yolo-thread-safe-inference/
# ============================================================================


@ThreadingLocked()
def thread_safe_predict(model: YOLO, img: np.ndarray):
    """
    Thread-safe single image inference for PyTorch models.

    The @ThreadingLocked decorator ensures that when multiple threads access
    the same model instance (via FastAPI's thread pool), only one inference
    runs at a time, preventing CUDA race conditions.

    **Why needed:**
    - PyTorch models load weights into GPU memory
    - Multiple concurrent calls can corrupt GPU state
    - @ThreadingLocked() serializes access to prevent this

    **Performance:**
    - Single-threaded inference (lock prevents parallelism)
    - But necessary for correctness with shared model instances
    - Alternative: Per-request model instances (not practical for PyTorch)

    Args:
        model: YOLO model instance (shared across requests)
        img: Image array (numpy, HWC, BGR)

    Returns:
        Detection results from YOLO model

    Example:
        >>> model = YOLO('/app/pytorch_models/yolo11n.pt')
        >>> results = thread_safe_predict(model, img)
    """
    # Use FP16 if GPU available (to match TensorRT models)
    use_half = torch.cuda.is_available()
    return model(img, verbose=False, half=use_half)


@ThreadingLocked()
def thread_safe_predict_batch(model: YOLO, images: list[np.ndarray]):
    """
    Thread-safe batch inference for PyTorch models.

    Similar to thread_safe_predict but handles multiple images in one call.

    Args:
        model: YOLO model instance (shared across requests)
        images: List of image arrays (numpy, HWC, BGR)

    Returns:
        Batch detection results from YOLO model

    Example:
        >>> model = YOLO('/app/pytorch_models/yolo11n.pt')
        >>> results = thread_safe_predict_batch(model, [img1, img2, img3])
    """
    # Use FP16 if GPU available (to match TensorRT models)
    use_half = torch.cuda.is_available()
    return model(images, verbose=False, half=use_half)


# ============================================================================
# Model Loading and Lifecycle Management
# ============================================================================


def load_pytorch_models(
    model_paths: dict[str, str], device: str = 'cuda', warmup: bool = True
) -> dict[str, YOLO]:
    """
    Load multiple YOLO PyTorch models with GPU placement and warmup.

    **Process:**
    1. Load model from .pt file
    2. Move to GPU (if available)
    3. Warmup with dummy inference (initializes CUDA kernels)

    **Why warmup:**
    - First inference triggers CUDA kernel compilation
    - Can take 2-5 seconds
    - Better to do at startup than on first user request

    Args:
        model_paths: Dict mapping model names to .pt file paths
                    e.g., {"nano": "/app/pytorch_models/yolo11n.pt"}
        device: Target device ('cuda' or 'cpu')
        warmup: Whether to run warmup inference

    Returns:
        Dict mapping model names to loaded YOLO instances

    Raises:
        Exception: If model loading fails (logged but not raised)

    Example:
        >>> models = load_pytorch_models(
        ...     {
        ...         'nano': '/app/pytorch_models/yolo11n.pt',
        ...         'small': '/app/pytorch_models/yolo11s.pt',
        ...     }
        ... )
        >>> # models["nano"] is ready for inference
    """
    loaded_models = {}

    logger.info(f'Loading {len(model_paths)} YOLO models...')

    for model_name, model_path in model_paths.items():
        try:
            logger.info(f'\n  → Loading {model_name} ({model_path})...')

            # Load model from .pt file
            model = YOLO(model_path)

            # Move to GPU if available
            if device == 'cuda' and torch.cuda.is_available():
                model.to(device)
                logger.info(f'    Moved to GPU: {torch.cuda.get_device_name(0)}')

            # Warmup inference (initialize CUDA kernels)
            if warmup:
                logger.info(f'    Warming up {model_name}...')
                dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
                _ = model(dummy_img, verbose=False)

            loaded_models[model_name] = model
            logger.info(f'    ✓ {model_name} ready')

        except Exception as e:
            logger.error(f'    ✗ Failed to load {model_name}: {e}')

    # Summary
    if loaded_models:
        logger.info(f'\n✓ Successfully loaded {len(loaded_models)}/{len(model_paths)} models')
        logger.info(f'  Available: {list(loaded_models.keys())}')
    else:
        logger.error('✗ No models loaded successfully!')

    return loaded_models


def log_gpu_info():
    """
    Log GPU information (name, memory, compute capability).

    Useful for debugging and verifying GPU availability.
    """
    if torch.cuda.is_available():
        logger.info(f'GPU: {torch.cuda.get_device_name(0)}')
        gpu_props = torch.cuda.get_device_properties(0)
        logger.info(f'GPU Memory: {gpu_props.total_memory / 1e9:.2f} GB')
        logger.info(f'Compute Capability: {gpu_props.major}.{gpu_props.minor}')
    else:
        logger.warning('GPU not available - using CPU')


def log_gpu_memory():
    """
    Log current GPU memory usage (allocated and reserved).

    Useful for monitoring memory consumption after loading models.
    """
    if torch.cuda.is_available():
        logger.info('GPU Memory Usage:')
        logger.info(f'  Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB')
        logger.info(f'  Reserved:  {torch.cuda.memory_reserved(0) / 1e9:.2f} GB')


def cleanup_pytorch_models(model_instances: dict[str, YOLO]):
    """
    Clean up PyTorch models and free GPU memory.

    **Process:**
    1. Delete model instances
    2. Clear GPU cache
    3. Verify cleanup

    Args:
        model_instances: Dict of loaded model instances

    Example:
        >>> cleanup_pytorch_models(MODEL_INSTANCES)
        >>> # GPU memory freed
    """
    if not model_instances:
        return

    if torch.cuda.is_available():
        logger.info('Clearing GPU memory...')

        # Delete all model instances
        for model_name in list(model_instances.keys()):
            del model_instances[model_name]

        # Clear CUDA cache
        torch.cuda.empty_cache()

        logger.info('✓ GPU memory cleared')

    # Clear dictionary
    model_instances.clear()
    logger.info('✓ Model cleanup complete')


@asynccontextmanager
async def pytorch_lifespan(
    _app, model_paths: dict[str, str], model_storage: dict[str, YOLO]
) -> AsyncGenerator[None, None]:
    """
    FastAPI lifespan context manager for PyTorch models.

    Handles startup (load models) and shutdown (cleanup) events.

    **Startup:**
    - Log GPU info
    - Load all models
    - Warmup models
    - Log memory usage

    **Shutdown:**
    - Delete models
    - Clear GPU cache
    - Log cleanup status

    Args:
        app: FastAPI application instance
        model_paths: Dict mapping model names to .pt file paths
        model_storage: Dict to store loaded model instances (modified in-place)

    Yields:
        None (application runs between startup and shutdown)

    Example:
        >>> MODEL_PATHS = {'nano': '/app/pytorch_models/yolo11n.pt'}
        >>> MODEL_INSTANCES = {}
        >>>
        >>> @asynccontextmanager
        >>> async def lifespan(app):
        >>>     async with pytorch_lifespan(app, MODEL_PATHS, MODEL_INSTANCES):
        >>>         yield
        >>>
        >>> app = FastAPI(lifespan=lifespan)
    """
    # ========================================================================
    # Startup
    # ========================================================================
    logger.info('=' * 60)
    logger.info('Starting PyTorch YOLO API')
    logger.info('=' * 60)

    # GPU info
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f'Device: {device}')
    log_gpu_info()

    # Load models
    loaded_models = load_pytorch_models(model_paths, device=device, warmup=True)

    # Update storage dict (in-place)
    model_storage.clear()
    model_storage.update(loaded_models)

    # Memory stats
    if device == 'cuda':
        logger.info('')
        log_gpu_memory()

    logger.info('=' * 60)
    logger.info('Service ready for inference requests')
    logger.info('=' * 60 + '\n')

    # ========================================================================
    # Application runs here
    # ========================================================================
    yield

    # ========================================================================
    # Shutdown
    # ========================================================================
    logger.info('\n' + '=' * 60)
    logger.info('Shutting down PyTorch YOLO API')
    logger.info('=' * 60)

    cleanup_pytorch_models(model_storage)

    logger.info('=' * 60)
    logger.info('Shutdown complete')
    logger.info('=' * 60 + '\n')


# ============================================================================
# Detection Formatting Utilities
# ============================================================================


def format_detections(results) -> list[dict]:
    """
    Extract and format detections from YOLO results with NORMALIZED coordinates.

    Uses Ultralytics' built-in `boxes.xyxyn` property which returns coordinates
    normalized to [0, 1] range relative to original image dimensions.

    **Output format** (API response):
    - List of dicts with x1, y1, x2, y2 in NORMALIZED [0,1] coordinates
    - confidence, class fields

    **Why normalized coordinates:**
    - Industry standard for object detection
    - Database-friendly (works with any image resolution)
    - Consistent with TensorRT End2End models (EfficientNMS normalize_boxes=True)
    - Easy to convert back: x_pixel = x_norm * width

    Args:
        results: YOLO detection results

    Returns:
        List of detection dictionaries with normalized coordinates

    Example:
        >>> results = model(img)  # 1920x1080 image
        >>> detections = format_detections(results)
        >>> # [{"x1": 0.052, "y1": 0.185, ..., "confidence": 0.95, "class": 0}]
    """
    # Use Ultralytics built-in xyxyn for normalized [0,1] coordinates
    boxes = results[0].boxes.xyxyn.cpu().numpy()
    scores = results[0].boxes.conf.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()

    return [
        {
            'x1': float(box[0]),
            'y1': float(box[1]),
            'x2': float(box[2]),
            'y2': float(box[3]),
            'confidence': float(score),
            'class': int(cls),
        }
        for box, score, cls in zip(boxes, scores, classes, strict=False)
    ]
