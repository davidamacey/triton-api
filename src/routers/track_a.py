"""
Track A: PyTorch Direct Inference Router

Provides baseline PyTorch inference endpoints using Ultralytics YOLO.
Returns industry-standard response format with timing injected by middleware.
"""

import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from src.core.dependencies import get_pytorch_models
from src.schemas.detection import BatchInferenceResult, InferenceResult
from src.services.inference import InferenceService


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix='/pytorch',
    tags=['Track A - PyTorch'],
)

# Service instance
inference_service = InferenceService()


@router.post('/predict/{model_name}', response_model=InferenceResult)
def predict_pytorch(
    model_name: str = 'small',
    image: UploadFile = File(...),
    models: dict = Depends(get_pytorch_models),
):
    """
    Track A: Direct PyTorch inference (baseline).

    Single image inference using native PyTorch YOLO models.
    Uses thread-safe prediction with locks for concurrent access.

    Response includes:
    - detections: List of normalized [0,1] bounding boxes
    - num_detections: Count of detections
    - image: Original image dimensions
    - model: Model name and backend info
    - total_time_ms: End-to-end request time (injected by middleware)

    Args:
        model_name: Model variant (default: 'small')
        image: Image file (JPEG/PNG)

    Returns:
        InferenceResult with detections and metadata
    """
    if not models:
        raise HTTPException(
            status_code=503,
            detail='Track A (PyTorch) is disabled. Set ENABLE_PYTORCH=true to enable.',
        )

    if model_name not in models:
        available = list(models.keys())
        raise HTTPException(
            status_code=400, detail=f"Invalid model '{model_name}'. Available: {available}"
        )

    model = models[model_name]
    filename = image.filename or 'uploaded_image'

    try:
        image_data = image.file.read()
        return inference_service.infer_pytorch(
            image_bytes=image_data,
            filename=filename,
            model=model,
            model_name=f'yolov11_{model_name}',
        )

    except ValueError as e:
        logger.warning(f'Client error for {filename}: {e}')
        raise HTTPException(status_code=400, detail=str(e)) from e

    except Exception as e:
        logger.error(f'Inference error for {filename}: {e}')
        raise HTTPException(status_code=500, detail=f'Inference failed: {e!s}') from e


@router.post('/predict_batch/{model_name}', response_model=BatchInferenceResult)
def predict_batch_pytorch(
    model_name: str = 'small',
    images: list[UploadFile] = File(...),
    models: dict = Depends(get_pytorch_models),
):
    """
    Track A: Batch PyTorch inference.

    Process multiple images in a single request using thread-safe batch prediction.

    Args:
        model_name: Model variant (default: 'small')
        images: List of image files

    Returns:
        BatchInferenceResult with per-image results and metadata
    """
    if not models:
        raise HTTPException(
            status_code=503,
            detail='Track A (PyTorch) is disabled. Set ENABLE_PYTORCH=true to enable.',
        )

    if model_name not in models:
        available = list(models.keys())
        raise HTTPException(
            status_code=400, detail=f"Invalid model '{model_name}'. Available: {available}"
        )

    model = models[model_name]

    try:
        # Collect image data
        images_data = []
        for image in images:
            filename = image.filename or f'image_{len(images_data)}'
            image_data = image.file.read()
            images_data.append((image_data, filename))

        return inference_service.infer_pytorch_batch(
            images_data=images_data,
            model=model,
            model_name=f'yolov11_{model_name}',
        )

    except Exception as e:
        logger.error(f'Batch inference error: {e}')
        raise HTTPException(status_code=500, detail=f'Batch inference failed: {e!s}') from e
