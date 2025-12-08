"""
Tracks B/C/D: Triton Inference Router

Unified router for all Triton-based inference:
- Track B: Standard TRT + CPU NMS
- Track C: End2End TRT + GPU NMS
- Track D: DALI + TRT (Full GPU Pipeline)
"""

import logging

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from ultralytics import YOLO

from src.clients.triton_client import get_triton_client
from src.config import get_settings
from src.schemas.detection import BatchInferenceResult, InferenceResult
from src.services.image import ImageService
from src.services.inference import InferenceService
from src.utils.image_processing import decode_image, validate_image
from src.utils.pytorch_utils import format_detections


logger = logging.getLogger(__name__)

router = APIRouter(
    tags=['Tracks B/C/D - Triton'],
)

# Service instances
inference_service = InferenceService()
image_service = ImageService()


@router.post('/predict/{model_name}', response_model=InferenceResult)
def predict_triton(
    model_name: str = 'small',
    image: UploadFile = File(...),
    resize: bool = Query(False, description='Enable pre-resize for large images'),
    max_size: int = Query(1024, description='Maximum dimension before resizing', ge=640, le=4096),
):
    """
    Tracks B/C/D: Triton Inference Server gateway.

    Routes to appropriate track based on model name:
    - 'small' → Track B (Standard TRT + CPU NMS)
    - 'small_end2end' → Track C (End2End TRT + GPU NMS)
    - 'small_gpu_e2e_streaming' → Track D streaming (low latency)
    - 'small_gpu_e2e_batch' → Track D batch (max throughput)
    - 'small_gpu_e2e' → Track D balanced

    Note: Always uses shared gRPC client pool for optimal batching performance.

    Args:
        model_name: Model variant with track suffix
        image: Image file (JPEG/PNG)
        resize: Pre-resize large images (>2000px recommended)
        max_size: Maximum dimension for resize (640-4096)

    Returns:
        InferenceResult with detections and track metadata
    """
    filename = image.filename or 'uploaded_image'

    try:
        image_data = image.file.read()

        # Pre-resize if enabled
        if resize:
            original_size = len(image_data)
            image_data = image_service.resize_if_needed(image_data, max_size=max_size)
            if len(image_data) < original_size:
                logger.info(
                    f'Resized image from {original_size / 1024 / 1024:.1f}MB '
                    f'to {len(image_data) / 1024 / 1024:.1f}MB'
                )

        # Resolve model name to track
        track, triton_model, is_auto = inference_service.resolve_model_name(model_name)

        # Route to appropriate track
        if track == 'D':
            return inference_service.infer_track_d(
                image_bytes=image_data,
                model_name=triton_model,
                auto_affine=is_auto,
            )

        if track == 'C':
            return inference_service.infer_track_c(
                image_bytes=image_data,
                filename=filename,
                model_name=triton_model,
            )

        # Track B
        return inference_service.infer_track_b(
            image_bytes=image_data,
            filename=filename,
            model_url=triton_model,
            model_name=f'yolov11_{model_name}',
        )

    except ValueError as e:
        logger.warning(f'Client error for {filename}: {e}')
        raise HTTPException(status_code=400, detail=str(e)) from e

    except Exception as e:
        logger.error(f'Inference error for {filename}: {e}')
        raise HTTPException(status_code=500, detail=f'Inference failed: {e!s}') from e


@router.post('/predict_batch/{model_name}', response_model=BatchInferenceResult)
def predict_batch_triton(
    model_name: str = 'small',
    images: list[UploadFile] = File(...),
):
    """
    Batch inference for Triton tracks (B/C only).

    Note: Track D batch is handled through the regular endpoint with
    appropriate model suffix (e.g., 'small_gpu_e2e_batch').
    Always uses shared gRPC client pool for optimal batching performance.

    Args:
        model_name: Model variant ('small', 'small_end2end')
        images: List of image files

    Returns:
        BatchInferenceResult with per-image results
    """
    settings = get_settings()
    all_results = []
    failed_images = []

    try:
        decoded_images = []
        decoded_filenames = []

        for idx, image in enumerate(images):
            filename = image.filename or f'image_{idx}'
            try:
                image_data = image.file.read()
                img = decode_image(image_data, filename)
                validate_image(img, filename)
                decoded_images.append(img)
                decoded_filenames.append(filename)
            except ValueError as e:
                logger.warning(f'Skipping {filename}: {e}')
                failed_images.append({'filename': filename, 'index': idx, 'error': str(e)})

        if not decoded_images:
            return BatchInferenceResult(
                total_images=len(images),
                processed_images=0,
                failed_images=len(failed_images),
                results=[],
                failures=failed_images,
                status='all_failed',
            )

        is_end2end = model_name.endswith('_end2end')
        models = settings.models

        if is_end2end:
            # Track C batch
            base_model = model_name.replace('_end2end', '')
            triton_model_name = models.END2END_MODELS.get(base_model)

            if not triton_model_name:
                available = [f'{k}_end2end' for k in models.END2END_MODELS]
                raise HTTPException(
                    status_code=400, detail=f"Invalid model '{model_name}'. Available: {available}"
                )

            client = get_triton_client(settings.triton_url)
            detections_batch = client.infer_track_c_batch(decoded_images, triton_model_name)

            for idx, (detections, filename, img) in enumerate(
                zip(detections_batch, decoded_filenames, decoded_images, strict=False)
            ):
                results = client.format_detections(detections)
                all_results.append(
                    {
                        'filename': filename,
                        'image_index': idx,
                        'detections': results,
                        'num_detections': len(results),
                        'status': 'success',
                        'track': 'C',
                        'image': {'height': img.shape[0], 'width': img.shape[1]},
                    }
                )
        else:
            # Track B batch
            model_url = models.STANDARD_MODELS.get(model_name)

            if not model_url:
                available = list(models.STANDARD_MODELS.keys())
                raise HTTPException(
                    status_code=400, detail=f"Invalid model '{model_name}'. Available: {available}"
                )

            model = YOLO(model_url, task='detect')
            detections_batch = model(decoded_images, verbose=False)

            for idx, (detections, filename, img) in enumerate(
                zip(detections_batch, decoded_filenames, decoded_images, strict=False)
            ):
                results = format_detections([detections])

                all_results.append(
                    {
                        'filename': filename,
                        'image_index': idx,
                        'detections': results,
                        'num_detections': len(results),
                        'status': 'success',
                        'track': 'B',
                        'image': {'height': img.shape[0], 'width': img.shape[1]},
                    }
                )

        return BatchInferenceResult(
            total_images=len(images),
            processed_images=len(all_results),
            failed_images=len(failed_images),
            results=all_results,
            failures=failed_images if failed_images else None,
            status='success',
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Batch inference error: {e}')
        raise HTTPException(status_code=500, detail=f'Batch inference failed: {e!s}') from e
