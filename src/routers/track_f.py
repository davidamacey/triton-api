"""
Track F: CPU Preprocessing + Direct TRT API Router.

Unlike Track E (DALI ensemble), Track F uses:
- CPU preprocessing (PIL + OpenCV)
- Direct TRT model calls (no ensemble scheduler)
- Custom letterbox/resize (no Ultralytics dependency in hot path)

Purpose: Compare CPU vs GPU preprocessing overhead and enable
higher TRT instance counts (DALI doesn't reserve VRAM).

Endpoints:
- /predict: YOLO detection + global embedding (SYNC)
"""

import logging

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import ORJSONResponse

from src.schemas.detection import ImageMetadata, ModelMetadata
from src.schemas.track_e import PredictResponse
from src.services.inference import InferenceService


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix='/track_f',
    tags=['Track F: CPU Preprocessing + Direct TRT'],
    default_response_class=ORJSONResponse,
)


@router.post('/predict', response_model=PredictResponse, tags=['Track F: Detection + Embedding'])
def predict_track_f(
    image: UploadFile = File(..., description='Image file (JPEG/PNG)'),
):
    """
    Track F: YOLO Detection + Global Image Embedding (CPU Preprocessing).

    Pipeline:
    1. CPU decode (PIL)
    2. CPU letterbox for YOLO (640x640, custom implementation)
    3. CPU resize/crop for CLIP (256x256, custom implementation)
    4. Direct YOLO TRT inference (yolov11_small_trt_end2end)
    5. Direct MobileCLIP TRT inference (mobileclip2_s2_image_encoder)

    Response format matches Track E for easy comparison.
    Timing is injected by middleware.
    """
    try:
        image_bytes = image.file.read()
        inference_service = InferenceService()

        # Use Track F inference (CPU preprocessing + direct TRT)
        response = inference_service.infer_track_f(image_bytes)

        return PredictResponse(
            detections=response['detections'],
            num_detections=response['num_detections'],
            image=ImageMetadata(
                width=response['image']['width'], height=response['image']['height']
            ),
            model=ModelMetadata(
                name=response['model']['name'],
                backend=response['model']['backend'],
                device=response['model']['device'],
            ),
            track='F',
            preprocessing='cpu',
            nms_location='gpu',
            embedding_norm=response.get('embedding_norm'),
            # total_time_ms injected by middleware
        )

    except Exception as e:
        logger.error(f'Track F prediction failed: {e}')
        raise HTTPException(500, f'Prediction failed: {e!s}') from e
