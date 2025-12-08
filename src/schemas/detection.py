"""
Detection-related Pydantic models.

Industry-standard response format for object detection inference.
All tracks (A, B, C, D, E) use consistent schema with timing and metadata.
"""

from pydantic import BaseModel, Field


class Detection(BaseModel):
    """
    Individual object detection result.

    Coordinates are normalized to [0, 1] relative to original image dimensions.
    This matches industry standard (COCO, YOLO) normalized output format.
    """

    x1: float = Field(..., ge=0.0, le=1.0, description='Left boundary (normalized 0-1)')
    y1: float = Field(..., ge=0.0, le=1.0, description='Top boundary (normalized 0-1)')
    x2: float = Field(..., ge=0.0, le=1.0, description='Right boundary (normalized 0-1)')
    y2: float = Field(..., ge=0.0, le=1.0, description='Bottom boundary (normalized 0-1)')
    confidence: float = Field(..., ge=0.0, le=1.0, description='Detection confidence score')
    class_id: int = Field(..., alias='class', description='COCO class ID (0-79)')

    class Config:
        populate_by_name = True  # Allow both 'class' and 'class_id'


class ImageMetadata(BaseModel):
    """Original image metadata."""

    width: int = Field(..., description='Original image width in pixels')
    height: int = Field(..., description='Original image height in pixels')


class ModelMetadata(BaseModel):
    """Model and inference metadata."""

    name: str = Field(..., description='Model name used for inference')
    backend: str = Field(..., description='Inference backend (pytorch, triton)')
    device: str = Field(default='gpu', description='Execution device (gpu, cpu)')


class InferenceResult(BaseModel):
    """
    Standard response schema for object detection inference.

    Industry-standard format with:
    - Detections with normalized coordinates
    - Image metadata (dimensions)
    - Model metadata (name, backend)
    - Performance timing (injected by middleware)
    - Track-specific metadata (preprocessing, NMS location)
    """

    # Core detection results
    detections: list = Field(
        default_factory=list,
        description='List of detected objects with normalized [0,1] coordinates',
    )
    num_detections: int = Field(default=0, description='Number of objects detected')
    status: str = Field(default='success', description="'success' or 'error'")

    # Image metadata
    image: ImageMetadata | None = Field(None, description='Original image dimensions')

    # Model metadata
    model: ModelMetadata | None = Field(None, description='Model and backend information')

    # Track metadata
    track: str | None = Field(None, description='Performance track used (A, B, C, D, E)')
    preprocessing: str | None = Field(
        None, description='Preprocessing method (cpu, gpu_dali, gpu_dali_auto)'
    )
    nms_location: str | None = Field(None, description='NMS execution location (cpu, gpu)')

    # Timing (injected by middleware for consistency)
    total_time_ms: float | None = Field(
        None, description='Total end-to-end request time in milliseconds'
    )


class BatchImageResult(BaseModel):
    """Single image result within a batch."""

    filename: str = Field(..., description='Original filename')
    image_index: int = Field(..., description='Index in batch')
    detections: list[Detection] = Field(
        default_factory=list, description='List of detected objects'
    )
    num_detections: int = Field(default=0, description='Number of detections')
    status: str = Field(default='success', description="'success' or 'error'")
    track: str | None = Field(None, description='Performance track used')
    image: ImageMetadata | None = Field(None, description='Image dimensions')
    error: str | None = Field(None, description='Error message if failed')


class BatchInferenceResult(BaseModel):
    """
    Response schema for batch inference endpoints.

    Includes per-image results and summary statistics.
    """

    total_images: int = Field(..., description='Total images in request')
    processed_images: int = Field(..., description='Successfully processed images')
    failed_images: int = Field(..., description='Failed images count')
    results: list[BatchImageResult] = Field(default_factory=list, description='Per-image results')
    failures: list[dict] | None = Field(None, description='Details of failed images')
    status: str = Field(default='success', description='Overall status')
    total_time_ms: float | None = Field(
        None, description='Total batch processing time in milliseconds'
    )
