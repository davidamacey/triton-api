"""
Track E (Visual Search) Pydantic models.

Models for visual search endpoints including ingestion, search, and index management.
Uses industry-standard response format consistent with other tracks.
"""

from typing import Any

from pydantic import BaseModel, Field

from src.schemas.detection import ImageMetadata, ModelMetadata


# =============================================================================
# Detection Responses (Standard Format)
# =============================================================================


class PredictResponse(BaseModel):
    """
    Response model for Track E predict endpoint.

    Matches standard detection format with additional embedding info.
    """

    # Core detection results
    detections: list[dict[str, Any]] = Field(
        default_factory=list,
        description='List of detected objects with normalized [0,1] coordinates',
    )
    num_detections: int = Field(default=0, description='Number of objects detected')
    status: str = Field(default='success')

    # Image metadata
    image: ImageMetadata | None = Field(None, description='Original image dimensions')

    # Model metadata
    model: ModelMetadata | None = Field(None, description='Model and backend information')

    # Track metadata
    track: str = Field(default='E')
    preprocessing: str = Field(default='gpu_dali')
    nms_location: str = Field(default='gpu')

    # Embedding info
    embedding_norm: float | None = Field(
        None, description='L2 norm of image embedding (should be ~1.0)'
    )

    # Timing (injected by middleware)
    total_time_ms: float | None = Field(
        None, description='Total end-to-end request time in milliseconds'
    )


class PredictFullResponse(BaseModel):
    """Response model for Track E full predict endpoint (with per-box embeddings)."""

    # Core detection results
    detections: list[dict[str, Any]] = Field(default_factory=list)
    num_detections: int = Field(default=0)
    status: str = Field(default='success')

    # Image metadata
    image: ImageMetadata | None = None
    model: ModelMetadata | None = None

    # Track metadata
    track: str = Field(default='E_full')
    preprocessing: str = Field(default='gpu_dali')
    nms_location: str = Field(default='gpu')

    # Box-level outputs
    normalized_boxes: list[list[float]] = Field(
        default_factory=list, description='Boxes in [0,1] range'
    )
    box_embeddings: list[list[float]] = Field(
        default_factory=list, description='Per-box 512-dim embeddings'
    )
    embedding_norm: float | None = None

    # Timing
    total_time_ms: float | None = None


class DetectOnlyResponse(BaseModel):
    """
    Response for detection-only endpoint.

    Matches standard detection format exactly.
    """

    # Core detection results
    detections: list[dict[str, Any]] = Field(
        default_factory=list,
        description='List of detected objects with normalized [0,1] coordinates',
    )
    num_detections: int = Field(default=0)
    status: str = Field(default='success')

    # Image metadata
    image: ImageMetadata | None = None
    model: ModelMetadata | None = None

    # Track metadata
    track: str = Field(default='E')
    preprocessing: str = Field(default='gpu_dali')
    nms_location: str = Field(default='gpu')

    # Timing
    total_time_ms: float | None = None


# =============================================================================
# Embedding Responses
# =============================================================================


class ImageEmbeddingResponse(BaseModel):
    """Response for image embedding endpoint."""

    embedding: list[float] = Field(..., description='512-dim L2-normalized embedding')
    embedding_norm: float = Field(..., description='L2 norm (should be ~1.0)')
    indexed: bool = Field(default=False, description='Whether image was stored')
    image_id: str | None = Field(None, description='ID if indexed')
    status: str = Field(default='success')
    track: str = Field(default='E')


class TextEmbeddingResponse(BaseModel):
    """Response for text embedding endpoint."""

    embedding: list[float] = Field(..., description='512-dim L2-normalized embedding')
    embedding_norm: float = Field(..., description='L2 norm (should be ~1.0)')
    text: str = Field(..., description='Input text (echoed)')
    status: str = Field(default='success')
    track: str = Field(default='E')


# =============================================================================
# Ingestion and Search
# =============================================================================


class ImageIngestRequest(BaseModel):
    """Request model for image ingestion."""

    image_id: str | None = Field(None, description='Unique identifier')
    metadata: dict[str, Any] | None = Field(None, description='Optional metadata')


class ImageIngestResponse(BaseModel):
    """Response model for image ingestion."""

    status: str = Field(..., description="'success' or 'error'")
    image_id: str = Field(..., description='Unique identifier for the ingested image')
    message: str = Field(..., description='Status message')
    num_detections: int = Field(..., description='Number of objects detected')
    global_embedding_norm: float = Field(..., description='L2 norm of global embedding')


class VisualSearchRequest(BaseModel):
    """Request model for visual search."""

    query_text: str | None = Field(None, description='Text query')
    top_k: int = Field(10, ge=1, le=100, description='Number of results')
    min_score: float | None = Field(None, ge=0.0, le=1.0, description='Min similarity')
    filter_metadata: dict[str, Any] | None = Field(None, description='Metadata filters')
    class_filter: list[int] | None = Field(None, description='Filter by COCO class IDs')


class SearchResult(BaseModel):
    """Individual search result."""

    image_id: str
    image_path: str
    score: float
    num_detections: int | None = None
    metadata: dict[str, Any] | None = None
    matched_objects: list[dict[str, Any]] | None = None


class VisualSearchResponse(BaseModel):
    """Response model for visual search."""

    status: str = Field(..., description="'success' or 'error'")
    query_type: str = Field(..., description="'image', 'object', or 'text'")
    results: list[SearchResult] = Field(..., description='Search results')
    total_results: int = Field(..., description='Total number of results')
    search_time_ms: float = Field(..., description='Search execution time')


class IndexStatsResponse(BaseModel):
    """Response model for index statistics."""

    status: str
    total_documents: int
    index_size_mb: float
