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


# =============================================================================
# Face Detection & Recognition Responses
# =============================================================================


class FaceDetection(BaseModel):
    """Individual face detection with landmarks."""

    box: list[float] = Field(..., description='Face bounding box [x1, y1, x2, y2] normalized [0,1]')
    landmarks: list[float] = Field(
        ..., description='5-point facial landmarks [lx1,ly1,...,lx5,ly5] normalized [0,1]'
    )
    score: float = Field(..., description='Detection confidence score')
    quality: float | None = Field(None, description='Face quality score (frontality, sharpness)')


class FaceDetectResponse(BaseModel):
    """Response for face detection only (SCRFD)."""

    num_faces: int = Field(..., description='Number of faces detected')
    faces: list[FaceDetection] = Field(default_factory=list, description='Detected faces')
    status: str = Field(default='success')

    # Image metadata
    image: ImageMetadata | None = None

    # Track metadata
    track: str = Field(default='E_faces')
    preprocessing: str = Field(default='gpu_dali')

    # Timing
    total_time_ms: float | None = None


class FaceRecognizeResponse(BaseModel):
    """Response for face detection + recognition (SCRFD + ArcFace)."""

    num_faces: int = Field(..., description='Number of faces detected')
    faces: list[FaceDetection] = Field(default_factory=list, description='Detected faces')
    embeddings: list[list[float]] = Field(
        default_factory=list, description='512-dim ArcFace embeddings per face'
    )
    status: str = Field(default='success')

    # Image metadata
    image: ImageMetadata | None = None

    # Track metadata
    track: str = Field(default='E_faces')
    preprocessing: str = Field(default='gpu_dali')
    model: str = Field(default='scrfd_10g + arcface_w600k_r50')

    # Timing
    total_time_ms: float | None = None


class FaceFullResponse(BaseModel):
    """Response for unified YOLO + Face + CLIP pipeline."""

    # YOLO detections
    detections: list[dict[str, Any]] = Field(default_factory=list)
    num_detections: int = Field(default=0)

    # Face detections and embeddings
    num_faces: int = Field(default=0)
    faces: list[FaceDetection] = Field(default_factory=list)
    face_embeddings: list[list[float]] = Field(default_factory=list)

    # Global embedding (MobileCLIP 512-dim)
    image_embedding: list[float] | None = Field(
        None, description='MobileCLIP global image embedding'
    )
    embedding_norm: float | None = Field(None, description='L2 norm of global embedding')

    status: str = Field(default='success')

    # Image metadata
    image: ImageMetadata | None = None
    model: ModelMetadata | None = None

    # Track metadata
    track: str = Field(default='E_full_faces')
    preprocessing: str = Field(default='gpu_dali')
    nms_location: str = Field(default='gpu')

    # Timing
    total_time_ms: float | None = None


# =============================================================================
# Face Search Response Models
# =============================================================================


class FaceSearchResult(BaseModel):
    """Single face search result."""

    face_id: str
    image_id: str
    image_path: str | None = None
    score: float
    person_id: str | None = None
    person_name: str | None = None
    box: list[float]
    confidence: float


class FaceSearchResponse(BaseModel):
    """Response for face similarity search."""

    status: str = 'success'
    query_face: FaceDetection
    results: list[FaceSearchResult]
    total_results: int
    search_time_ms: float


class FaceIdentifyResponse(BaseModel):
    """Response for face identification."""

    status: str = 'success'
    query_face: FaceDetection
    identified: bool
    person_id: str | None = None
    person_name: str | None = None
    match_score: float | None = None
    associated_faces: list[FaceSearchResult] = Field(default_factory=list)


class PersonFacesResponse(BaseModel):
    """Response for person faces lookup."""

    person_id: str
    person_name: str | None = None
    face_count: int
    faces: list[FaceSearchResult]


# =============================================================================
# Face Identity Management Schemas
# =============================================================================


class FaceIngestRequest(BaseModel):
    """Request for face ingestion."""

    person_id: str | None = Field(None, description='Person ID to assign face to')
    face_id: str | None = Field(None, description='Face ID (auto-generated if not provided)')
    source_image_id: str | None = Field(None, description='ID of source image')
    metadata: dict | None = Field(None, description='Additional metadata')


class FaceIngestResponse(BaseModel):
    """Response for face ingestion."""

    status: str = 'success'
    num_faces: int = Field(description='Number of faces detected and ingested')
    faces: list[dict] = Field(description='List of ingested face info (face_id, person_id, etc)')
    source_image_id: str | None = None


class FaceIdentityIdentifyRequest(BaseModel):
    """Request for 1:N face identification."""

    top_k: int = Field(5, ge=1, le=100, description='Number of matches to return per face')
    threshold: float = Field(0.6, ge=0.0, le=1.0, description='Similarity threshold for match')
    face_detector: str = Field('scrfd', description='Face detector to use')


class FaceIdentityIdentifyResponse(BaseModel):
    """Response for face identification."""

    status: str = 'success'
    num_faces: int = Field(description='Number of faces detected in query image')
    matches: list[dict] = Field(description='Matches for each detected face')
    query_time_ms: float = Field(description='Query processing time')


class PersonFacesListResponse(BaseModel):
    """Response for getting all faces of a person."""

    status: str = 'success'
    person_id: str
    num_faces: int
    faces: list[dict] = Field(description='List of face records')


class FaceAssignRequest(BaseModel):
    """Request to assign face to person."""

    person_id: str = Field(description='Person ID to assign face to')


class FaceAssignResponse(BaseModel):
    """Response for face assignment."""

    status: str = 'success'
    face_id: str
    previous_person_id: str | None
    new_person_id: str
