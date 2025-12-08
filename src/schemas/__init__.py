"""
Pydantic schemas for API request/response models.

Consolidated models used across all API endpoints for consistent typing.
"""

from src.schemas.common import (
    CacheStatsResponse,
    ConnectionPoolInfo,
    HealthResponse,
    ServiceInfoResponse,
)
from src.schemas.detection import BatchImageResult, BatchInferenceResult, Detection, InferenceResult
from src.schemas.track_e import (
    ImageIngestRequest,
    ImageIngestResponse,
    IndexStatsResponse,
    PredictFullResponse,
    PredictResponse,
    SearchResult,
    VisualSearchRequest,
    VisualSearchResponse,
)


__all__ = [
    # Detection schemas
    'BatchImageResult',
    'BatchInferenceResult',
    # Common schemas
    'CacheStatsResponse',
    'ConnectionPoolInfo',
    'Detection',
    'HealthResponse',
    # Track E schemas
    'ImageIngestRequest',
    'ImageIngestResponse',
    'IndexStatsResponse',
    'InferenceResult',
    'PredictFullResponse',
    'PredictResponse',
    'SearchResult',
    'ServiceInfoResponse',
    'VisualSearchRequest',
    'VisualSearchResponse',
]
