"""
Common Pydantic models used across multiple endpoints.

Health checks, service info, and monitoring responses.
"""

from typing import Any

from pydantic import BaseModel, Field


class TrackInfo(BaseModel):
    """Information about a single track."""

    endpoint: str | None = None
    endpoints: list[str] | None = None
    description: str
    models: Any  # Can be list or dict


class PerformanceMetrics(BaseModel):
    """Performance metrics for health check."""

    memory_mb: float
    cpu_percent: float
    max_file_size_mb: int
    slow_request_threshold_ms: int
    optimizations: dict[str, bool]
    gpu_memory_allocated_mb: float | None = None
    gpu_memory_reserved_mb: float | None = None


class TrackHealthInfo(BaseModel):
    """Health info for a specific track."""

    models: dict[str, str] | None = None
    gpu_available: bool | None = None
    backend: str | None = None
    protocol: str | None = None
    url: str | None = None
    client_creation: str | None = None


class HealthResponse(BaseModel):
    """Enhanced health check response."""

    status: str = Field(default='healthy', description='Service health status')
    tracks: dict[str, TrackHealthInfo] = Field(..., description='Health info per track')
    performance: PerformanceMetrics = Field(..., description='Performance metrics')


class ServiceInfoResponse(BaseModel):
    """Root endpoint response with service information."""

    service: str = Field(default='Unified YOLO Inference API')
    status: str = Field(default='running')
    tracks: dict[str, TrackInfo]
    triton_backend: str
    gpu_available: bool


class ConnectionPoolStats(BaseModel):
    """Statistics for a connection pool."""

    active: bool
    connection_count: int
    triton_urls: list[str]


class ConnectionPoolUsageInfo(BaseModel):
    """Usage information for connection pools."""

    shared_client_enabled: str
    per_request_client: str
    performance_impact: dict[str, str]


class ConnectionPoolTesting(BaseModel):
    """Testing examples for connection pools."""

    example_shared: str
    example_per_request: str
    check_batching: str


class ConnectionPoolInfo(BaseModel):
    """Response for connection pool info endpoint."""

    shared_client_pool: ConnectionPoolStats
    usage_info: ConnectionPoolUsageInfo
    testing: ConnectionPoolTesting


class CacheStats(BaseModel):
    """Statistics for a single cache."""

    hits: int
    misses: int
    size: int
    max_size: int
    hit_rate: float


class CacheStatsResponse(BaseModel):
    """Response for cache statistics endpoint."""

    status: str = 'success'
    caches: dict[str, CacheStats]
