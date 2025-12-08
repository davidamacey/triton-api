"""
Centralized configuration using Pydantic Settings.

All configuration is loaded from environment variables with sensible defaults.
Use get_settings() to access the singleton settings instance.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class TritonModelConfig:
    """Model configurations for all tracks."""

    # Track A: PyTorch models (loaded at startup)
    PYTORCH_MODELS: dict[str, str] = {
        'small': '/app/pytorch_models/yolo11s.pt',
    }

    # Track B: Standard TRT (CPU NMS)
    STANDARD_MODELS: dict[str, str] = {
        'small': 'grpc://triton-api:8001/yolov11_small_trt',
    }

    # Track C: End2End TRT (GPU NMS)
    END2END_MODELS: dict[str, str] = {
        'small': 'yolov11_small_trt_end2end',
    }

    # Track D: DALI+TRT (Full GPU) - three variants
    GPU_E2E_MODELS: dict[str, str] = {
        'small': 'yolov11_small_gpu_e2e',
    }

    GPU_E2E_BATCH_MODELS: dict[str, str] = {
        'small': 'yolov11_small_gpu_e2e_batch',
    }

    GPU_E2E_STREAMING_MODELS: dict[str, str] = {
        'small': 'yolov11_small_gpu_e2e_streaming',
    }

    # Track E: Visual Search ensembles
    ENSEMBLE_MODELS: dict[str, str] = {
        'yolo_clip': 'yolo_clip_ensemble',
        'yolo_mobileclip': 'yolo_mobileclip_ensemble',
    }

    # MobileCLIP encoder models
    CLIP_MODELS: dict[str, str] = {
        'image_encoder': 'mobileclip2_s2_image_encoder',
        'text_encoder': 'mobileclip2_s2_text_encoder',
    }


class Settings(BaseSettings):
    """
    Application settings with environment variable support.

    All settings can be overridden via environment variables.
    Example: ENABLE_PYTORCH=true TRITON_URL=localhost:8001 python -m src.main
    """

    # ==========================================================================
    # Feature Flags
    # ==========================================================================
    enable_pytorch: bool = Field(default=False, description='Enable Track A PyTorch endpoints')

    # ==========================================================================
    # Triton Configuration
    # ==========================================================================
    triton_url: str = Field(
        default='triton-api:8001', description='Triton Inference Server gRPC endpoint'
    )

    triton_grpc_timeout: float = Field(default=30.0, description='gRPC timeout in seconds')

    # ==========================================================================
    # OpenSearch Configuration (Track E)
    # ==========================================================================
    opensearch_url: str = Field(
        default='http://opensearch:9200', description='OpenSearch endpoint URL'
    )

    opensearch_index: str = Field(
        default='visual_search', description='OpenSearch index name for visual search'
    )

    opensearch_timeout: int = Field(default=30, description='OpenSearch timeout in seconds')

    # ==========================================================================
    # Performance Configuration
    # ==========================================================================
    max_file_size_mb: int = Field(default=50, description='Maximum upload file size in MB')

    slow_request_threshold_ms: int = Field(
        default=100, description='Log requests slower than this threshold'
    )

    # Image preprocessing defaults
    default_max_resize: int = Field(
        default=1024, ge=640, le=4096, description='Default maximum dimension for image resizing'
    )

    min_model_size: int = Field(
        default=640, description='Minimum model input size (no upscaling below this)'
    )

    # ==========================================================================
    # Cache Configuration
    # ==========================================================================
    embedding_cache_size: int = Field(
        default=1000, description='Maximum number of embeddings to cache'
    )

    embedding_cache_ttl: int = Field(default=3600, description='Embedding cache TTL in seconds')

    affine_cache_size: int = Field(
        default=1000, description='Maximum number of affine matrices to cache'
    )

    # ==========================================================================
    # API Configuration
    # ==========================================================================
    api_title: str = Field(
        default='Unified YOLO Inference API (All Tracks)', description='API title for OpenAPI docs'
    )

    api_description: str = Field(
        default='All-in-one YOLO inference service - Tracks A/B/C/D/E',
        description='API description for OpenAPI docs',
    )

    api_version: str = Field(default='5.0.0', description='API version')

    # ==========================================================================
    # Computed Properties
    # ==========================================================================
    @property
    def max_file_size_bytes(self) -> int:
        """Maximum file size in bytes."""
        return self.max_file_size_mb * 1024 * 1024

    @property
    def triton_host(self) -> str:
        """Triton host without port."""
        return self.triton_url.split(':')[0]

    @property
    def triton_port(self) -> int:
        """Triton gRPC port."""
        parts = self.triton_url.split(':')
        return int(parts[1]) if len(parts) > 1 else 8001

    # ==========================================================================
    # Model Configurations (class-level, not from env)
    # ==========================================================================
    models: TritonModelConfig = TritonModelConfig()

    class Config:
        env_prefix = ''  # No prefix for env vars
        case_sensitive = False
        extra = 'ignore'


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance (singleton pattern).

    Returns:
        Settings: Application settings
    """
    return Settings()
