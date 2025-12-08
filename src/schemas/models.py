"""
Pydantic schemas for Model Management API.

Defines request/response models for:
- Model upload and export
- Export task tracking
- Model listing and management
"""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class ExportFormat(str, Enum):
    """Available export formats."""

    TRT = 'trt'
    TRT_END2END = 'trt_end2end'
    ONNX = 'onnx'
    ONNX_END2END = 'onnx_end2end'
    ALL = 'all'


class ExportStatus(str, Enum):
    """Export task status."""

    PENDING = 'pending'
    VALIDATING = 'validating'
    EXPORTING = 'exporting'
    LOADING = 'loading'
    COMPLETED = 'completed'
    FAILED = 'failed'


class ModelUploadRequest(BaseModel):
    """Request parameters for model upload (form fields)."""

    triton_name: str | None = Field(
        default=None,
        description='Custom Triton model name. If not provided, auto-generated from filename.',
        examples=['vehicle_detector', 'custom_yolo_v1'],
    )
    max_batch: int = Field(
        default=32,
        ge=1,
        le=128,
        description='Maximum batch size for dynamic batching',
    )
    formats: list[ExportFormat] = Field(
        default=[ExportFormat.TRT_END2END],
        description='Export formats to generate',
    )
    normalize_boxes: bool = Field(
        default=True,
        description='Output boxes in [0,1] normalized range (recommended)',
    )
    auto_load: bool = Field(
        default=True,
        description='Automatically load model into Triton after export',
    )


class ExportTaskResponse(BaseModel):
    """Response for export task creation."""

    task_id: str = Field(description='Unique task identifier for status polling')
    model_name: str = Field(description='Model name being exported')
    triton_name: str = Field(description='Triton model name')
    status: ExportStatus = Field(description='Current task status')
    message: str = Field(description='Status message')
    created_at: datetime = Field(description='Task creation timestamp')


class ExportTaskStatus(BaseModel):
    """Detailed export task status."""

    task_id: str
    model_name: str
    triton_name: str
    status: ExportStatus
    progress: float = Field(ge=0.0, le=100.0, description='Export progress percentage')
    current_step: str = Field(description='Current export step')
    message: str
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None
    formats_completed: list[str] = Field(default_factory=list)
    formats_pending: list[str] = Field(default_factory=list)
    num_classes: int | None = None
    class_names: list[str] | None = None
    triton_loaded: bool = False
    # Timing information
    export_duration_seconds: float | None = Field(
        default=None, description='Total export duration in seconds'
    )
    step_times: dict[str, float] | None = Field(
        default=None, description='Timing for each export step in seconds'
    )


class ModelInfo(BaseModel):
    """Information about a deployed model."""

    name: str = Field(description='Model name in Triton')
    status: str = Field(description='Model status (READY, LOADING, UNAVAILABLE)')
    versions: list[str] = Field(default_factory=list, description='Available model versions')
    backend: str | None = Field(default=None, description='Model backend (tensorrt, onnxruntime)')
    max_batch_size: int | None = None
    has_labels: bool = Field(default=False, description='Whether labels.txt exists')
    num_classes: int | None = None
    input_shape: list[int] | None = None


class ModelListResponse(BaseModel):
    """Response for listing models."""

    models: list[ModelInfo]
    total: int
    triton_status: str = Field(description='Triton server status')


class ModelLoadResponse(BaseModel):
    """Response for model load/unload operations."""

    model_name: str
    action: str = Field(description='load or unload')
    success: bool
    message: str


class ModelDeleteResponse(BaseModel):
    """Response for model deletion."""

    model_name: str
    deleted_files: list[str]
    unloaded_from_triton: bool
    message: str
