"""
Model Management Router.

Provides endpoints for uploading, exporting, and managing YOLO models.
Delegates all business logic to service layer.
"""

import logging
import shutil
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile

from src.schemas.models import (
    ExportFormat,
    ExportTaskResponse,
    ExportTaskStatus,
    ModelDeleteResponse,
    ModelInfo,
    ModelListResponse,
    ModelLoadResponse,
)
from src.services.model_export import (
    create_export_task,
    generate_triton_name,
    get_export_task,
    list_export_tasks,
    run_export,
    save_uploaded_file,
)
from src.services.triton_control import TritonControlService


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix='/models',
    tags=['Model Management'],
)

TRITON_MODELS_DIR = Path('/app/models')


@router.post('/upload', response_model=ExportTaskResponse)
async def upload_model(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description='YOLO11 .pt model file'),
    triton_name: str | None = Form(default=None, description='Custom Triton model name'),
    max_batch: int = Form(default=32, ge=1, le=128, description='Max batch size'),
    formats: list[ExportFormat] = Form(default=[ExportFormat.TRT_END2END]),
    normalize_boxes: bool = Form(default=True),
    auto_load: bool = Form(default=True),
):
    """
    Upload a YOLO11 model and start export to TensorRT.

    The export runs in the background. Use /models/export/{task_id} to check status.
    """
    if not file.filename or not file.filename.endswith('.pt'):
        raise HTTPException(status_code=400, detail='File must be a .pt model file')

    content = await file.read()
    name = triton_name or generate_triton_name(file.filename)

    try:
        pt_file, model_info = await save_uploaded_file(content, name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    format_strs = [f.value for f in formats]

    # TRT End2End requires ONNX End2End as prerequisite
    if 'trt_end2end' in format_strs and 'onnx_end2end' not in format_strs:
        format_strs.insert(0, 'onnx_end2end')

    task_id = create_export_task(file.filename, name, model_info, format_strs)

    background_tasks.add_task(
        run_export, task_id, pt_file, name, max_batch, format_strs, normalize_boxes, auto_load
    )

    task = get_export_task(task_id)
    return ExportTaskResponse(
        task_id=task_id,
        model_name=task['model_name'],
        triton_name=task['triton_name'],
        status=task['status'],
        message='Export started',
        created_at=task['created_at'],
    )


@router.get('/export/{task_id}', response_model=ExportTaskStatus)
async def get_export_status(task_id: str):
    """Get detailed status of an export task."""
    task = get_export_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f'Export task {task_id} not found')

    return ExportTaskStatus(**task)


@router.get('/exports', response_model=list[ExportTaskStatus])
async def list_exports():
    """List all export tasks."""
    return [ExportTaskStatus(**t) for t in list_export_tasks()]


@router.get('/', response_model=ModelListResponse)
async def list_models():
    """List all models in Triton repository."""
    triton = TritonControlService()
    ready = await triton.server_ready()
    models = await triton.get_repository_index()

    model_list = []
    for m in models:
        name = m.get('name', '')
        state = m.get('state', 'UNKNOWN')

        labels_path = TRITON_MODELS_DIR / name / 'labels.txt'
        has_labels = labels_path.exists()
        num_classes = None
        if has_labels:
            num_classes = len(labels_path.read_text().strip().split('\n'))

        model_list.append(
            ModelInfo(
                name=name,
                status=state,
                versions=m.get('versions', []),
                has_labels=has_labels,
                num_classes=num_classes,
            )
        )

    return ModelListResponse(
        models=model_list,
        total=len(model_list),
        triton_status='READY' if ready else 'UNAVAILABLE',
    )


@router.post('/{model_name}/load', response_model=ModelLoadResponse)
async def load_model(model_name: str):
    """Load a model into Triton server."""
    triton = TritonControlService()
    success, message = await triton.load_model(model_name)

    if not success:
        raise HTTPException(status_code=500, detail=message)

    return ModelLoadResponse(model_name=model_name, action='load', success=True, message=message)


@router.post('/{model_name}/unload', response_model=ModelLoadResponse)
async def unload_model(model_name: str):
    """Unload a model from Triton server."""
    triton = TritonControlService()
    success, message = await triton.unload_model(model_name)

    if not success:
        raise HTTPException(status_code=500, detail=message)

    return ModelLoadResponse(model_name=model_name, action='unload', success=True, message=message)


@router.delete('/{model_name}', response_model=ModelDeleteResponse)
async def delete_model(model_name: str):
    """Delete a model from the repository and unload from Triton."""
    triton = TritonControlService()
    model_dir = TRITON_MODELS_DIR / model_name

    if not model_dir.exists():
        raise HTTPException(status_code=404, detail=f'Model {model_name} not found')

    # Unload from Triton first
    unloaded = False
    success, _ = await triton.unload_model(model_name)
    if success:
        unloaded = True

    # Delete model directory
    deleted_files = [
        str(f.relative_to(TRITON_MODELS_DIR)) for f in model_dir.rglob('*') if f.is_file()
    ]

    shutil.rmtree(model_dir)
    logger.info(f'Deleted model directory: {model_dir}')

    return ModelDeleteResponse(
        model_name=model_name,
        deleted_files=deleted_files,
        unloaded_from_triton=unloaded,
        message=f'Model {model_name} deleted successfully',
    )
