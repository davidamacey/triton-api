"""
Model Export Service.

Handles YOLO model validation, export task management, and TensorRT export execution.
"""

import asyncio
import logging
import shutil
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.schemas.models import ExportStatus
from src.services.triton_control import TritonControlService


logger = logging.getLogger(__name__)

# Paths
PYTORCH_MODELS_DIR = Path('/app/pytorch_models')
TRITON_MODELS_DIR = Path('/app/models')
EXPORT_SCRIPT = Path('/app/export/export_models.py')

# In-memory task storage (use Redis in production)
export_tasks: dict[str, dict[str, Any]] = {}


def validate_pytorch_model(file_path: Path) -> tuple[bool, str, dict[str, Any]]:
    """
    Validate uploaded file is a valid YOLO11 detection model.

    Returns:
        Tuple of (is_valid, error_message, model_info)
    """
    try:
        import torch
        from ultralytics import YOLO

        # Check file size
        file_size = file_path.stat().st_size
        if file_size < 1000:
            return False, 'File too small to be a valid model', {}
        if file_size > 500 * 1024 * 1024:
            return False, 'File too large (max 500MB)', {}

        # Load and validate model
        model = YOLO(str(file_path))

        model_info = {
            'task': getattr(model, 'task', 'detect'),
            'num_classes': len(model.names) if hasattr(model, 'names') else 0,
            'class_names': list(model.names.values()) if isinstance(model.names, dict) else [],
        }

        if model_info['task'] != 'detect':
            return False, f'Only detection models supported, got: {model_info["task"]}', model_info

        del model
        torch.cuda.empty_cache()

        return True, 'Valid YOLO detection model', model_info

    except Exception as e:
        return False, f'Validation error: {e}', {}


def generate_triton_name(filename: str, custom_name: str | None = None) -> str:
    """Generate Triton-compatible model name from filename."""
    if custom_name:
        return custom_name.lower().replace('-', '_').replace(' ', '_')

    stem = Path(filename).stem.lower().replace('-', '_').replace(' ', '_')

    # Remove common suffixes
    for suffix in ['_best', '_last', '_final']:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]

    return stem


def create_export_task(
    filename: str,
    triton_name: str,
    model_info: dict[str, Any],
    formats: list[str],
) -> str:
    """Create a new export task and return task_id."""
    task_id = str(uuid.uuid4())[:8]
    now = datetime.now(UTC)

    export_tasks[task_id] = {
        'task_id': task_id,
        'model_name': Path(filename).stem,
        'triton_name': triton_name,
        'status': ExportStatus.PENDING,
        'progress': 0.0,
        'current_step': 'Queued for export',
        'message': 'Export starting...',
        'created_at': now,
        'started_at': None,
        'completed_at': None,
        'error': None,
        'formats_completed': [],
        'formats_pending': formats,
        'num_classes': model_info.get('num_classes'),
        'class_names': model_info.get('class_names'),
        'triton_loaded': False,
    }

    return task_id


def get_export_task(task_id: str) -> dict[str, Any] | None:
    """Get export task by ID."""
    return export_tasks.get(task_id)


def list_export_tasks() -> list[dict[str, Any]]:
    """List all export tasks."""
    return list(export_tasks.values())


class StepTimer:
    """Context manager for tracking step timing in export pipeline."""

    def __init__(self, step_times: dict[str, float], step_name: str):
        self.step_times = step_times
        self.step_name = step_name
        self.start: float = 0

    def __enter__(self):
        from time import perf_counter

        self.start = perf_counter()
        return self

    def __exit__(self, *args):
        from time import perf_counter

        self.step_times[self.step_name] = round(perf_counter() - self.start, 2)


async def run_export(
    task_id: str,
    pt_file: Path,
    triton_name: str,
    max_batch: int,
    formats: list[str],
    normalize_boxes: bool,
    auto_load: bool,
) -> None:
    """
    Run the export process in background.

    Updates task status throughout the process.
    Tracks timing for each step to provide user feedback.
    """
    from time import perf_counter

    task = export_tasks[task_id]
    step_times: dict[str, float] = {}
    total_start = perf_counter()

    try:
        task['status'] = ExportStatus.EXPORTING
        task['started_at'] = datetime.now(UTC)
        task['current_step'] = 'Starting export'
        task['progress'] = 10.0
        task['step_times'] = step_times

        # Build export command
        cmd = [
            'python',
            str(EXPORT_SCRIPT),
            '--custom-model',
            f'{pt_file}:{triton_name}:{max_batch}',
            '--formats',
            *formats,
            '--save-labels',
            '--generate-config',
        ]
        if normalize_boxes:
            cmd.append('--normalize-boxes')

        logger.info(f'[{task_id}] Running: {" ".join(cmd)}')
        task['current_step'] = f'Exporting: {", ".join(formats)}'
        task['progress'] = 20.0

        # Run export subprocess with timing
        with StepTimer(step_times, 'total_export'):
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd='/app',
            )

            output_lines = []
            current_phase = 'initialization'
            phase_start = perf_counter()

            while True:
                line = await process.stdout.readline()
                if not line:
                    break

                line_str = line.decode('utf-8', errors='replace').strip()
                output_lines.append(line_str)

                # Update progress and track step timing based on output
                if 'ONNX' in line_str and 'Export' in line_str:
                    if current_phase != 'onnx_export':
                        step_times[current_phase] = round(perf_counter() - phase_start, 2)
                        current_phase = 'onnx_export'
                        phase_start = perf_counter()
                    task['progress'] = 40.0
                    task['current_step'] = 'Exporting ONNX End2End'
                elif 'TensorRT' in line_str or 'Building' in line_str:
                    if current_phase != 'tensorrt_build':
                        step_times[current_phase] = round(perf_counter() - phase_start, 2)
                        current_phase = 'tensorrt_build'
                        phase_start = perf_counter()
                    task['progress'] = 60.0
                    task['current_step'] = 'Building TensorRT engine (this takes 2-5 min)'
                elif 'Engine saved' in line_str:
                    step_times[current_phase] = round(perf_counter() - phase_start, 2)
                    current_phase = 'finalization'
                    phase_start = perf_counter()
                    task['progress'] = 85.0
                    task['current_step'] = 'Finalizing export'

                # Update step_times in real-time
                task['step_times'] = step_times

            await process.wait()

            # Record final phase time
            step_times[current_phase] = round(perf_counter() - phase_start, 2)

        if process.returncode != 0:
            error_output = '\n'.join(output_lines[-20:])
            raise RuntimeError(f'Export failed:\n{error_output}')

        task['progress'] = 90.0
        task['formats_completed'] = formats
        task['formats_pending'] = []

        # Auto-load into Triton with timing
        if auto_load:
            task['status'] = ExportStatus.LOADING
            task['current_step'] = 'Loading into Triton'

            with StepTimer(step_times, 'triton_load'):
                triton = TritonControlService()
                models_to_load = []

                if 'trt' in formats or 'all' in formats:
                    models_to_load.append(f'{triton_name}_trt')
                if 'trt_end2end' in formats or 'all' in formats:
                    models_to_load.append(f'{triton_name}_trt_end2end')

                for model in models_to_load:
                    success, msg = await triton.load_model(model)
                    if success:
                        task['triton_loaded'] = True
                        logger.info(f'[{task_id}] Loaded {model}')
                    else:
                        logger.warning(f'[{task_id}] Failed to load {model}: {msg}')

        task['status'] = ExportStatus.COMPLETED
        task['completed_at'] = datetime.now(UTC)
        task['progress'] = 100.0
        task['current_step'] = 'Complete'

        # Calculate total duration
        total_duration = round(perf_counter() - total_start, 2)
        task['export_duration_seconds'] = total_duration
        task['step_times'] = step_times
        task['message'] = f'Model {triton_name} exported successfully in {total_duration:.1f}s'

        logger.info(f'[{task_id}] Export completed in {total_duration:.1f}s')
        logger.info(f'[{task_id}] Step times: {step_times}')

    except Exception as e:
        logger.exception(f'[{task_id}] Export failed')
        task['status'] = ExportStatus.FAILED
        task['error'] = str(e)
        task['message'] = f'Export failed: {e}'
        task['completed_at'] = datetime.now(UTC)
        task['export_duration_seconds'] = round(perf_counter() - total_start, 2)


async def save_uploaded_file(
    content: bytes,
    triton_name: str,
) -> tuple[Path, dict[str, Any]]:
    """
    Save uploaded file and validate it.

    Returns:
        Tuple of (saved_path, model_info)

    Raises:
        ValueError: If validation fails
    """
    PYTORCH_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Save to temp file first for validation
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        is_valid, error_msg, model_info = validate_pytorch_model(tmp_path)
        if not is_valid:
            tmp_path.unlink(missing_ok=True)
            raise ValueError(error_msg)

        # Move to final location
        final_path = PYTORCH_MODELS_DIR / f'{triton_name}.pt'
        shutil.move(tmp_path, final_path)

        return final_path, model_info

    except ValueError:
        raise
    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        raise ValueError(f'Failed to save file: {e}') from e
