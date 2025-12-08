#!/usr/bin/env python3
"""
Unified YOLO Model Export Script
=================================

Export YOLO models in multiple formats for Triton deployment.

Formats:
--------
1. onnx         - Standard ONNX (no NMS, needs CPU post-processing)
2. trt          - Native TensorRT engine (no NMS, needs CPU post-processing)
3. onnx_end2end - ONNX with TensorRT EfficientNMS operators (GPU NMS)
4. trt_end2end  - TRT engine with compiled NMS (fastest, built from onnx_end2end)
5. all          - Export all formats (default)

Usage:
------
# Export all formats (default)
docker compose exec yolo-api python /app/export/export_models.py

# Export only end2end models
docker compose exec yolo-api python /app/export/export_models.py --formats onnx_end2end trt_end2end

# Export specific models
docker compose exec yolo-api python /app/export/export_models.py --models nano small

# Export everything
docker compose exec yolo-api python /app/export/export_models.py --formats all
"""

import sys


sys.path.insert(0, '/app/src')

import argparse
import gc
import json
import logging
import re
import shutil
from pathlib import Path
from typing import Any

import yaml

# Apply end2end patch for onnx_trt format
from ultralytics_patches import apply_end2end_patch


apply_end2end_patch()

# Import after patch (REQUIRED - patch must be applied before importing ultralytics)
import onnx  # noqa: E402
import tensorrt as trt  # noqa: E402
import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from ultralytics.cfg import get_cfg  # noqa: E402
from ultralytics.engine.exporter import Exporter  # noqa: E402


# ============================================================================
# Logging Configuration
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

# Default configurations for standard YOLO11 model sizes
# These can be overridden via --config-file or extended via --custom-model
DEFAULT_MODELS: dict[str, dict[str, Any]] = {
    'nano': {
        'pt_file': '/app/pytorch_models/yolo11n.pt',
        'triton_name': 'yolov11_nano',
        'max_batch': 128,  # Nano is small, A6000 can handle this
        'topk': 300,
    },
    'small': {
        'pt_file': '/app/pytorch_models/yolo11s.pt',
        'triton_name': 'yolov11_small',
        'max_batch': 64,
        'topk': 300,
    },
    'medium': {
        'pt_file': '/app/pytorch_models/yolo11m.pt',
        'triton_name': 'yolov11_medium',
        'max_batch': 32,
        'topk': 300,
    },
    'large': {
        'pt_file': '/app/pytorch_models/yolo11l.pt',
        'triton_name': 'yolov11_large',
        'max_batch': 16,  # Large model needs more memory
        'topk': 300,
    },
    'xlarge': {
        'pt_file': '/app/pytorch_models/yolo11x.pt',
        'triton_name': 'yolov11_xlarge',
        'max_batch': 8,  # XLarge is very memory intensive
        'topk': 300,
    },
}

# Export settings
IMG_SIZE = 640
DEVICE = 0  # GPU 0
HALF = True  # FP16 precision
WORKSPACE_GB = 4  # TensorRT workspace size in GB

# NMS settings (for end2end exports)
IOU_THRESHOLD = 0.7
CONF_THRESHOLD = 0.25

# ============================================================================
# Helper Functions
# ============================================================================


def validate_pt_file(pt_file: str) -> bool:
    """Validate that PyTorch model file exists and is readable."""
    path = Path(pt_file)
    if not path.exists():
        logger.error(f'Model file not found: {pt_file}')
        return False
    if not path.is_file():
        logger.error(f'Model path is not a file: {pt_file}')
        return False
    if path.stat().st_size == 0:
        logger.error(f'Model file is empty: {pt_file}')
        return False
    return True


def check_gpu_memory(required_gb: float = 4.0) -> bool:
    """Check if sufficient GPU memory is available."""
    try:
        if torch.cuda.is_available():
            free_memory = torch.cuda.get_device_properties(DEVICE).total_memory
            free_memory_gb = free_memory / (1024**3)
            if free_memory_gb < required_gb:
                logger.warning(
                    f'Low GPU memory: {free_memory_gb:.1f}GB available, {required_gb}GB recommended'
                )
            return True
        logger.error('CUDA not available')
        return False
    except Exception as e:
        logger.warning(f'Could not check GPU memory: {e}')
        return True  # Continue anyway


def backup_existing_file(file_path: Path, suffix: str = '.old') -> None:
    """Backup existing file if it exists."""
    if file_path.exists():
        backup_path = file_path.with_suffix(file_path.suffix + suffix)
        if backup_path.exists():
            backup_path.unlink()
        shutil.move(file_path, backup_path)
        logger.debug(f'Backed up existing file to: {backup_path}')


def setup_trt_builder(
    workspace_gb: int = WORKSPACE_GB,
) -> tuple[trt.Builder, trt.IBuilderConfig, trt.INetworkDefinition, trt.Logger]:
    """
    Set up TensorRT builder with common configuration.

    Args:
        workspace_gb: Workspace size in GB

    Returns:
        Tuple of (builder, config, network, logger)
    """
    trt_logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(trt_logger, '')

    builder = trt.Builder(trt_logger)
    config = builder.create_builder_config()

    # Set workspace size
    workspace_bytes = int(workspace_gb * (1 << 30))
    is_trt10 = int(trt.__version__.split('.')[0]) >= 10
    if is_trt10:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    else:
        config.max_workspace_size = workspace_bytes

    # Create network with explicit batch flag
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)

    return builder, config, network, trt_logger


def add_optimization_profile(
    builder: trt.Builder,
    config: trt.IBuilderConfig,
    network: trt.INetworkDefinition,
    max_batch: int,
) -> None:
    """Add optimization profile for dynamic batching."""
    profile = builder.create_optimization_profile()

    min_shape = (1, 3, IMG_SIZE, IMG_SIZE)
    opt_shape = (max_batch // 2, 3, IMG_SIZE, IMG_SIZE)
    max_shape = (max_batch, 3, IMG_SIZE, IMG_SIZE)

    for i in range(network.num_inputs):
        inp = network.get_input(i)
        profile.set_shape(inp.name, min=min_shape, opt=opt_shape, max=max_shape)
        logger.debug(f'  {inp.name}: min={min_shape}, opt={opt_shape}, max={max_shape}')

    config.add_optimization_profile(profile)


# ============================================================================
# Configuration Loading Functions
# ============================================================================


def load_config_file(config_path: str) -> dict[str, dict[str, Any]]:
    """
    Load model configurations from a YAML file.

    Expected YAML format:
    ```yaml
    models:
      my_custom_model:
        pt_file: /path/to/model.pt
        triton_name: my_model_name  # optional, auto-generated if not provided
        max_batch: 32  # optional, default 32
        topk: 300  # optional, default 300
        num_classes: 80  # optional, auto-detected from model
        class_names:  # optional, auto-detected from model
          - person
          - car
          - ...
    ```

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Dictionary of model configurations
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f'Config file not found: {config_path}')

    with open(path) as f:
        config = yaml.safe_load(f)

    if 'models' not in config:
        raise ValueError("Config file must have a 'models' section")

    models = {}
    for model_id, model_config in config['models'].items():
        if 'pt_file' not in model_config:
            raise ValueError(f"Model '{model_id}' must have 'pt_file' specified")

        # Set defaults
        models[model_id] = {
            'pt_file': model_config['pt_file'],
            'triton_name': model_config.get('triton_name', _generate_triton_name(model_id)),
            'max_batch': model_config.get('max_batch', 32),
            'topk': model_config.get('topk', 300),
        }

        # Optional class configuration
        if 'num_classes' in model_config:
            models[model_id]['num_classes'] = model_config['num_classes']
        if 'class_names' in model_config:
            models[model_id]['class_names'] = model_config['class_names']

    logger.info(f'Loaded {len(models)} model(s) from config: {config_path}')
    return models


def parse_custom_model(custom_model_arg: str) -> tuple[str, dict[str, Any]]:
    """
    Parse a custom model argument string.

    Format: path/to/model.pt[:name][:max_batch]
    Examples:
      - /path/to/my_model.pt                  -> auto name, batch=32
      - /path/to/my_model.pt:custom_name      -> custom_name, batch=32
      - /path/to/my_model.pt:custom_name:64   -> custom_name, batch=64
      - /path/to/my_model.pt::16              -> auto name, batch=16

    Args:
        custom_model_arg: Custom model specification string

    Returns:
        Tuple of (model_id, config_dict)
    """
    parts = custom_model_arg.split(':')
    pt_file = parts[0]

    if not Path(pt_file).exists():
        # Try with /app prefix for container paths
        container_path = f'/app/{pt_file.lstrip("/")}'
        if Path(container_path).exists():
            pt_file = container_path
        else:
            raise FileNotFoundError(f'Model file not found: {parts[0]}')

    # Extract model ID from filename
    model_id = Path(pt_file).stem

    # Parse optional name
    triton_name = parts[1] if len(parts) > 1 and parts[1] else _generate_triton_name(model_id)

    # Parse optional max_batch
    max_batch = int(parts[2]) if len(parts) > 2 and parts[2] else 32

    config = {
        'pt_file': pt_file,
        'triton_name': triton_name,
        'max_batch': max_batch,
        'topk': 300,
    }

    logger.info(f'Custom model: {model_id} -> {triton_name} (batch={max_batch})')
    return model_id, config


def _generate_triton_name(model_id: str) -> str:
    """
    Generate a Triton-compatible model name from model ID.

    Converts names like 'yolo11s' to 'yolov11_small', 'my-custom-model' to 'my_custom_model'.
    """
    # Known YOLO11 size mappings
    size_map = {
        'n': 'nano',
        's': 'small',
        'm': 'medium',
        'l': 'large',
        'x': 'xlarge',
    }

    # Check for yolo11X pattern
    match = re.match(r'yolo11([nsmxl])$', model_id.lower())
    if match:
        size = size_map.get(match.group(1), match.group(1))
        return f'yolov11_{size}'

    # General cleanup: replace hyphens with underscores, lowercase
    name = model_id.lower().replace('-', '_')

    # Remove .pt extension if present
    if name.endswith('.pt'):
        name = name[:-3]

    return name


# ============================================================================
# Class Name Extraction and Labels
# ============================================================================


def extract_class_names(model: 'YOLO') -> list[str]:
    """
    Extract class names from a loaded YOLO model.

    Args:
        model: Loaded YOLO model instance

    Returns:
        List of class names in order (index = class_id)
    """
    try:
        # YOLO models store class names in model.names
        if hasattr(model, 'names'):
            names = model.names
            if isinstance(names, dict):
                # Convert dict {0: 'person', 1: 'bicycle', ...} to list
                max_idx = max(names.keys())
                return [names.get(i, f'class_{i}') for i in range(max_idx + 1)]
            if isinstance(names, list):
                return names

        # Fallback: check model.model.names
        if hasattr(model, 'model') and hasattr(model.model, 'names'):
            names = model.model.names
            if isinstance(names, dict):
                max_idx = max(names.keys())
                return [names.get(i, f'class_{i}') for i in range(max_idx + 1)]
            if isinstance(names, list):
                return names

        logger.warning('Could not extract class names from model')
        return []
    except Exception as e:
        logger.warning(f'Error extracting class names: {e}')
        return []


def save_labels_file(class_names: list[str], model_dir: Path) -> Path | None:
    """
    Save class names to labels.txt file in model directory.

    Args:
        class_names: List of class names
        model_dir: Model directory path (e.g., /app/models/yolov11_small_trt)

    Returns:
        Path to saved labels file, or None if no classes to save
    """
    if not class_names:
        return None

    labels_path = model_dir / 'labels.txt'
    with open(labels_path, 'w') as f:
        for name in class_names:
            f.write(f'{name}\n')

    logger.info(f'Saved {len(class_names)} class names to: {labels_path}')
    return labels_path


# ============================================================================
# Triton Config Generation
# ============================================================================


def generate_triton_config(
    triton_name: str,
    model_format: str,
    max_batch: int,
    num_classes: int = 80,
    has_nms: bool = False,
) -> str:
    """
    Generate Triton config.pbtxt content for a model.

    Args:
        triton_name: Triton model name
        model_format: Model format ('onnx', 'trt', 'trt_end2end')
        max_batch: Maximum batch size
        num_classes: Number of classes (default 80 for COCO)
        has_nms: Whether model has built-in NMS (end2end models)

    Returns:
        config.pbtxt content as string
    """
    # Determine backend
    if model_format in ('trt', 'trt_end2end'):
        backend = 'tensorrt'
    else:
        backend = 'onnxruntime'

    # Calculate preferred batch sizes
    preferred_batches = [size for size in [8, 16, 32, 64] if size <= max_batch]
    if not preferred_batches:
        preferred_batches = [1]

    preferred_batch_str = ', '.join(str(b) for b in preferred_batches)

    # Determine platform string
    platform = f'{backend}_plan' if backend == 'tensorrt' else 'onnxruntime_onnx'

    if has_nms:
        # End2End model with NMS outputs
        config = f"""name: "{triton_name}"
platform: "{platform}"
max_batch_size: {max_batch}

input [
  {{
    name: "images"
    data_type: TYPE_FP32
    dims: [ 3, {IMG_SIZE}, {IMG_SIZE} ]
  }}
]

output [
  {{
    name: "num_dets"
    data_type: TYPE_INT32
    dims: [ 1 ]
  }},
  {{
    name: "det_boxes"
    data_type: TYPE_FP32
    dims: [ 300, 4 ]
  }},
  {{
    name: "det_scores"
    data_type: TYPE_FP32
    dims: [ 300 ]
  }},
  {{
    name: "det_classes"
    data_type: TYPE_INT32
    dims: [ 300 ]
  }}
]

dynamic_batching {{
  preferred_batch_size: [ {preferred_batch_str} ]
  max_queue_delay_microseconds: 5000
}}

instance_group [
  {{
    count: 2
    kind: KIND_GPU
    gpus: [ 0 ]
  }}
]
"""
    else:
        # Standard model without NMS
        # Output shape: [batch, 84, 8400] for YOLO11 with 80 classes
        output_dim_1 = 4 + num_classes  # 4 bbox coords + num_classes
        config = f"""name: "{triton_name}"
platform: "{platform}"
max_batch_size: {max_batch}

input [
  {{
    name: "images"
    data_type: TYPE_FP32
    dims: [ 3, {IMG_SIZE}, {IMG_SIZE} ]
  }}
]

output [
  {{
    name: "output0"
    data_type: TYPE_FP32
    dims: [ {output_dim_1}, 8400 ]
  }}
]

dynamic_batching {{
  preferred_batch_size: [ {preferred_batch_str} ]
  max_queue_delay_microseconds: 5000
}}

instance_group [
  {{
    count: 2
    kind: KIND_GPU
    gpus: [ 0 ]
  }}
]
"""

    return config


def save_triton_config(
    model_dir: Path,
    triton_name: str,
    model_format: str,
    max_batch: int,
    num_classes: int = 80,
    has_nms: bool = False,
) -> Path:
    """
    Save Triton config.pbtxt file for a model.

    Args:
        model_dir: Model directory path
        triton_name: Triton model name
        model_format: Model format
        max_batch: Maximum batch size
        num_classes: Number of classes
        has_nms: Whether model has built-in NMS

    Returns:
        Path to saved config file
    """
    config_content = generate_triton_config(
        triton_name=triton_name,
        model_format=model_format,
        max_batch=max_batch,
        num_classes=num_classes,
        has_nms=has_nms,
    )

    config_path = model_dir / 'config.pbtxt'
    with open(config_path, 'w') as f:
        f.write(config_content)

    logger.info(f'Generated Triton config: {config_path}')
    return config_path


def enable_fp16_if_available(builder: trt.Builder, config: trt.IBuilderConfig) -> bool:
    """Enable FP16 precision if hardware supports it."""
    if HALF and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        logger.info('FP16 precision enabled')
        return True
    logger.info('Using FP32 precision (FP16 not available or disabled)')
    return False


def parse_onnx_model(
    parser: trt.OnnxParser,
    onnx_path: Path,
) -> bool:
    """Parse ONNX model and report any errors."""
    if not parser.parse_from_file(str(onnx_path)):
        logger.error('Failed to parse ONNX model:')
        for i in range(parser.num_errors):
            error = parser.get_error(i)
            logger.error(f'  [{i}] {error}')
        return False
    return True


def build_and_save_engine(
    builder: trt.Builder,
    network: trt.INetworkDefinition,
    config: trt.IBuilderConfig,
    output_path: Path,
) -> bool:
    """Build TensorRT engine and save to file."""
    # Free memory before building
    gc.collect()
    torch.cuda.empty_cache()

    logger.info('Building TensorRT engine (this may take 5-10 minutes)...')

    is_trt10 = int(trt.__version__.split('.')[0]) >= 10

    try:
        if is_trt10:
            serialized_engine = builder.build_serialized_network(network, config)
            if serialized_engine is None:
                logger.error('Failed to build engine - builder returned None')
                return False
            with open(output_path, 'wb') as f:
                f.write(serialized_engine)
        else:
            engine = builder.build_engine(network, config)
            if engine is None:
                logger.error('Failed to build engine - builder returned None')
                return False
            with open(output_path, 'wb') as f:
                f.write(engine.serialize())

        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f'Engine saved: {output_path} ({file_size_mb:.2f} MB)')
        return True

    except Exception as e:
        logger.error(f'Engine build failed: {e}')
        return False


# ============================================================================
# Export Functions
# ============================================================================


def export_onnx_standard(model: YOLO, config: dict[str, Any]) -> dict[str, Any]:
    """
    Export standard ONNX model (no NMS).

    Args:
        model: Loaded YOLO model
        config: Model configuration dict with triton_name, max_batch, etc.

    Returns:
        Dict with status, path, and output format info
    """
    logger.info('=' * 60)
    logger.info('[1/4] Standard ONNX Export (ONNX Runtime + TensorRT EP)')
    logger.info('=' * 60)

    try:
        logger.info('Exporting ONNX with dynamic batching...')
        logger.info('  - dynamic=True: Variable batch, height, width')
        logger.info('  - simplify=True: Optimizes graph for TensorRT')

        onnx_path = model.export(
            format='onnx',
            imgsz=IMG_SIZE,
            device=DEVICE,
            dynamic=True,
            simplify=True,
            half=HALF,
            verbose=False,
        )

        # Move to Triton model repository
        triton_name = config['triton_name']
        onnx_model_dir = Path(f'/app/models/{triton_name}/1')
        onnx_model_dir.mkdir(parents=True, exist_ok=True)

        onnx_dest = onnx_model_dir / 'model.onnx'
        backup_existing_file(onnx_dest)

        shutil.copy2(onnx_path, onnx_dest)
        logger.info(f'ONNX saved to: {onnx_dest}')
        logger.info('Output: [84, 8400] - needs CPU NMS post-processing')

        return {
            'status': 'success',
            'path': str(onnx_dest),
            'host_path': str(onnx_dest).replace('/app/models', './models'),
            'output_format': '[84, 8400] - raw detections',
        }
    except Exception as e:
        logger.exception(f'ONNX export failed: {e}')
        return {'status': 'error', 'error': str(e)}


def export_trt_standard(model: YOLO, config: dict[str, Any]) -> dict[str, Any]:
    """
    Export native TensorRT engine from standard ONNX (no NMS).

    Args:
        model: Loaded YOLO model
        config: Model configuration dict

    Returns:
        Dict with status, path, and output format info
    """
    logger.info('=' * 60)
    logger.info('[2/4] Standard TensorRT Engine Export (no NMS)')
    logger.info('=' * 60)

    try:
        triton_name = config['triton_name']
        max_batch = config['max_batch']

        # First, check if we have standard ONNX, if not export it
        onnx_path = Path(f'/app/models/{triton_name}/1/model.onnx')
        if not onnx_path.exists():
            logger.info('Standard ONNX not found, exporting first...')
            result = export_onnx_standard(model, config)
            if result.get('status') != 'success':
                return result

        logger.info(f'Using standard ONNX: {onnx_path}')
        logger.info(f'Dynamic batch: min=1, opt={max_batch // 2}, max={max_batch}')
        logger.info(f'Workspace: {WORKSPACE_GB}GB, Precision: {"FP16" if HALF else "FP32"}')

        # Check GPU memory
        check_gpu_memory(WORKSPACE_GB)

        # Setup TensorRT builder using helper
        builder, config_trt, network, trt_logger = setup_trt_builder()

        # Parse ONNX
        logger.info('Parsing ONNX model...')
        parser = trt.OnnxParser(network, trt_logger)
        if not parse_onnx_model(parser, onnx_path):
            return {'status': 'error', 'error': 'ONNX parsing failed'}

        # Set optimization profile
        logger.info('Setting optimization profile...')
        add_optimization_profile(builder, config_trt, network, max_batch)

        # Set FP16
        enable_fp16_if_available(builder, config_trt)

        # Prepare output path
        trt_model_dir = Path(f'/app/models/{triton_name}_trt/1')
        trt_model_dir.mkdir(parents=True, exist_ok=True)
        trt_dest = trt_model_dir / 'model.plan'
        backup_existing_file(trt_dest)

        # Build and save engine
        if not build_and_save_engine(builder, network, config_trt, trt_dest):
            return {'status': 'error', 'error': 'Failed to build engine'}

        logger.info('Output format: [84, 8400] - raw detections (CPU NMS needed)')

        return {
            'status': 'success',
            'path': str(trt_dest),
            'host_path': str(trt_dest).replace('/app/models', './models'),
            'output_format': '[84, 8400] - raw detections',
        }
    except Exception as e:
        logger.exception(f'TensorRT export failed: {e}')
        return {'status': 'error', 'error': str(e)}


def export_onnx_end2end(
    model: YOLO, config: dict[str, Any], normalize_boxes: bool = True
) -> dict[str, Any]:
    """
    Export ONNX with TensorRT EfficientNMS operators (GPU NMS).

    Args:
        model: Loaded YOLO model
        config: Model configuration dict
        normalize_boxes: If True, output boxes in [0,1] range; otherwise pixel coords

    Returns:
        Dict with status, path, and output format info
    """
    logger.info('=' * 60)
    logger.info('[3/4] ONNX End2End Export (with GPU NMS operators)')
    logger.info('=' * 60)

    try:
        topk = config['topk']
        box_format_str = '[0,1] range' if normalize_boxes else 'pixel coords (640x640)'

        logger.info('Exporting ONNX with EfficientNMS plugin...')
        logger.info(f'  topk_all={topk}, iou_thres={IOU_THRESHOLD}, conf_thres={CONF_THRESHOLD}')
        logger.info(f'  normalize_boxes={normalize_boxes}: {box_format_str}')

        # Create export arguments with standard ONNX format
        args = get_cfg(
            overrides={
                'format': 'onnx',
                'imgsz': IMG_SIZE,
                'dynamic': True,
                'simplify': True,
                'half': HALF,
                'device': DEVICE,
                'opset': 17,
            }
        )

        # Add custom End2End arguments
        args.topk_all = topk
        args.iou_thres = IOU_THRESHOLD
        args.conf_thres = CONF_THRESHOLD
        args.class_agnostic = True
        args.mask_resolution = 56
        args.pooler_scale = 0.25
        args.sampling_ratio = 0
        args.normalize_boxes = normalize_boxes

        # Create and configure exporter
        exporter = Exporter(cfg=args, _callbacks=model.callbacks)
        exporter.args = args
        exporter.model = model.model.to(DEVICE)
        if hasattr(model, 'overrides'):
            exporter.model.args = model.overrides

        exporter.im = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
        exporter.file = Path(config['pt_file'])

        # Call export_onnx_trt directly
        logger.info('Calling patched export_onnx_trt() method...')
        export_path, _ = exporter.export_onnx_trt(prefix='ONNX TRT:')

        # Move to Triton model repository
        triton_name = config['triton_name']
        onnx_model_dir = Path(f'/app/models/{triton_name}_end2end/1')
        onnx_model_dir.mkdir(parents=True, exist_ok=True)

        onnx_dest = onnx_model_dir / 'model.onnx'
        export_file = Path(export_path)

        if not export_file.exists():
            logger.error(f'Export failed, file not found: {export_file}')
            return {'status': 'error', 'error': 'Export file not found'}

        backup_existing_file(onnx_dest)
        shutil.copy2(export_file, onnx_dest)

        logger.info(f'ONNX End2End saved to: {onnx_dest}')
        logger.info('Contains: TRT::EfficientNMS_TRT operators')
        logger.info('Output: num_dets, det_boxes, det_scores, det_classes')

        # Verify NMS plugin is present
        try:
            onnx_model = onnx.load(str(onnx_dest))
            ops = [node.op_type for node in onnx_model.graph.node]
            has_nms = any('NMS' in op or 'TRT' in op for op in ops)
            if has_nms:
                logger.info('Verified: NMS plugin in ONNX graph')
            else:
                logger.warning('No NMS operators found in ONNX graph!')
        except Exception as e:
            logger.debug(f'Could not verify ONNX operators: {e}')

        box_format = '[0,1] normalized' if normalize_boxes else 'pixel coords (640x640)'
        return {
            'status': 'success',
            'path': str(onnx_dest),
            'host_path': str(onnx_dest).replace('/app/models', './models'),
            'has_nms': True,
            'output_format': 'num_dets, det_boxes, det_scores, det_classes',
            'box_format': box_format,
        }

    except Exception as e:
        logger.exception(f'ONNX End2End export failed: {e}')
        return {'status': 'error', 'error': str(e)}


def export_trt_end2end(config: dict[str, Any]) -> dict[str, Any]:
    """
    Export TensorRT engine from end2end ONNX (compiled GPU NMS).

    Args:
        config: Model configuration dict

    Returns:
        Dict with status, path, and output format info
    """
    logger.info('=' * 60)
    logger.info('[4/4] TensorRT End2End Export (compiled GPU NMS)')
    logger.info('=' * 60)

    try:
        triton_name = config['triton_name']
        max_batch = config['max_batch']

        # Check for end2end ONNX
        onnx_end2end_path = Path(f'/app/models/{triton_name}_end2end/1/model.onnx')
        if not onnx_end2end_path.exists():
            logger.error(f'End2End ONNX not found: {onnx_end2end_path}')
            logger.error('Run with --formats onnx_end2end first!')
            return {'status': 'error', 'error': 'End2End ONNX not found'}

        logger.info(f'Using end2end ONNX: {onnx_end2end_path}')
        logger.info(f'Dynamic batch: min=1, opt={max_batch // 2}, max={max_batch}')
        logger.info(f'Workspace: {WORKSPACE_GB}GB, Precision: {"FP16" if HALF else "FP32"}')

        # Check GPU memory
        check_gpu_memory(WORKSPACE_GB)

        # Setup TensorRT builder using helper
        builder, config_trt, network, trt_logger = setup_trt_builder()
        logger.info('TensorRT plugins initialized (EfficientNMS available)')

        # Parse ONNX
        logger.info('Parsing ONNX model...')
        parser = trt.OnnxParser(network, trt_logger)
        if not parse_onnx_model(parser, onnx_end2end_path):
            return {'status': 'error', 'error': 'ONNX parsing failed'}

        # Log network structure
        logger.info('Network inputs:')
        for i in range(network.num_inputs):
            inp = network.get_input(i)
            logger.info(f'  {inp.name}: shape={inp.shape}, dtype={inp.dtype}')

        logger.info('Network outputs:')
        for i in range(network.num_outputs):
            out = network.get_output(i)
            logger.info(f'  {out.name}: shape={out.shape}, dtype={out.dtype}')

        # Set optimization profile
        logger.info('Setting optimization profile...')
        add_optimization_profile(builder, config_trt, network, max_batch)

        # Set FP16
        enable_fp16_if_available(builder, config_trt)

        # Prepare output path
        trt_model_dir = Path(f'/app/models/{triton_name}_trt_end2end/1')
        trt_model_dir.mkdir(parents=True, exist_ok=True)
        trt_dest = trt_model_dir / 'model.plan'
        backup_existing_file(trt_dest)

        # Build and save engine
        if not build_and_save_engine(builder, network, config_trt, trt_dest):
            return {'status': 'error', 'error': 'Failed to build engine'}

        logger.info('Contains: Compiled GPU NMS (EfficientNMS)')
        logger.info('Output format: num_dets, det_boxes, det_scores, det_classes')

        return {
            'status': 'success',
            'path': str(trt_dest),
            'host_path': str(trt_dest).replace('/app/models', './models'),
            'has_nms': True,
            'output_format': 'num_dets, det_boxes, det_scores, det_classes',
            'source': 'end2end_onnx',
        }
    except Exception as e:
        logger.exception(f'TensorRT End2End export failed: {e}')
        return {'status': 'error', 'error': str(e)}


# ============================================================================
# Main Export Logic
# ============================================================================


def export_model(
    model_id: str,
    config: dict[str, Any],
    formats: list[str],
    normalize_boxes: bool = True,
    generate_config: bool = False,
    save_labels: bool = False,
) -> dict[str, Any]:
    """
    Export a single model in specified formats.

    Args:
        model_id: Model identifier (nano, small, medium, large, xlarge, or custom)
        config: Model configuration dict
        formats: List of formats to export
        normalize_boxes: If True, output boxes in [0,1] range
        generate_config: If True, auto-generate Triton config.pbtxt files
        save_labels: If True, save class names to labels.txt

    Returns:
        Dict with export results for each format
    """
    logger.info('=' * 70)
    logger.info(f'Processing {model_id} -> {config["triton_name"]}')
    logger.info('=' * 70)

    pt_file = config['pt_file']

    # Validate input file
    if not validate_pt_file(pt_file):
        return {'model': model_id, 'status': 'error', 'error': 'Model file not found'}

    results = {
        'model': model_id,
        'triton_name': config['triton_name'],
        'max_batch': config['max_batch'],
        'topk': config['topk'],
    }

    # Load model once
    model = YOLO(pt_file)

    # Extract class information from model
    class_names = config.get('class_names') or extract_class_names(model)
    num_classes = config.get('num_classes') or len(class_names) or 80

    results['num_classes'] = num_classes
    results['class_names_count'] = len(class_names)

    logger.info(f'Model has {num_classes} classes')
    if class_names and len(class_names) <= 10:
        logger.info(f'Classes: {", ".join(class_names)}')
    elif class_names:
        logger.info(f'Classes: {", ".join(class_names[:5])}... (and {len(class_names) - 5} more)')

    # Export in requested formats
    if 'onnx' in formats or 'all' in formats:
        results['onnx'] = export_onnx_standard(model, config)

        # Save labels and generate config for ONNX model
        if results['onnx']['status'] == 'success':
            triton_name = config['triton_name']
            model_dir = Path(f'/app/models/{triton_name}')

            if save_labels and class_names:
                save_labels_file(class_names, model_dir)

            if generate_config:
                save_triton_config(
                    model_dir=model_dir,
                    triton_name=triton_name,
                    model_format='onnx',
                    max_batch=config['max_batch'],
                    num_classes=num_classes,
                    has_nms=False,
                )

    if 'trt' in formats or 'all' in formats:
        # Reload model for clean state
        model = YOLO(pt_file)
        results['trt'] = export_trt_standard(model, config)

        # Save labels and generate config for TRT model
        if results['trt']['status'] == 'success':
            triton_name = config['triton_name']
            model_dir = Path(f'/app/models/{triton_name}_trt')

            if save_labels and class_names:
                save_labels_file(class_names, model_dir)

            if generate_config:
                save_triton_config(
                    model_dir=model_dir,
                    triton_name=f'{triton_name}_trt',
                    model_format='trt',
                    max_batch=config['max_batch'],
                    num_classes=num_classes,
                    has_nms=False,
                )

    if 'onnx_end2end' in formats or 'all' in formats:
        # Reload model for clean state
        model = YOLO(pt_file)
        results['onnx_end2end'] = export_onnx_end2end(
            model, config, normalize_boxes=normalize_boxes
        )

        # Save labels for end2end ONNX model
        if results['onnx_end2end']['status'] == 'success':
            triton_name = config['triton_name']
            model_dir = Path(f'/app/models/{triton_name}_end2end')

            if save_labels and class_names:
                save_labels_file(class_names, model_dir)

    if 'trt_end2end' in formats or 'all' in formats:
        # This uses the end2end ONNX, doesn't need model reload
        results['trt_end2end'] = export_trt_end2end(config)

        # Save labels and generate config for TRT end2end model
        if results['trt_end2end']['status'] == 'success':
            triton_name = config['triton_name']
            model_dir = Path(f'/app/models/{triton_name}_trt_end2end')

            if save_labels and class_names:
                save_labels_file(class_names, model_dir)

            if generate_config:
                save_triton_config(
                    model_dir=model_dir,
                    triton_name=f'{triton_name}_trt_end2end',
                    model_format='trt_end2end',
                    max_batch=config['max_batch'],
                    num_classes=num_classes,
                    has_nms=True,
                )

    return results


def print_summary(results: list[dict[str, Any]]) -> None:
    """Print export summary to logger."""
    logger.info('=' * 70)
    logger.info('EXPORT SUMMARY')
    logger.info('=' * 70)

    for result in results:
        logger.info(f'{result["model"]} ({result["triton_name"]}):')

        if 'onnx' in result:
            status = '[OK]' if result['onnx']['status'] == 'success' else '[FAIL]'
            logger.info(f'  {status} ONNX Standard: {result["onnx"]["status"]}')
            if result['onnx']['status'] == 'success':
                logger.info(f'       Host: {result["onnx"]["host_path"]}')

        if 'trt' in result:
            status = '[OK]' if result['trt']['status'] == 'success' else '[FAIL]'
            logger.info(f'  {status} TRT Standard: {result["trt"]["status"]}')
            if result['trt']['status'] == 'success':
                logger.info(f'       Host: {result["trt"]["host_path"]}')

        if 'onnx_end2end' in result:
            status = '[OK]' if result['onnx_end2end']['status'] == 'success' else '[FAIL]'
            logger.info(f'  {status} ONNX End2End: {result["onnx_end2end"]["status"]}')
            if result['onnx_end2end']['status'] == 'success':
                logger.info(f'       Host: {result["onnx_end2end"]["host_path"]}')
                logger.info('       NMS: GPU (TRT operators)')

        if 'trt_end2end' in result:
            status = '[OK]' if result['trt_end2end']['status'] == 'success' else '[FAIL]'
            logger.info(f'  {status} TRT End2End: {result["trt_end2end"]["status"]}')
            if result['trt_end2end']['status'] == 'success':
                logger.info(f'       Host: {result["trt_end2end"]["host_path"]}')
                logger.info('       NMS: Compiled (maximum performance!)')

    logger.info('-' * 70)
    logger.info('FORMAT COMPARISON')
    logger.info('-' * 70)
    logger.info('1. ONNX Standard: [84, 8400] -> CPU NMS, baseline speed')
    logger.info('2. TRT Standard:  [84, 8400] -> CPU NMS, 1.5x faster')
    logger.info('3. ONNX End2End:  NMS outputs -> GPU NMS via TRT EP, 2-3x faster')
    logger.info('4. TRT End2End:   NMS outputs -> compiled GPU NMS, 3-5x faster')
    logger.info('=' * 70)


def main() -> None:
    """Main entry point for export script."""
    parser = argparse.ArgumentParser(
        description='Export YOLO models in multiple formats for Triton',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export all formats for default models
  python export_models.py

  # Export only end2end models
  python export_models.py --formats onnx_end2end trt_end2end

  # Export specific built-in models (nano, small, medium, large, xlarge)
  python export_models.py --models nano small large

  # Export a custom trained model
  python export_models.py --custom-model /path/to/my_model.pt

  # Export custom model with specific name and batch size
  python export_models.py --custom-model /path/to/my_model.pt:custom_name:64

  # Export using a YAML configuration file
  python export_models.py --config-file /path/to/models.yaml

  # Generate Triton configs and labels for exported models
  python export_models.py --models small --generate-config --save-labels
        """,
    )

    parser.add_argument(
        '--formats',
        nargs='+',
        choices=['onnx', 'trt', 'onnx_end2end', 'trt_end2end', 'all'],
        default=['all'],
        help='Formats to export (default: all)',
    )

    parser.add_argument(
        '--models',
        nargs='+',
        default=None,
        help=(
            'Built-in models to export: nano, small, medium, large, xlarge. '
            'If not specified and no --custom-model or --config-file, exports small only.'
        ),
    )

    parser.add_argument(
        '--custom-model',
        nargs='+',
        dest='custom_models',
        metavar='PATH[:NAME][:BATCH]',
        help=(
            'Custom model(s) to export. Format: path/to/model.pt[:triton_name][:max_batch]. '
            'Examples: my_model.pt, my_model.pt:custom_name, my_model.pt:custom_name:64'
        ),
    )

    parser.add_argument(
        '--config-file',
        dest='config_file',
        metavar='PATH',
        help='YAML configuration file with model definitions (overrides --models)',
    )

    parser.add_argument(
        '--normalize-boxes',
        action='store_true',
        default=False,
        help='Output boxes in [0, 1] normalized range instead of pixel coords (Track E recommended)',
    )

    parser.add_argument(
        '--generate-config',
        action='store_true',
        default=False,
        help='Auto-generate Triton config.pbtxt files for exported models',
    )

    parser.add_argument(
        '--save-labels',
        action='store_true',
        default=False,
        help='Save class names to labels.txt in each model directory',
    )

    parser.add_argument(
        '--list-models',
        action='store_true',
        help='List available built-in models and exit',
    )

    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose (DEBUG) logging',
    )

    args = parser.parse_args()

    # Handle --list-models
    if args.list_models:
        print('Available built-in YOLO11 models:')
        print('-' * 50)
        for model_id, cfg in DEFAULT_MODELS.items():
            print(f'  {model_id:10} -> {cfg["triton_name"]:25} (batch={cfg["max_batch"]})')
        print('-' * 50)
        print('\nUsage: --models nano small medium large xlarge')
        print('Custom: --custom-model /path/to/model.pt[:name][:batch]')
        sys.exit(0)

    # Set log level based on verbosity
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Build model configurations from various sources
    models_to_export: dict[str, dict[str, Any]] = {}

    # 1. Load from config file if specified
    if args.config_file:
        try:
            models_to_export.update(load_config_file(args.config_file))
        except (FileNotFoundError, ValueError) as e:
            logger.error(f'Config file error: {e}')
            sys.exit(1)

    # 2. Add built-in models if specified
    if args.models:
        for model_id in args.models:
            if model_id in DEFAULT_MODELS:
                models_to_export[model_id] = DEFAULT_MODELS[model_id].copy()
            else:
                logger.error(
                    f"Unknown model '{model_id}'. "
                    f'Available: {", ".join(DEFAULT_MODELS.keys())}. '
                    f'Use --custom-model for custom models.'
                )
                sys.exit(1)

    # 3. Add custom models if specified
    if args.custom_models:
        for custom_arg in args.custom_models:
            try:
                model_id, model_config = parse_custom_model(custom_arg)
                models_to_export[model_id] = model_config
            except (FileNotFoundError, ValueError) as e:
                logger.error(f'Custom model error: {e}')
                sys.exit(1)

    # 4. Default: export 'small' if nothing specified
    if not models_to_export:
        logger.info('No models specified, defaulting to small')
        models_to_export['small'] = DEFAULT_MODELS['small'].copy()

    logger.info('=' * 70)
    logger.info('YOLO11 Unified Export Script')
    logger.info('=' * 70)
    logger.info(f'Models: {", ".join(models_to_export.keys())}')
    logger.info(f'Formats: {", ".join(args.formats)}')
    logger.info(f'Image size: {IMG_SIZE}x{IMG_SIZE}')
    logger.info(f'Device: cuda:{DEVICE}')
    logger.info(f'Precision: {"FP16" if HALF else "FP32"}')
    logger.info(f'Workspace: {WORKSPACE_GB}GB')
    if 'onnx_end2end' in args.formats or 'trt_end2end' in args.formats or 'all' in args.formats:
        logger.info(f'NMS IoU threshold: {IOU_THRESHOLD}')
        logger.info(f'Confidence threshold: {CONF_THRESHOLD}')
        box_mode = '(Track E recommended)' if args.normalize_boxes else '(pixel coords 640x640)'
        logger.info(f'Normalize boxes: {args.normalize_boxes} {box_mode}')
    if args.generate_config:
        logger.info('Auto-generate Triton configs: enabled')
    if args.save_labels:
        logger.info('Save labels.txt: enabled')
    logger.info('=' * 70)

    # Check GPU availability
    if not check_gpu_memory():
        logger.error('GPU check failed. Aborting.')
        sys.exit(1)

    results = []

    for model_id, model_config in models_to_export.items():
        result = export_model(
            model_id,
            model_config,
            args.formats,
            normalize_boxes=args.normalize_boxes,
            generate_config=args.generate_config,
            save_labels=args.save_labels,
        )
        results.append(result)

    # Print summary
    print_summary(results)

    # Save results
    results_file = Path('/app/scripts/export_results.json')
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(
            {
                'formats': args.formats,
                'models': list(models_to_export.keys()),
                'results': results,
            },
            f,
            indent=2,
        )
    logger.info(f'Results saved to: {results_file}')

    # Show current model structure
    logger.info('=' * 70)
    logger.info('CURRENT MODEL REPOSITORY')
    logger.info('=' * 70)
    models_dir = Path('/app/models')
    if models_dir.exists():
        for model_dir in sorted(models_dir.iterdir()):
            if model_dir.is_dir() and (
                model_dir.name.startswith('yolov11') or model_dir.name.startswith('yolo11')
            ):
                has_model = (model_dir / '1' / 'model.onnx').exists() or (
                    model_dir / '1' / 'model.plan'
                ).exists()
                has_config = (model_dir / 'config.pbtxt').exists()
                has_labels = (model_dir / 'labels.txt').exists()
                status = '[OK]' if has_model else '[--]'
                config_status = '[OK]' if has_config else '[!!]'
                labels_status = '[OK]' if has_labels else '[--]'
                logger.info(
                    f'{status} {model_dir.name:35} config: {config_status} labels: {labels_status}'
                )
    logger.info('=' * 70)


if __name__ == '__main__':
    main()
