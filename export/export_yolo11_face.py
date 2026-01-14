#!/usr/bin/env python3
"""
Export YOLO11-face models to TensorRT for Triton deployment.

YOLO11-face uses pose/keypoint architecture:
- Output includes boxes + 5 keypoints (face landmarks)
- Each keypoint has (x, y, visibility) coordinates
- Landmark order: left_eye, right_eye, nose, left_mouth, right_mouth

Export Modes:
1. Standard (default): Raw pose output, Python backend handles NMS
2. End2End (--end2end): TensorRT EfficientNMS plugin, GPU-fused NMS
   - Note: End2End mode outputs ONLY boxes/scores (no keypoints)
   - Use with MTCNN-style face cropping (yolo11_face_pipeline)

Key steps:
1. Load YOLO11-face PyTorch model
2. Export to ONNX (with or without EfficientNMS)
3. Convert to TensorRT FP16 with dynamic batch
4. Create Triton configuration

Usage:
    # Standard export (with Python NMS):
    docker compose exec yolo-api python /app/export/export_yolo11_face.py

    # End2End export (with GPU NMS - faster):
    docker compose exec yolo-api python /app/export/export_yolo11_face.py --end2end

    # With custom model:
    docker compose exec yolo-api python /app/export/export_yolo11_face.py \
        --model /app/pytorch_models/yolo11_face/yolo11s-face.pt --end2end
"""

import argparse
import gc
import logging
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import torch

sys.path.insert(0, '/app/src')

# Apply end2end patch for GPU NMS support (must be before ultralytics import)
from ultralytics_patches import apply_end2end_patch, is_patch_applied

if not is_patch_applied():
    apply_end2end_patch()


# =============================================================================
# Logging Configuration
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

# Default paths (from YapaLab/yolo-face releases)
DEFAULT_MODEL_PATH = Path('/app/pytorch_models/yolo11_face/yolov11s-face.pt')
if not Path('/app').exists():
    DEFAULT_MODEL_PATH = Path('pytorch_models/yolo11_face/yolov11s-face.pt')

OUTPUT_DIR = Path('/app/models') if Path('/app').exists() else Path('models')

# Export settings
IMG_SIZE = 640
DEVICE = 0
HALF = True  # FP16
MAX_BATCH_SIZE = 64
WORKSPACE_GB = 4

# Model size mapping
SIZE_MAP = {'n': 'nano', 's': 'small', 'm': 'medium', 'l': 'large', 'x': 'xlarge'}


# =============================================================================
# Helper Functions
# =============================================================================


def get_model_size(model_path: Path) -> str:
    """Extract model size from filename (e.g., yolov11s-face.pt -> small)."""
    stem = model_path.stem.lower()
    # Extract size letter from yolov11X-face or yolo11X-face pattern
    # Look for pattern after "yolov11" or "yolo11"
    import re
    match = re.search(r'yolov?11([nslmx])', stem)
    if match:
        size_char = match.group(1)
        return SIZE_MAP.get(size_char, 'small')
    return 'small'  # Default


def generate_triton_name(model_path: Path) -> str:
    """Generate Triton model name from path."""
    size = get_model_size(model_path)
    return f'yolo11_face_{size}_trt'


def backup_existing_file(file_path: Path, suffix: str = '.old') -> None:
    """Backup existing file if it exists."""
    if file_path.exists():
        backup_path = file_path.with_suffix(file_path.suffix + suffix)
        if backup_path.exists():
            backup_path.unlink()
        shutil.move(file_path, backup_path)
        logger.debug(f'Backed up existing file to: {backup_path}')


# =============================================================================
# ONNX Export
# =============================================================================


def export_to_onnx(
    model_path: Path,
    output_path: Path,
    imgsz: int = IMG_SIZE,
    opset: int = 17,
) -> bool:
    """
    Export YOLO11-face to ONNX format.

    YOLO11-face is a pose model, so output includes keypoints.
    Output shape: [batch, num_detections, 6 + 3*num_keypoints]
    Where 6 = [x, y, w, h, obj_conf, cls_conf] and 3*num_keypoints = [x, y, vis] * 5

    Args:
        model_path: Path to .pt model
        output_path: Output ONNX path
        imgsz: Input image size
        opset: ONNX opset version

    Returns:
        True if successful
    """
    logger.info(f'Exporting to ONNX: {model_path} -> {output_path}')

    try:
        from ultralytics import YOLO

        # Load model
        model = YOLO(str(model_path))
        logger.info(f'  Loaded model: {model.task} with {len(model.names)} classes')

        # Export to ONNX using ultralytics built-in
        # The model is a pose model so export handles keypoints automatically
        export_path = model.export(
            format='onnx',
            imgsz=imgsz,
            half=False,  # ONNX export in FP32, TRT will use FP16
            simplify=True,
            opset=opset,
            dynamic=True,  # Dynamic batch
            device=DEVICE,
        )

        # Move to desired output path
        export_path = Path(export_path)
        if export_path != output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(export_path, output_path)

        logger.info(f'  ONNX exported: {output_path} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)')

        # Analyze output
        analyze_onnx_model(output_path)

        # Cleanup
        del model
        gc.collect()
        torch.cuda.empty_cache()

        return True

    except Exception as e:
        logger.error(f'ONNX export failed: {e}')
        import traceback
        traceback.print_exc()
        return False


def analyze_onnx_model(onnx_path: Path) -> dict:
    """Analyze ONNX model structure."""
    logger.info(f'Analyzing ONNX model: {onnx_path}')

    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)

    logger.info(f'  Nodes: {len(model.graph.node)}, Opset: {model.opset_import[0].version}')

    # Inputs
    for inp in model.graph.input:
        shape = [d.dim_value if d.dim_value > 0 else d.dim_param for d in inp.type.tensor_type.shape.dim]
        logger.info(f'  Input: {inp.name} {shape}')

    # Outputs
    outputs = {}
    for out in model.graph.output:
        shape = [d.dim_value if d.dim_value > 0 else d.dim_param for d in out.type.tensor_type.shape.dim]
        logger.info(f'  Output: {out.name} {shape}')
        outputs[out.name] = shape

    return outputs


def test_onnx_inference(onnx_path: Path, batch_size: int = 1) -> bool:
    """Test ONNX model inference."""
    logger.info(f'Testing ONNX inference (batch={batch_size})...')

    try:
        import onnxruntime as ort

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(str(onnx_path), providers=providers)
        logger.info(f'  Provider: {session.get_providers()[0]}')

        # Get input info
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        logger.info(f'  Input: {input_name} {input_shape}')

        # Create test input
        # Handle dynamic dimensions
        shape = [batch_size, 3, IMG_SIZE, IMG_SIZE]
        test_input = np.random.rand(*shape).astype(np.float32)

        # Run inference
        start = time.time()
        outputs = session.run(None, {input_name: test_input})
        inference_time = (time.time() - start) * 1000

        logger.info(f'  Inference time: {inference_time:.2f}ms')

        for i, out in enumerate(outputs):
            output_name = session.get_outputs()[i].name
            logger.info(f'  Output {output_name}: shape={out.shape}, dtype={out.dtype}')

        return True

    except Exception as e:
        logger.error(f'ONNX inference test failed: {e}')
        return False


# =============================================================================
# End2End ONNX Export (GPU NMS via TensorRT EfficientNMS)
# =============================================================================


def export_to_onnx_end2end(
    model_path: Path,
    output_path: Path,
    imgsz: int = IMG_SIZE,
    max_det: int = 100,
    iou_thres: float = 0.4,
    conf_thres: float = 0.5,
    normalize_boxes: bool = True,
) -> bool:
    """
    Export YOLO11-face to ONNX with TensorRT EfficientNMS plugin.

    This creates an End2End model with GPU-accelerated NMS baked into the graph.
    Note: Keypoints are NOT preserved in End2End mode - only boxes, scores, classes.

    Args:
        model_path: Path to .pt model
        output_path: Output ONNX path
        imgsz: Input image size
        max_det: Maximum detections (faces) per image
        iou_thres: NMS IoU threshold
        conf_thres: Confidence threshold
        normalize_boxes: If True, output boxes in [0,1] range

    Returns:
        True if successful
    """
    logger.info(f'Exporting End2End ONNX: {model_path} -> {output_path}')
    logger.info(f'  NMS config: max_det={max_det}, iou={iou_thres}, conf={conf_thres}')
    logger.info(f'  Normalize boxes: {normalize_boxes}')

    try:
        from ultralytics import YOLO
        from ultralytics.cfg import get_cfg
        from ultralytics.engine.exporter import Exporter

        # Load model
        model = YOLO(str(model_path))
        logger.info(f'  Loaded model: {model.task} with {len(model.names)} classes')

        # Create exporter with STANDARD args only (custom End2End args set separately)
        overrides = {
            'format': 'onnx',  # Base format - we call export_onnx_trt directly
            'imgsz': imgsz,
            'half': False,  # ONNX in FP32, TRT will use FP16
            'simplify': True,
            'dynamic': True,
            'device': DEVICE,
        }

        # Setup exporter with standard config
        cfg = get_cfg(overrides=overrides)
        exporter = Exporter(cfg)
        exporter.model = model.model
        exporter.args = cfg

        # Set End2End specific args AFTER init (these are recognized by the patch)
        exporter.args.topk_all = max_det
        exporter.args.iou_thres = iou_thres
        exporter.args.conf_thres = conf_thres
        exporter.args.class_agnostic = True  # Face detection is single class
        exporter.args.normalize_boxes = normalize_boxes
        exporter.args.mask_resolution = 56  # Default (not used for detection)
        exporter.args.pooler_scale = 0.25
        exporter.args.sampling_ratio = 0

        # Set required attributes
        if hasattr(model, 'overrides'):
            exporter.model.args = model.overrides

        exporter.im = torch.zeros(1, 3, imgsz, imgsz).to(DEVICE)
        exporter.file = model_path

        # Call the patched export_onnx_trt method
        logger.info('  Calling export_onnx_trt() for End2End export...')
        export_path, _ = exporter.export_onnx_trt(prefix='ONNX TRT:')

        # Move to desired output path
        export_path = Path(export_path)
        if export_path != output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(export_path, output_path)

        logger.info(f'  End2End ONNX exported: {output_path} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)')

        # Cleanup
        del model, exporter
        gc.collect()
        torch.cuda.empty_cache()

        return True

    except Exception as e:
        logger.error(f'End2End ONNX export failed: {e}')
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# TensorRT Export
# =============================================================================


def export_to_tensorrt(
    onnx_path: Path,
    plan_path: Path,
    max_batch_size: int = MAX_BATCH_SIZE,
    fp16: bool = True,
    workspace_gb: int = WORKSPACE_GB,
) -> bool:
    """
    Convert ONNX model to TensorRT engine.

    Args:
        onnx_path: Input ONNX path
        plan_path: Output TensorRT plan path
        max_batch_size: Maximum batch size
        fp16: Use FP16 precision
        workspace_gb: TensorRT workspace size in GB

    Returns:
        True if successful
    """
    logger.info(f'Converting to TensorRT: {onnx_path} -> {plan_path}')
    logger.info(f'  FP16: {fp16}, Max batch: {max_batch_size}, Workspace: {workspace_gb}GB')

    try:
        import tensorrt as trt

        trt.init_libnvinfer_plugins(None, '')
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

        # Create builder
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, TRT_LOGGER)

        # Parse ONNX
        logger.info('  Parsing ONNX...')
        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    logger.error(f'  Parse error {i}: {parser.get_error(i)}')
                raise RuntimeError('Failed to parse ONNX')

        logger.info(f'  Parsed: {network.num_layers} layers, {network.num_inputs} inputs, {network.num_outputs} outputs')

        # Configure builder
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)

        if fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            logger.info('  FP16 mode enabled')

        # Optimization profile for dynamic batching
        profile = builder.create_optimization_profile()

        for i in range(network.num_inputs):
            inp = network.get_input(i)
            # Get current dims (with dynamic batch)
            dims = inp.shape  # e.g., (-1, 3, -1, -1) or (-1, 3, 640, 640)

            # Handle dynamic dimensions - use IMG_SIZE for height/width
            c = dims[1] if dims[1] > 0 else 3
            h = dims[2] if dims[2] > 0 else IMG_SIZE
            w = dims[3] if dims[3] > 0 else IMG_SIZE

            min_shape = (1, c, h, w)
            opt_shape = (max_batch_size // 2, c, h, w)
            max_shape = (max_batch_size, c, h, w)

            profile.set_shape(inp.name, min=min_shape, opt=opt_shape, max=max_shape)
            logger.info(f'  Profile {inp.name}: min={min_shape}, opt={opt_shape}, max={max_shape}')

        config.add_optimization_profile(profile)

        # Build engine
        logger.info('  Building TensorRT engine (this may take 3-10 minutes)...')
        start_time = time.time()
        serialized_engine = builder.build_serialized_network(network, config)

        if serialized_engine is None:
            raise RuntimeError('Failed to build TensorRT engine')

        build_time = time.time() - start_time
        logger.info(f'  Build time: {build_time:.1f}s')

        # Save engine
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        with open(plan_path, 'wb') as f:
            f.write(serialized_engine)

        size_mb = plan_path.stat().st_size / (1024 * 1024)
        logger.info(f'  TensorRT saved: {plan_path} ({size_mb:.1f} MB)')

        return True

    except ImportError:
        logger.error('TensorRT not available. Install with: pip install tensorrt')
        return False
    except Exception as e:
        logger.error(f'TensorRT conversion failed: {e}')
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Triton Configuration
# =============================================================================


def create_triton_config(
    triton_name: str,
    plan_path: Path,
    max_batch_size: int = MAX_BATCH_SIZE,
    is_end2end: bool = False,
    max_det: int = 100,
) -> Path:
    """
    Create Triton config.pbtxt for YOLO11-face model.

    The config defines:
    - Input: [3, 640, 640] FP32 images
    - Output: Raw pose output OR End2End NMS output

    Args:
        triton_name: Triton model name
        plan_path: Path to TensorRT plan
        max_batch_size: Maximum batch size
        is_end2end: If True, configure for End2End output format
        max_det: Maximum detections (for End2End config)

    Returns:
        Path to config file
    """
    config_path = plan_path.parent.parent / 'config.pbtxt'

    if is_end2end:
        # End2End model with TensorRT EfficientNMS
        config_content = f'''# YOLO11-face TensorRT End2End model
# Face detection with built-in GPU NMS (no keypoints in End2End mode)
# Input: RGB images normalized [0,1]
# Output: Post-NMS detections (boxes in [0,1] range)

name: "{triton_name}"
platform: "tensorrt_plan"
max_batch_size: {max_batch_size}

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
    dims: [ {max_det}, 4 ]
  }},
  {{
    name: "det_scores"
    data_type: TYPE_FP32
    dims: [ {max_det} ]
  }},
  {{
    name: "det_classes"
    data_type: TYPE_INT32
    dims: [ {max_det} ]
  }}
]

dynamic_batching {{
  preferred_batch_size: [ 4, 8, 16, 32 ]
  max_queue_delay_microseconds: 10000
}}

instance_group [
  {{
    count: 2
    kind: KIND_GPU
    gpus: [ 0 ]
  }}
]

version_policy {{
  latest {{
    num_versions: 1
  }}
}}
'''
    else:
        # Standard model - raw pose output for Python NMS
        config_content = f'''# YOLO11-face TensorRT model
# Face detection with 5-point landmarks
# Input: RGB images normalized [0,1]
# Output: Raw pose predictions (requires post-processing)

name: "{triton_name}"
platform: "tensorrt_plan"
max_batch_size: {max_batch_size}

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
    dims: [ -1, -1 ]
  }}
]

dynamic_batching {{
  preferred_batch_size: [ 1, 4, 8, 16, 32 ]
  max_queue_delay_microseconds: 10000
}}

instance_group [
  {{
    count: 2
    kind: KIND_GPU
    gpus: [ 0 ]
  }}
]

version_policy {{
  latest {{
    num_versions: 1
  }}
}}
'''

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(config_content)
    logger.info(f'Triton config written: {config_path}')

    # Create labels file
    labels_path = config_path.parent / 'labels.txt'
    labels_path.write_text('face\n')
    logger.info(f'Labels file written: {labels_path}')

    return config_path


# =============================================================================
# Main Export Function
# =============================================================================


def export_yolo11_face(
    model_path: Path,
    output_name: str | None = None,
    output_dir: Path = OUTPUT_DIR,
    max_batch_size: int = MAX_BATCH_SIZE,
    fp16: bool = True,
    skip_onnx: bool = False,
    end2end: bool = False,
    max_det: int = 100,
    iou_thres: float = 0.4,
    conf_thres: float = 0.5,
) -> dict:
    """
    Full export pipeline for YOLO11-face.

    Args:
        model_path: Path to .pt model
        output_name: Triton model name (auto-generated if None)
        output_dir: Output directory for Triton models
        max_batch_size: Maximum batch size for TensorRT
        fp16: Use FP16 precision
        skip_onnx: Skip ONNX export if it exists
        end2end: If True, export with TensorRT EfficientNMS (GPU NMS)
        max_det: Maximum detections per image (for End2End)
        iou_thres: NMS IoU threshold (for End2End)
        conf_thres: Confidence threshold (for End2End)

    Returns:
        Dictionary with export results
    """
    logger.info('=' * 60)
    logger.info('YOLO11-Face Export')
    logger.info('=' * 60)

    # Validate input
    if not model_path.exists():
        logger.error(f'Model not found: {model_path}')
        return {'success': False, 'error': 'Model not found'}

    # Generate names
    if output_name is None:
        base_name = generate_triton_name(model_path)
        output_name = f'{base_name}_end2end' if end2end else base_name

    logger.info(f'  Model: {model_path}')
    logger.info(f'  Triton name: {output_name}')
    logger.info(f'  Output dir: {output_dir}')
    logger.info(f'  End2End (GPU NMS): {end2end}')
    if end2end:
        logger.info(f'  NMS config: max_det={max_det}, iou={iou_thres}, conf={conf_thres}')

    # Setup paths
    triton_model_dir = output_dir / output_name
    triton_version_dir = triton_model_dir / '1'
    onnx_path = triton_version_dir / 'model.onnx'
    plan_path = triton_version_dir / 'model.plan'

    triton_version_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'model_path': str(model_path),
        'triton_name': output_name,
        'triton_dir': str(triton_model_dir),
        'end2end': end2end,
    }

    # Step 1: Export to ONNX
    if skip_onnx and onnx_path.exists():
        logger.info(f'Skipping ONNX export (exists): {onnx_path}')
    else:
        backup_existing_file(onnx_path)
        if end2end:
            # End2End export with TensorRT EfficientNMS
            if not export_to_onnx_end2end(
                model_path, onnx_path,
                max_det=max_det,
                iou_thres=iou_thres,
                conf_thres=conf_thres,
                normalize_boxes=True,
            ):
                return {'success': False, 'error': 'End2End ONNX export failed'}
        else:
            # Standard export
            if not export_to_onnx(model_path, onnx_path):
                return {'success': False, 'error': 'ONNX export failed'}

    results['onnx_path'] = str(onnx_path)

    # Step 2: Test ONNX inference (skip for End2End - custom ops)
    if not end2end:
        if not test_onnx_inference(onnx_path):
            logger.warning('ONNX inference test failed, continuing anyway...')

    # Step 3: Convert to TensorRT
    backup_existing_file(plan_path)
    if not export_to_tensorrt(onnx_path, plan_path, max_batch_size, fp16):
        return {'success': False, 'error': 'TensorRT conversion failed'}

    results['plan_path'] = str(plan_path)

    # Step 4: Create Triton config
    config_path = create_triton_config(
        output_name, plan_path, max_batch_size,
        is_end2end=end2end, max_det=max_det
    )
    results['config_path'] = str(config_path)

    results['success'] = True

    logger.info('')
    logger.info('=' * 60)
    logger.info('Export Complete!')
    logger.info('=' * 60)
    logger.info(f'  Triton model: {triton_model_dir}')
    logger.info(f'  Mode: {"End2End (GPU NMS)" if end2end else "Standard (Python NMS)"}')
    logger.info('')
    logger.info('Next steps:')
    logger.info('  1. Restart Triton: make restart-triton')
    logger.info('  2. Test: curl -X POST http://localhost:4603/track_e/faces/yolo11/detect -F "file=@test.jpg"')

    return results


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description='Export YOLO11-face to TensorRT',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Standard export (Python NMS):
  python export_yolo11_face.py

  # End2End export (GPU NMS - faster):
  python export_yolo11_face.py --end2end

  # Custom model with End2End:
  python export_yolo11_face.py --model /path/to/model.pt --end2end --max-det 50
''',
    )
    parser.add_argument(
        '--model',
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help='Path to YOLO11-face .pt model',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=OUTPUT_DIR,
        help='Output directory for Triton models',
    )
    parser.add_argument(
        '--output-name',
        type=str,
        default=None,
        help='Triton model name (auto-generated if not provided)',
    )
    parser.add_argument(
        '--max-batch',
        type=int,
        default=MAX_BATCH_SIZE,
        help='Maximum batch size for TensorRT',
    )
    parser.add_argument(
        '--no-fp16',
        action='store_true',
        help='Disable FP16 precision',
    )
    parser.add_argument(
        '--skip-onnx',
        action='store_true',
        help='Skip ONNX export if it exists',
    )
    # End2End options
    parser.add_argument(
        '--end2end',
        action='store_true',
        help='Export with TensorRT EfficientNMS (GPU NMS). Note: No keypoints in output.',
    )
    parser.add_argument(
        '--max-det',
        type=int,
        default=100,
        help='Maximum detections per image (End2End only)',
    )
    parser.add_argument(
        '--iou-thres',
        type=float,
        default=0.4,
        help='NMS IoU threshold (End2End only)',
    )
    parser.add_argument(
        '--conf-thres',
        type=float,
        default=0.5,
        help='Confidence threshold (End2End only)',
    )
    args = parser.parse_args()

    result = export_yolo11_face(
        model_path=args.model,
        output_name=args.output_name,
        output_dir=args.output_dir,
        max_batch_size=args.max_batch,
        fp16=not args.no_fp16,
        skip_onnx=args.skip_onnx,
        end2end=args.end2end,
        max_det=args.max_det,
        iou_thres=args.iou_thres,
        conf_thres=args.conf_thres,
    )

    return 0 if result.get('success') else 1


if __name__ == '__main__':
    sys.exit(main())
