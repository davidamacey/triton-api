#!/usr/bin/env python3
"""
Export PP-OCRv5 English Recognition Model to TensorRT with Dynamic Width.

This script converts the PP-OCRv5 English recognition model (en_PP-OCRv5_mobile_rec)
to TensorRT for high-performance text recognition on GPU.

Model: PP-OCRv5 English Recognition (SVTR-LCNet architecture)
- Input: [B, 3, 48, W] text crop images (W: 48-2048, dynamic)
- Output: [B, T, 438] character sequence logits (T dynamic, 438 chars = 436 + blank + space)
- Preprocessing: (x / 255 - 0.5) / 0.5 = x / 127.5 - 1, BGR format

Dynamic shapes enable proper handling of variable-width text without distortion.

Run from host (uses Triton container for TensorRT conversion):
    python export/export_paddleocr_rec.py

Or from yolo-api container for ONNX export only:
    docker compose exec yolo-api python /app/export/export_paddleocr_rec.py --skip-tensorrt
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

# Model configuration
MODEL_NAME = 'en_PP-OCRv5_mobile_rec'
INPUT_NAME = 'x'
OUTPUT_NAME = 'fetch_name_0'

# Fixed height, dynamic width
REC_HEIGHT = 48
MIN_WIDTH = 48      # Minimum text crop width
OPT_WIDTH = 320     # Optimal width for TensorRT
MAX_WIDTH = 2048    # Maximum width for long text

# Dynamic output
NUM_CHARS = 438     # 436 English chars + blank + space

# Batch sizes (recognition processes many text crops per image)
MIN_BATCH = 1
OPT_BATCH = 32    # Higher optimal batch for text crops
MAX_BATCH = 64    # Match Triton max_batch_size

# Paths
PADDLEX_MODEL_DIR = Path.home() / '.paddlex/official_models' / MODEL_NAME
EXPORT_DIR = Path('/mnt/nvm/repos/triton-api/models/exports/ocr')
MODELS_DIR = Path('/mnt/nvm/repos/triton-api/models')

ONNX_PATH = EXPORT_DIR / 'en_ppocrv5_mobile_rec.onnx'
PLAN_OUTPUT = MODELS_DIR / 'paddleocr_rec_trt/1/model.plan'
DICT_OUTPUT = MODELS_DIR / 'paddleocr_rec_trt/en_ppocrv5_dict.txt'


def check_paddlex_model() -> bool:
    """Check if PaddleX model is downloaded."""
    required_files = ['inference.json', 'inference.pdiparams', 'inference.yml']
    if not PADDLEX_MODEL_DIR.exists():
        return False
    for f in required_files:
        if not (PADDLEX_MODEL_DIR / f).exists():
            return False
    return True


def export_to_onnx() -> Path | None:
    """Export PaddlePaddle model to ONNX format."""
    print('\n' + '=' * 60)
    print('Step 1: Export to ONNX')
    print('=' * 60)

    if ONNX_PATH.exists():
        print(f'ONNX model already exists: {ONNX_PATH}')
        print('  Delete it to re-export.')
        return ONNX_PATH

    try:
        import paddle2onnx

        print(f'Exporting {MODEL_NAME} to ONNX...')
        print(f'  Source: {PADDLEX_MODEL_DIR}')
        print(f'  Output: {ONNX_PATH}')

        EXPORT_DIR.mkdir(parents=True, exist_ok=True)

        paddle2onnx.export(
            model_filename=str(PADDLEX_MODEL_DIR / 'inference.json'),
            params_filename=str(PADDLEX_MODEL_DIR / 'inference.pdiparams'),
            save_file=str(ONNX_PATH),
            opset_version=14,
            enable_onnx_checker=True,
            deploy_backend='tensorrt',
        )

        print(f'  ONNX exported: {ONNX_PATH.stat().st_size / 1024 / 1024:.2f} MB')
        return ONNX_PATH

    except ImportError:
        print('ERROR: paddle2onnx not installed. Run: pip install paddle2onnx')
        return None
    except Exception as e:
        print(f'ERROR: ONNX export failed: {e}')
        import traceback
        traceback.print_exc()
        return None


def verify_onnx_model(onnx_path: Path) -> dict | None:
    """Verify ONNX model structure."""
    print('\n' + '=' * 60)
    print('Step 2: Verify ONNX Model')
    print('=' * 60)

    try:
        import onnx

        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)

        input_info = model.graph.input[0]
        input_name = input_info.name
        input_shape = [d.dim_param if d.dim_param else d.dim_value
                       for d in input_info.type.tensor_type.shape.dim]

        output_info = model.graph.output[0]
        output_name = output_info.name
        output_shape = [d.dim_param if d.dim_param else d.dim_value
                        for d in output_info.type.tensor_type.shape.dim]

        print(f'  Input: {input_name} {input_shape}')
        print(f'  Output: {output_name} {output_shape}')
        print(f'  Nodes: {len(model.graph.node)}')

        # Verify dynamic shapes
        has_dynamic = any(isinstance(d, str) for d in input_shape)
        print(f'  Dynamic shapes: {has_dynamic}')

        return {
            'input_name': input_name,
            'input_shape': input_shape,
            'output_name': output_name,
            'output_shape': output_shape,
            'has_dynamic': has_dynamic,
        }

    except Exception as e:
        print(f'ERROR: ONNX verification failed: {e}')
        return None


def test_onnx_inference(onnx_path: Path) -> bool:
    """Test ONNX model with different widths."""
    print('\n' + '=' * 60)
    print('Step 3: Test ONNX Inference')
    print('=' * 60)

    try:
        import onnxruntime as ort

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(str(onnx_path), providers=providers)
        print(f'  Provider: {session.get_providers()[0]}')

        # Test with different widths (dynamic)
        test_shapes = [
            (1, 3, REC_HEIGHT, MIN_WIDTH),    # Min width
            (1, 3, REC_HEIGHT, OPT_WIDTH),    # Optimal width
            (8, 3, REC_HEIGHT, OPT_WIDTH),    # Optimal batch
            (1, 3, REC_HEIGHT, 640),          # Medium width
            (4, 3, REC_HEIGHT, 1280),         # Large width
        ]

        for shape in test_shapes:
            try:
                test_input = np.random.randn(*shape).astype(np.float32)
                output = session.run(None, {INPUT_NAME: test_input})[0]
                print(f'  {shape} -> {output.shape} OK')
            except Exception as e:
                print(f'  {shape} FAILED: {e}')
                return False

        return True

    except Exception as e:
        print(f'ERROR: ONNX test failed: {e}')
        return False


def check_triton_container() -> bool:
    """Check if Triton container is running."""
    try:
        result = subprocess.run(
            ['docker', 'ps', '--filter', 'name=triton-api', '--format', '{{.Names}}'],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return 'triton-api' in result.stdout
    except Exception:
        return False


def unload_models_for_memory():
    """Unload Triton models to free GPU memory for TensorRT build."""
    print('\nFreeing GPU memory by unloading models...')

    models_to_unload = [
        'ocr_pipeline',
        'paddleocr_det_trt',
        'paddleocr_rec_trt',
        'yolov11_small_trt',
        'yolov11_small_trt_end2end',
        'mobileclip_image_trt',
        'mobileclip_text_trt',
    ]

    import requests
    for model in models_to_unload:
        try:
            requests.post(f'http://localhost:4600/v2/repository/models/{model}/unload', timeout=5)
        except Exception:
            pass

    import time
    time.sleep(2)
    print('  Models unloaded')


def convert_to_tensorrt_via_trtexec(onnx_path: Path, plan_path: Path) -> Path | None:
    """Convert ONNX to TensorRT using trtexec in Triton container."""
    print('\n' + '=' * 60)
    print('Step 4: Convert to TensorRT')
    print('=' * 60)

    # Check Triton container is running
    if not check_triton_container():
        print('ERROR: triton-api container is not running')
        print('  Start it with: docker compose up -d triton-api')
        return None

    # Unload models to free GPU memory
    unload_models_for_memory()

    # Copy ONNX to models dir for container access
    container_onnx = MODELS_DIR / onnx_path.name
    if not container_onnx.exists() or container_onnx.stat().st_size != onnx_path.stat().st_size:
        shutil.copy(onnx_path, container_onnx)
        print(f'Copied ONNX to: {container_onnx}')

    # Remove old plan file
    if plan_path.exists():
        plan_path.unlink()
        print(f'Removed old plan: {plan_path}')

    # Build trtexec command with dynamic shapes
    # IMPORTANT: Use --workspace=N format (MB) not --memPoolSize which requires different syntax
    min_shapes = f'{INPUT_NAME}:{MIN_BATCH}x3x{REC_HEIGHT}x{MIN_WIDTH}'
    opt_shapes = f'{INPUT_NAME}:{OPT_BATCH}x3x{REC_HEIGHT}x{OPT_WIDTH}'
    max_shapes = f'{INPUT_NAME}:{MAX_BATCH}x3x{REC_HEIGHT}x{MAX_WIDTH}'

    # Build command - write output to file inside container for reliable capture
    # NOTE: --workspace is deprecated in TRT 10+, use --memPoolSize=workspace:8192MiB instead
    trtexec_cmd = f'''
/usr/src/tensorrt/bin/trtexec \\
    --onnx=/models/{onnx_path.name} \\
    --saveEngine=/models/paddleocr_rec_trt/1/model.plan \\
    --minShapes={min_shapes} \\
    --optShapes={opt_shapes} \\
    --maxShapes={max_shapes} \\
    --fp16 \\
    --memPoolSize=workspace:8192M \\
    2>&1 | tee /tmp/trtexec.log

EXIT_CODE=${{PIPESTATUS[0]}}
echo ""
echo "=== BUILD RESULT ==="
echo "Exit code: $EXIT_CODE"
if [ -f /models/paddleocr_rec_trt/1/model.plan ]; then
    ls -lh /models/paddleocr_rec_trt/1/model.plan
else
    echo "ERROR: model.plan not created"
fi
exit $EXIT_CODE
'''

    cmd = ['docker', 'exec', 'triton-api', 'bash', '-c', trtexec_cmd]

    print(f'Running trtexec with dynamic shapes:')
    print(f'  Min: {min_shapes}')
    print(f'  Opt: {opt_shapes}')
    print(f'  Max: {max_shapes}')
    print(f'  Memory Pool: 8192 MiB')
    print(f'\nThis may take 10-20 minutes for dynamic shapes...')
    print('  (Output saved to /tmp/trtexec.log in container)')
    print('')

    try:
        # Run with live output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        # Stream output
        last_lines = []
        for line in process.stdout:
            print(line, end='')
            last_lines.append(line)
            if len(last_lines) > 100:
                last_lines.pop(0)

        process.wait(timeout=1800)

        if process.returncode == 0 and plan_path.exists() and plan_path.stat().st_size > 0:
            print('\nTensorRT conversion successful!')
            print(f'  Engine size: {plan_path.stat().st_size / 1024 / 1024:.2f} MB')
            return plan_path
        else:
            print(f'\nERROR: trtexec failed (exit code {process.returncode})')
            if plan_path.exists():
                print(f'  Plan file size: {plan_path.stat().st_size} bytes (0 = failed)')
            return None

    except subprocess.TimeoutExpired:
        print('ERROR: TensorRT conversion timed out after 30 minutes')
        process.kill()
        return None
    except Exception as e:
        print(f'ERROR: TensorRT conversion failed: {e}')
        import traceback
        traceback.print_exc()
        return None


def extract_dictionary() -> Path | None:
    """Extract English dictionary from PaddleX model config."""
    print('\n' + '=' * 60)
    print('Step 5: Extract Dictionary')
    print('=' * 60)

    try:
        import yaml

        config_path = PADDLEX_MODEL_DIR / 'inference.yml'
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        char_dict = config.get('PostProcess', {}).get('character_dict', [])
        if not char_dict:
            print('ERROR: No character_dict found in config')
            return None

        # Write dictionary (one char per line)
        DICT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        with open(DICT_OUTPUT, 'w', encoding='utf-8') as f:
            for char in char_dict:
                f.write(f'{char}\n')

        print(f'  Dictionary: {DICT_OUTPUT}')
        print(f'  Characters: {len(char_dict)}')
        print(f'  Sample: {char_dict[:10]}...')

        return DICT_OUTPUT

    except Exception as e:
        print(f'ERROR: Dictionary extraction failed: {e}')
        return None


def create_triton_config(plan_path: Path):
    """Create Triton config.pbtxt with dynamic dimensions."""
    print('\n' + '=' * 60)
    print('Step 6: Create Triton Config')
    print('=' * 60)

    config_path = plan_path.parent.parent / 'config.pbtxt'

    config_content = f"""# PP-OCRv5 English Text Recognition Model (TensorRT)
#
# SVTR-LCNet architecture for text sequence recognition
# Supports dynamic width for variable-length text recognition
#
# Input:
#   - x: [B, 3, 48, W] FP32, preprocessed (x / 127.5 - 1), BGR
#        Text crops with height=48, width={MIN_WIDTH}-{MAX_WIDTH}
#
# Output:
#   - fetch_name_0: [B, T, {NUM_CHARS}] FP32, character probabilities
#        T timesteps (dynamic), {NUM_CHARS} character classes

name: "paddleocr_rec_trt"
platform: "tensorrt_plan"
max_batch_size: {MAX_BATCH}

input [
  {{
    name: "{INPUT_NAME}"
    data_type: TYPE_FP32
    dims: [ 3, {REC_HEIGHT}, -1 ]
  }}
]

output [
  {{
    name: "{OUTPUT_NAME}"
    data_type: TYPE_FP32
    dims: [ -1, {NUM_CHARS} ]
  }}
]

instance_group [
  {{
    count: 2
    kind: KIND_GPU
    gpus: [0]
  }}
]

# Higher queue delay for recognition to accumulate text crops from multiple images
dynamic_batching {{
  preferred_batch_size: [ 16, 32, 64 ]
  max_queue_delay_microseconds: 10000
}}
"""

    with open(config_path, 'w') as f:
        f.write(config_content)

    print(f'  Config: {config_path}')
    print(f'  Dynamic width: {MIN_WIDTH}-{MAX_WIDTH}')
    print(f'  Max batch: {MAX_BATCH}')


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Export PP-OCRv5 English Recognition to TensorRT')
    parser.add_argument(
        '--skip-tensorrt',
        action='store_true',
        help='Skip TensorRT conversion (ONNX export only)',
    )
    parser.add_argument(
        '--force-onnx',
        action='store_true',
        help='Force re-export of ONNX model',
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    print('=' * 70)
    print('PP-OCRv5 English Recognition Model Export')
    print('  Model: en_PP-OCRv5_mobile_rec')
    print(f'  Dynamic width: {MIN_WIDTH} - {MAX_WIDTH}')
    print(f'  Characters: {NUM_CHARS} (436 + blank + space)')
    print('=' * 70)

    # Check PaddleX model
    if not check_paddlex_model():
        print(f'\nERROR: PaddleX model not found: {PADDLEX_MODEL_DIR}')
        print('\nTo download, run PaddleOCR:')
        print('  from paddleocr import PaddleOCR')
        print('  ocr = PaddleOCR(lang="en")')
        return 1

    # Force re-export if requested
    if args.force_onnx and ONNX_PATH.exists():
        ONNX_PATH.unlink()
        print(f'Removed existing ONNX: {ONNX_PATH}')

    # Step 1: Export to ONNX
    onnx_path = export_to_onnx()
    if not onnx_path:
        return 1

    # Step 2: Verify ONNX
    model_info = verify_onnx_model(onnx_path)
    if not model_info:
        return 1

    # Step 3: Test ONNX
    if not test_onnx_inference(onnx_path):
        return 1

    # Step 4: Convert to TensorRT
    plan_path = None
    if not args.skip_tensorrt:
        plan_path = convert_to_tensorrt_via_trtexec(onnx_path, PLAN_OUTPUT)
        if not plan_path:
            print('\nWARNING: TensorRT conversion failed')
            print('  You can retry manually or check GPU memory')

    # Step 5: Extract dictionary
    dict_path = extract_dictionary()

    # Step 6: Create Triton config
    if plan_path:
        create_triton_config(plan_path)

    # Summary
    print('\n' + '=' * 70)
    print('Export Complete!')
    print('=' * 70)
    print('\nOutputs:')
    print(f'  ONNX:       {onnx_path}')
    if plan_path:
        print(f'  TensorRT:   {plan_path}')
    if dict_path:
        print(f'  Dictionary: {dict_path}')

    print('\nModel specifications:')
    print(f'  - Input: x [B, 3, 48, W] FP32 (W: {MIN_WIDTH}-{MAX_WIDTH})')
    print(f'  - Output: fetch_name_0 [B, T, {NUM_CHARS}] FP32')
    print('  - Preprocessing: (x / 127.5) - 1, BGR format')
    print(f'  - Max batch size: {MAX_BATCH}')

    if plan_path:
        print('\nNext steps:')
        print('  1. Reload Triton models:')
        print('     curl -X POST localhost:4600/v2/repository/models/paddleocr_rec_trt/load')
        print('  2. Update BLS model to use new dictionary')
        print('  3. Test OCR: curl -X POST http://localhost:4603/track_e/ocr/predict -F "image=@test.png"')

    return 0


if __name__ == '__main__':
    sys.exit(main())
