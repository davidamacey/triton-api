#!/usr/bin/env python3
"""
Export PP-OCRv5 Detection Model to TensorRT.

This script converts the PP-OCRv5 detection ONNX model to TensorRT for
high-performance text detection on GPU.

Model: PP-OCRv5 Detection (DB++ architecture)
- Input: [B, 3, H, W] where H,W are multiples of 32 (max 960)
- Output: [B, 1, H, W] probability map for text regions
- Preprocessing: (x / 255 - 0.5) / 0.5 = x / 127.5 - 1, BGR format

Run from: yolo-api container
    docker compose exec yolo-api python /app/export/export_paddleocr_det.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np


# Model configuration
INPUT_NAME = 'x'
OUTPUT_NAME = 'fetch_name_0'  # PP-OCRv5 output name
MAX_RESOLUTION = 960
MIN_RESOLUTION = 32
OPTIMAL_RESOLUTION = 736

# Output paths
ONNX_PATH = Path('/app/pytorch_models/paddleocr/ppocr_det_v5_mobile.onnx')
PLAN_OUTPUT = Path('/app/models/paddleocr_det_trt/1/model.plan')


def verify_onnx_model(onnx_path: Path) -> dict:
    """Verify ONNX model and return input/output info."""
    import onnx

    print(f'\nVerifying ONNX model: {onnx_path}')

    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)

    # Get input info
    input_info = model.graph.input[0]
    input_name = input_info.name
    input_shape = [d.dim_value if d.dim_value else d.dim_param for d in input_info.type.tensor_type.shape.dim]

    # Get output info
    output_info = model.graph.output[0]
    output_name = output_info.name
    output_shape = [d.dim_value if d.dim_value else d.dim_param for d in output_info.type.tensor_type.shape.dim]

    print(f'  Input: {input_name} {input_shape}')
    print(f'  Output: {output_name} {output_shape}')
    print(f'  Nodes: {len(model.graph.node)}')

    return {
        'input_name': input_name,
        'input_shape': input_shape,
        'output_name': output_name,
        'output_shape': output_shape,
    }


def test_onnx_inference(onnx_path: Path, input_name: str) -> bool:
    """Test ONNX model inference."""
    import onnxruntime as ort

    print('\nTesting ONNX inference...')

    # Create session
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    print(f'  Provider: {session.get_providers()[0]}')

    # Test with different resolutions
    test_shapes = [
        (1, 3, 736, 736),   # Optimal
        (1, 3, 960, 960),   # Max
        (1, 3, 640, 480),   # Common aspect ratio
        (2, 3, 736, 736),   # Batch of 2
    ]

    for shape in test_shapes:
        try:
            test_input = np.random.randn(*shape).astype(np.float32)
            output = session.run(None, {input_name: test_input})[0]
            print(f'  {shape} -> {output.shape} OK')
        except Exception as e:
            print(f'  {shape} FAILED: {e}')
            return False

    return True


def convert_to_tensorrt(
    onnx_path: Path,
    plan_path: Path,
    fp16: bool = True,
    max_batch_size: int = 4,
) -> Path | None:
    """
    Convert ONNX model to TensorRT engine.

    Args:
        onnx_path: Path to ONNX model
        plan_path: Output path for TensorRT plan
        fp16: Use FP16 precision
        max_batch_size: Maximum batch size
    """
    print('\nConverting to TensorRT engine...')
    print(f'  Input: {onnx_path}')
    print(f'  Output: {plan_path}')
    print(f'  FP16: {fp16}')
    print(f'  Max batch size: {max_batch_size}')

    try:
        import tensorrt as trt

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

        # Create builder
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, TRT_LOGGER)

        # Parse ONNX
        print('  Parsing ONNX model...')
        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(f'    Error {i}: {parser.get_error(i)}')
                raise RuntimeError('Failed to parse ONNX model')

        print('  ONNX parsed successfully')

        # Builder config
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4GB workspace

        if fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print('  FP16 mode enabled')

        # Optimization profile for dynamic shapes
        # Detection model has dynamic H,W (multiples of 32)
        profile = builder.create_optimization_profile()

        # Min: small image
        # Opt: typical resolution
        # Max: maximum resolution
        profile.set_shape(
            INPUT_NAME,
            min=(1, 3, MIN_RESOLUTION, MIN_RESOLUTION),
            opt=(1, 3, OPTIMAL_RESOLUTION, OPTIMAL_RESOLUTION),
            max=(max_batch_size, 3, MAX_RESOLUTION, MAX_RESOLUTION),
        )
        config.add_optimization_profile(profile)

        print('  Optimization profile configured')
        print(f'    - Min: (1, 3, {MIN_RESOLUTION}, {MIN_RESOLUTION})')
        print(f'    - Opt: (1, 3, {OPTIMAL_RESOLUTION}, {OPTIMAL_RESOLUTION})')
        print(f'    - Max: ({max_batch_size}, 3, {MAX_RESOLUTION}, {MAX_RESOLUTION})')

        # Build engine (this takes a few minutes)
        print('\n  Building TensorRT engine (this may take 2-5 minutes)...')
        serialized_engine = builder.build_serialized_network(network, config)

        if serialized_engine is None:
            raise RuntimeError('Failed to build TensorRT engine')

        # Save engine
        plan_path.parent.mkdir(parents=True, exist_ok=True)

        with open(plan_path, 'wb') as f:
            f.write(serialized_engine)

        print('\n  TensorRT engine saved!')
        print(f'    File size: {plan_path.stat().st_size / (1024 * 1024):.2f} MB')

        return plan_path

    except ImportError:
        print('\n  TensorRT not available in this container')
        print('  To convert to TensorRT, use the Triton container:')
        print('    docker compose exec triton-api trtexec \\')
        print(f'      --onnx={onnx_path} \\')
        print(f'      --saveEngine={plan_path} \\')
        print('      --fp16 \\')
        print(f'      --minShapes={INPUT_NAME}:1x3x{MIN_RESOLUTION}x{MIN_RESOLUTION} \\')
        print(f'      --optShapes={INPUT_NAME}:1x3x{OPTIMAL_RESOLUTION}x{OPTIMAL_RESOLUTION} \\')
        print(f'      --maxShapes={INPUT_NAME}:{max_batch_size}x3x{MAX_RESOLUTION}x{MAX_RESOLUTION}')
        return None

    except Exception as e:
        print(f'\n  TensorRT conversion failed: {e}')
        import traceback

        traceback.print_exc()
        return None


def create_triton_config(model_dir: Path, max_batch_size: int = 4):
    """Create Triton config.pbtxt for the detection model."""
    config_path = model_dir / 'config.pbtxt'

    config_content = f"""# PP-OCRv5 Text Detection Model (TensorRT)
#
# DB++ architecture for text region detection
#
# Input:
#   - x: [B, 3, H, W] FP32, preprocessed (x / 127.5 - 1), BGR
#        H,W must be multiples of 32, max {MAX_RESOLUTION}
#
# Output:
#   - fetch_name_0: [B, 1, H, W] FP32, probability map [0, 1]

name: "paddleocr_det_trt"
platform: "tensorrt_plan"
max_batch_size: {max_batch_size}

input [
  {{
    name: "x"
    data_type: TYPE_FP32
    dims: [ 3, -1, -1 ]  # Dynamic H, W (multiples of 32)
  }}
]

output [
  {{
    name: "fetch_name_0"
    data_type: TYPE_FP32
    dims: [ 1, -1, -1 ]  # Same H, W as input
  }}
]

instance_group [
  {{
    count: 2
    kind: KIND_GPU
    gpus: [0]
  }}
]

dynamic_batching {{
  preferred_batch_size: [ 1, 2, 4 ]
  max_queue_delay_microseconds: 5000
}}
"""

    with open(config_path, 'w') as f:
        f.write(config_content)

    print(f'\nTriton config created: {config_path}')


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Export PP-OCRv5 Detection to TensorRT')
    parser.add_argument(
        '--onnx-path',
        type=Path,
        default=ONNX_PATH,
        help='Path to ONNX model',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=PLAN_OUTPUT,
        help='Output path for TensorRT plan',
    )
    parser.add_argument('--fp32', action='store_true', help='Use FP32 instead of FP16')
    parser.add_argument(
        '--max-batch-size',
        type=int,
        default=4,
        help='Maximum batch size for TensorRT engine',
    )
    parser.add_argument('--skip-tensorrt', action='store_true', help='Skip TensorRT conversion')
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    print('=' * 70)
    print('PP-OCRv5 Detection Model Export')
    print('=' * 70)

    # Check ONNX exists
    if not args.onnx_path.exists():
        print(f'\nERROR: ONNX model not found: {args.onnx_path}')
        print('\nTo download, run:')
        print('  docker compose exec yolo-api python /app/export/download_paddleocr.py')
        return 1

    # Verify ONNX
    model_info = verify_onnx_model(args.onnx_path)

    # Test ONNX inference
    if not test_onnx_inference(args.onnx_path, model_info['input_name']):
        print('\nERROR: ONNX inference test failed')
        return 1

    # Convert to TensorRT
    plan_path = None
    if not args.skip_tensorrt:
        plan_path = convert_to_tensorrt(
            args.onnx_path,
            args.output,
            fp16=not args.fp32,
            max_batch_size=args.max_batch_size,
        )

    # Create Triton config
    if plan_path:
        create_triton_config(plan_path.parent.parent, args.max_batch_size)

    # Summary
    print('\n' + '=' * 70)
    print('Export Complete!')
    print('=' * 70)
    print('\nOutputs:')
    print(f'  ONNX:     {args.onnx_path}')
    if plan_path:
        print(f'  TensorRT: {plan_path}')

    print('\nModel specifications:')
    print(f'  - Input: x [B, 3, H, W] FP32, H,W multiples of 32, max {MAX_RESOLUTION}')
    print('  - Output: sigmoid_0.tmp_0 [B, 1, H, W] FP32, probability map')
    print('  - Preprocessing: (x / 127.5) - 1, BGR format')

    if plan_path:
        print('\nNext steps:')
        print('  1. Export recognition: python export/export_paddleocr_rec.py')
        print('  2. Restart Triton: docker compose restart triton-api')

    return 0


if __name__ == '__main__':
    sys.exit(main())
