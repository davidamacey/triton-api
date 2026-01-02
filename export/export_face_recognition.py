#!/usr/bin/env python3
"""
Track E: Export ArcFace Face Recognition to TensorRT

This script converts the ArcFace face recognition model to TensorRT for Triton deployment.

Model: ArcFace w600k_r50 (WebFace600K trained ResNet-50)
- Input: [B, 3, 112, 112] FP32, RGB, normalized (x-127.5)/128
- Output: [B, 512] FP32, L2-normalized embeddings
- LFW Accuracy: 99.8%

Key steps:
1. Validate ONNX model structure
2. Test inference with ONNX Runtime
3. Convert to TensorRT FP16 with dynamic batch
4. Validate TensorRT output

Usage:
    # From host with venv:
    python export/export_face_recognition.py

    # From yolo-api container:
    docker compose exec yolo-api python /app/export/export_face_recognition.py
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np


# =============================================================================
# Configuration
# =============================================================================

# ArcFace configuration
INPUT_SIZE = 112  # Standard ArcFace aligned face size
EMBEDDING_DIM = 512  # Output embedding dimension
MAX_BATCH_SIZE = 128  # Max batch for TensorRT (faces per image)

# Paths
ONNX_PATH = Path('pytorch_models/arcface_w600k_r50.onnx')
PLAN_PATH = Path('models/arcface_w600k_r50/1/model.plan')

# TensorRT settings
FP16_MODE = True
WORKSPACE_GB = 4


# =============================================================================
# ONNX Preprocessing
# =============================================================================


def make_batch_dynamic(onnx_path: Path, output_path: Path | None = None) -> Path:
    """
    Convert ArcFace ONNX model to have dynamic batch dimension if needed.

    Args:
        onnx_path: Path to original ONNX model
        output_path: Path for output (default: adds _dynamic suffix)

    Returns:
        Path to dynamic batch ONNX model
    """
    import onnx
    from onnx import TensorProto, helper

    print('\nChecking batch dimension...')
    print(f'  Input: {onnx_path}')

    model = onnx.load(str(onnx_path))

    # Get current input shape
    input_tensor = model.graph.input[0]
    old_shape = [
        d.dim_value if d.dim_value > 0 else d.dim_param
        for d in input_tensor.type.tensor_type.shape.dim
    ]
    print(f'  Original input shape: {old_shape}')

    # Check if already dynamic
    if isinstance(old_shape[0], str) or old_shape[0] == 0:
        print('  ✓ Already has dynamic batch dimension')
        return onnx_path

    # Create new input with dynamic batch
    new_input = helper.make_tensor_value_info(
        input_tensor.name,
        TensorProto.FLOAT,
        ['batch', 3, INPUT_SIZE, INPUT_SIZE],
    )

    # Replace input
    model.graph.input.remove(input_tensor)
    model.graph.input.insert(0, new_input)

    # Update output to have dynamic batch
    output_tensor = model.graph.output[0]
    new_output = helper.make_tensor_value_info(
        output_tensor.name,
        TensorProto.FLOAT,
        ['batch', EMBEDDING_DIM],
    )
    model.graph.output.remove(output_tensor)
    model.graph.output.insert(0, new_output)

    # Save modified model
    if output_path is None:
        output_path = onnx_path.parent / f'{onnx_path.stem}_dynamic.onnx'

    onnx.save(model, str(output_path))
    print(f'  ✓ Dynamic batch model saved: {output_path}')

    return output_path


# =============================================================================
# ONNX Validation
# =============================================================================


def analyze_onnx_model(onnx_path: Path) -> dict:
    """
    Analyze ArcFace ONNX model structure.

    Returns model info including inputs, outputs, and shapes.
    """
    print(f'\nAnalyzing ONNX model: {onnx_path}')

    import onnx

    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)

    print(f'  ✓ Model valid: {len(model.graph.node)} nodes')
    print(f'  ✓ IR version: {model.ir_version}')
    print(f'  ✓ Opset version: {model.opset_import[0].version}')

    # Get input info
    for inp in model.graph.input:
        shape = [
            d.dim_value if d.dim_value > 0 else d.dim_param for d in inp.type.tensor_type.shape.dim
        ]
        print(f'  Input: {inp.name} {shape}')

    # Get output info
    for out in model.graph.output:
        shape = [
            d.dim_value if d.dim_value > 0 else d.dim_param for d in out.type.tensor_type.shape.dim
        ]
        print(f'  Output: {out.name} {shape}')

    return {
        'num_nodes': len(model.graph.node),
    }


def test_onnx_inference(onnx_path: Path, batch_size: int = 1) -> dict:
    """
    Test ArcFace inference with ONNX Runtime.

    Returns inference results for validation.
    """
    print(f'\nTesting ONNX inference (batch_size={batch_size})...')

    import onnxruntime as ort

    # Create session
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    print(f'  ✓ Using provider: {session.get_providers()[0]}')

    # Get input name
    input_name = session.get_inputs()[0].name
    print(f'  Input name: {input_name}')

    # Create test input (aligned face, normalized)
    # ArcFace expects (x - 127.5) / 128.0 normalization
    test_input = np.random.rand(batch_size, 3, INPUT_SIZE, INPUT_SIZE).astype(np.float32)
    test_input = (test_input * 255 - 127.5) / 128.0  # Simulate normalized input

    # Run inference
    start = time.time()
    outputs = session.run(None, {input_name: test_input})
    inference_time = (time.time() - start) * 1000

    embeddings = outputs[0]
    print(f'  ✓ Inference time: {inference_time:.2f}ms')
    print(f'  Output shape: {embeddings.shape}')

    # Check L2 normalization
    norms = np.linalg.norm(embeddings, axis=-1)
    print(
        f'  Embedding norms: min={norms.min():.4f}, max={norms.max():.4f}, mean={norms.mean():.4f}'
    )

    # Verify normalized (should be ~1.0)
    if np.allclose(norms, 1.0, atol=0.01):
        print('  ✓ Embeddings are L2-normalized')
    else:
        print('  ⚠ Embeddings may not be L2-normalized')

    return {
        'input_name': input_name,
        'embeddings': embeddings,
        'inference_time_ms': inference_time,
    }


def benchmark_onnx(onnx_path: Path, num_iterations: int = 100):
    """Benchmark ONNX inference speed."""

    print(f'\nBenchmarking ONNX inference ({num_iterations} iterations)...')

    import onnxruntime as ort

    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    input_name = session.get_inputs()[0].name

    for batch_size in [1, 4, 8, 16, 32]:
        test_input = np.random.rand(batch_size, 3, INPUT_SIZE, INPUT_SIZE).astype(np.float32)
        test_input = (test_input * 255 - 127.5) / 128.0

        # Warmup
        for _ in range(10):
            session.run(None, {input_name: test_input})

        # Benchmark
        latencies = []
        for _ in range(num_iterations):
            start = time.time()
            session.run(None, {input_name: test_input})
            latencies.append((time.time() - start) * 1000)

        mean_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        per_face = mean_latency / batch_size

        print(
            f'  Batch size {batch_size:2d}: {mean_latency:.2f}ms total, '
            f'{per_face:.2f}ms/face (p95: {p95_latency:.2f}ms)'
        )


# =============================================================================
# TensorRT Conversion
# =============================================================================


def convert_to_tensorrt(
    onnx_path: Path,
    plan_path: Path,
    fp16: bool = True,
    max_batch_size: int = 128,
) -> bool:
    """
    Convert ArcFace ONNX model to TensorRT engine.

    Args:
        onnx_path: Path to ONNX model
        plan_path: Output path for TensorRT plan
        fp16: Use FP16 precision
        max_batch_size: Maximum batch size for dynamic batching

    Returns:
        True if conversion successful
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

        print(f'  ✓ ONNX parsed successfully ({network.num_layers} layers)')

        # Builder config
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE_GB << 30)

        if fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print('  ✓ FP16 mode enabled')

        # Get input tensor name from network
        input_tensor = network.get_input(0)
        input_name = input_tensor.name
        print(f'  Input tensor: {input_name}')

        # Optimization profile for dynamic batch sizes
        profile = builder.create_optimization_profile()
        profile.set_shape(
            input_name,
            min=(1, 3, INPUT_SIZE, INPUT_SIZE),
            opt=(16, 3, INPUT_SIZE, INPUT_SIZE),  # Optimal: 16 faces
            max=(max_batch_size, 3, INPUT_SIZE, INPUT_SIZE),
        )
        config.add_optimization_profile(profile)

        print('  ✓ Optimization profile configured')
        print('    - Min batch: 1')
        print('    - Optimal batch: 16')
        print(f'    - Max batch: {max_batch_size}')

        # Build engine
        print('\n  Building TensorRT engine (this may take 5-10 minutes)...')
        serialized_engine = builder.build_serialized_network(network, config)

        if serialized_engine is None:
            raise RuntimeError('Failed to build TensorRT engine')

        # Save engine
        plan_path = Path(plan_path)
        plan_path.parent.mkdir(parents=True, exist_ok=True)

        with open(plan_path, 'wb') as f:
            f.write(serialized_engine)

        size_mb = plan_path.stat().st_size / (1024 * 1024)
        print(f'  ✓ TensorRT engine saved: {plan_path} ({size_mb:.1f} MB)')

        return True

    except ImportError:
        print('  ✗ TensorRT not available')
        return False
    except Exception as e:
        print(f'  ✗ TensorRT conversion failed: {e}')
        return False


# =============================================================================
# Triton Config
# =============================================================================


def create_triton_config(plan_path: Path) -> Path:
    """
    Create Triton config.pbtxt for ArcFace model.
    """
    config_path = plan_path.parent.parent / 'config.pbtxt'

    config_content = f"""name: "arcface_w600k_r50"
platform: "tensorrt_plan"
max_batch_size: {MAX_BATCH_SIZE}

input {{
    name: "input.1"
    data_type: TYPE_FP32
    dims: [3, {INPUT_SIZE}, {INPUT_SIZE}]
}}

output {{
    name: "683"
    data_type: TYPE_FP32
    dims: [{EMBEDDING_DIM}]
}}

dynamic_batching {{
    preferred_batch_size: [8, 16, 32, 64]
    max_queue_delay_microseconds: 15000
}}

instance_group [{{
    count: 4
    kind: KIND_GPU
    gpus: [0]
}}]

version_policy {{
    latest {{
        num_versions: 1
    }}
}}
"""

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(config_content)
    print(f'\n✓ Triton config written: {config_path}')

    return config_path


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description='Export ArcFace to TensorRT')
    parser.add_argument('--onnx', type=Path, default=ONNX_PATH, help='Input ONNX path')
    parser.add_argument('--plan', type=Path, default=PLAN_PATH, help='Output TensorRT plan path')
    parser.add_argument('--no-fp16', action='store_true', help='Disable FP16 mode')
    parser.add_argument('--max-batch', type=int, default=MAX_BATCH_SIZE, help='Max batch size')
    parser.add_argument('--benchmark', action='store_true', help='Run ONNX benchmark')
    parser.add_argument('--skip-trt', action='store_true', help='Skip TensorRT conversion')
    args = parser.parse_args()

    print('=' * 60)
    print('ArcFace Face Recognition Export')
    print('=' * 60)

    # Check ONNX exists
    if not args.onnx.exists():
        print(f'\n✗ ONNX model not found: {args.onnx}')
        print('  Run: python export/download_face_models.py')
        return 1

    # Step 1: Analyze ONNX
    analyze_onnx_model(args.onnx)

    # Step 2: Test ONNX inference
    test_onnx_inference(args.onnx, batch_size=1)

    # Step 3: Optional benchmark
    if args.benchmark:
        benchmark_onnx(args.onnx)

    # Step 4: Make batch dimension dynamic if needed
    dynamic_onnx = make_batch_dynamic(args.onnx)

    # Step 5: Convert to TensorRT
    if not args.skip_trt:
        success = convert_to_tensorrt(
            dynamic_onnx,
            args.plan,
            fp16=not args.no_fp16,
            max_batch_size=args.max_batch,
        )

        if not success:
            print('\n✗ TensorRT conversion failed')
            return 1

        # Step 6: Create Triton config
        create_triton_config(args.plan)

    print('\n' + '=' * 60)
    print('ArcFace Export Complete!')
    print('=' * 60)
    print(f'  ONNX: {args.onnx}')
    if not args.skip_trt:
        print(f'  TensorRT: {args.plan}')
    print('\nNext steps:')
    print('  1. Restart Triton: docker compose restart triton-api')
    print('  2. Test inference via API')

    return 0


if __name__ == '__main__':
    sys.exit(main())
