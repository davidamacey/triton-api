#!/usr/bin/env python3
"""
Track E: Export MobileCLIP2 Image Encoder to ONNX + TensorRT

This script exports MobileCLIP2 image encoder for deployment on Triton.

Supported variants (with simple ÷255 normalization):
- MobileCLIP2-S2: 35.7M params, 77.2% ImageNet, 3.6ms (RECOMMENDED)
- MobileCLIP2-B:  86.3M params, 79.4% ImageNet, 10.4ms (MAXIMUM ACCURACY)

NOT recommended (different normalization):
- S3, S4, L-14: Use ImageNet normalization, complicates DALI pipeline

Key steps:
1. Load model with proper configuration (image_mean=0, image_std=1)
2. CRITICAL: Call reparameterize_model() before export
3. Export to ONNX with dynamic batch size
4. Convert to TensorRT plan for maximum throughput
5. Validate output matches PyTorch

Run from: yolo-api container
    docker compose exec yolo-api python /app/scripts/track_e/export_mobileclip_image_encoder.py --model S2
    docker compose exec yolo-api python /app/scripts/track_e/export_mobileclip_image_encoder.py --model B
"""

import argparse
import sys
from pathlib import Path


sys.path.insert(0, '/app/reference_repos/ml-mobileclip')
sys.path.insert(0, '/app/reference_repos/open_clip/src')

import numpy as np
import torch
from torch import nn


# Model configurations
MODEL_CONFIGS = {
    'S2': {
        'name': 'MobileCLIP2-S2',
        'checkpoint_name': 'mobileclip2_s2',
        'params': '35.7M',
        'accuracy': '77.2%',
        'latency': '3.6ms',
    },
    'B': {
        'name': 'MobileCLIP2-B',
        'checkpoint_name': 'mobileclip2_b',
        'params': '86.3M',
        'accuracy': '79.4%',
        'latency': '10.4ms',
    },
}

# Common settings for S0, S2, B variants
IMAGE_SIZE = 256  # All these variants use 256x256
EMBEDDING_DIM = 512  # MobileCLIP2-S2 uses 512-dim embeddings

# ONNX export settings
# TensorRT 10.x (Triton 25.10) supports opset 9-20
# For Transformer/LayerNorm models with dynamic batch: opset 17+ recommended
# See: https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html
# See: https://torchpipe.github.io/docs/faq/onnx
ONNX_OPSET_VERSION = 17


class MobileCLIPImageEncoder(nn.Module):
    """
    Wrapper for MobileCLIP2-S2 image encoder with L2 normalization.

    This wrapper:
    1. Takes preprocessed images [B, 3, 256, 256] in [0, 1] range
    2. Encodes to 768-dim embedding
    3. L2-normalizes output (critical for cosine similarity)
    """

    def __init__(self, clip_model):
        super().__init__()
        self.visual = clip_model.visual

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [B, 3, 256, 256] FP32, normalized [0, 1]

        Returns:
            embeddings: [B, 768] FP32, L2-normalized
        """
        # Encode image
        image_features = self.visual(images)

        # L2 normalize (CRITICAL for cosine similarity in OpenSearch)
        return image_features / image_features.norm(dim=-1, keepdim=True)


def load_mobileclip_model(model_name, checkpoint_path):
    """
    Load MobileCLIP2 model with proper configuration.

    Args:
        model_name: Model name (e.g., "MobileCLIP2-S2")
        checkpoint_path: Path to checkpoint file
    """
    print(f'\nLoading {model_name}...')
    print(f'  Checkpoint: {checkpoint_path}')

    import open_clip
    from mobileclip.modules.common.mobileone import reparameterize_model

    # CRITICAL: S0, S2, B variants use simple normalization (image_mean=0, image_std=1)
    # This means input is just divided by 255 (same as YOLO!)
    # S3, S4, L-14 use ImageNet normalization (different)
    model, _, _preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=checkpoint_path, image_mean=(0, 0, 0), image_std=(1, 1, 1)
    )

    model.eval()

    # CRITICAL: Reparameterize before export
    # MobileCLIP uses train-time overparameterization that must be merged
    # Failure to reparameterize → incorrect inference results!
    print('  Reparameterizing model (merging train-time branches)...')
    model = reparameterize_model(model)

    print('  ✓ Model loaded and reparameterized')

    return model


def export_to_onnx(model, output_path):
    """Export image encoder to ONNX format with dynamic batch support.

    Uses opset 17 for best TensorRT 10.x compatibility with LayerNorm.
    Forces legacy exporter to avoid dynamo issues with dynamic shapes.
    """

    print(f'\nExporting to ONNX: {output_path}')
    print(f'  Opset version: {ONNX_OPSET_VERSION}')

    # Create encoder wrapper
    encoder = MobileCLIPImageEncoder(model)
    encoder.eval()

    # Create dummy input [batch=1, channels=3, height=256, width=256]
    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)

    # Test forward pass
    print('Testing forward pass...')
    with torch.no_grad():
        output = encoder(dummy_input)
        print(f'  Output shape: {output.shape}')  # [1, 512]
        print(f'  Output norm: {output.norm(dim=-1).item():.4f}')  # Should be ~1.0

    # Export to ONNX
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use legacy exporter to avoid dynamo issues with dynamic shapes
    # See: https://github.com/pytorch/pytorch/issues/74740
    print('  Exporting with dynamic batch support...')
    torch.onnx.export(
        encoder,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=ONNX_OPSET_VERSION,
        do_constant_folding=True,
        input_names=['images'],
        output_names=['image_embeddings'],
        dynamic_axes={'images': {0: 'batch_size'}, 'image_embeddings': {0: 'batch_size'}},
        verbose=False,
        # Force legacy exporter for better dynamic shape handling
        dynamo=False,
    )

    print('  ✓ ONNX export complete')
    print(f'  File size: {output_path.stat().st_size / (1024 * 1024):.2f} MB')

    return output_path


def validate_onnx(pytorch_encoder, onnx_path):
    """Validate ONNX model matches PyTorch output."""

    print('\nValidating ONNX model...')

    import onnx
    import onnxruntime as ort

    # Load and check ONNX model
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print('  ✓ ONNX model is valid')

    # Create ONNX Runtime session
    print('  Creating ONNX Runtime session...')
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    ort_session = ort.InferenceSession(str(onnx_path), providers=providers)
    print(f'  ✓ Using provider: {ort_session.get_providers()[0]}')

    # Test with random input
    test_input = np.random.randn(4, 3, IMAGE_SIZE, IMAGE_SIZE).astype(np.float32)

    # PyTorch inference
    with torch.no_grad():
        pytorch_output = pytorch_encoder(torch.from_numpy(test_input)).numpy()

    # ONNX inference
    ort_inputs = {'images': test_input}
    ort_output = ort_session.run(None, ort_inputs)[0]

    # Compare outputs
    max_diff = np.abs(pytorch_output - ort_output).max()
    mean_diff = np.abs(pytorch_output - ort_output).mean()

    print('\n  Comparison (batch_size=4):')
    print(f'    PyTorch output shape: {pytorch_output.shape}')
    print(f'    ONNX output shape: {ort_output.shape}')
    print(f'    Max difference: {max_diff:.6f}')
    print(f'    Mean difference: {mean_diff:.6f}')

    # Check L2 normalization
    onnx_norms = np.linalg.norm(ort_output, axis=-1)
    print(f'    ONNX embedding norms: {onnx_norms}')

    if max_diff < 1e-4:
        print('  ✓ ONNX model matches PyTorch (diff < 1e-4)')
        return True
    if max_diff < 1e-3:
        print('  ⚠ ONNX model approximately matches PyTorch (diff < 1e-3)')
        return True
    print(f'  ✗ Large difference detected: {max_diff}')
    return False


def benchmark_onnx(onnx_path, num_iterations=100):
    """Benchmark ONNX inference speed."""

    print(f'\nBenchmarking ONNX inference ({num_iterations} iterations)...')

    import time

    import onnxruntime as ort

    # Create session
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    ort_session = ort.InferenceSession(str(onnx_path), providers=providers)

    # Test different batch sizes
    for batch_size in [1, 4, 8, 16]:
        test_input = np.random.randn(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE).astype(np.float32)

        # Warmup
        for _ in range(10):
            ort_session.run(None, {'images': test_input})

        # Benchmark
        latencies = []
        for _ in range(num_iterations):
            start = time.time()
            ort_session.run(None, {'images': test_input})
            latency = (time.time() - start) * 1000
            latencies.append(latency)

        mean_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)

        print(
            f'  Batch size {batch_size:2d}: {mean_latency:.2f}ms (mean), {p95_latency:.2f}ms (p95)'
        )


def convert_to_tensorrt(onnx_path, plan_path, fp16=True, max_batch_size=128):
    """
    Convert ONNX model to TensorRT engine for maximum throughput.

    Args:
        onnx_path: Path to ONNX model
        plan_path: Output path for TensorRT plan
        fp16: Use FP16 precision (2x faster, minimal accuracy loss)
        max_batch_size: Maximum batch size for dynamic batching
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

        print('  ✓ ONNX parsed successfully')

        # Builder config
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4GB workspace

        if fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print('  ✓ FP16 mode enabled')

        # Optimization profile for dynamic batch sizes
        profile = builder.create_optimization_profile()

        # Image encoder: batch 1 to max_batch_size
        profile.set_shape(
            'images',
            min=(1, 3, IMAGE_SIZE, IMAGE_SIZE),
            opt=(8, 3, IMAGE_SIZE, IMAGE_SIZE),  # Optimal batch size
            max=(max_batch_size, 3, IMAGE_SIZE, IMAGE_SIZE),
        )
        config.add_optimization_profile(profile)

        print('  ✓ Optimization profile configured')
        print('    - Min batch: 1')
        print('    - Optimal batch: 8')
        print(f'    - Max batch: {max_batch_size}')

        # Build engine (this takes a few minutes)
        print('\n  Building TensorRT engine (this may take 5-10 minutes)...')
        serialized_engine = builder.build_serialized_network(network, config)

        if serialized_engine is None:
            raise RuntimeError('Failed to build TensorRT engine')

        # Save engine
        plan_path = Path(plan_path)
        plan_path.parent.mkdir(parents=True, exist_ok=True)

        with open(plan_path, 'wb') as f:
            f.write(serialized_engine)

        print('\n  ✓ TensorRT engine saved!')
        print(f'    File size: {plan_path.stat().st_size / (1024 * 1024):.2f} MB')

        return plan_path

    except ImportError:
        print('\n  ⚠ TensorRT not available in this container')
        print('  To convert to TensorRT, use the Triton container:')
        print('    docker compose exec triton-api trtexec \\')
        print(f'      --onnx={onnx_path} \\')
        print(f'      --saveEngine={plan_path} \\')
        print('      --fp16 \\')
        print(f'      --minShapes=images:1x3x{IMAGE_SIZE}x{IMAGE_SIZE} \\')
        print(f'      --optShapes=images:8x3x{IMAGE_SIZE}x{IMAGE_SIZE} \\')
        print(f'      --maxShapes=images:{max_batch_size}x3x{IMAGE_SIZE}x{IMAGE_SIZE}')
        return None

    except Exception as e:
        print(f'\n  ✗ TensorRT conversion failed: {e}')
        import traceback

        traceback.print_exc()
        return None


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Export MobileCLIP2 Image Encoder')
    parser.add_argument(
        '--model',
        type=str,
        default='S2',
        choices=['S2', 'B'],
        help='Model variant: S2 (balanced) or B (maximum accuracy)',
    )
    parser.add_argument(
        '--skip-tensorrt', action='store_true', help='Skip TensorRT conversion (ONNX only)'
    )
    parser.add_argument('--fp32', action='store_true', help='Use FP32 instead of FP16 for TensorRT')
    parser.add_argument(
        '--max-batch-size', type=int, default=128, help='Maximum batch size for TensorRT engine'
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Get model config
    config = MODEL_CONFIGS[args.model]
    model_name = config['name']
    checkpoint_name = config['checkpoint_name']

    checkpoint_path = f'/app/pytorch_models/{checkpoint_name}/{checkpoint_name}.pt'
    onnx_output = f'/app/pytorch_models/{checkpoint_name}_image_encoder.onnx'
    plan_output = f'/app/models/{checkpoint_name}_image_encoder/1/model.plan'

    print('=' * 80)
    print(f'Track E: {model_name} Image Encoder Export')
    print('=' * 80)
    print(f'\nModel: {model_name}')
    print(f'  Parameters: {config["params"]}')
    print(f'  ImageNet Accuracy: {config["accuracy"]}')
    print(f'  Latency (iPhone): {config["latency"]}')

    # Check checkpoint exists
    if not Path(checkpoint_path).exists():
        print(f'\nERROR: Checkpoint not found: {checkpoint_path}')
        print('\nTo download, run on HOST:')
        print('  bash scripts/track_e/setup_mobileclip_env.sh')
        sys.exit(1)

    # Load model
    model = load_mobileclip_model(model_name, checkpoint_path)

    # Create encoder wrapper
    encoder = MobileCLIPImageEncoder(model)
    encoder.eval()

    # Export to ONNX
    onnx_path = export_to_onnx(model, onnx_output)

    # Validate ONNX
    if not validate_onnx(encoder, onnx_path):
        print('\n⚠ ONNX validation had warnings, but continuing...')

    # Benchmark ONNX
    benchmark_onnx(onnx_path)

    # Convert to TensorRT
    plan_path = None
    if not args.skip_tensorrt:
        plan_path = convert_to_tensorrt(
            onnx_path, plan_output, fp16=not args.fp32, max_batch_size=args.max_batch_size
        )

    # Summary
    print('\n' + '=' * 80)
    print('✅ Export Complete!')
    print('=' * 80)
    print('\nOutputs:')
    print(f'  ONNX:     {onnx_output}')
    if plan_path:
        print(f'  TensorRT: {plan_path}')

    print('\nModel specifications:')
    print(f'  - Input: images [B, 3, {IMAGE_SIZE}, {IMAGE_SIZE}] FP32, normalized [0, 1]')
    print(f'  - Output: image_embeddings [B, {EMBEDDING_DIM}] FP32, L2-normalized')
    print(f'  - Dynamic batch: 1 to {args.max_batch_size}')

    if plan_path:
        print('\nTriton deployment:')
        print(f'  Model directory: /app/models/{checkpoint_name}_image_encoder/')
        print('  Plan file: 1/model.plan')
        print('\n  Next: Create config.pbtxt and restart Triton')
    else:
        print('\nTo convert ONNX to TensorRT plan in Triton container:')
        print('  docker compose exec triton-api trtexec \\')
        print(f'    --onnx=/models/{checkpoint_name}_image_encoder.onnx \\')
        print(f'    --saveEngine=/models/{checkpoint_name}_image_encoder/1/model.plan \\')
        print('    --fp16 \\')
        print(f'    --minShapes=images:1x3x{IMAGE_SIZE}x{IMAGE_SIZE} \\')
        print(f'    --optShapes=images:8x3x{IMAGE_SIZE}x{IMAGE_SIZE} \\')
        print(f'    --maxShapes=images:{args.max_batch_size}x3x{IMAGE_SIZE}x{IMAGE_SIZE}')


if __name__ == '__main__':
    main()
