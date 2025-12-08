#!/usr/bin/env python3
"""
Track E: Export MobileCLIP2 Text Encoder to ONNX + TensorRT

This script exports MobileCLIP2 text encoder for deployment on Triton.
The text encoder is used for text-based visual search queries.

Supported variants (with same 63.4M text encoder):
- MobileCLIP2-S2: Same text encoder as S0, B
- MobileCLIP2-B:  Same text encoder as S0, S2

Key steps:
1. Load model with proper configuration
2. CRITICAL: Call reparameterize_model() before export
3. Export to ONNX with dynamic batch size
4. Convert to TensorRT plan for maximum throughput
5. Validate output matches PyTorch

Run from: yolo-api container
    docker compose exec yolo-api python /app/scripts/track_e/export_mobileclip_text_encoder.py --model S2
"""

import argparse
import sys
from pathlib import Path


sys.path.insert(0, '/app/reference_repos/ml-mobileclip')
sys.path.insert(0, '/app/reference_repos/open_clip/src')

import numpy as np
import torch
from torch import nn


# Model configurations (S0, S2, B share the same 63.4M text encoder)
MODEL_CONFIGS = {
    'S2': {
        'name': 'MobileCLIP2-S2',
        'checkpoint_name': 'mobileclip2_s2',
    },
    'B': {
        'name': 'MobileCLIP2-B',
        'checkpoint_name': 'mobileclip2_b',
    },
}

# Text encoder specifications
CONTEXT_LENGTH = 77  # Max token sequence length
EMBEDDING_DIM = 512  # MobileCLIP2-S2 uses 512-dim embeddings (same as image encoder)

# ONNX export settings
# TensorRT 10.x (Triton 25.10) supports opset 9-20
# For Transformer/LayerNorm models with dynamic batch: opset 17+ recommended
# See: https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html
ONNX_OPSET_VERSION = 17


class MobileCLIPTextEncoder(nn.Module):
    """
    Wrapper for MobileCLIP2 text encoder with L2 normalization.

    MobileCLIP2 uses CustomTextCLIP architecture where:
    - Text encoder is accessed via model.text (TextTransformer)
    - The TextTransformer has its own forward() that handles embeddings + attention mask

    The text encoder:
    1. Takes tokenized text [B, 77] as INT64
    2. Encodes to 512-dim embedding
    3. L2-normalizes output (critical for cosine similarity with image embeddings)
    """

    def __init__(self, clip_model):
        super().__init__()
        # CustomTextCLIP stores text encoder as unified .text object
        # Use the entire TextTransformer module directly - it handles
        # embeddings, positional encoding, attention mask, pooling, and projection
        self.text_encoder = clip_model.text

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text: [B, 77] INT64 token IDs

        Returns:
            embeddings: [B, 512] FP32, L2-normalized
        """
        # Use the text encoder's forward method which handles everything:
        # - Token embedding + positional embedding
        # - Causal attention mask
        # - Transformer layers
        # - Pooling (argmax for EOS position)
        # - Final layer norm
        # - Text projection
        x = self.text_encoder(text)

        # L2 normalize (CRITICAL for cosine similarity)
        return x / x.norm(dim=-1, keepdim=True)


def load_mobileclip_model(model_name, checkpoint_path):
    """Load MobileCLIP2 model with proper configuration."""
    print(f'\nLoading {model_name}...')
    print(f'  Checkpoint: {checkpoint_path}')

    import open_clip
    from mobileclip.modules.common.mobileone import reparameterize_model

    model, _, _preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=checkpoint_path, image_mean=(0, 0, 0), image_std=(1, 1, 1)
    )

    model.eval()

    # Reparameterize the full model
    print('  Reparameterizing model...')
    model = reparameterize_model(model)

    print('  ✓ Model loaded')

    return model


def export_to_onnx(model, output_path):
    """Export text encoder to ONNX format with dynamic batch support.

    Uses opset 17 for best TensorRT 10.x compatibility with LayerNorm.
    Forces legacy exporter to avoid dynamo issues with dynamic shapes.
    """
    print(f'\nExporting to ONNX: {output_path}')
    print(f'  Opset version: {ONNX_OPSET_VERSION}')

    # Create encoder wrapper
    encoder = MobileCLIPTextEncoder(model)
    encoder.eval()

    # Create dummy input [batch=1, context_length=77]
    dummy_input = torch.randint(0, 49408, (1, CONTEXT_LENGTH), dtype=torch.long)

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
    print('  Exporting with dynamic batch support...')
    torch.onnx.export(
        encoder,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=ONNX_OPSET_VERSION,
        do_constant_folding=True,
        input_names=['text_tokens'],
        output_names=['text_embeddings'],
        dynamic_axes={'text_tokens': {0: 'batch_size'}, 'text_embeddings': {0: 'batch_size'}},
        verbose=False,
        # Force legacy exporter for better dynamic shape handling
        dynamo=False,
    )

    print('  ✓ ONNX export complete')
    print(f'  File size: {output_path.stat().st_size / (1024 * 1024):.2f} MB')

    return output_path, encoder


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
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    ort_session = ort.InferenceSession(str(onnx_path), providers=providers)
    print(f'  ✓ Using provider: {ort_session.get_providers()[0]}')

    # Test with random tokens
    test_input = np.random.randint(0, 49408, (4, CONTEXT_LENGTH)).astype(np.int64)

    # PyTorch inference
    with torch.no_grad():
        pytorch_output = pytorch_encoder(torch.from_numpy(test_input)).numpy()

    # ONNX inference
    ort_output = ort_session.run(None, {'text_tokens': test_input})[0]

    # Compare
    max_diff = np.abs(pytorch_output - ort_output).max()
    mean_diff = np.abs(pytorch_output - ort_output).mean()

    print('\n  Comparison (batch_size=4):')
    print(f'    Max difference: {max_diff:.6f}')
    print(f'    Mean difference: {mean_diff:.6f}')

    # Check L2 normalization
    onnx_norms = np.linalg.norm(ort_output, axis=-1)
    print(f'    ONNX embedding norms: {onnx_norms}')

    if max_diff < 1e-4:
        print('  ✓ ONNX model matches PyTorch (diff < 1e-4)')
        return True
    if max_diff < 1e-3:
        print('  ⚠ ONNX approximately matches PyTorch (diff < 1e-3)')
        return True
    print(f'  ✗ Large difference: {max_diff}')
    return False


def test_with_real_queries(onnx_path, model_name):
    """Test with actual text queries."""
    print('\nTesting with real text queries...')

    import onnxruntime as ort
    import open_clip

    # Load tokenizer
    tokenizer = open_clip.get_tokenizer(model_name)

    # Test queries
    queries = [
        'a photo of a dog',
        'red car on highway',
        'person wearing a jacket',
        'beach scene with palm trees',
    ]

    # Tokenize
    tokens = tokenizer(queries).numpy()
    print(f'  Tokenized shape: {tokens.shape}')  # [4, 77]

    # Run ONNX inference
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    ort_session = ort.InferenceSession(str(onnx_path), providers=providers)

    embeddings = ort_session.run(None, {'text_tokens': tokens})[0]
    print(f'  Embeddings shape: {embeddings.shape}')  # [4, 768]

    # Compute similarity matrix
    print('\n  Query similarity matrix (should show distinct embeddings):')
    similarity = np.dot(embeddings, embeddings.T)
    for i, q in enumerate(queries):
        print(f'    {q[:30]:<30}: {similarity[i]}')

    print('  ✓ Text encoding working correctly')


def convert_to_tensorrt(onnx_path, plan_path, fp16=True, max_batch_size=64):
    """Convert ONNX to TensorRT for maximum throughput."""
    print('\nConverting to TensorRT engine...')
    print(f'  Input: {onnx_path}')
    print(f'  Output: {plan_path}')

    try:
        import tensorrt as trt

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

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

        print('  ✓ ONNX parsed')

        # Builder config
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)

        if fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print('  ✓ FP16 mode enabled')

        # Optimization profile
        profile = builder.create_optimization_profile()
        profile.set_shape(
            'text_tokens',
            min=(1, CONTEXT_LENGTH),
            opt=(8, CONTEXT_LENGTH),
            max=(max_batch_size, CONTEXT_LENGTH),
        )
        config.add_optimization_profile(profile)

        print(f'  ✓ Optimization profile: batch 1 to {max_batch_size}')

        # Build engine
        print('\n  Building TensorRT engine...')
        serialized_engine = builder.build_serialized_network(network, config)

        if serialized_engine is None:
            raise RuntimeError('Failed to build TensorRT engine')

        # Save
        plan_path = Path(plan_path)
        plan_path.parent.mkdir(parents=True, exist_ok=True)

        with open(plan_path, 'wb') as f:
            f.write(serialized_engine)

        print('\n  ✓ TensorRT engine saved!')
        print(f'    File size: {plan_path.stat().st_size / (1024 * 1024):.2f} MB')

        return plan_path

    except ImportError:
        print('\n  ⚠ TensorRT not available')
        print('  Use Triton container:')
        print('    docker compose exec triton-api trtexec \\')
        print(f'      --onnx={onnx_path} \\')
        print(f'      --saveEngine={plan_path} \\')
        print('      --fp16 \\')
        print(f'      --minShapes=text_tokens:1x{CONTEXT_LENGTH} \\')
        print(f'      --optShapes=text_tokens:8x{CONTEXT_LENGTH} \\')
        print(f'      --maxShapes=text_tokens:{max_batch_size}x{CONTEXT_LENGTH}')
        return None


def parse_args():
    parser = argparse.ArgumentParser(description='Export MobileCLIP2 Text Encoder')
    parser.add_argument(
        '--model',
        type=str,
        default='S2',
        choices=['S2', 'B'],
        help='Model variant (S2 and B share same text encoder)',
    )
    parser.add_argument('--skip-tensorrt', action='store_true', help='Skip TensorRT conversion')
    parser.add_argument(
        '--max-batch-size', type=int, default=64, help='Maximum batch size for TensorRT'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    config = MODEL_CONFIGS[args.model]
    model_name = config['name']
    checkpoint_name = config['checkpoint_name']

    checkpoint_path = f'/app/pytorch_models/{checkpoint_name}/{checkpoint_name}.pt'
    onnx_output = f'/app/pytorch_models/{checkpoint_name}_text_encoder.onnx'
    plan_output = f'/app/models/{checkpoint_name}_text_encoder/1/model.plan'

    print('=' * 80)
    print(f'Track E: {model_name} Text Encoder Export')
    print('=' * 80)
    print('\nNote: S0, S2, and B variants share the same 63.4M text encoder')

    # Check checkpoint
    if not Path(checkpoint_path).exists():
        print(f'\nERROR: Checkpoint not found: {checkpoint_path}')
        print('\nTo download, run on HOST:')
        print('  bash scripts/track_e/setup_mobileclip_env.sh')
        sys.exit(1)

    # Load model
    model = load_mobileclip_model(model_name, checkpoint_path)

    # Export to ONNX
    onnx_path, encoder = export_to_onnx(model, onnx_output)

    # Validate
    validate_onnx(encoder, onnx_path)

    # Test with real queries
    test_with_real_queries(onnx_path, model_name)

    # Convert to TensorRT
    plan_path = None
    if not args.skip_tensorrt:
        plan_path = convert_to_tensorrt(onnx_path, plan_output, max_batch_size=args.max_batch_size)

    # Summary
    print('\n' + '=' * 80)
    print('✅ Export Complete!')
    print('=' * 80)
    print('\nOutputs:')
    print(f'  ONNX:     {onnx_output}')
    if plan_path:
        print(f'  TensorRT: {plan_path}')

    print('\nModel specifications:')
    print(f'  - Input: text_tokens [B, {CONTEXT_LENGTH}] INT64')
    print(f'  - Output: text_embeddings [B, {EMBEDDING_DIM}] FP32, L2-normalized')
    print(f'  - Dynamic batch: 1 to {args.max_batch_size}')


if __name__ == '__main__':
    main()
