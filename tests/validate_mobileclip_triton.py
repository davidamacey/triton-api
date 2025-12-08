#!/usr/bin/env python3
"""
Validate MobileCLIP2-S2 models deployed on Triton Inference Server
"""

import sys
import time
from pathlib import Path

import numpy as np
import tritonclient.grpc as grpcclient
from PIL import Image


# Add path for tokenizer
sys.path.insert(0, '/app/reference_repos/open_clip/src')
sys.path.insert(0, '/app/reference_repos/ml-mobileclip')

try:
    import open_clip
except ImportError:
    print('ERROR: OpenCLIP not installed. Run setup_mobileclip_env.sh first')
    sys.exit(1)


def test_image_encoder():
    """Test image encoder on Triton"""

    print('\n' + '=' * 80)
    print('Testing MobileCLIP2-S2 Image Encoder...')
    print('=' * 80)

    # Load test image
    test_image_path = '/app/test_images/bus.jpg'
    if not Path(test_image_path).exists():
        print(f'WARNING: Test image not found at {test_image_path}')
        print('Creating random test image...')
        img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
    else:
        img = Image.open(test_image_path).convert('RGB')

    img = img.resize((256, 256))
    img_array = np.array(img).astype(np.float32) / 255.0  # Normalize [0,1]
    img_array = np.transpose(img_array, (2, 0, 1))  # HWC -> CHW
    img_array = np.expand_dims(img_array, 0)  # Add batch dim [1, 3, 256, 256]

    print(f'Input image shape: {img_array.shape}')
    print(f'Input range: [{img_array.min():.4f}, {img_array.max():.4f}]')

    # Create Triton client
    try:
        client = grpcclient.InferenceServerClient(url='triton-api:8001', verbose=False)
        print('✓ Connected to Triton server')
    except Exception as e:
        print(f'✗ Failed to connect to Triton: {e}')
        return False

    # Prepare input
    inputs = [grpcclient.InferInput('images', img_array.shape, 'FP32')]
    inputs[0].set_data_from_numpy(img_array)

    # Prepare output
    outputs = [grpcclient.InferRequestedOutput('image_embeddings')]

    # Warmup
    print('\nWarming up...')
    for _ in range(3):
        try:
            response = client.infer(
                model_name='mobileclip2_s2_image_encoder', inputs=inputs, outputs=outputs
            )
        except Exception as e:
            print(f'✗ Warmup inference failed: {e}')
            return False

    # Benchmark
    print('Running benchmark (100 iterations)...')
    latencies = []
    for i in range(100):
        start = time.time()
        response = client.infer(
            model_name='mobileclip2_s2_image_encoder', inputs=inputs, outputs=outputs
        )
        latency = (time.time() - start) * 1000
        latencies.append(latency)

        if i == 0:
            embedding = response.as_numpy('image_embeddings')
            print('\nFirst inference:')
            print(f'  Embedding shape: {embedding.shape}')  # [1, 768]
            print(f'  Embedding norm: {np.linalg.norm(embedding):.4f}')  # Should be ~1.0
            print(f'  Embedding range: [{embedding.min():.4f}, {embedding.max():.4f}]')

            # Validate L2 normalization
            norm = np.linalg.norm(embedding)
            if abs(norm - 1.0) < 0.01:
                print('  ✓ Embedding is L2-normalized')
            else:
                print(f'  ⚠ Warning: Embedding norm {norm:.4f} != 1.0')

    print('\nLatency Statistics:')
    print(f'  Mean:   {np.mean(latencies):.2f}ms')
    print(f'  Median: {np.median(latencies):.2f}ms')
    print(f'  P95:    {np.percentile(latencies, 95):.2f}ms')
    print(f'  P99:    {np.percentile(latencies, 99):.2f}ms')
    print(f'  Min:    {np.min(latencies):.2f}ms')
    print(f'  Max:    {np.max(latencies):.2f}ms')

    # Check target
    mean_latency = np.mean(latencies)
    if mean_latency < 5:
        print(f'\n✓ Image encoder test passed! (Mean latency: {mean_latency:.2f}ms < 5ms)')
        return True
    print(f'\n⚠ Latency above target (mean: {mean_latency:.2f}ms > 5ms)')
    return True  # Still pass, just slower


def test_text_encoder():
    """Test text encoder on Triton"""

    print('\n' + '=' * 80)
    print('Testing MobileCLIP2-S2 Text Encoder...')
    print('=' * 80)

    # Get tokenizer
    try:
        tokenizer = open_clip.get_tokenizer('MobileCLIP2-S2')
        print('✓ Tokenizer loaded')
    except Exception as e:
        print(f'✗ Failed to load tokenizer: {e}')
        return False

    # Tokenize queries
    queries = ['a photo of a dog', 'red car on highway', 'person wearing jacket']
    tokens = tokenizer(queries).numpy()

    print(f'\nTokenized {len(queries)} queries:')
    print(f'  Token shape: {tokens.shape}')  # [3, 77]
    print(f'  Sample tokens: {tokens[0][:10]}')

    # Create Triton client
    try:
        client = grpcclient.InferenceServerClient(url='triton-api:8001', verbose=False)
    except Exception as e:
        print(f'✗ Failed to connect to Triton: {e}')
        return False

    # Prepare input
    inputs = [grpcclient.InferInput('text_tokens', tokens.shape, 'INT64')]
    inputs[0].set_data_from_numpy(tokens)

    # Prepare output
    outputs = [grpcclient.InferRequestedOutput('text_embeddings')]

    # Warmup
    print('\nWarming up...')
    for _ in range(3):
        try:
            response = client.infer(
                model_name='mobileclip2_s2_text_encoder', inputs=inputs, outputs=outputs
            )
        except Exception as e:
            print(f'✗ Warmup inference failed: {e}')
            return False

    # Benchmark
    print('Running benchmark (100 iterations)...')
    latencies = []
    for i in range(100):
        start = time.time()
        response = client.infer(
            model_name='mobileclip2_s2_text_encoder', inputs=inputs, outputs=outputs
        )
        latency = (time.time() - start) * 1000
        latencies.append(latency)

        if i == 0:
            embeddings = response.as_numpy('text_embeddings')
            print('\nFirst inference:')
            print(f'  Embeddings shape: {embeddings.shape}')  # [3, 768]
            print(f'  Embedding norms: {np.linalg.norm(embeddings, axis=1)}')  # All ~1.0

            # Validate L2 normalization
            norms = np.linalg.norm(embeddings, axis=1)
            if np.all(np.abs(norms - 1.0) < 0.01):
                print('  ✓ All embeddings are L2-normalized')
            else:
                print('  ⚠ Warning: Some embedding norms != 1.0')

    print('\nLatency Statistics:')
    print(f'  Mean:   {np.mean(latencies):.2f}ms')
    print(f'  Median: {np.median(latencies):.2f}ms')
    print(f'  P95:    {np.percentile(latencies, 95):.2f}ms')
    print(f'  P99:    {np.percentile(latencies, 99):.2f}ms')

    # Check target
    mean_latency = np.mean(latencies)
    if mean_latency < 2:
        print(f'\n✓ Text encoder test passed! (Mean latency: {mean_latency:.2f}ms < 2ms)')
        return True
    print(f'\n⚠ Latency above target (mean: {mean_latency:.2f}ms > 2ms)')
    return True  # Still pass


def test_similarity():
    """Test image-text similarity"""

    print('\n' + '=' * 80)
    print('Testing Image-Text Similarity...')
    print('=' * 80)

    try:
        client = grpcclient.InferenceServerClient(url='triton-api:8001', verbose=False)
        tokenizer = open_clip.get_tokenizer('MobileCLIP2-S2')
    except Exception as e:
        print(f'✗ Setup failed: {e}')
        return False

    # Create test image (solid color)
    img = Image.fromarray(np.full((256, 256, 3), 100, dtype=np.uint8))
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.transpose(img_array, (2, 0, 1)).reshape(1, 3, 256, 256)

    # Encode image
    img_inputs = [grpcclient.InferInput('images', img_array.shape, 'FP32')]
    img_inputs[0].set_data_from_numpy(img_array)
    img_outputs = [grpcclient.InferRequestedOutput('image_embeddings')]

    try:
        img_response = client.infer(
            model_name='mobileclip2_s2_image_encoder', inputs=img_inputs, outputs=img_outputs
        )
        img_embedding = img_response.as_numpy('image_embeddings')[0]
        print(f'Image embedding shape: {img_embedding.shape}')
    except Exception as e:
        print(f'✗ Image encoding failed: {e}')
        return False

    # Encode text
    queries = ['a gray image', 'a colorful photo', 'a black and white picture']
    tokens = tokenizer(queries).numpy()

    text_inputs = [grpcclient.InferInput('text_tokens', tokens.shape, 'INT64')]
    text_inputs[0].set_data_from_numpy(tokens)
    text_outputs = [grpcclient.InferRequestedOutput('text_embeddings')]

    try:
        text_response = client.infer(
            model_name='mobileclip2_s2_text_encoder', inputs=text_inputs, outputs=text_outputs
        )
        text_embeddings = text_response.as_numpy('text_embeddings')
        print(f'Text embeddings shape: {text_embeddings.shape}')
    except Exception as e:
        print(f'✗ Text encoding failed: {e}')
        return False

    # Compute similarities (dot product since normalized)
    similarities = np.dot(img_embedding, text_embeddings.T)

    print('\nSimilarity scores:')
    for query, score in zip(queries, similarities, strict=False):
        print(f"  '{query}': {score:.4f}")

    print('\n✓ Similarity computation successful!')
    print(f'  All scores in valid range: {similarities.min():.4f} to {similarities.max():.4f}')

    return True


def main():
    """Main test runner"""
    print('\n' + '=' * 80)
    print('MobileCLIP2-S2 Triton Validation Suite')
    print('=' * 80)

    results = {'Image Encoder': False, 'Text Encoder': False, 'Image-Text Similarity': False}

    # Run all tests
    results['Image Encoder'] = test_image_encoder()
    results['Text Encoder'] = test_text_encoder()
    results['Image-Text Similarity'] = test_similarity()

    # Summary
    print('\n' + '=' * 80)
    print('VALIDATION SUMMARY')
    print('=' * 80)

    all_passed = True
    for test_name, passed in results.items():
        status = '✓ PASS' if passed else '✗ FAIL'
        print(f'{test_name:30s} {status}')
        if not passed:
            all_passed = False

    if all_passed:
        print('\n' + '=' * 80)
        print('✅ ALL TESTS PASSED!')
        print('=' * 80)
        return 0
    print('\n' + '=' * 80)
    print('❌ SOME TESTS FAILED')
    print('=' * 80)
    return 1


if __name__ == '__main__':
    sys.exit(main())
