#!/usr/bin/env python3
"""
Track E Phase 1 Test Script

Tests the simplified pipeline without Python backend:
1. dual_preprocess_dali - JPEG → YOLO (640x640) + CLIP (256x256) + Original
2. yolov11_small_trt_end2end - Object detection with GPU NMS
3. mobileclip2_s2_image_encoder - Global image embedding (512-dim)
4. mobileclip2_s2_text_encoder - Text query embedding (512-dim)

Usage:
    python scripts/track_e/test_phase1_pipeline.py [image_path]

    # Or run inside container:
    docker compose exec yolo-api python /app/scripts/track_e/test_phase1_pipeline.py /app/test_images/bus.jpg
"""

import sys
import time
from pathlib import Path

import numpy as np

# Triton client
import tritonclient.grpc as grpcclient


def test_dali_preprocessing(
    client: grpcclient.InferenceServerClient, image_bytes: bytes, img_width: int, img_height: int
):
    """Test dual_preprocess_dali model"""
    print('\n' + '=' * 60)
    print('Testing: dual_preprocess_dali')
    print('=' * 60)

    # Calculate affine matrix for YOLO letterbox
    # Same format as Track D: [[scale, 0, pad_x], [0, scale, pad_y]]
    target_size = 640
    scale = min(target_size / img_height, target_size / img_width)
    scale = min(scale, 1.0)

    new_h = round(img_height * scale)
    new_w = round(img_width * scale)
    pad_x = (target_size - new_w) / 2.0
    pad_y = (target_size - new_h) / 2.0

    affine_matrix = np.array([[scale, 0.0, pad_x], [0.0, scale, pad_y]], dtype=np.float32)

    print(f'  Input image: {img_width}x{img_height}')
    print(f'  Affine matrix: scale={scale:.4f}, pad=({pad_x:.1f}, {pad_y:.1f})')

    # Prepare inputs - batch size 1
    # For DALI with variable-length input, shape is [1, -1] for batch of 1
    image_data = np.frombuffer(image_bytes, dtype=np.uint8)

    input_encoded = grpcclient.InferInput('encoded_images', [1, len(image_bytes)], 'UINT8')
    input_encoded.set_data_from_numpy(image_data.reshape(1, -1))

    input_affine = grpcclient.InferInput('affine_matrices', [1, 2, 3], 'FP32')
    input_affine.set_data_from_numpy(affine_matrix.reshape(1, 2, 3))

    # Request outputs
    outputs = [
        grpcclient.InferRequestedOutput('yolo_images'),
        grpcclient.InferRequestedOutput('clip_images'),
        grpcclient.InferRequestedOutput('original_images'),
    ]

    # Inference
    start = time.time()
    response = client.infer(
        model_name='dual_preprocess_dali', inputs=[input_encoded, input_affine], outputs=outputs
    )
    elapsed = (time.time() - start) * 1000

    # Parse outputs
    yolo_images = response.as_numpy('yolo_images')
    clip_images = response.as_numpy('clip_images')
    original_images = response.as_numpy('original_images')

    print(f'\n  ✓ DALI preprocessing completed in {elapsed:.2f}ms')
    print('  Output shapes:')
    print(f'    - yolo_images: {yolo_images.shape} (expected: [3, 640, 640])')
    print(f'    - clip_images: {clip_images.shape} (expected: [3, 256, 256])')
    print(f'    - original_images: {original_images.shape} (expected: [3, H, W])')
    print('  Value ranges:')
    print(f'    - yolo_images: [{yolo_images.min():.4f}, {yolo_images.max():.4f}]')
    print(f'    - clip_images: [{clip_images.min():.4f}, {clip_images.max():.4f}]')

    return yolo_images, clip_images, original_images, affine_matrix


def test_yolo_detection(client: grpcclient.InferenceServerClient, yolo_images: np.ndarray):
    """Test yolov11_small_trt_end2end model"""
    print('\n' + '=' * 60)
    print('Testing: yolov11_small_trt_end2end')
    print('=' * 60)

    # Add batch dimension if needed
    if yolo_images.ndim == 3:
        yolo_images = yolo_images[np.newaxis, ...]

    print(f'  Input shape: {yolo_images.shape}')

    # Prepare input
    input_tensor = grpcclient.InferInput('images', list(yolo_images.shape), 'FP32')
    input_tensor.set_data_from_numpy(yolo_images.astype(np.float32))

    # Request outputs
    outputs = [
        grpcclient.InferRequestedOutput('num_dets'),
        grpcclient.InferRequestedOutput('det_boxes'),
        grpcclient.InferRequestedOutput('det_scores'),
        grpcclient.InferRequestedOutput('det_classes'),
    ]

    # Inference
    start = time.time()
    response = client.infer(
        model_name='yolov11_small_trt_end2end', inputs=[input_tensor], outputs=outputs
    )
    elapsed = (time.time() - start) * 1000

    # Parse outputs
    num_dets = response.as_numpy('num_dets')[0][0]
    det_boxes = response.as_numpy('det_boxes')[0]
    det_scores = response.as_numpy('det_scores')[0]
    det_classes = response.as_numpy('det_classes')[0]

    print(f'\n  ✓ YOLO detection completed in {elapsed:.2f}ms')
    print(f'  Detections: {num_dets}')

    if num_dets > 0:
        print('  Top 5 detections:')
        for i in range(min(5, num_dets)):
            box = det_boxes[i]
            score = det_scores[i]
            cls = det_classes[i]
            print(
                f'    [{i}] class={cls}, score={score:.3f}, box=[{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]'
            )

    return num_dets, det_boxes, det_scores, det_classes


def test_image_encoder(client: grpcclient.InferenceServerClient, clip_images: np.ndarray):
    """Test mobileclip2_s2_image_encoder model"""
    print('\n' + '=' * 60)
    print('Testing: mobileclip2_s2_image_encoder')
    print('=' * 60)

    # Add batch dimension if needed
    if clip_images.ndim == 3:
        clip_images = clip_images[np.newaxis, ...]

    print(f'  Input shape: {clip_images.shape}')

    # Prepare input
    input_tensor = grpcclient.InferInput('images', list(clip_images.shape), 'FP32')
    input_tensor.set_data_from_numpy(clip_images.astype(np.float32))

    # Request output
    output = grpcclient.InferRequestedOutput('image_embeddings')

    # Inference
    start = time.time()
    response = client.infer(
        model_name='mobileclip2_s2_image_encoder', inputs=[input_tensor], outputs=[output]
    )
    elapsed = (time.time() - start) * 1000

    # Parse output
    image_embedding = response.as_numpy('image_embeddings')[0]

    # Calculate L2 norm (should be ~1.0 for normalized embeddings)
    l2_norm = np.linalg.norm(image_embedding)

    print(f'\n  ✓ Image encoding completed in {elapsed:.2f}ms')
    print(f'  Embedding shape: {image_embedding.shape} (expected: [512])')
    print(f'  L2 norm: {l2_norm:.6f} (should be ~1.0)')
    print(f'  First 10 values: {image_embedding[:10]}')

    return image_embedding


def test_text_encoder(client: grpcclient.InferenceServerClient, text: str = 'a photo of a bus'):
    """Test mobileclip2_s2_text_encoder model"""
    print('\n' + '=' * 60)
    print('Testing: mobileclip2_s2_text_encoder')
    print('=' * 60)

    print(f"  Query text: '{text}'")

    # Tokenization with CLIP tokenizer
    # CRITICAL: MobileCLIP/OpenCLIP uses pad_token_id=0, not EOS token (49407)
    # HuggingFace CLIPTokenizer defaults to padding with EOS, which breaks similarity!
    try:
        from transformers import CLIPTokenizer

        tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
        tokenizer.pad_token_id = 0  # CRITICAL: OpenCLIP pads with 0, not EOS (49407)
        tokens = tokenizer(
            text, padding='max_length', max_length=77, truncation=True, return_tensors='np'
        )
        text_tokens = tokens['input_ids'].astype(np.int64)
    except ImportError:
        print('  Warning: transformers not available, using dummy tokens')
        # Create dummy tokens for testing (with correct padding = 0)
        text_tokens = np.zeros((1, 77), dtype=np.int64)
        text_tokens[0, 0] = 49406  # Start token
        text_tokens[0, 1:6] = [320, 1125, 539, 320, 2840]  # Approximate tokens
        text_tokens[0, 6] = 49407  # End token
        # Remaining positions stay 0 (correct padding)

    print(f'  Token shape: {text_tokens.shape}')

    # Prepare input
    input_tensor = grpcclient.InferInput('text_tokens', list(text_tokens.shape), 'INT64')
    input_tensor.set_data_from_numpy(text_tokens)

    # Request output
    output = grpcclient.InferRequestedOutput('text_embeddings')

    # Inference
    start = time.time()
    response = client.infer(
        model_name='mobileclip2_s2_text_encoder', inputs=[input_tensor], outputs=[output]
    )
    elapsed = (time.time() - start) * 1000

    # Parse output
    text_embedding = response.as_numpy('text_embeddings')[0]

    # Calculate L2 norm
    l2_norm = np.linalg.norm(text_embedding)

    print(f'\n  ✓ Text encoding completed in {elapsed:.2f}ms')
    print(f'  Embedding shape: {text_embedding.shape} (expected: [512])')
    print(f'  L2 norm: {l2_norm:.6f} (should be ~1.0)')
    print(f'  First 10 values: {text_embedding[:10]}')

    return text_embedding


def test_similarity(image_embedding: np.ndarray, text_embedding: np.ndarray):
    """Test cosine similarity between image and text embeddings"""
    print('\n' + '=' * 60)
    print('Testing: Cosine Similarity')
    print('=' * 60)

    # Cosine similarity (embeddings should already be L2 normalized)
    similarity = np.dot(image_embedding, text_embedding)

    print(f'  Image-Text similarity: {similarity:.4f}')
    print('  (Higher = more similar, range: -1 to 1)')

    return similarity


def main():
    # Default image path
    image_path = '/app/test_images/bus.jpg'
    if len(sys.argv) > 1:
        image_path = sys.argv[1]

    print('=' * 60)
    print('Track E Phase 1 Pipeline Test')
    print('=' * 60)
    print(f'Image: {image_path}')

    # Load image
    image_path = Path(image_path)
    if not image_path.exists():
        print(f'Error: Image not found: {image_path}')
        sys.exit(1)

    image_bytes = image_path.read_bytes()

    # Get image dimensions
    import io

    from PIL import Image

    img = Image.open(io.BytesIO(image_bytes))
    img_width, img_height = img.size

    print(f'Image size: {img_width}x{img_height}')
    print(f'Image bytes: {len(image_bytes):,}')

    # Connect to Triton
    print('\nConnecting to Triton server...')
    try:
        client = grpcclient.InferenceServerClient(url='triton-api:8001')
        if not client.is_server_live():
            print('Error: Triton server is not live')
            sys.exit(1)
        print('✓ Connected to Triton')
    except Exception as e:
        print(f'Error connecting to Triton: {e}')
        sys.exit(1)

    # Check model status
    print('\nModel status:')
    models = [
        'dual_preprocess_dali',
        'yolov11_small_trt_end2end',
        'mobileclip2_s2_image_encoder',
        'mobileclip2_s2_text_encoder',
    ]
    for model in models:
        try:
            if client.is_model_ready(model):
                print(f'  ✓ {model}: READY')
            else:
                print(f'  ✗ {model}: NOT READY')
        except Exception as e:
            print(f'  ✗ {model}: ERROR - {e}')

    # Run tests
    total_start = time.time()

    # 1. Test DALI preprocessing
    yolo_images, clip_images, _original_images, _affine_matrix = test_dali_preprocessing(
        client, image_bytes, img_width, img_height
    )

    # 2. Test YOLO detection
    num_dets, _det_boxes, _det_scores, _det_classes = test_yolo_detection(client, yolo_images)

    # 3. Test image encoder
    image_embedding = test_image_encoder(client, clip_images)

    # 4. Test text encoder
    text_embedding = test_text_encoder(client, 'a photo of a bus on the street')

    # 5. Test similarity
    similarity = test_similarity(image_embedding, text_embedding)

    total_elapsed = (time.time() - total_start) * 1000

    # Summary
    print('\n' + '=' * 60)
    print('SUMMARY')
    print('=' * 60)
    print(f'  Total pipeline time: {total_elapsed:.2f}ms')
    print(f'  Detections: {num_dets}')
    print(f'  Image embedding: {image_embedding.shape}, L2={np.linalg.norm(image_embedding):.4f}')
    print(f'  Text embedding: {text_embedding.shape}, L2={np.linalg.norm(text_embedding):.4f}')
    print(f'  Similarity: {similarity:.4f}')
    print('\n✓ All Phase 1 tests completed successfully!')


if __name__ == '__main__':
    main()
