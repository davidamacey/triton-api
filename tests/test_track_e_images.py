#!/usr/bin/env python3
"""
Track E: Comprehensive Image Test Script

Tests the full Track E pipeline (YOLO + MobileCLIP) on all images in a directory.

Usage:
    # Test all images in test_images/
    docker compose exec yolo-api python /app/scripts/track_e/test_track_e_images.py

    # Test specific directory
    docker compose exec yolo-api python /app/scripts/track_e/test_track_e_images.py --images /app/test_images

    # Quick test (first 3 images only)
    docker compose exec yolo-api python /app/scripts/track_e/test_track_e_images.py --limit 3
"""

import argparse
import io
import sys
import time
from pathlib import Path

import numpy as np
import tritonclient.grpc as grpcclient
from PIL import Image


# Text prompts for similarity testing
TEST_PROMPTS = [
    'a photo of a bus',
    'a photo of a car',
    'a photo of a person',
    'a street scene',
]


def get_tokenizer():
    """Get CLIP tokenizer with correct padding (pad_token_id=0)."""
    from transformers import CLIPTokenizer

    tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
    tokenizer.pad_token_id = 0  # CRITICAL: OpenCLIP pads with 0, not EOS (49407)
    return tokenizer


def calculate_affine_matrix(img_width: int, img_height: int, target_size: int = 640) -> np.ndarray:
    """Calculate letterbox affine matrix for YOLO preprocessing."""
    scale = target_size / max(img_height, img_width)
    scale = min(scale, 1.0)

    new_w = round(img_width * scale)
    new_h = round(img_height * scale)

    pad_x = (target_size - new_w) / 2.0
    pad_y = (target_size - new_h) / 2.0

    return np.array([[scale, 0.0, pad_x], [0.0, scale, pad_y]], dtype=np.float32)


def process_image(client: grpcclient.InferenceServerClient, image_path: Path, tokenizer) -> dict:
    """Process a single image through the Track E pipeline."""
    results = {
        'image': image_path.name,
        'success': False,
        'error': None,
        'timings': {},
        'detections': 0,
        'similarities': {},
    }

    try:
        # Load image
        image_bytes = image_path.read_bytes()
        img = Image.open(io.BytesIO(image_bytes))
        img_width, img_height = img.size
        results['size'] = f'{img_width}x{img_height}'

        # Calculate affine matrix
        affine_matrix = calculate_affine_matrix(img_width, img_height)

        # ===== DALI Preprocessing =====
        start = time.time()
        input_data = np.frombuffer(image_bytes, dtype=np.uint8)[np.newaxis, :]
        input_encoded = grpcclient.InferInput('encoded_images', list(input_data.shape), 'UINT8')
        input_encoded.set_data_from_numpy(input_data)

        input_affine = grpcclient.InferInput('affine_matrices', [1, 2, 3], 'FP32')
        input_affine.set_data_from_numpy(affine_matrix[np.newaxis, :])

        response = client.infer(
            model_name='dual_preprocess_dali',
            inputs=[input_encoded, input_affine],
            outputs=[
                grpcclient.InferRequestedOutput('yolo_images'),
                grpcclient.InferRequestedOutput('clip_images'),
                grpcclient.InferRequestedOutput('original_images'),
            ],
        )
        yolo_images = response.as_numpy('yolo_images')
        clip_images = response.as_numpy('clip_images')
        results['timings']['dali_ms'] = (time.time() - start) * 1000

        # ===== YOLO Detection =====
        start = time.time()
        input_tensor = grpcclient.InferInput('images', list(yolo_images.shape), 'FP32')
        input_tensor.set_data_from_numpy(yolo_images.astype(np.float32))

        response = client.infer(
            model_name='yolov11_small_trt_end2end',
            inputs=[input_tensor],
            outputs=[
                grpcclient.InferRequestedOutput('num_dets'),
                grpcclient.InferRequestedOutput('det_boxes'),
                grpcclient.InferRequestedOutput('det_scores'),
                grpcclient.InferRequestedOutput('det_classes'),
            ],
        )
        num_dets = int(response.as_numpy('num_dets')[0][0])
        results['detections'] = num_dets
        results['timings']['yolo_ms'] = (time.time() - start) * 1000

        # ===== MobileCLIP Image Encoding =====
        start = time.time()
        input_tensor = grpcclient.InferInput('images', list(clip_images.shape), 'FP32')
        input_tensor.set_data_from_numpy(clip_images.astype(np.float32))

        response = client.infer(
            model_name='mobileclip2_s2_image_encoder',
            inputs=[input_tensor],
            outputs=[grpcclient.InferRequestedOutput('image_embeddings')],
        )
        image_embedding = response.as_numpy('image_embeddings')[0]
        results['timings']['clip_image_ms'] = (time.time() - start) * 1000

        # ===== Text Similarity for each prompt =====
        for prompt in TEST_PROMPTS:
            start = time.time()
            tokens = tokenizer(
                prompt, padding='max_length', max_length=77, truncation=True, return_tensors='np'
            )['input_ids'].astype(np.int64)

            input_tensor = grpcclient.InferInput('text_tokens', list(tokens.shape), 'INT64')
            input_tensor.set_data_from_numpy(tokens)

            response = client.infer(
                model_name='mobileclip2_s2_text_encoder',
                inputs=[input_tensor],
                outputs=[grpcclient.InferRequestedOutput('text_embeddings')],
            )
            text_embedding = response.as_numpy('text_embeddings')[0]

            similarity = float(np.dot(image_embedding, text_embedding))
            results['similarities'][prompt] = similarity

        # Calculate total time
        results['timings']['total_ms'] = sum(
            [
                results['timings']['dali_ms'],
                results['timings']['yolo_ms'],
                results['timings']['clip_image_ms'],
            ]
        )

        results['success'] = True

    except Exception as e:
        results['error'] = str(e)

    return results


def print_results_table(all_results: list):
    """Print results in a formatted table."""
    print('\n' + '=' * 100)
    print('RESULTS SUMMARY')
    print('=' * 100)

    # Header
    print(
        f'{"Image":<25} {"Size":<12} {"Dets":>5} {"DALI":>8} {"YOLO":>8} {"CLIP":>8} {"Total":>8} {"Best Match":<25}'
    )
    print('-' * 100)

    for r in all_results:
        if r['success']:
            # Find best matching prompt
            best_prompt = max(r['similarities'], key=r['similarities'].get)
            best_sim = r['similarities'][best_prompt]

            print(
                f'{r["image"]:<25} {r["size"]:<12} {r["detections"]:>5} '
                f'{r["timings"]["dali_ms"]:>7.1f}ms {r["timings"]["yolo_ms"]:>7.1f}ms '
                f'{r["timings"]["clip_image_ms"]:>7.1f}ms {r["timings"]["total_ms"]:>7.1f}ms '
                f'{best_prompt[:20]:<20} ({best_sim:.3f})'
            )
        else:
            print(
                f'{r["image"]:<25} {"FAILED":<12} {"-":>5} {"-":>8} {"-":>8} {"-":>8} {"-":>8} {r["error"][:25]}'
            )

    print('-' * 100)


def print_timing_stats(all_results: list):
    """Print timing statistics."""
    successful = [r for r in all_results if r['success']]
    if not successful:
        return

    print('\n' + '=' * 60)
    print('TIMING STATISTICS')
    print('=' * 60)

    for stage in ['dali_ms', 'yolo_ms', 'clip_image_ms', 'total_ms']:
        times = [r['timings'][stage] for r in successful]
        print(f'\n{stage.replace("_ms", "").upper():}')
        print(f'  Mean:   {np.mean(times):>7.2f} ms')
        print(f'  Median: {np.median(times):>7.2f} ms')
        print(f'  Min:    {np.min(times):>7.2f} ms')
        print(f'  Max:    {np.max(times):>7.2f} ms')
        print(f'  Std:    {np.std(times):>7.2f} ms')


def main():
    parser = argparse.ArgumentParser(description='Track E Image Test Script')
    parser.add_argument(
        '--images', default='/app/test_images', help='Directory containing test images'
    )
    parser.add_argument('--limit', type=int, default=0, help='Limit number of images (0 = all)')
    parser.add_argument('--triton', default='triton-api:8001', help='Triton server URL')
    args = parser.parse_args()

    print('=' * 60)
    print('Track E: Comprehensive Image Test')
    print('=' * 60)
    print(f'Image directory: {args.images}')
    print(f'Triton server: {args.triton}')

    # Find images
    image_dir = Path(args.images)
    if not image_dir.exists():
        print(f'ERROR: Directory not found: {args.images}')
        sys.exit(1)

    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    images = sorted(
        [f for f in image_dir.iterdir() if f.is_file() and f.suffix.lower() in extensions]
    )

    if args.limit > 0:
        images = images[: args.limit]

    print(f'Found {len(images)} images to test')

    if not images:
        print('ERROR: No images found!')
        sys.exit(1)

    # Connect to Triton
    print('\nConnecting to Triton server...')
    try:
        client = grpcclient.InferenceServerClient(url=args.triton)
        if not client.is_server_live():
            print('ERROR: Triton server is not live')
            sys.exit(1)
        print('✓ Connected to Triton')
    except Exception as e:
        print(f'ERROR: Failed to connect to Triton: {e}')
        sys.exit(1)

    # Check model status
    print('\nModel status:')
    models = [
        'dual_preprocess_dali',
        'yolov11_small_trt_end2end',
        'mobileclip2_s2_image_encoder',
        'mobileclip2_s2_text_encoder',
    ]
    all_ready = True
    for model in models:
        try:
            ready = client.is_model_ready(model)
            status = '✓ READY' if ready else '✗ NOT READY'
            print(f'  {model}: {status}')
            if not ready:
                all_ready = False
        except Exception as e:
            print(f'  {model}: ✗ ERROR - {e}')
            all_ready = False

    if not all_ready:
        print('\nERROR: Not all models are ready!')
        sys.exit(1)

    # Load tokenizer (once)
    print('\nLoading tokenizer...')
    tokenizer = get_tokenizer()
    print('✓ Tokenizer loaded')

    # Process images
    print(f'\nProcessing {len(images)} images...')
    print('-' * 60)

    all_results = []
    for i, image_path in enumerate(images, 1):
        print(f'[{i}/{len(images)}] Processing {image_path.name}...', end=' ', flush=True)
        result = process_image(client, image_path, tokenizer)
        all_results.append(result)

        if result['success']:
            print(f'✓ {result["detections"]} dets, {result["timings"]["total_ms"]:.1f}ms')
        else:
            print(f'✗ {result["error"]}')

    # Print summary
    print_results_table(all_results)
    print_timing_stats(all_results)

    # Final summary
    successful = sum(1 for r in all_results if r['success'])
    failed = len(all_results) - successful

    print('\n' + '=' * 60)
    print('FINAL SUMMARY')
    print('=' * 60)
    print(f'Total images: {len(all_results)}')
    print(f'Successful:   {successful}')
    print(f'Failed:       {failed}')

    if successful == len(all_results):
        print('\n✓ ALL TESTS PASSED!')
        return 0
    if successful > 0:
        print(f'\n⚠ PARTIAL SUCCESS ({successful}/{len(all_results)})')
        return 1
    print('\n✗ ALL TESTS FAILED!')
    return 1


if __name__ == '__main__':
    sys.exit(main())
