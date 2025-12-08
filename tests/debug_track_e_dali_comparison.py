#!/usr/bin/env python3
"""
Debug Script: Compare Track D DALI vs Track E DALI

This script directly compares the outputs of:
- yolo_preprocess_dali (Track D - WORKING)
- dual_preprocess_dali (Track E - BROKEN)

Using the EXACT same image and affine matrix to identify the difference.

Usage:
    docker compose exec yolo-api python /app/scripts/track_e/debug_dali_comparison.py
"""

import io
import sys
from pathlib import Path

import numpy as np
import tritonclient.grpc as grpcclient
from PIL import Image


def calculate_affine_matrix(img_width: int, img_height: int, target_size: int = 640) -> tuple:
    """Calculate letterbox affine matrix - SAME as Track D's triton_end2end_client.py"""
    # This matches: scale = self.input_size / max(orig_h, orig_w)
    scale = target_size / max(img_height, img_width)
    scale = min(scale, 1.0)

    new_w = round(img_width * scale)
    new_h = round(img_height * scale)

    pad_x = (target_size - new_w) / 2.0
    pad_y = (target_size - new_h) / 2.0

    affine_matrix = np.array([[scale, 0.0, pad_x], [0.0, scale, pad_y]], dtype=np.float32)

    return affine_matrix, scale, pad_x, pad_y, new_w, new_h


def test_track_d_dali(client, image_bytes: bytes, affine_matrix: np.ndarray) -> np.ndarray:
    """Call Track D's yolo_preprocess_dali model"""
    # Prepare inputs - same format as triton_end2end_client.py
    input_data = np.frombuffer(image_bytes, dtype=np.uint8)
    input_data = np.expand_dims(input_data, axis=0)  # [1, N]

    affine_batch = np.expand_dims(affine_matrix, axis=0)  # [1, 2, 3]

    input_encoded = grpcclient.InferInput('encoded_images', list(input_data.shape), 'UINT8')
    input_encoded.set_data_from_numpy(input_data)

    input_affine = grpcclient.InferInput('affine_matrices', list(affine_batch.shape), 'FP32')
    input_affine.set_data_from_numpy(affine_batch)

    output = grpcclient.InferRequestedOutput('preprocessed_images')

    response = client.infer(
        model_name='yolo_preprocess_dali', inputs=[input_encoded, input_affine], outputs=[output]
    )

    return response.as_numpy('preprocessed_images')[0]  # Remove batch dim


def test_track_e_dali(client, image_bytes: bytes, affine_matrix: np.ndarray) -> np.ndarray:
    """Call Track E's dual_preprocess_dali model"""
    # Prepare inputs - same format
    input_data = np.frombuffer(image_bytes, dtype=np.uint8)
    input_data = np.expand_dims(input_data, axis=0)  # [1, N]

    affine_batch = np.expand_dims(affine_matrix, axis=0)  # [1, 2, 3]

    input_encoded = grpcclient.InferInput('encoded_images', list(input_data.shape), 'UINT8')
    input_encoded.set_data_from_numpy(input_data)

    input_affine = grpcclient.InferInput('affine_matrices', list(affine_batch.shape), 'FP32')
    input_affine.set_data_from_numpy(affine_batch)

    outputs = [
        grpcclient.InferRequestedOutput('yolo_images'),
        grpcclient.InferRequestedOutput('clip_images'),
        grpcclient.InferRequestedOutput('original_images'),
    ]

    response = client.infer(
        model_name='dual_preprocess_dali', inputs=[input_encoded, input_affine], outputs=outputs
    )

    return response.as_numpy('yolo_images')[0]  # Remove batch dim


def visualize_differences(track_d: np.ndarray, track_e: np.ndarray):
    """Visualize the differences between the two outputs"""
    print('\n' + '=' * 70)
    print('DETAILED COMPARISON')
    print('=' * 70)

    # Basic stats
    diff = np.abs(track_d - track_e)
    print('\nDifference statistics:')
    print(f'  Max absolute diff:  {diff.max():.6f}')
    print(f'  Mean absolute diff: {diff.mean():.6f}')
    print(f'  Min absolute diff:  {diff.min():.6f}')
    print(f'  Std of diff:        {diff.std():.6f}')

    # Value ranges
    print(f'\nTrack D value range: [{track_d.min():.4f}, {track_d.max():.4f}]')
    print(f'Track E value range: [{track_e.min():.4f}, {track_e.max():.4f}]')

    # Check specific regions
    # For an 810x1080 image scaled to 640, we expect:
    # - Left padding: columns 0-79 should be gray (0.447)
    # - Right padding: columns 560-639 should be gray (0.447)
    # - Center: columns 80-559 should have image content

    print('\n' + '-' * 70)
    print('EDGE ANALYSIS (checking padding regions)')
    print('-' * 70)

    # Left edge (should be gray padding ~0.447 for 114/255)
    left_edge_d = track_d[:, :, 0:10].mean()  # First 10 columns
    left_edge_e = track_e[:, :, 0:10].mean()
    print('\nLeft edge mean (columns 0-9, should be ~0.447 if gray padding):')
    print(
        f'  Track D: {left_edge_d:.4f} {"✓ GRAY PADDING" if 0.4 < left_edge_d < 0.5 else "✗ NOT PADDING"}'
    )
    print(
        f'  Track E: {left_edge_e:.4f} {"✓ GRAY PADDING" if 0.4 < left_edge_e < 0.5 else "✗ NOT PADDING"}'
    )

    # Right edge
    right_edge_d = track_d[:, :, -10:].mean()
    right_edge_e = track_e[:, :, -10:].mean()
    print('\nRight edge mean (columns 630-639, should be ~0.447 if gray padding):')
    print(
        f'  Track D: {right_edge_d:.4f} {"✓ GRAY PADDING" if 0.4 < right_edge_d < 0.5 else "✗ NOT PADDING"}'
    )
    print(
        f'  Track E: {right_edge_e:.4f} {"✓ GRAY PADDING" if 0.4 < right_edge_e < 0.5 else "✗ NOT PADDING"}'
    )

    # Top edge (should have image content for this aspect ratio)
    top_edge_d = track_d[:, 0:10, :].mean()
    top_edge_e = track_e[:, 0:10, :].mean()
    print('\nTop edge mean (rows 0-9):')
    print(f'  Track D: {top_edge_d:.4f}')
    print(f'  Track E: {top_edge_e:.4f}')

    # Center
    center_d = track_d[:, 310:330, 310:330].mean()
    center_e = track_e[:, 310:330, 310:330].mean()
    print('\nCenter region mean (rows/cols 310-329):')
    print(f'  Track D: {center_d:.4f}')
    print(f'  Track E: {center_e:.4f}')

    # Check if images are identical
    if np.allclose(track_d, track_e, atol=1e-5):
        print('\n✓ IMAGES ARE IDENTICAL (within tolerance)')
    else:
        print('\n✗ IMAGES ARE DIFFERENT')

        # Find where they differ most
        diff_per_pixel = diff.mean(axis=0)  # Average across channels
        max_diff_idx = np.unravel_index(diff_per_pixel.argmax(), diff_per_pixel.shape)
        print(f'  Maximum difference location: row={max_diff_idx[0]}, col={max_diff_idx[1]}')
        print(f'  Track D value at that location: {track_d[:, max_diff_idx[0], max_diff_idx[1]]}')
        print(f'  Track E value at that location: {track_e[:, max_diff_idx[0], max_diff_idx[1]]}')

        # Check if it's a scaling/positioning issue
        print('\n  Checking for offset/scaling issues:')
        # Sample some specific pixel positions
        for row, col in [(0, 80), (320, 320), (639, 560)]:
            d_val = track_d[:, row, col].mean()
            e_val = track_e[:, row, col].mean()
            print(
                f'    Pixel ({row}, {col}): Track D={d_val:.4f}, Track E={e_val:.4f}, diff={abs(d_val - e_val):.4f}'
            )


def main():
    # Load test image
    image_path = Path('/app/test_images/bus.jpg')
    if not image_path.exists():
        print(f'Error: {image_path} not found')
        sys.exit(1)

    image_bytes = image_path.read_bytes()
    img = Image.open(io.BytesIO(image_bytes))
    img_width, img_height = img.size

    print('=' * 70)
    print('DALI Pipeline Comparison: Track D vs Track E')
    print('=' * 70)
    print(f'\nTest image: {image_path}')
    print(f'Image dimensions: {img_width}x{img_height} (WxH)')
    print(f'Image bytes: {len(image_bytes):,}')

    # Calculate affine matrix
    affine_matrix, scale, pad_x, pad_y, new_w, new_h = calculate_affine_matrix(
        img_width, img_height
    )
    print('\nAffine transformation:')
    print(f'  Scale: {scale:.6f}')
    print(f'  New size: {new_w}x{new_h} (WxH)')
    print(f'  Padding: x={pad_x:.1f}, y={pad_y:.1f}')
    print(f'  Matrix:\n{affine_matrix}')

    # Connect to Triton
    print('\nConnecting to Triton...')
    client = grpcclient.InferenceServerClient(url='triton-api:8001')

    # Check model status
    print('\nModel status:')
    for model in ['yolo_preprocess_dali', 'dual_preprocess_dali']:
        ready = client.is_model_ready(model)
        print(f'  {model}: {"READY" if ready else "NOT READY"}')
        if not ready:
            print(f'    ERROR: {model} is not loaded!')
            sys.exit(1)

    # Run both pipelines
    print('\n' + '=' * 70)
    print('Running Track D DALI (yolo_preprocess_dali)...')
    print('=' * 70)
    track_d_output = test_track_d_dali(client, image_bytes, affine_matrix)
    print(f'  Output shape: {track_d_output.shape}')
    print(f'  Output dtype: {track_d_output.dtype}')

    print('\n' + '=' * 70)
    print('Running Track E DALI (dual_preprocess_dali)...')
    print('=' * 70)
    track_e_output = test_track_e_dali(client, image_bytes, affine_matrix)
    print(f'  Output shape: {track_e_output.shape}')
    print(f'  Output dtype: {track_e_output.dtype}')

    # Compare outputs
    visualize_differences(track_d_output, track_e_output)

    # Test YOLO with both outputs
    print('\n' + '=' * 70)
    print('Testing YOLO detection with both outputs')
    print('=' * 70)

    for name, output in [('Track D', track_d_output), ('Track E', track_e_output)]:
        # Add batch dimension
        yolo_input = output[np.newaxis, ...].astype(np.float32)

        input_tensor = grpcclient.InferInput('images', list(yolo_input.shape), 'FP32')
        input_tensor.set_data_from_numpy(yolo_input)

        outputs = [
            grpcclient.InferRequestedOutput('num_dets'),
            grpcclient.InferRequestedOutput('det_boxes'),
            grpcclient.InferRequestedOutput('det_scores'),
            grpcclient.InferRequestedOutput('det_classes'),
        ]

        response = client.infer(
            model_name='yolov11_small_trt_end2end', inputs=[input_tensor], outputs=outputs
        )

        num_dets = response.as_numpy('num_dets')[0][0]
        det_scores = response.as_numpy('det_scores')[0]

        print(f'\n{name} YOLO results:')
        print(f'  Detections: {num_dets}')
        if num_dets > 0:
            print(f'  Top scores: {det_scores[: min(5, num_dets)]}')

    print('\n' + '=' * 70)
    print('DIAGNOSIS COMPLETE')
    print('=' * 70)


if __name__ == '__main__':
    main()
