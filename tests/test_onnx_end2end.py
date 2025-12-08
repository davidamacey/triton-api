#!/usr/bin/env python3
"""
Test ONNX End2End model locally to verify outputs before Triton deployment.

This script loads the ONNX End2End model and runs inference directly using
ONNX Runtime to isolate model behavior from Triton serving issues.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def letterbox(
    img: np.ndarray,
    new_shape=(640, 640),
    color=(114, 114, 114),
    auto=False,
    scaleFill=False,
    scaleup=True,
    stride=32,
):
    """
    Resize and pad image to target shape while maintaining aspect ratio.
    Returns: (padded_image, scale_ratio, padding)
    """
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = round(shape[1] * r), round(shape[0] * r)
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return img, ratio, (dw, dh)


def preprocess_image(img_path: str):
    """Preprocess image for YOLO inference."""
    # Read image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f'Failed to load image: {img_path}')

    orig_h, orig_w = img.shape[:2]
    print(f'Original image shape: {orig_h}x{orig_w}')

    # Letterbox resize
    img_letterbox, ratio, padding = letterbox(img, new_shape=(640, 640))
    print(f'Letterbox - Scale: {ratio[0]:.4f}, Padding: {padding}')

    # Normalize to 0-1
    img_norm = img_letterbox.astype(np.float32) / 255.0

    # HWC to CHW
    img_chw = np.transpose(img_norm, (2, 0, 1))

    # Add batch dimension
    img_batch = np.expand_dims(img_chw, axis=0)

    return img_batch, (orig_h, orig_w), ratio[0], padding


def run_onnx_inference(model_path: str, img_path: str):
    """Run inference using ONNX Runtime."""
    print('=' * 60)
    print('Testing ONNX End2End Model Locally')
    print('=' * 60)

    # Load ONNX model
    print(f'\nLoading model: {model_path}')
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

    # Print model info
    print('\nModel Inputs:')
    for inp in session.get_inputs():
        print(f'  {inp.name}: {inp.shape} ({inp.type})')

    print('\nModel Outputs:')
    for out in session.get_outputs():
        print(f'  {out.name}: {out.shape} ({out.type})')

    # Preprocess image
    print(f'\nPreprocessing image: {img_path}')
    input_data, orig_shape, scale, padding = preprocess_image(img_path)
    print(f'Input tensor shape: {input_data.shape}')

    # Run inference
    print('\nRunning ONNX inference...')
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_data})

    # Parse outputs
    num_dets = int(outputs[0][0][0])
    boxes = outputs[1][0][:num_dets]  # [N, 4]
    scores = outputs[2][0][:num_dets]  # [N]
    classes = outputs[3][0][:num_dets]  # [N]

    print(f'\n{"=" * 60}')
    print(f'RESULTS: {num_dets} detections')
    print(f'{"=" * 60}')

    # Display raw outputs (in 640x640 space)
    print('\nRaw detections (in padded 640x640 space):')
    print(f'{"Box (x,y,w,h)":<30} {"Score":<10} {"Class"}')
    print('-' * 60)
    for box, score, cls in zip(boxes, scores, classes, strict=False):
        print(f'{box!s:<30} {score:<10.4f} {int(cls)}')

    # Apply inverse letterbox transformation
    print(f'\nTransformed detections (original {orig_shape[1]}x{orig_shape[0]} space):')
    print(f'{"Box (x1,y1,x2,y2)":<40} {"Score":<10} {"Class"}')
    print('-' * 60)

    pad_x, pad_y = padding
    for box, score, cls in zip(boxes, scores, classes, strict=False):
        x, y, w, h = box

        # Remove padding and scale back to original
        x_orig = (x - pad_x) / scale
        y_orig = (y - pad_y) / scale
        w_orig = w / scale
        h_orig = h / scale

        # Convert to x1,y1,x2,y2
        x1 = x_orig - w_orig / 2
        y1 = y_orig - h_orig / 2
        x2 = x_orig + w_orig / 2
        y2 = y_orig + h_orig / 2

        print(f'({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}){"":<13} {score:<10.4f} {int(cls)}')

    print(f'\n{"=" * 60}')
    print('Test completed successfully!')
    print(f'{"=" * 60}\n')


if __name__ == '__main__':
    # Default paths
    model_path = '/mnt/nvm/repos/triton-api/models/yolov11_small_end2end/1/model.onnx'
    img_path = '/mnt/nvm/KILLBOY_SAMPLE_PICTURES/DSC00002.JPG'

    # Allow override from command line
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    if len(sys.argv) > 2:
        model_path = sys.argv[2]

    # Check files exist
    if not Path(model_path).exists():
        print(f'Error: Model not found: {model_path}')
        sys.exit(1)

    if not Path(img_path).exists():
        print(f'Error: Image not found: {img_path}')
        sys.exit(1)

    # Run test
    try:
        run_onnx_inference(model_path, img_path)
    except Exception as e:
        print(f'\nError: {e}')
        import traceback

        traceback.print_exc()
        sys.exit(1)
