#!/usr/bin/env python3
"""
Visualize unified pipeline results on an image.
Draws bounding boxes for YOLO detections, faces, and OCR text regions.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import requests

# COCO class names for reference
COCO_CLASSES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
    5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
    10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
    14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
    20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
    25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
    30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
    35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
    39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife',
    44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich',
    49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza',
    54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant',
    59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop',
    64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
    69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
    74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
    79: 'toothbrush'
}


def draw_results(image_path: str, result: dict, output_path: str):
    """Draw all detections on image and save."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return False

    h, w = img.shape[:2]

    # Colors (BGR)
    COLOR_YOLO = (0, 255, 0)      # Green for YOLO detections
    COLOR_FACE = (255, 0, 255)    # Magenta for faces
    COLOR_OCR = (0, 255, 255)     # Yellow for OCR
    COLOR_PERSON = (0, 200, 0)    # Dark green for person class

    # Draw YOLO detections
    for det in result.get('detections', []):
        box = det['box']  # normalized [x1, y1, x2, y2]
        x1, y1, x2, y2 = int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h)
        class_id = det['class_id']
        score = det['score']
        class_name = COCO_CLASSES.get(class_id, f'class_{class_id}')

        color = COLOR_PERSON if class_id == 0 else COLOR_YOLO
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{class_name}: {score:.2f}"
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Draw faces
    for i, face in enumerate(result.get('faces', [])):
        box = face['box']  # normalized [x1, y1, x2, y2]
        x1, y1, x2, y2 = int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h)
        score = face['score']

        cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_FACE, 2)
        label = f"face: {score:.2f}"
        cv2.putText(img, label, (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_FACE, 2)

        # Draw landmarks if available
        landmarks = face.get('landmarks', [])
        if landmarks and any(l != 0 for l in landmarks):
            # 5-point landmarks: left_eye, right_eye, nose, left_mouth, right_mouth
            for j in range(0, len(landmarks), 2):
                lx, ly = int(landmarks[j] * w), int(landmarks[j+1] * h)
                cv2.circle(img, (lx, ly), 2, (0, 0, 255), -1)

    # Draw OCR text boxes
    for text_item in result.get('texts', []):
        text = text_item.get('text', '')
        box_norm = text_item.get('box_normalized', [])
        rec_score = text_item.get('rec_score', 0)

        if len(box_norm) == 4 and text:
            x1, y1, x2, y2 = int(box_norm[0] * w), int(box_norm[1] * h), int(box_norm[2] * w), int(box_norm[3] * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_OCR, 2)
            label = f'"{text}" ({rec_score:.2f})'
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_OCR, 1)

    # Add legend
    legend_y = 20
    cv2.putText(img, f"Face model: {result.get('face_model_used', 'N/A')}", (10, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    legend_y += 20
    cv2.putText(img, f"Detections: {result.get('num_detections', 0)}", (10, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_YOLO, 2)
    legend_y += 20
    cv2.putText(img, f"Faces: {result.get('num_faces', 0)}", (10, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_FACE, 2)
    legend_y += 20
    cv2.putText(img, f"Texts: {result.get('num_texts', 0)}", (10, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_OCR, 2)

    # Save
    cv2.imwrite(output_path, img)
    print(f"Saved: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description='Visualize unified pipeline results')
    parser.add_argument('image', help='Input image path')
    parser.add_argument('--face-model', choices=['scrfd', 'yolo11'], default='scrfd',
                        help='Face detection model (default: scrfd)')
    parser.add_argument('--output', '-o', help='Output image path')
    parser.add_argument('--api-url', default='http://localhost:4603',
                        help='FastAPI server URL (default: http://localhost:4603)')
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)

    # Default output path
    if args.output:
        output_path = args.output
    else:
        output_dir = Path('/mnt/nvm/repos/triton-api/outputs/verification')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / f"{image_path.stem}_{args.face_model}_labeled.jpg")

    # Call API
    url = f"{args.api_url}/track_e/analyze?face_model={args.face_model}"
    print(f"Calling: {url}")

    with open(image_path, 'rb') as f:
        response = requests.post(url, files={'image': f})

    if response.status_code != 200:
        print(f"API error: {response.status_code}")
        print(response.text)
        sys.exit(1)

    result = response.json()
    print(f"\nResults ({args.face_model}):")
    print(f"  Detections: {result.get('num_detections', 0)}")
    print(f"  Faces: {result.get('num_faces', 0)}")
    print(f"  Texts: {result.get('num_texts', 0)}")
    print(f"  Face model used: {result.get('face_model_used', 'N/A')}")
    print(f"  Time: {result.get('total_time_ms', 0):.1f}ms")

    # Draw and save
    draw_results(str(image_path), result, output_path)

    # Also save JSON result
    json_path = output_path.replace('.jpg', '.json')
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {json_path}")


if __name__ == '__main__':
    main()
