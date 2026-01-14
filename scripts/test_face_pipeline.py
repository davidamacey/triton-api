#!/usr/bin/env python3
"""
Test script for face detection and ingestion pipeline.

Tests:
1. Face detection with SCRFD and YOLO11 models
2. Visualization with bounding boxes
3. Face ingestion to OpenSearch
4. Embedding consistency comparison
"""

import argparse
import io
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image

# Configuration
API_BASE = os.environ.get("API_BASE", "http://localhost:4603")
OUTPUT_DIR = Path("/mnt/nvm/repos/triton-api/test_results/face_pipeline")


def draw_faces(image_path: str, faces: list, output_path: str, title: str = ""):
    """Draw bounding boxes and landmarks on image."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"  Error: Could not read {image_path}")
        return

    h, w = img.shape[:2]

    for i, face in enumerate(faces):
        # Draw bounding box (normalized coords -> pixel coords)
        box = face.get('box', [])
        if len(box) == 4:
            x1 = int(box[0] * w)
            y1 = int(box[1] * h)
            x2 = int(box[2] * w)
            y2 = int(box[3] * h)

            # Draw box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw score
            score = face.get('score', 0)
            label = f"Face {i+1}: {score:.2f}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw landmarks if present
        landmarks = face.get('landmarks', [])
        if len(landmarks) == 10:
            for j in range(0, 10, 2):
                lx = int(landmarks[j] * w)
                ly = int(landmarks[j+1] * h)
                cv2.circle(img, (lx, ly), 3, (0, 0, 255), -1)

    # Add title
    if title:
        cv2.putText(img, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imwrite(output_path, img)
    print(f"  Saved: {output_path}")


def test_face_detection(image_path: str, detector: str = "scrfd") -> dict:
    """Test face detection endpoint."""
    url = f"{API_BASE}/track_e/faces/detect"
    if detector == "yolo11":
        url = f"{API_BASE}/track_e/faces/yolo11/detect"

    with open(image_path, "rb") as f:
        files = {"image": f} if detector == "scrfd" else {"file": f}
        params = {"confidence": 0.5}
        response = requests.post(url, files=files, params=params)

    if response.status_code != 200:
        print(f"  Error: {response.status_code} - {response.text[:200]}")
        return {}

    return response.json()


def test_face_recognition(image_path: str, detector: str = "scrfd") -> dict:
    """Test face recognition endpoint (detection + embeddings)."""
    url = f"{API_BASE}/track_e/faces/recognize"
    if detector == "yolo11":
        url = f"{API_BASE}/track_e/faces/yolo11/recognize"

    with open(image_path, "rb") as f:
        files = {"image": f} if detector == "scrfd" else {"file": f}
        params = {"confidence": 0.5}
        response = requests.post(url, files=files, params=params)

    if response.status_code != 200:
        print(f"  Error: {response.status_code} - {response.text[:200]}")
        return {}

    return response.json()


def test_face_full(image_path: str, face_detector: str = "scrfd") -> dict:
    """Test unified face pipeline (YOLO + Face + CLIP)."""
    url = f"{API_BASE}/track_e/faces/full"

    with open(image_path, "rb") as f:
        files = {"image": f}
        params = {"face_detector": face_detector, "confidence": 0.5}
        response = requests.post(url, files=files, params=params)

    if response.status_code != 200:
        print(f"  Error: {response.status_code} - {response.text[:200]}")
        return {}

    return response.json()


def test_face_ingest(image_path: str, person_id: str = None) -> dict:
    """Test face ingestion endpoint."""
    url = f"{API_BASE}/track_e/faces/identity/ingest"

    with open(image_path, "rb") as f:
        files = {"file": f}
        params = {}
        if person_id:
            params["person_id"] = person_id
        response = requests.post(url, files=files, params=params)

    if response.status_code != 200:
        print(f"  Error: {response.status_code} - {response.text[:200]}")
        return {}

    return response.json()


def cosine_similarity(a: list, b: list) -> float:
    """Compute cosine similarity between two embeddings."""
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def test_embedding_consistency(image_path: str) -> dict:
    """Test that embeddings are consistent across detectors."""
    results = {}

    # Get embeddings from both detectors
    scrfd_result = test_face_recognition(image_path, "scrfd")
    yolo11_result = test_face_recognition(image_path, "yolo11")

    results["scrfd"] = {
        "num_faces": scrfd_result.get("num_faces", 0),
        "embeddings": scrfd_result.get("embeddings", [])
    }
    results["yolo11"] = {
        "num_faces": yolo11_result.get("num_faces", 0),
        "embeddings": yolo11_result.get("embeddings", [])
    }

    # Compare if same number of faces detected
    if results["scrfd"]["num_faces"] > 0 and results["yolo11"]["num_faces"] > 0:
        # Compare first face embeddings
        scrfd_emb = results["scrfd"]["embeddings"][0] if results["scrfd"]["embeddings"] else []
        yolo11_emb = results["yolo11"]["embeddings"][0] if results["yolo11"]["embeddings"] else []

        if scrfd_emb and yolo11_emb:
            similarity = cosine_similarity(scrfd_emb, yolo11_emb)
            results["cross_detector_similarity"] = similarity
            print(f"  Cross-detector similarity: {similarity:.4f}")

    return results


def run_tests(image_paths: list, output_dir: Path):
    """Run all tests on given images."""
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "images": [],
        "summary": {
            "total_images": 0,
            "scrfd_detections": 0,
            "yolo11_detections": 0,
            "ingested_faces": 0,
            "avg_cross_similarity": []
        }
    }

    for img_path in image_paths:
        img_name = Path(img_path).stem
        print(f"\n{'='*60}")
        print(f"Testing: {img_name}")
        print(f"{'='*60}")

        img_result = {"path": str(img_path), "name": img_name}

        # Test 1: SCRFD detection
        print("\n1. Testing SCRFD face detection...")
        scrfd_detect = test_face_detection(img_path, "scrfd")
        num_scrfd = scrfd_detect.get("num_faces", 0)
        print(f"   Detected {num_scrfd} faces")
        img_result["scrfd_faces"] = num_scrfd
        results["summary"]["scrfd_detections"] += num_scrfd

        if num_scrfd > 0:
            output_path = output_dir / f"{img_name}_scrfd.jpg"
            draw_faces(img_path, scrfd_detect.get("faces", []), str(output_path), f"SCRFD: {num_scrfd} faces")

        # Test 2: YOLO11 detection
        print("\n2. Testing YOLO11 face detection...")
        yolo11_detect = test_face_detection(img_path, "yolo11")
        num_yolo11 = yolo11_detect.get("num_faces", 0)
        print(f"   Detected {num_yolo11} faces")
        img_result["yolo11_faces"] = num_yolo11
        results["summary"]["yolo11_detections"] += num_yolo11

        if num_yolo11 > 0:
            output_path = output_dir / f"{img_name}_yolo11.jpg"
            draw_faces(img_path, yolo11_detect.get("faces", []), str(output_path), f"YOLO11: {num_yolo11} faces")

        # Test 3: Unified pipeline
        print("\n3. Testing unified pipeline (/faces/full)...")
        full_result = test_face_full(img_path, "scrfd")
        if full_result:
            print(f"   YOLO detections: {full_result.get('num_detections', 0)}")
            print(f"   Face detections: {full_result.get('num_faces', 0)}")
            print(f"   Has CLIP embedding: {full_result.get('embedding_norm', 0) > 0}")

        # Test 4: Embedding consistency
        print("\n4. Testing embedding consistency...")
        embed_result = test_embedding_consistency(img_path)
        if "cross_detector_similarity" in embed_result:
            img_result["cross_similarity"] = embed_result["cross_detector_similarity"]
            results["summary"]["avg_cross_similarity"].append(embed_result["cross_detector_similarity"])

        # Test 5: Face ingestion
        print("\n5. Testing face ingestion...")
        ingest_result = test_face_ingest(img_path)
        if ingest_result.get("status") == "success":
            num_ingested = ingest_result.get("num_faces", 0)
            print(f"   Ingested {num_ingested} faces")
            img_result["ingested_faces"] = num_ingested
            results["summary"]["ingested_faces"] += num_ingested
            if ingest_result.get("faces"):
                print(f"   Face IDs: {[f.get('face_id', 'N/A')[:8] for f in ingest_result.get('faces', [])]}")
        else:
            print(f"   Ingestion failed: {ingest_result.get('error', 'Unknown')}")

        results["images"].append(img_result)
        results["summary"]["total_images"] += 1

    # Calculate averages
    if results["summary"]["avg_cross_similarity"]:
        avg_sim = np.mean(results["summary"]["avg_cross_similarity"])
        results["summary"]["avg_cross_similarity"] = float(avg_sim)
        print(f"\n\nAverage cross-detector similarity: {avg_sim:.4f}")

    # Save results
    results_path = output_dir / "test_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Test face pipeline")
    parser.add_argument("--images-dir", type=str, default="/mnt/nvm/FACE_TEST_IMAGES",
                        help="Directory with test images")
    parser.add_argument("--max-images", type=int, default=5,
                        help="Maximum number of images to test")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR),
                        help="Output directory for visualizations")
    args = parser.parse_args()

    # Get image paths
    images_dir = Path(args.images_dir)
    image_paths = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.jpeg")) + list(images_dir.glob("*.png"))
    image_paths = sorted(image_paths)[:args.max_images]

    if not image_paths:
        print(f"No images found in {images_dir}")
        return 1

    print(f"Testing {len(image_paths)} images from {images_dir}")
    print(f"Output directory: {args.output_dir}")

    # Run tests
    results = run_tests([str(p) for p in image_paths], Path(args.output_dir))

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total images tested: {results['summary']['total_images']}")
    print(f"SCRFD total detections: {results['summary']['scrfd_detections']}")
    print(f"YOLO11 total detections: {results['summary']['yolo11_detections']}")
    print(f"Total faces ingested: {results['summary']['ingested_faces']}")
    if isinstance(results['summary']['avg_cross_similarity'], float):
        print(f"Avg cross-detector similarity: {results['summary']['avg_cross_similarity']:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
