# Tests Directory

Comprehensive testing tools for the Triton YOLO Inference System with five performance tracks.

---

## Overview

This directory contains test scripts and utilities for validating the Triton Inference Server deployment across all five performance tracks:

| Track | Description | Backend |
|-------|-------------|---------|
| **A** | PyTorch Baseline | CPU preprocessing + CPU NMS |
| **B** | TensorRT Standard | CPU preprocessing + CPU NMS |
| **C** | TensorRT End2End | CPU preprocessing + GPU NMS |
| **D** | DALI + TensorRT | GPU preprocessing + GPU NMS |
| **E** | Visual Search | DALI + YOLO + MobileCLIP embeddings |

---

## Test Files

### Integration Tests

| File | Purpose | Tracks Covered |
|------|---------|----------------|
| `test_inference.sh` | Main integration test via HTTP API | A, B, C, D |
| `compare_tracks.py` | Compare detection outputs across all tracks | A, B, C, D, E |

### Validation Tests

| File | Purpose | Tracks Covered |
|------|---------|----------------|
| `test_end2end_patch.py` | Verify ultralytics End2End TRT NMS patch | C, D |
| `test_onnx_end2end.py` | Debug ONNX models locally (bypasses Triton) | C |
| `compare_padding_methods.py` | Compare DALI padding methods (center vs simple) | D |

### Track E Tests

| File | Purpose | Description |
|------|---------|-------------|
| `test_track_e_ensemble.py` | Test Track E ensemble pipeline | YOLO + MobileCLIP via Triton |
| `test_track_e_images.py` | Test Track E image processing | Image embedding generation |
| `test_track_e_integration.py` | Full Track E integration test | Ingest, search, index management |
| `test_track_e_phase1_pipeline.py` | Phase 1 pipeline validation | DALI + encoder pipeline |
| `validate_mobileclip_triton.py` | Validate MobileCLIP models | Encoder output verification |
| `debug_track_e_dali_comparison.py` | Debug DALI vs PyTorch preprocessing | Preprocessing comparison |

### Utilities

| File | Purpose |
|------|---------|
| `create_test_images.py` | Generate test images in various sizes and aspect ratios |
| `detection_comparison_utils.py` | Shared utilities for IoU calculation, detection matching, and metrics |

---

## Quick Start

### Prerequisites

Before running tests, ensure:

1. **Services are running:**
   ```bash
   docker compose up -d
   # or
   make up
   ```

2. **Check service health:**
   ```bash
   make status
   # or
   bash scripts/check_services.sh
   ```

3. **Dependencies installed:**
   - `jq` for JSON parsing in shell scripts
   - Python packages from `requirements.txt`

### Run All Tests

```bash
# Via Makefile (recommended)
make test-all

# Direct execution
bash tests/test_inference.sh
```

---

## Running Tests

### Using Makefile Targets

The Makefile provides convenient targets for all test operations:

| Target | Description |
|--------|-------------|
| `make test-inference` | Test all tracks via shell script |
| `make test-validate-models` | Compare detections across tracks (alias for compare-tracks) |
| `make compare-tracks` | Run compare_tracks.py with default settings |
| `make test-shared-client` | Test shared vs per-request client performance |
| `make test-patch` | Verify End2End TRT NMS patch is applied |
| `make test-onnx` | Test ONNX model locally |
| `make test-compare-padding` | Compare DALI padding methods |
| `make test-create-images` | Generate test images (requires SOURCE param) |
| `make test-all` | Run comprehensive test suite |

**Track-specific quick tests:**

| Target | Description |
|--------|-------------|
| `make test-track-a` | Quick test Track A (PyTorch) |
| `make test-track-b` | Quick test Track B (TensorRT) |
| `make test-track-c` | Quick test Track C (TRT + GPU NMS) |
| `make test-track-d` | Quick test Track D (DALI pipeline) |
| `make test-track-e` | Quick test Track E (Visual Search) |
| `make test-all-tracks` | Test all tracks sequentially |

### Direct Script Execution

#### test_inference.sh - Main Integration Test

```bash
# Default: 10 images, small model
bash tests/test_inference.sh

# Custom image directory
bash tests/test_inference.sh /path/to/images

# Specify model size and image count
bash tests/test_inference.sh /path/to/images small 50

# Test specific Track D variant (streaming, batch, or balanced)
bash tests/test_inference.sh /path/to/images small 10 batch
```

**Arguments:**
- `IMAGE_DIR`: Directory containing test images (default: `./test_images`)
- `MODEL_SIZE`: Model size - `nano`, `small`, or `medium` (default: `small`)
- `NUM_TESTS`: Number of images to test per track (default: `10`)
- `TRACK_D_VARIANT`: Track D variant - `streaming`, `batch`, or empty for balanced

#### compare_tracks.py - Cross-Track Validation

```bash
# Inside container (recommended)
docker compose exec yolo-api python /app/tests/compare_tracks.py

# Test specific tracks
docker compose exec yolo-api python /app/tests/compare_tracks.py --tracks A,C,D

# Include Track E
docker compose exec yolo-api python /app/tests/compare_tracks.py --tracks A,C,D,E

# Custom IoU threshold
docker compose exec yolo-api python /app/tests/compare_tracks.py --iou-threshold 0.7

# Limit number of images
docker compose exec yolo-api python /app/tests/compare_tracks.py --max-images 5

# Save detailed results to JSON
docker compose exec yolo-api python /app/tests/compare_tracks.py --output /app/results.json

# From host (uses HTTP API)
python tests/compare_tracks.py --host localhost --port-main 4603 --tracks A,C,E
```

**Arguments:**
- `--images`: Directory containing test images (default: `/app/test_images`)
- `--host`: API host (default: `localhost`)
- `--port-main`: Port for Track A/C/E (default: `4603`)
- `--port-trackd`: Port for Track D (default: `4613`)
- `--iou-threshold`: IoU threshold for matching (default: `0.5`)
- `--max-images`: Maximum images to process (default: all)
- `--tracks`: Comma-separated list of tracks (default: `A,C,D,E`)
- `--output`: Output JSON file for detailed results
- `--quiet`: Suppress per-image output

#### test_end2end_patch.py - Patch Verification

```bash
docker compose exec yolo-api python /app/tests/test_end2end_patch.py
```

Verifies:
1. Patch module imports successfully
2. `export_onnx_trt` method exists on Exporter class
3. TRT operators are available
4. Can load YOLO model (if available)

#### test_onnx_end2end.py - Local ONNX Testing

```bash
# Default paths
docker compose exec yolo-api python /app/tests/test_onnx_end2end.py

# Custom image
docker compose exec yolo-api python /app/tests/test_onnx_end2end.py /path/to/image.jpg

# Custom model and image
docker compose exec yolo-api python /app/tests/test_onnx_end2end.py /path/to/image.jpg /path/to/model.onnx
```

#### compare_padding_methods.py - DALI Padding Comparison

```bash
# Default settings
docker compose exec yolo-api python /app/tests/compare_padding_methods.py

# Custom image directory
docker compose exec yolo-api python /app/tests/compare_padding_methods.py --images /path/to/images

# Different IoU threshold
docker compose exec yolo-api python /app/tests/compare_padding_methods.py --iou-threshold 0.7

# Save results
docker compose exec yolo-api python /app/tests/compare_padding_methods.py --output-dir /app/results
```

#### test_shared_vs_per_request.sh - Client Performance Test

```bash
bash tests/test_shared_vs_per_request.sh
```

Compares:
- Shared gRPC client (enables batching)
- Per-request gRPC client (no batching)

#### create_test_images.py - Test Image Generator

```bash
# Generate standard test suite (8 sizes)
python tests/create_test_images.py --source /path/to/image.jpg

# Custom output directory
python tests/create_test_images.py --source image.jpg --output ./my_tests

# Generate specific sizes
python tests/create_test_images.py --source image.jpg --sizes 640x640 1920x1080

# Mix standard + custom sizes
python tests/create_test_images.py --source image.jpg --sizes standard 800x600

# Add filename prefix
python tests/create_test_images.py --source image.jpg --prefix test01
```

**Standard sizes:**
- Square: 640x640
- Portrait 2:3: 400x600
- Portrait 1:2: 320x640
- Landscape 3:2: 600x400
- Landscape 2:1: 640x320
- Wide 16:9: 1920x1080
- Tall 9:16: 1080x1920
- Small: 128x128

---

## Shared Utilities

### detection_comparison_utils.py

Provides shared utilities for comparing detection outputs:

**Classes:**
- `Detection`: Dataclass representing a single detection with box coordinates, confidence, and class ID
- `ComparisonMetrics`: Dataclass containing precision, recall, F1, IoU, and other comparison metrics

**Functions:**
- `calculate_iou(box1, box2)`: Calculate IoU between two boxes
- `calculate_iou_matrix(boxes1, boxes2)`: Calculate IoU matrix between two sets of boxes
- `match_detections(reference, test, iou_threshold)`: Match detections using greedy IoU matching
- `calculate_comparison_metrics(reference, test, iou_threshold)`: Calculate comprehensive metrics
- `parse_detections(detections)`: Parse detection dictionaries to Detection objects
- `format_metrics_table(metrics_dict)`: Format metrics as a table string

**Usage Example:**
```python
from tests.detection_comparison_utils import (
    parse_detections,
    calculate_comparison_metrics,
    format_metrics_table
)

# Parse API responses
ref_dets = parse_detections(track_a_response['detections'])
test_dets = parse_detections(track_c_response['detections'])

# Calculate metrics
metrics = calculate_comparison_metrics(ref_dets, test_dets, iou_threshold=0.5)

print(f"F1 Score: {metrics.f1_score:.3f}")
print(f"Mean IoU: {metrics.mean_iou:.3f}")
```

---

## Track Coverage Matrix

| Test File | A | B | C | D | E |
|-----------|:-:|:-:|:-:|:-:|:-:|
| `test_inference.sh` | Y | Y | Y | Y | - |
| `compare_tracks.py` | Y | Y | Y | Y | Y |
| `test_end2end_patch.py` | - | - | Y | Y | - |
| `test_onnx_end2end.py` | - | - | Y | - | - |
| `compare_padding_methods.py` | Y | - | - | Y | - |
| `test_track_e_ensemble.py` | - | - | - | - | Y |
| `test_track_e_images.py` | - | - | - | - | Y |
| `test_track_e_integration.py` | - | - | - | - | Y |
| `test_track_e_phase1_pipeline.py` | - | - | - | - | Y |
| `validate_mobileclip_triton.py` | - | - | - | - | Y |

---

## Common Workflows

### Pre-Deployment Validation

```bash
# 1. Start services
make up

# 2. Wait for services to be ready
sleep 30

# 3. Verify patch is applied
make test-patch

# 4. Test ONNX models locally
make test-onnx

# 5. Run comprehensive integration test
make test-inference

# 6. Validate cross-track consistency
make compare-tracks
```

### DALI Pipeline Validation

```bash
# 1. Generate test images with various aspect ratios
make test-create-images SOURCE=/path/to/source.jpg

# 2. Run DALI validation script
docker compose exec yolo-api python /app/dali/validate_dali_letterbox.py

# 3. Compare padding methods
make test-compare-padding

# 4. Test Track D via API
bash tests/test_inference.sh test_images small 10
```

### Debugging Track C Issues

```bash
# 1. Verify patch is applied
make test-patch

# 2. Test ONNX model locally (bypasses Triton)
make test-onnx

# 3. Test via Triton API
make test-track-c

# 4. Compare against baseline
docker compose exec yolo-api python /app/tests/compare_tracks.py --tracks A,C --max-images 5
```

### Track E Validation

```bash
# 1. Quick test
make test-track-e

# 2. Full Track E test suite
make test-track-e-full

# 3. Run integration tests
make test-integration

# 4. Compare with detection tracks
docker compose exec yolo-api python /app/tests/compare_tracks.py --tracks A,C,E
```

---

## Troubleshooting

### All Tests Fail

```bash
# Check services are running
docker compose ps
make status

# Check logs for errors
docker compose logs triton-api
docker compose logs yolo-api

# Restart services
make restart
```

### Specific Track Fails

| Track | Troubleshooting Steps |
|-------|----------------------|
| **A** | Check yolo-api logs, verify PyTorch models loaded |
| **B** | Check triton-api logs, verify TRT model.plan exists |
| **C** | Run `make test-patch`, check End2End model exists |
| **D** | Run `make validate-dali`, check DALI pipeline |
| **E** | Check OpenSearch status, verify MobileCLIP models |

### Missing Dependencies

```bash
# Install jq for shell scripts
sudo apt-get install jq

# Install Python dependencies
pip install -r requirements.txt
```

### Performance Issues

```bash
# Check GPU utilization
nvidia-smi -l 1

# Check batching is working
docker compose logs triton-api | grep "batch size"

# Run benchmarks for accurate measurements
cd benchmarks && ./triton_bench --mode full
```

---

## CI/CD Integration

Example CI script:

```bash
#!/bin/bash
set -e

echo "Starting services..."
docker compose up -d

echo "Waiting for services..."
sleep 45

echo "Running integration tests..."
bash tests/test_inference.sh ./test_images small 20

echo "Comparing track outputs..."
docker compose exec yolo-api python /app/tests/compare_tracks.py \
    --tracks A,C,D \
    --max-images 10 \
    --quiet

echo "All tests passed!"
```

---

## Related Documentation

- **[../CLAUDE.md](../CLAUDE.md)** - Project overview and development commands
- **[../benchmarks/README.md](../benchmarks/README.md)** - Performance benchmarking guide
- **[../dali/README.md](../dali/README.md)** - DALI pipeline documentation
- **[../docs/](../docs/)** - Additional documentation

---

**Last Updated:** December 2025
