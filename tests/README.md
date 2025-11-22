# Testing Guide

Comprehensive testing tools for the Triton YOLO Inference System.

---

## Test Files

| File | Purpose | Type |
|------|---------|------|
| `test_inference.sh` | **Main integration test** - Tests all 4 tracks via API | Integration |
| `test_onnx_end2end.py` | Debug ONNX models locally (bypasses Triton) | Unit/Debug |
| `test_end2end_patch.py` | Verify ultralytics end2end patch is applied | Validation |
| `create_test_images.py` | Generate test images in various sizes | Utility |

---

## Quick Start

### Test All 4 Tracks

```bash
# Test all tracks with default settings (10 images each)
bash tests/test_inference.sh

# Test with custom image directory
bash tests/test_inference.sh /path/to/images

# Test specific model size
bash tests/test_inference.sh /path/to/images medium

# Test with more images
bash tests/test_inference.sh /path/to/images small 50

# Test specific Track D variant
bash tests/test_inference.sh /path/to/images small 10 batch
```

**What it tests:**
- ✅ Track A: PyTorch + CPU NMS (localhost:9600/pytorch/predict/)
- ✅ Track B: TensorRT + CPU NMS (localhost:9600/predict/)
- ✅ Track C: TensorRT + GPU NMS (localhost:9600/predict/*_end2end)
- ✅ Track D: DALI + TRT + GPU NMS (localhost:9600/predict/*_gpu_e2e*)

**Output:** Pass/fail for each track with latency measurements

---

## Test Details

### test_inference.sh - Main Integration Test

**Purpose:** Comprehensive test of all 4 performance tracks via FastAPI endpoints

**Usage:**
```bash
bash tests/test_inference.sh [IMAGE_DIR] [MODEL_SIZE] [NUM_TESTS] [TRACK_D_VARIANT]
```

**Arguments:**
- `IMAGE_DIR`: Directory containing test images (default: `/mnt/nvm/KILLBOY_SAMPLE_PICTURES`)
- `MODEL_SIZE`: Model size - `nano`, `small`, or `medium` (default: `small`)
- `NUM_TESTS`: Number of images to test per track (default: `10`)
- `TRACK_D_VARIANT`: Track D variant - `streaming`, `batch`, or empty for balanced (default: balanced)

**Examples:**
```bash
# Quick test (10 images)
bash tests/test_inference.sh

# Full test (50 images)
bash tests/test_inference.sh /mnt/nvm/KILLBOY_SAMPLE_PICTURES small 50

# Test Track D batch variant
bash tests/test_inference.sh /mnt/nvm/KILLBOY_SAMPLE_PICTURES small 10 batch

# Test with nano model
bash tests/test_inference.sh /mnt/nvm/KILLBOY_SAMPLE_PICTURES nano 10
```

**Requirements:**
- Services running: `docker compose up -d`
- Test images available
- `jq` installed (for JSON parsing)

---

### test_onnx_end2end.py - Local ONNX Debugging

**Purpose:** Test ONNX End2End models locally using ONNX Runtime (bypasses Triton)

**When to use:**
- Debugging model export issues
- Isolating model behavior from Triton serving
- Validating ONNX model outputs before deploying to Triton

**Usage:**
```bash
# Test with default paths
docker compose exec yolo-api python /app/tests/test_onnx_end2end.py

# Test specific image
docker compose exec yolo-api python /app/tests/test_onnx_end2end.py /path/to/image.jpg

# Test specific model
docker compose exec yolo-api python /app/tests/test_onnx_end2end.py /path/to/image.jpg /path/to/model.onnx
```

**What it does:**
1. Loads ONNX End2End model using ONNX Runtime
2. Applies letterbox preprocessing (same as YOLO)
3. Runs inference
4. Shows raw outputs (640x640 space) and transformed outputs (original image space)

**Output:**
- Model inputs/outputs structure
- Detection results in both padded and original coordinates
- Useful for debugging coordinate transformation issues

**Requirements:**
- ONNX model file (`models/yolov11_small_end2end/1/model.onnx`)
- `onnxruntime` package installed

---

### test_end2end_patch.py - Patch Verification

**Purpose:** Verify that the ultralytics end2end patch is correctly applied

**When to use:**
- After installing/updating dependencies
- Troubleshooting Track C export issues
- Verifying patch is active before model export

**Usage:**
```bash
docker compose exec yolo-api python /app/tests/test_end2end_patch.py
```

**What it tests:**
1. Patch module imports successfully
2. `export_onnx_trt` method exists on Exporter class
3. TRT operators are available
4. Can load YOLO model (if .pt file exists)

**Requirements:**
- ultralytics package installed
- Patch applied via `src/ultralytics_patches/`

---

### create_test_images.py - Test Image Generator

**Purpose:** Generate test images in various sizes from a source image

**When to use:**
- Creating test suite for DALI letterbox validation
- Testing various aspect ratios
- Generating consistent test data

**Usage:**
```bash
# Generate standard test suite (8 sizes)
python tests/create_test_images.py --source /path/to/image.jpg

# Custom output directory
python tests/create_test_images.py --source image.jpg --output ./my_tests

# Generate specific custom sizes
python tests/create_test_images.py --source image.jpg --sizes 640x640 1920x1080

# Mix standard + custom sizes
python tests/create_test_images.py --source image.jpg --sizes standard 800x600

# Add filename prefix
python tests/create_test_images.py --source image.jpg --prefix test01
```

**Standard sizes:**
- Square: 640×640
- Portrait 2:3: 400×600
- Portrait 1:2: 320×640
- Landscape 3:2: 600×400
- Landscape 2:1: 640×320
- Wide 16:9: 1920×1080
- Tall 9:16: 1080×1920
- Small: 128×128

**Arguments:**
- `--source`, `-s`: Source image file path (required)
- `--output`, `-o`: Output directory (default: `./test_images/generated`)
- `--sizes`: Sizes to generate - `standard` or custom like `640x480` (default: `standard`)
- `--quality`, `-q`: JPEG quality 1-100 (default: `95`)
- `--prefix`: Prefix for output filenames

---

## Common Testing Workflows

### Pre-Deployment Validation

```bash
# 1. Verify patch is applied
docker compose exec yolo-api python /app/tests/test_end2end_patch.py

# 2. Test ONNX models locally
docker compose exec yolo-api python /app/tests/test_onnx_end2end.py

# 3. Start services
docker compose up -d

# 4. Run comprehensive integration test
bash tests/test_inference.sh
```

### DALI Letterbox Validation

```bash
# 1. Generate test images with various aspect ratios
python tests/create_test_images.py --source /path/to/image.jpg

# 2. Run DALI validation script
docker compose exec yolo-api python /app/dali/validate_dali_letterbox.py

# 3. Test Track D via API
bash tests/test_inference.sh /path/to/images small 10
```

### Debugging Track C Issues

```bash
# 1. Verify patch
docker compose exec yolo-api python /app/tests/test_end2end_patch.py

# 2. Test ONNX model locally
docker compose exec yolo-api python /app/tests/test_onnx_end2end.py

# 3. Test via Triton
bash tests/test_inference.sh /path/to/images small 5
```

---

## Troubleshooting

### Test Failures

**All tracks fail:**
```bash
# Check services are running
docker compose ps

# Check logs
docker compose logs triton-api
docker compose logs yolo-api
docker compose logs yolo-api

# Restart services
docker compose restart
```

**Specific track fails:**
- **Track A fails:** Check yolo-api logs
- **Track B/C/D fail:** Check triton-api and yolo-api logs
- **Track C fails:** Run `test_end2end_patch.py` to verify patch
- **Track D fails:** Run DALI validation in `dali/validate_dali_letterbox.py`

**Missing dependencies:**
```bash
# Install jq for test_inference.sh
sudo apt-get install jq

# Install PIL for create_test_images.py (should be in requirements.txt)
pip install Pillow
```

### Performance Issues

If tests pass but latency is high:
```bash
# Check GPU utilization
nvidia-smi -l 1

# Check batching is working
docker compose logs triton-api | grep "batch size"

# Check worker count (should be 33 for yolo-api)
docker compose exec yolo-api ps aux | grep uvicorn | wc -l

# Run proper benchmarks
cd benchmarks && ./triton_bench --mode full
```

---

## CI/CD Integration

For automated testing:

```bash
#!/bin/bash
# ci-test.sh

set -e

echo "Starting services..."
docker compose up -d

echo "Waiting for services to be ready..."
sleep 30

echo "Running integration tests..."
bash tests/test_inference.sh /path/to/test/images small 20

echo "All tests passed!"
```

---

## Test Coverage

| Component | Integration Test | Unit Test | Debug Tool |
|-----------|-----------------|-----------|------------|
| PyTorch API | ✅ test_inference.sh | - | - |
| Triton TRT | ✅ test_inference.sh | - | - |
| End2End TRT | ✅ test_inference.sh | ✅ test_onnx_end2end.py | - |
| DALI Pipeline | ✅ test_inference.sh | ✅ dali/validate_dali_letterbox.py | - |
| Ultralytics Patch | - | ✅ test_end2end_patch.py | - |

---

## Related Documentation

- **[AUTOMATION.md](../AUTOMATION.md)** - Complete automation guide
- **[dali/](../dali/)** - DALI validation scripts
- **[benchmarks/README.md](../benchmarks/README.md)** - Performance benchmarking
- **[docs/TESTING.md](../docs/TESTING.md)** - Testing strategy

---

**Last Updated:** November 2025
