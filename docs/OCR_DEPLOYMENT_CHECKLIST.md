# PP-OCRv5 Deployment Checklist

> Step-by-step checklist for deploying PP-OCRv5 text detection and recognition on NVIDIA Triton Inference Server.

---

## Prerequisites Checklist

### Hardware Requirements

- [ ] NVIDIA GPU with at least 4GB VRAM (tested on RTX A6000)
- [ ] 16GB+ system RAM for TensorRT engine building
- [ ] Fast SSD storage for model files (~50MB total)

### Software Requirements

- [ ] Docker with NVIDIA Container Toolkit
- [ ] Docker Compose v2+
- [ ] CUDA 12.x compatible GPU drivers
- [ ] TensorRT 10.x (included in Triton container)

### Container Requirements

The deployment uses two containers from the triton-api project:

| Container | Purpose | Image Base |
|-----------|---------|------------|
| `triton-api` | Triton Inference Server (TRT models) | `nvcr.io/nvidia/tritonserver:24.XX-py3` |
| `yolo-api` | FastAPI service (model export, API endpoints) | `python:3.12` |

---

## Step 1: Prepare Environment

### 1.1 Start Containers

```bash
# Navigate to project directory
cd /mnt/nvm/repos/triton-api

# Start both containers
docker compose up -d triton-api yolo-api

# Verify containers are running
docker compose ps
```

### 1.2 Verify GPU Access

```bash
# Check Triton has GPU access
docker compose exec triton-api nvidia-smi

# Expected: GPU information displayed
```

### 1.3 Check Triton Health

```bash
# Wait for Triton to be ready
curl -s localhost:4600/v2/health/ready

# Expected: HTTP 200 response
```

---

## Step 2: Download Models

### 2.1 Run Download Script

```bash
# Download PP-OCRv5 ONNX models and dictionaries
docker compose exec yolo-api python /app/export/download_paddleocr.py
```

### 2.2 Verify Downloads

```bash
# Check ONNX models exist
ls -la pytorch_models/paddleocr/

# Expected files:
#   ppocr_det_v5_mobile.onnx   (~4-5 MB)
#   ppocr_rec_v5_mobile.onnx   (~16 MB)
#   ppocr_keys_v1.txt          (dictionary)
#   en_dict.txt                (English dictionary)
```

---

## Step 3: Export to TensorRT

### 3.1 Free GPU Memory

Before exporting, unload existing models to free GPU memory:

```bash
# Unload models via Triton API
curl -X POST localhost:4600/v2/repository/models/paddleocr_det_trt/unload
curl -X POST localhost:4600/v2/repository/models/paddleocr_rec_trt/unload
curl -X POST localhost:4600/v2/repository/models/yolov11_small_trt/unload
curl -X POST localhost:4600/v2/repository/models/mobileclip_image_trt/unload
```

### 3.2 Export Detection Model

```bash
# Create model directory
mkdir -p models/paddleocr_det_trt/1

# Copy ONNX to models dir for container access
cp pytorch_models/paddleocr/ppocr_det_v5_mobile.onnx models/

# Run trtexec
docker compose exec triton-api /usr/src/tensorrt/bin/trtexec \
    --onnx=/models/ppocr_det_v5_mobile.onnx \
    --saveEngine=/models/paddleocr_det_trt/1/model.plan \
    --fp16 \
    --minShapes=x:1x3x32x32 \
    --optShapes=x:1x3x736x736 \
    --maxShapes=x:4x3x960x960 \
    --memPoolSize=workspace:4G
```

**CRITICAL**: Use `--memPoolSize=workspace:4G` syntax, NOT `--workspace=4096`.

### 3.3 Verify Detection Export

```bash
# Check model.plan was created
ls -la models/paddleocr_det_trt/1/model.plan

# Expected: ~5-10 MB file
```

### 3.4 Export Recognition Model

```bash
# Create model directory
mkdir -p models/paddleocr_rec_trt/1

# Copy ONNX to models dir for container access
cp pytorch_models/paddleocr/ppocr_rec_v5_mobile.onnx models/ 2>/dev/null || \
cp models/exports/ocr/en_ppocrv5_mobile_rec.onnx models/

# Run trtexec (takes 10-20 minutes for dynamic shapes)
docker compose exec triton-api /usr/src/tensorrt/bin/trtexec \
    --onnx=/models/ppocr_rec_v5_mobile.onnx \
    --saveEngine=/models/paddleocr_rec_trt/1/model.plan \
    --fp16 \
    --minShapes=x:1x3x48x8 \
    --optShapes=x:32x3x48x320 \
    --maxShapes=x:64x3x48x2048 \
    --memPoolSize=workspace:4G
```

### 3.5 Verify Recognition Export

```bash
# Check model.plan was created
ls -la models/paddleocr_rec_trt/1/model.plan

# Expected: ~15-30 MB file
```

---

## Step 4: Setup Dictionary

### 4.1 Copy Dictionary File

The dictionary file must be in the recognition model directory:

```bash
# Verify dictionary exists (should have 436 lines, plus blank+space = 438 total)
wc -l models/paddleocr_rec_trt/en_ppocrv5_dict.txt

# Expected output: 437 (436 chars + 1 empty line)
# Total classes: 438 (436 dict + blank token + space token)
```

### 4.2 Dictionary Format

The dictionary should contain one character per line:
- Lines 1-10: digits 0-9
- Lines 11-36: uppercase A-Z
- Lines 37-62: lowercase a-z
- Lines 63+: punctuation and special characters

---

## Step 5: Verify Configuration Files

### 5.1 Detection Config

File: `models/paddleocr_det_trt/config.pbtxt`

Key settings to verify:
- [ ] `platform: "tensorrt_plan"`
- [ ] `max_batch_size: 4`
- [ ] Input name: `x`
- [ ] Input dims: `[ 3, -1, -1 ]` (dynamic H, W)
- [ ] Output name: `fetch_name_0`

### 5.2 Recognition Config

File: `models/paddleocr_rec_trt/config.pbtxt`

Key settings to verify:
- [ ] `platform: "tensorrt_plan"`
- [ ] `max_batch_size: 1` (variable width per crop)
- [ ] Input name: `x`
- [ ] Input dims: `[ 3, 48, -1 ]` (fixed height, dynamic width)
- [ ] Output name: `fetch_name_0`
- [ ] Output dims: `[ -1, 438 ]` (dynamic timesteps, 438 chars)

### 5.3 Pipeline Config

File: `models/ocr_pipeline/config.pbtxt`

Key settings to verify:
- [ ] `backend: "python"`
- [ ] `max_batch_size: 0` (handles batching internally)
- [ ] Input names: `ocr_images`, `original_image`, `orig_shape`
- [ ] Output names: `num_texts`, `text_boxes`, `texts`, etc.

---

## Step 6: Deploy and Load Models

### 6.1 Restart Triton

```bash
docker compose restart triton-api

# Wait for ready
sleep 10
curl -s localhost:4600/v2/health/ready
```

### 6.2 Load OCR Models

```bash
# Load detection model
curl -X POST localhost:4600/v2/repository/models/paddleocr_det_trt/load

# Load recognition model
curl -X POST localhost:4600/v2/repository/models/paddleocr_rec_trt/load

# Load pipeline model
curl -X POST localhost:4600/v2/repository/models/ocr_pipeline/load
```

### 6.3 Verify Models Loaded

```bash
# Check all OCR models are READY
curl -s localhost:4600/v2/models | jq '.models[] | select(.name | contains("ocr") or contains("paddle"))'
```

Expected output for each model:
```json
{"name": "paddleocr_det_trt", "state": "READY"}
{"name": "paddleocr_rec_trt", "state": "READY"}
{"name": "ocr_pipeline", "state": "READY"}
```

---

## Step 7: Test OCR Pipeline

### 7.1 Create Test Image

```bash
# Create a simple test image with text
docker compose exec yolo-api python /app/scripts/create_ocr_test_images.py
```

### 7.2 Test via API

```bash
# Test OCR endpoint
curl -X POST http://localhost:4603/track_e/ocr/predict \
    -F "image=@test_images/ocr-synthetic/hello_world.jpg"
```

### 7.3 Expected Response

```json
{
  "status": "success",
  "num_texts": 2,
  "texts": ["Hello", "World"],
  "boxes_normalized": [[0.1, 0.2, 0.3, 0.25], [0.4, 0.2, 0.6, 0.25]],
  "det_scores": [0.95, 0.92],
  "rec_scores": [0.98, 0.97]
}
```

---

## Troubleshooting

### Problem: "Cudnn Error: CUDNN_STATUS_NOT_SUPPORTED"

**Cause**: Insufficient TensorRT workspace memory.

**Solution**: Use correct workspace syntax:
```bash
# Correct (TensorRT 10+):
--memPoolSize=workspace:4G

# Incorrect:
--workspace=4096
```

### Problem: "Engine deserialization failed"

**Cause**: TensorRT engine built on different GPU/driver version.

**Solution**: Rebuild engines on target machine:
```bash
rm models/paddleocr_det_trt/1/model.plan
rm models/paddleocr_rec_trt/1/model.plan
# Then re-run export steps
```

### Problem: Model fails to load in Triton

**Cause**: Config mismatch or missing files.

**Solution**: Check Triton logs:
```bash
docker compose logs triton-api | grep -i "paddleocr\|error"
```

### Problem: Empty OCR results

**Cause**: Detection threshold too high or preprocessing mismatch.

**Solution**:
1. Lower detection threshold in ocr_pipeline/model.py:
   - `thresh=0.2` (default 0.3)
   - `box_thresh=0.4` (default 0.5)
2. Verify input preprocessing:
   - Image should be BGR format
   - Normalized to [-1, 1] range

### Problem: GPU out of memory during export

**Cause**: Other models loaded using GPU memory.

**Solution**: Unload all models before export:
```bash
./scripts/export_paddleocr.sh status  # Check current state
# Then unload models as shown in Step 3.1
```

---

## Quick Reference

### Model Specifications

| Model | Input Shape | Output Shape | Precision |
|-------|-------------|--------------|-----------|
| Detection | `[B, 3, H, W]` H,W multiples of 32, max 960 | `[B, 1, H, W]` probability map | FP16 |
| Recognition | `[B, 3, 48, W]` W=8-2048 | `[B, T, 438]` logits | FP16 |

### File Locations

| File | Path |
|------|------|
| Detection TRT | `models/paddleocr_det_trt/1/model.plan` |
| Recognition TRT | `models/paddleocr_rec_trt/1/model.plan` |
| Dictionary | `models/paddleocr_rec_trt/en_ppocrv5_dict.txt` |
| Pipeline code | `models/ocr_pipeline/1/model.py` |
| Export script | `scripts/export_paddleocr.sh` |

### Useful Commands

```bash
# Full export pipeline
./scripts/export_paddleocr.sh all

# Check status
./scripts/export_paddleocr.sh status

# Export only detection
./scripts/export_paddleocr.sh det

# Export only recognition
./scripts/export_paddleocr.sh rec

# Restart Triton
docker compose restart triton-api
```

---

## Verification Checklist

Before marking deployment complete:

- [ ] Detection model.plan exists and is >1MB
- [ ] Recognition model.plan exists and is >5MB
- [ ] Dictionary file has 436+ characters
- [ ] All three models show "READY" in Triton
- [ ] Test image OCR returns expected text
- [ ] No errors in Triton logs

---

## Version Information

| Component | Version |
|-----------|---------|
| PP-OCR | v5 (Mobile) |
| TensorRT | 10.x |
| Triton | 24.XX |
| CUDA | 12.x |
| Python | 3.12 |

Created: 2025-01-10
Last Updated: 2025-01-10
