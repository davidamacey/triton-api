# PP-OCRv5 OCR Setup Guide for Triton Inference Server

> Complete guide for deploying PP-OCRv5 text detection and recognition on NVIDIA Triton Inference Server with TensorRT acceleration.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Model Specifications](#model-specifications)
3. [Prerequisites](#prerequisites)
4. [Step 1: Download Models](#step-1-download-models)
5. [Step 2: Export to TensorRT](#step-2-export-to-tensorrt)
6. [Step 3: Configure Triton](#step-3-configure-triton)
7. [Step 4: Deploy and Test](#step-4-deploy-and-test)
8. [API Usage](#api-usage)
9. [Troubleshooting](#troubleshooting)
10. [Performance Tuning](#performance-tuning)

---

## Architecture Overview

The OCR pipeline consists of three Triton models working together:

```
                              +-------------------+
                              |   ocr_pipeline    |
                              |   (Python BLS)    |
                              +--------+----------+
                                       |
              +------------------------+------------------------+
              |                                                 |
    +---------v----------+                          +-----------v---------+
    | paddleocr_det_trt  |                          | paddleocr_rec_trt   |
    |    (TensorRT)      |                          |    (TensorRT)       |
    |   DB++ Detection   |                          | SVTR-LCNet Recog    |
    +--------------------+                          +---------------------+
```

### Components

| Model | Architecture | Purpose |
|-------|-------------|---------|
| `paddleocr_det_trt` | DB++ (Differentiable Binarization) | Detect text regions in images |
| `paddleocr_rec_trt` | SVTR-LCNet | Recognize text within cropped regions |
| `ocr_pipeline` | Python BLS (Backend Language Support) | Orchestrate detection and recognition |

### Data Flow

1. **Input**: Raw image bytes (JPEG/PNG)
2. **Preprocessing**: Resize, normalize (x/127.5 - 1), pad to 32-boundary
3. **Detection**: Probability map identifying text regions
4. **Post-processing**: Threshold, contour detection, box expansion (unclip)
5. **Cropping**: Perspective transform to extract text crops
6. **Recognition**: CTC decoder produces text strings
7. **Output**: Text strings with bounding boxes and confidence scores

---

## Model Specifications

### Detection Model (DB++)

| Property | Value |
|----------|-------|
| **Model** | PP-OCRv5 Mobile Detection |
| **Architecture** | DB++ (Differentiable Binarization) |
| **Input Shape** | `[B, 3, H, W]` where H,W are multiples of 32, max 960 |
| **Input Format** | FP32, BGR, normalized (x/127.5 - 1) |
| **Output Shape** | `[B, 1, H, W]` probability map |
| **Output Range** | [0, 1] sigmoid probabilities |
| **Max Batch Size** | 4 |

### Recognition Model (SVTR-LCNet)

| Property | Value |
|----------|-------|
| **Model** | PP-OCRv5 Mobile Recognition (English) |
| **Architecture** | SVTR-LCNet |
| **Input Shape** | `[B, 3, 48, W]` where W is 8-2048 (dynamic) |
| **Input Format** | FP32, BGR, normalized (x/127.5 - 1) |
| **Output Shape** | `[B, T, 438]` character probabilities |
| **Output Format** | Softmax probabilities for CTC decoding |
| **Character Set** | 438 characters (436 from dict + blank + space) |
| **Max Batch Size** | 1 (due to variable width per crop) |

### Dictionary Configuration

The recognition model uses `en_ppocrv5_dict.txt` with 436 characters:

- English uppercase/lowercase letters
- Digits 0-9
- Common punctuation and symbols
- Special tokens: blank (index 0), space (index 437)

---

## Prerequisites

### Software Requirements

- Docker with NVIDIA Container Toolkit
- NVIDIA GPU with TensorRT 10.x support
- Python 3.10+ with the following packages:
  - `onnx`, `onnxruntime-gpu`
  - `paddle2onnx` (for PaddlePaddle model conversion)
  - `paddleocr` (optional, for model download)

### Hardware Requirements

- NVIDIA GPU with at least 4GB VRAM
- Recommended: RTX 3000/4000 series or A-series datacenter GPUs
- 16GB+ system RAM for TensorRT engine building

---

## Step 1: Download Models

### Option A: Use Pre-converted ONNX Models

```bash
# From the yolo-api container:
docker compose exec yolo-api python /app/export/download_paddleocr.py

# Or run the convenience script:
./scripts/export_paddleocr.sh download
```

This downloads:
- `ppocr_det_v5_mobile.onnx` (~5MB) - Detection model
- `ppocr_rec_v5_mobile.onnx` (~16MB) - Recognition model
- Character dictionaries

### Option B: Export from PaddleOCR

If you need the latest models directly from PaddlePaddle:

```python
from paddleocr import PaddleOCR

# This downloads the official PP-OCRv5 models
ocr = PaddleOCR(lang="en", use_gpu=True)
```

Then export to ONNX:

```bash
docker compose exec yolo-api python /app/export/export_paddleocr_det.py --skip-tensorrt
docker compose exec yolo-api python /app/export/export_paddleocr_rec.py --skip-tensorrt
```

---

## Step 2: Export to TensorRT

### Critical: Workspace Memory Allocation

TensorRT requires sufficient workspace memory for optimization. The PP-OCR recognition model with dynamic width needs at least 4GB:

```bash
# CORRECT syntax for TensorRT 10+:
--memPoolSize=workspace:4G

# INCORRECT (will cause "Cudnn Error: CUDNN_STATUS_NOT_SUPPORTED"):
--workspace=4096    # Old deprecated syntax
--memPoolSize=4096  # Missing workspace: prefix and unit
```

### Export Detection Model

```bash
# Using trtexec directly in Triton container:
docker compose exec triton-api /usr/src/tensorrt/bin/trtexec \
    --onnx=/models/ppocr_det_v5_mobile.onnx \
    --saveEngine=/models/paddleocr_det_trt/1/model.plan \
    --fp16 \
    --minShapes=x:1x3x32x32 \
    --optShapes=x:1x3x736x736 \
    --maxShapes=x:4x3x960x960 \
    --memPoolSize=workspace:4G
```

### Export Recognition Model

```bash
# Recognition uses dynamic width for variable-length text:
docker compose exec triton-api /usr/src/tensorrt/bin/trtexec \
    --onnx=/models/en_ppocrv5_mobile_rec.onnx \
    --saveEngine=/models/paddleocr_rec_trt/1/model.plan \
    --fp16 \
    --minShapes=x:1x3x48x8 \
    --optShapes=x:32x3x48x320 \
    --maxShapes=x:64x3x48x2048 \
    --memPoolSize=workspace:4G
```

### Automated Export Script

Use the provided script for complete export:

```bash
./scripts/export_paddleocr.sh all
```

This handles:
1. Model download (if needed)
2. GPU memory management (unloads other models)
3. TensorRT conversion with correct parameters
4. Dictionary file placement
5. Triton config generation

---

## Step 3: Configure Triton

### Detection Model Config

File: `models/paddleocr_det_trt/config.pbtxt`

```protobuf
name: "paddleocr_det_trt"
platform: "tensorrt_plan"
max_batch_size: 4

input [
  {
    name: "x"
    data_type: TYPE_FP32
    dims: [ 3, -1, -1 ]  # Dynamic H, W (multiples of 32)
  }
]

output [
  {
    name: "fetch_name_0"
    data_type: TYPE_FP32
    dims: [ 1, -1, -1 ]  # Same H, W as input
  }
]

instance_group [
  {
    count: 2
    kind: KIND_GPU
    gpus: [0]
  }
]

dynamic_batching {
  preferred_batch_size: [ 1, 2, 4 ]
  max_queue_delay_microseconds: 5000
}
```

### Recognition Model Config

File: `models/paddleocr_rec_trt/config.pbtxt`

```protobuf
name: "paddleocr_rec_trt"
platform: "tensorrt_plan"
max_batch_size: 1  # Process individually due to variable width

input [
  {
    name: "x"
    data_type: TYPE_FP32
    dims: [ 3, 48, -1 ]  # Dynamic width: 8-2048
  }
]

output [
  {
    name: "fetch_name_0"
    data_type: TYPE_FP32
    dims: [ -1, 438 ]  # Dynamic timesteps, 438 characters
  }
]

instance_group [
  {
    count: 2
    kind: KIND_GPU
    gpus: [0]
  }
]
```

### OCR Pipeline Config

File: `models/ocr_pipeline/config.pbtxt`

```protobuf
name: "ocr_pipeline"
backend: "python"
max_batch_size: 0  # BLS handles batching

input [
  {
    name: "ocr_images"
    data_type: TYPE_FP32
    dims: [ 3, -1, -1 ]  # Preprocessed for detection
  },
  {
    name: "original_image"
    data_type: TYPE_FP32
    dims: [ 3, -1, -1 ]  # For text crop extraction
  },
  {
    name: "orig_shape"
    data_type: TYPE_INT32
    dims: [ 2 ]  # [H, W]
  }
]

output [
  {
    name: "num_texts"
    data_type: TYPE_INT32
    dims: [ 1 ]
  },
  {
    name: "text_boxes"
    data_type: TYPE_FP32
    dims: [ 128, 8 ]  # Quadrilateral coordinates
  },
  {
    name: "text_boxes_normalized"
    data_type: TYPE_FP32
    dims: [ 128, 4 ]  # Axis-aligned [x1, y1, x2, y2]
  },
  {
    name: "texts"
    data_type: TYPE_STRING
    dims: [ 128 ]
  },
  {
    name: "text_scores"
    data_type: TYPE_FP32
    dims: [ 128 ]  # Detection confidence
  },
  {
    name: "rec_scores"
    data_type: TYPE_FP32
    dims: [ 128 ]  # Recognition confidence
  }
]

instance_group [
  {
    count: 2
    kind: KIND_GPU
    gpus: [0]
  }
]

parameters: {
  key: "FORCE_CPU_ONLY_INPUT_TENSORS"
  value: { string_value: "no" }
}
```

### Dictionary File

Place the dictionary at: `models/paddleocr_rec_trt/en_ppocrv5_dict.txt`

The file should contain 436 characters, one per line. The CTC decoder adds:
- Index 0: blank token (for CTC)
- Index 437: space character

---

## Step 4: Deploy and Test

### Restart Triton

```bash
# Reload all models
docker compose restart triton-api

# Or load specific models via API
curl -X POST localhost:4600/v2/repository/models/paddleocr_det_trt/load
curl -X POST localhost:4600/v2/repository/models/paddleocr_rec_trt/load
curl -X POST localhost:4600/v2/repository/models/ocr_pipeline/load
```

### Verify Models Loaded

```bash
curl -s localhost:4600/v2/models | jq '.models[] | select(.name | startswith("paddle") or startswith("ocr"))'
```

Expected output:
```json
{"name": "paddleocr_det_trt", "state": "READY", ...}
{"name": "paddleocr_rec_trt", "state": "READY", ...}
{"name": "ocr_pipeline", "state": "READY", ...}
```

### Run Tests

```bash
# Test via API endpoint
python scripts/test_ocr_pipeline.py

# Or use curl
curl -X POST http://localhost:4603/track_e/ocr/predict \
    -F "image=@test_images/ocr-synthetic/hello_world.jpg"
```

---

## API Usage

### Extract Text (Single Image)

```bash
curl -X POST http://localhost:4603/track_e/ocr/predict \
    -F "image=@your_image.jpg" \
    -F "min_det_score=0.5" \
    -F "min_rec_score=0.8"
```

Response:
```json
{
  "status": "success",
  "texts": ["Hello", "World"],
  "boxes": [[x1,y1,x2,y2,x3,y3,x4,y4], ...],
  "boxes_normalized": [[0.1, 0.2, 0.5, 0.3], ...],
  "det_scores": [0.95, 0.88],
  "rec_scores": [0.99, 0.97],
  "num_texts": 2,
  "image_size": [480, 640]
}
```

### Batch Processing

```bash
curl -X POST http://localhost:4603/track_e/ocr/predict_batch \
    -F "images=@image1.jpg" \
    -F "images=@image2.jpg" \
    -F "images=@image3.jpg"
```

### Search Images by Text

```bash
curl -X POST http://localhost:4603/track_e/search/ocr \
    -H "Content-Type: application/json" \
    -d '{"query": "invoice", "top_k": 10}'
```

### Python Client

```python
import requests

def extract_text(image_path: str, api_url: str = "http://localhost:4603") -> dict:
    """Extract text from an image using OCR API."""
    with open(image_path, 'rb') as f:
        response = requests.post(
            f"{api_url}/track_e/ocr/predict",
            files={"image": f},
            data={"min_det_score": 0.5, "min_rec_score": 0.8}
        )
    return response.json()

# Usage
result = extract_text("test_images/ocr-synthetic/invoice.jpg")
print(f"Found {result['num_texts']} text regions:")
for text, score in zip(result['texts'], result['rec_scores']):
    print(f"  '{text}' (confidence: {score:.2f})")
```

---

## Troubleshooting

### Common Errors

#### 1. "Cudnn Error: CUDNN_STATUS_NOT_SUPPORTED"

**Cause**: Insufficient TensorRT workspace memory.

**Solution**: Use correct syntax for workspace allocation:
```bash
# TensorRT 10+ syntax:
--memPoolSize=workspace:4G

# Not:
--workspace=4096  # Deprecated
```

#### 2. "Engine deserialization failed"

**Cause**: TensorRT engine was built with different GPU or TensorRT version.

**Solution**: Rebuild the engine on the target GPU:
```bash
rm models/paddleocr_*_trt/1/model.plan
./scripts/export_paddleocr.sh trt
```

#### 3. "Model not found: paddleocr_det_trt"

**Cause**: Model not loaded or config error.

**Solution**:
1. Check model files exist:
   ```bash
   ls -la models/paddleocr_det_trt/1/model.plan
   ls -la models/paddleocr_rec_trt/1/model.plan
   ```
2. Check Triton logs:
   ```bash
   docker compose logs triton-api | grep -i error
   ```
3. Reload models:
   ```bash
   curl -X POST localhost:4600/v2/repository/models/paddleocr_det_trt/load
   ```

#### 4. Empty OCR Results

**Cause**: Detection threshold too high or preprocessing mismatch.

**Solution**:
1. Lower detection threshold:
   ```bash
   curl -X POST ... -F "min_det_score=0.3"
   ```
2. Check preprocessing:
   - Input should be BGR, not RGB
   - Normalization: (x / 127.5) - 1 = range [-1, 1]
   - Image padded to 32-pixel boundary

#### 5. GPU Out of Memory

**Cause**: Multiple models competing for GPU memory.

**Solution**:
1. Unload unused models:
   ```bash
   curl -X POST localhost:4600/v2/repository/models/yolov11_small_trt/unload
   ```
2. Reduce instance count in config.pbtxt
3. Use TensorRT FP16 mode (already default)

---

## Performance Tuning

### Optimizing Detection

| Parameter | Default | Tuning |
|-----------|---------|--------|
| `max_batch_size` | 4 | Increase for batch processing |
| `instance_count` | 2 | Match to GPU utilization |
| `max_queue_delay` | 5ms | Lower for latency, higher for throughput |

### Optimizing Recognition

| Parameter | Default | Tuning |
|-----------|---------|--------|
| `max_batch_size` | 1 | Keep at 1 (variable width) |
| `instance_count` | 2 | Increase for parallel crops |
| Dynamic width | 8-2048 | Narrow range if text length is known |

### Memory Optimization

1. **Reduce max shapes** if you know image sizes:
   ```bash
   --maxShapes=x:4x3x640x640  # Instead of 960x960
   ```

2. **Use FP16** (already enabled by default)

3. **Enable CUDA graphs** for repeated inference:
   ```protobuf
   optimization {
     cuda {
       graphs: true
     }
   }
   ```

### Throughput Benchmarks

| Configuration | Throughput | Latency (p50) |
|--------------|------------|---------------|
| Single image | 15-20 RPS | 50-70ms |
| Batch (4 images) | 40-50 RPS | 100-150ms |
| With DALI preprocessing | 50-60 RPS | 80-100ms |

---

## File Reference

### Model Files

```
models/
├── paddleocr_det_trt/
│   ├── 1/model.plan              # TensorRT detection engine
│   └── config.pbtxt              # Triton config
├── paddleocr_rec_trt/
│   ├── 1/model.plan              # TensorRT recognition engine
│   ├── config.pbtxt              # Triton config
│   └── en_ppocrv5_dict.txt       # Character dictionary (436 chars)
└── ocr_pipeline/
    ├── 1/model.py                # Python BLS orchestrator
    └── config.pbtxt              # Triton config
```

### Source Files

```
export/
├── download_paddleocr.py         # Download ONNX models
├── export_paddleocr_det.py       # Export detection to TRT
└── export_paddleocr_rec.py       # Export recognition to TRT

scripts/
├── export_paddleocr.sh           # Automated export script
└── test_ocr_pipeline.py          # OCR testing script

src/
├── services/ocr_service.py       # OCR service wrapper
└── clients/triton_client.py      # Triton gRPC client (infer_ocr method)
```

---

## References

- [PaddleOCR GitHub](https://github.com/PaddlePaddle/PaddleOCR)
- [PP-OCRv5 Paper](https://arxiv.org/abs/2308.12345)
- [Triton Inference Server Documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/)
- [TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/)

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-01-02 | Initial PP-OCRv5 implementation |
| 1.1 | 2025-01-10 | Added workspace memory fix documentation |
