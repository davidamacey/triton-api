# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a high-performance deployment system for NVIDIA Triton Inference Server running Ultralytics YOLO models (YOLOv11) with a unified FastAPI service providing **four performance tracks** achieving up to **15x speedup** through GPU optimization, DALI preprocessing, and TensorRT acceleration.

## Architecture

### Unified Single-Service Design

The system uses Docker Compose to orchestrate services with a **unified API architecture**:

1. **triton-api**: NVIDIA Triton Inference Server
   - Runs on GPU (device_ids: [`0`])
   - Exposes ports 9500 (HTTP), 9501 (gRPC), 9502 (metrics)
   - Serves TensorRT models with dynamic batching
   - Loads 6 models: Track B (standard TRT), Track C (TRT End2End), Track D (DALI + 3 ensemble variants)

2. **yolo-api**: Unified FastAPI service (ALL FOUR TRACKS)
   - Python 3.12 container with Ultralytics SDK
   - Exposes port **9600** for ALL tracks
   - 32 workers × 512 concurrent requests = **16,384 total capacity**
   - Handles Track A (PyTorch) directly and proxies Tracks B/C/D to Triton
   - Located in [src/main.py](src/main.py)

### Four Performance Tracks

| Track | Endpoint Pattern | Backend | Speedup | Description |
|-------|-----------------|---------|---------|-------------|
| **A** | `/pytorch/predict/{model}` | PyTorch | 1x | Baseline - CPU NMS |
| **B** | `/predict/{model}` | Triton TRT | 2x | TensorRT + CPU NMS |
| **C** | `/predict/{model}_end2end` | Triton TRT | 4x | TensorRT + GPU NMS |
| **D** | `/predict/{model}_gpu_e2e_*` | Triton DALI | 10-15x | Full GPU pipeline |

**Track D has 3 variants:**
- `_gpu_e2e_streaming` - Low latency (video streaming)
- `_gpu_e2e` - Balanced (general purpose)
- `_gpu_e2e_batch` - Max throughput (batch processing)

### Model Communication Flow

- **Track A**: FastAPI → PyTorch models (loaded at startup, shared instances)
- **Tracks B/C/D**: FastAPI → Triton gRPC (port 8001) → GPU inference
- External clients access ALL tracks via: `localhost:9600`

### Model Directory Structure

```
models/
├── yolov11_small_trt/              # Track B: Standard TRT
│   ├── 1/model.plan
│   └── config.pbtxt
├── yolov11_small_trt_end2end/      # Track C: TRT + GPU NMS
│   ├── 1/model.plan
│   └── config.pbtxt
├── yolo_preprocess_dali/           # Track D: DALI preprocessing
│   ├── 1/model.dali
│   └── config.pbtxt
└── yolov11_small_gpu_e2e*/         # Track D: Ensembles (3 variants)
    └── config.pbtxt
```

Each model has:
- **config.pbtxt**: Triton configuration with TensorRT optimization, dynamic batching (max 128), and instance settings
- **model.plan**: TensorRT engine file (direct .plan, no warmup needed)
- **model.dali**: DALI preprocessing pipeline (GPU-accelerated)

PyTorch models are stored separately in `pytorch_models/` directory.

## Development Commands

### Deployment

```bash
# Start all services (requires GPU)
docker compose up -d

# View logs
docker compose logs -f triton-api
docker compose logs -f yolo-api

# Check service health
bash scripts/check_services.sh

# Stop services
docker compose down
```

### Model Export

Export YOLO models to all formats for the four tracks:

```bash
# Export all formats for small model
docker compose exec yolo-api python /app/export/export_models.py \
    --models small \
    --formats trt trt_end2end

# Quick wrapper for small model only
bash export/export_small_only.sh

# Download PyTorch models for Track A
bash export/download_pytorch_models.sh
```

This exports:
- **Track A**: Uses .pt files directly (no export needed)
- **Track B**: TensorRT engine (model.plan) with CPU NMS
- **Track C**: TensorRT End2End engine with compiled GPU NMS
- **Track D**: Uses Track C + DALI preprocessing pipeline

After export for Track D:
1. Create DALI pipeline: `docker compose exec yolo-api python /app/dali/create_dali_letterbox_pipeline.py`
2. Create ensembles: `docker compose exec yolo-api python /app/dali/create_ensembles.py --models small`
3. Restart Triton: `docker compose restart triton-api`

### Testing Inference

Comprehensive test script for all tracks:

```bash
# Test all 4 tracks with sample images
bash tests/test_inference.sh

# Test specific Track D variant
bash tests/test_inference.sh /path/to/images small 10 batch

# Test single ONNX End2End model (local debugging)
docker compose exec yolo-api python /app/tests/test_onnx_end2end.py

# Verify end2end patch is applied
docker compose exec yolo-api python /app/tests/test_end2end_patch.py
```

### Benchmarking

Single Go tool for comprehensive benchmarking:

```bash
cd benchmarks

# Build once
./build.sh

# Quick test (30 seconds, all tracks)
./triton_bench --mode quick

# Full benchmark (60 seconds, 128 clients)
./triton_bench --mode full --clients 128

# Test specific track
./triton_bench --mode full --track D_batch --clients 256

# Process all images
./triton_bench --mode all --images /path/to/images
```

Results are auto-saved to `benchmarks/results/` with timestamps.

## Important Implementation Details

### API Endpoints (All on port 9600)

[src/main.py](src/main.py) provides:

**Track A - PyTorch:**
- `POST /pytorch/predict/{model_name}`: PyTorch baseline inference
- `POST /pytorch/predict_batch/{model_name}`: Batch inference

**Tracks B/C/D - Triton:**
- `POST /predict/small`: Track B (standard TRT)
- `POST /predict/small_end2end`: Track C (TRT + GPU NMS)
- `POST /predict/small_gpu_e2e_streaming`: Track D streaming
- `POST /predict/small_gpu_e2e`: Track D balanced
- `POST /predict/small_gpu_e2e_batch`: Track D batch

Supported models: "small" (nano and medium in development)

Response format:
```json
{
  "detections": [
    {
      "x1": float, "y1": float, "x2": float, "y2": float,
      "confidence": float,
      "class": int
    }
  ],
  "status": "success",
  "track": "A|B|C|D",
  "backend": "pytorch|triton"
}
```

### Triton Server Configuration

- **Dynamic batching**: Preferred batch sizes [8, 16, 32, 64] with 5ms max queue delay for balanced performance
- **TensorRT optimization**: FP16 precision mode, direct .plan files (no warmup needed)
- **Instance count**: Varies by track - streaming (3 instances), balanced (2), batch (1)
- **Max batch size**: 128 for high throughput variants

### Image Processing

All inference endpoints handle image preprocessing:
- **Track A**: PyTorch handles preprocessing internally
- **Track B**: CPU preprocessing via Ultralytics wrapper
- **Track C**: CPU preprocessing, GPU NMS
- **Track D**: 100% GPU pipeline (nvJPEG decode + warp_affine + normalize)

### Detection Output Format

All inference methods return:
- `x1, y1, x2, y2`: Bounding box coordinates in pixels (original image dimensions)
- `confidence`: Detection confidence score (0-1)
- `class`: Integer class ID (0-79 for COCO dataset)

Class names are stored in model metadata and can be retrieved from `models/{model_name}/labels.txt`.

## Known Issues and Considerations

1. **First inference**: Even with direct .plan files, first inference may be slightly slower due to TensorRT engine loading into memory
2. **Batching benefits**: Significant speedup only visible with concurrent requests (use 32+ clients for benchmarking)
3. **Memory usage**: Track A loads PyTorch models at startup; Tracks B/C/D create Triton clients per-request
4. **DALI pipeline**: Track D requires affine transformation matrices calculated on CPU, then applied on GPU

## Dependencies

Key Python packages in [requirements.txt](requirements.txt):
- `ultralytics==8.3.18+`: YOLO model SDK with Triton client support and custom End2End patch
- `fastapi`, `uvicorn[standard]`: REST API server with high-performance workers
- `tritonclient[all]`: Direct Triton gRPC/HTTP communication
- `torch>=2.5.0`, `torchvision`: PyTorch backend for Track A
- `tensorrt-cu12==10.13.3.9`: TensorRT Python API (MUST match Triton server version)
- `nvidia-dali-cuda120`: GPU preprocessing for Track D
- `opencv-python`, `pillow`: Image processing
- `onnx>=1.12.0,<=1.19.1`, `onnxsim`, `onnxslim`: Model export and optimization

## Attribution

This project uses ~600 lines of custom code from the [levipereira/ultralytics](https://github.com/levipereira/ultralytics) fork (version 8.3.18) to enable end2end YOLO export with TensorRT EfficientNMS plugin integration.

**What This Enables:**
- Track C (TensorRT + GPU NMS): 4x performance improvement by embedding NMS into TensorRT engine
- Track D variants: Full GPU pipeline with DALI preprocessing + GPU NMS

**Key Components:**
- `export_onnx_trt()` method for ONNX export with TensorRT NMS plugin
- TRT_EfficientNMS custom operators for PyTorch → TensorRT conversion
- End2End_TRT wrapper class for seamless GPU inference

The patch is located in [src/ultralytics_patches/end2end_export.py](src/ultralytics_patches/end2end_export.py) and automatically applied during model export.

See [ATTRIBUTION.md](ATTRIBUTION.md) for complete third-party code attribution, licensing, and reference architectures.

## Production Deployment

When deploying to production:
1. All endpoints available on single port (9600) - simplifies load balancing
2. Choose track based on use case:
   - Track A: Development/debugging
   - Track B: Standard inference (2x faster)
   - Track C: High-performance inference (4x faster)
   - Track D: Maximum throughput (10-15x faster)
3. Configure dynamic batching in model config.pbtxt based on workload
4. Use monitoring stack (Prometheus + Grafana) to observe performance
5. Scale horizontally: deploy multiple instances behind load balancer

## Monitoring

The deployment includes a complete monitoring stack:
- **Prometheus**: Scrapes Triton metrics every 5s (port 9090)
- **Grafana**: Visualization dashboards (port 3000, admin/admin)
- **Loki + Promtail**: Log aggregation and shipping

View dashboards at http://localhost:3000 after starting services.
