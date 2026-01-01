# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a high-performance deployment system for NVIDIA Triton Inference Server running Ultralytics YOLO models (YOLOv11) with a unified FastAPI service providing **five performance tracks** achieving up to **15x speedup** through GPU optimization, DALI preprocessing, TensorRT acceleration, and visual search via MobileCLIP embeddings.

## Architecture

### Unified Single-Service Design

The system uses Docker Compose to orchestrate services with a **unified API architecture**:

1. **triton-api**: NVIDIA Triton Inference Server
   - Runs on GPU (device_ids: [`0`, `2`])
   - Exposes ports 4600 (HTTP), 4601 (gRPC), 4602 (metrics)
   - Serves TensorRT models with dynamic batching
   - Loads models for all tracks: Track B (TRT), Track C (TRT End2End), Track D (DALI ensemble), Track E (MobileCLIP + embeddings)

2. **yolo-api**: Unified FastAPI service (ALL FIVE TRACKS)
   - Python 3.12 container with Ultralytics SDK
   - Exposes port **4603** for ALL tracks
   - Worker count: 2 (dev with PyTorch) or 64 (production) × 512 concurrent = up to **32,768 capacity**
   - Handles Track A (PyTorch) directly and proxies Tracks B/C/D/E to Triton
   - Located in [src/main.py](src/main.py)

3. **opensearch**: Vector database for Track E visual search
   - OpenSearch 3.3.1 with k-NN plugin
   - Exposes port **4607** (REST API)
   - Security disabled for development

### Six Performance Tracks

| Track | Endpoint Pattern | Backend | Speedup | Description |
|-------|-----------------|---------|---------|-------------|
| **A** | `/pytorch/predict/{model}` | PyTorch | 1x | Baseline - CPU NMS |
| **B** | `/predict/{model}` | Triton TRT | 2x | TensorRT + CPU NMS |
| **C** | `/predict/{model}_end2end` | Triton TRT | 4x | TensorRT + GPU NMS |
| **D** | `/predict/{model}_gpu_e2e_*` | Triton DALI | 10-15x | Full GPU pipeline |
| **E** | `/track_e/*` | Triton DALI + OpenSearch | N/A | Visual search with MobileCLIP (DALI preprocessing) |
| **F** | `/track_f/*` | Triton TRT | N/A | Visual search with MobileCLIP (CPU preprocessing) |

**Track D has 3 variants:**
- `_gpu_e2e_streaming` - Low latency (video streaming)
- `_gpu_e2e` - Balanced (general purpose)
- `_gpu_e2e_batch` - Max throughput (batch processing)

**Track E endpoints (DALI GPU preprocessing):**
- `/track_e/predict` - YOLO + global embedding (single image)
- `/track_e/predict_batch` - **Batch processing (up to 64 images per request)**
- `/track_e/ingest` - Ingest images with embeddings
- `/track_e/search/image` - Image-to-image similarity search
- `/track_e/search/text` - Text-to-image search
- `/track_e/search/object` - Object-level search

**Track F endpoints (CPU preprocessing):**
- `/track_f/predict` - YOLO + global embedding (direct TRT, no DALI)

### DALI GPU Preprocessing Advantages

**Benchmark Results (RTX A6000):**
| Track | Preprocessing | Throughput | Notes |
|-------|--------------|------------|-------|
| E (DALI) | GPU | **130 RPS** | nvJPEG decode, GPU letterbox |
| F (CPU) | CPU | 30 RPS | PIL decode, cv2 letterbox |

**Why DALI is 4x faster than CPU preprocessing:**
1. **GPU-accelerated decode**: nvJPEG is 10-20x faster than PIL/cv2
2. **Parallel processing**: DALI processes multiple images concurrently on GPU
3. **Zero CPU-GPU transfer**: Preprocessed tensors stay on GPU memory
4. **Optimal batching**: Triton batches DALI requests efficiently

**For large photo libraries (50K+ images):**
- Use `/track_e/predict_batch` with batches of 16-64 images
- Reduces HTTP overhead and ensures full GPU utilization
- Expected throughput: 200+ images/second

### Model Communication Flow

- **Track A**: FastAPI → PyTorch models (loaded at startup, shared instances)
- **Tracks B/C/D**: FastAPI → Triton gRPC (port 8001) → GPU inference
- **Track E**: FastAPI → Triton gRPC → OpenSearch k-NN
- External clients access ALL tracks via: `localhost:4603`

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

**IMPORTANT: Code Hot Reloading**
- The `yolo-api` container uses volume mounts for `./src:/app/src` enabling hot reloading
- **DO NOT rebuild containers** when changing Python code - just restart the service
- To pick up code changes: `docker compose stop yolo-api && docker compose rm -f yolo-api && docker compose up -d yolo-api`
- Simple `docker compose restart yolo-api` may not reload all modules due to Python bytecode caching
- Only rebuild containers when `Dockerfile` or `requirements.txt` changes

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

Export YOLO models to all formats using Makefile commands:

```bash
# Export TRT + End2End for small model (recommended)
make export-models

# Or manually:
docker compose exec yolo-api python /app/export/export_models.py \
    --models small \
    --formats trt trt_end2end \
    --normalize-boxes

# Download PyTorch models for Track A
make download-pytorch
```

This exports:
- **Track A**: Uses .pt files directly (no export needed)
- **Track B**: TensorRT engine (model.plan) with CPU NMS
- **Track C**: TensorRT End2End engine with compiled GPU NMS
- **Track D**: Uses Track C + DALI preprocessing pipeline

After export for Track D:
```bash
make create-dali  # Creates DALI pipeline + ensembles and restarts Triton
```

For Track E (Visual Search):
```bash
make setup-track-e           # Complete Track E setup
# Or step-by-step:
make export-mobileclip       # Export MobileCLIP image/text encoders
make create-dali-dual        # Create triple-branch DALI pipeline
make restart-triton          # Load new models
```

### Testing Inference

Comprehensive test script for all tracks:

```bash
# Test all 5 tracks with sample images
make test-all-tracks

# Or individual tracks:
make test-track-a    # PyTorch
make test-track-b    # TensorRT
make test-track-c    # TensorRT + GPU NMS
make test-track-d    # DALI + GPU pipeline
make test-track-e    # Visual Search

# Compare detections across all tracks
make compare-tracks

# Verify end2end patch is applied
make test-patch
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

### API Endpoints (All on port 4603)

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

**Track E - Visual Search:**
- `POST /track_e/detect`: YOLO detection only
- `POST /track_e/predict`: Detection + global embedding
- `POST /track_e/predict_full`: Detection + global + per-box embeddings
- `POST /track_e/embed/image`: Image embedding only
- `POST /track_e/embed/text`: Text embedding only
- `POST /track_e/ingest`: Ingest image into OpenSearch index
- `POST /track_e/search/image`: Image-to-image similarity search
- `POST /track_e/search/text`: Text-to-image search
- `POST /track_e/search/object`: Object-level search
- `GET /track_e/index/stats`: Index statistics
- `POST /track_e/index/create`: Create/recreate index
- `DELETE /track_e/index`: Delete index

Supported models: "small" (nano and medium in development)

Response format (Tracks A-D):
```json
{
  "detections": [
    {
      "x1": float, "y1": float, "x2": float, "y2": float,
      "confidence": float,
      "class_id": int
    }
  ],
  "image": {"width": int, "height": int},
  "model": {"name": str, "backend": "pytorch|triton"},
  "track": "A|B|C|D",
  "total_time_ms": float
}
```

Response format (Track E search):
```json
{
  "results": [
    {"image_id": str, "score": float, "image_path": str}
  ],
  "total_results": int,
  "search_time_ms": float
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

All inference methods return normalized coordinates:
- `x1, y1, x2, y2`: Bounding box coordinates normalized to [0, 1] range
- `confidence`: Detection confidence score (0-1)
- `class_id`: Integer class ID (0-79 for COCO dataset)

To convert to pixel coordinates: `x1_px = x1 * image_width`, etc.

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
- `nvidia-dali-cuda120`: GPU preprocessing for Tracks D and E
- `opencv-python`, `pillow`: Image processing
- `onnx>=1.12.0,<=1.19.1`, `onnxsim`, `onnxslim`: Model export and optimization
- `opensearch-py>=2.3.0`: Async OpenSearch client for Track E vector search
- `transformers>=4.30.0`: CLIP tokenizer for Track E text search
- `timm`, `huggingface_hub`: Model loading for MobileCLIP export

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
1. All endpoints available on single port (4603) - simplifies load balancing
2. Choose track based on use case:
   - Track A: Development/debugging
   - Track B: Standard inference (2x faster)
   - Track C: High-performance inference (4x faster)
   - Track D: Maximum throughput (10-15x faster)
   - Track E: Visual search (image/text-to-image similarity)
3. Configure dynamic batching in model config.pbtxt based on workload
4. Use monitoring stack (Prometheus + Grafana) to observe performance
5. Scale horizontally: deploy multiple instances behind load balancer
6. **Track E production**: Enable OpenSearch security, configure multi-node cluster
7. **Workers**: Set `--workers=64` in docker-compose.yml for production throughput

## Monitoring

The deployment includes a complete monitoring stack:
- **Prometheus**: Scrapes Triton metrics every 5s (port 4604)
- **Grafana**: Visualization dashboards (port 4605, admin/admin)
- **Loki + Promtail**: Log aggregation and shipping (port 4606)
- **OpenSearch Dashboards**: Vector search management (port 4608)

View dashboards at http://localhost:4605 after starting services.
