# Unified YOLO Inference FastAPI Service

**Single service providing all 5 performance tracks** through one unified API.

## Service Overview

**Unified Architecture:** All tracks run in a **single FastAPI service** at port **4603**.

| Track | Technology | Endpoint Pattern | Speedup |
|-------|-----------|------------------|---------|
| **A** | PyTorch Direct | `/pytorch/predict/{model}` | 1x (baseline) |
| **B** | TRT + CPU NMS | `/predict/{model}` | 2x |
| **C** | TRT + GPU NMS | `/predict/{model}_end2end` | 4x |
| **D** | DALI + TRT + GPU NMS | `/predict/{model}_gpu_e2e_*` | 10-15x |
| **E** | MobileCLIP + OpenSearch | `/track_e/*` | Visual Search |

**Key Innovation:** Track A (PyTorch) is **embedded** in the same service as Tracks B/C/D/E, eliminating the need for separate containers and simplifying deployment.

---

## File Structure

```
src/
├── main.py                      # Application factory with lifespan management
│
├── routers/                     # FastAPI route handlers
│   ├── __init__.py              # Router exports
│   ├── health.py                # Health check endpoints (/, /health)
│   ├── track_a.py               # Track A: PyTorch inference
│   ├── triton.py                # Tracks B/C/D: Triton inference
│   └── track_e.py               # Track E: Visual search endpoints
│
├── services/                    # Business logic layer
│   ├── __init__.py              # Service exports
│   ├── inference.py             # Core inference logic for all tracks
│   ├── embedding.py             # MobileCLIP embedding generation
│   ├── image.py                 # Image processing service
│   └── visual_search.py         # OpenSearch visual search operations
│
├── clients/                     # External service clients
│   ├── __init__.py              # Client exports
│   ├── triton_client.py         # Triton gRPC client wrapper
│   ├── triton_pool.py           # Connection pooling for Triton
│   └── opensearch.py            # OpenSearch async client
│
├── schemas/                     # Pydantic models
│   ├── __init__.py              # Schema exports
│   ├── common.py                # Shared schemas (ImageInfo, etc.)
│   ├── detection.py             # Detection response schemas
│   └── track_e.py               # Track E specific schemas
│
├── config/                      # Configuration management
│   ├── __init__.py              # Config exports
│   └── settings.py              # Pydantic settings with validation
│
├── core/                        # Core application components
│   ├── __init__.py              # Core exports
│   ├── dependencies.py          # FastAPI dependencies & factories
│   └── exceptions.py            # Custom exception classes
│
├── utils/                       # Utility functions
│   ├── __init__.py              # Utility exports
│   ├── image_processing.py      # Image decode & validation
│   ├── pytorch_utils.py         # PyTorch inference helpers
│   ├── affine.py                # Affine transformation for DALI
│   └── cache.py                 # LRU caching utilities
│
└── ultralytics_patches/         # Custom Ultralytics modifications
    ├── __init__.py              # Patch exports
    └── end2end_export.py        # TensorRT NMS operator integration
```

---

## Architecture Overview

### Application Factory Pattern

The service uses FastAPI's application factory pattern in `main.py`:

```python
# main.py
from src.routers import health_router, track_a_router, track_e_router, triton_router

def create_app() -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    app.include_router(health_router)
    app.include_router(track_a_router)
    app.include_router(triton_router)
    app.include_router(track_e_router)
    return app
```

### Dependency Injection

Centralized in `core/dependencies.py`:
- `TritonClientFactory` - Manages Triton gRPC connections
- `OpenSearchClientFactory` - Manages OpenSearch connections
- `app_state` - Application-wide state (PyTorch models, settings)

### Service Layer

Business logic separated into `services/`:
- `InferenceService` - Handles Track A/B/C/D inference
- `EmbeddingService` - MobileCLIP image/text embeddings
- `VisualSearchService` - OpenSearch k-NN operations
- `ImageService` - Image processing and validation

---

## Track A: PyTorch Baseline

**Router:** `routers/track_a.py`

Direct PyTorch inference using Ultralytics YOLO - provides performance baseline.

### Endpoints

```
POST /pytorch/predict/{model_name}         # Single image
POST /pytorch/predict_batch/{model_name}   # Batch inference
```

### Example Usage

```bash
curl -X POST http://localhost:4603/pytorch/predict/small \
  -F "image=@test.jpg"
```

### Performance

- **Latency:** ~12-18ms (small model)
- **Throughput:** 80-120 images/sec
- **Speedup:** 1.0x (baseline reference)

---

## Track B: Standard TRT + CPU NMS

**Router:** `routers/triton.py`

TensorRT-optimized inference via Triton with Ultralytics wrapper.

### Endpoints

```
POST /predict/{model_name}         # Standard TRT
POST /predict_batch/{model_name}   # Batch inference
```

### Example Usage

```bash
curl -X POST http://localhost:4603/predict/small \
  -F "image=@test.jpg"
```

### Performance

- **Latency:** ~8-12ms
- **Throughput:** 150-250 images/sec
- **Speedup:** 1.5-2.5x faster than Track A

---

## Track C: End2End TRT + GPU NMS

**Router:** `routers/triton.py`
**Client:** `clients/triton_client.py`

Maximum performance with GPU NMS compiled into TensorRT engine.

### Endpoints

```
POST /predict/{model}_end2end         # GPU NMS
POST /predict_batch/{model}_end2end   # Batch inference
```

### Example Usage

```bash
curl -X POST http://localhost:4603/predict/small_end2end \
  -F "image=@test.jpg"
```

### Performance

- **Latency:** ~3-5ms
- **Throughput:** 300-500 images/sec
- **Speedup:** 3-5x faster than Track A

---

## Track D: DALI + TRT + GPU NMS (Full GPU Pipeline)

**Router:** `routers/triton.py`

100% GPU pipeline - preprocessing, inference, and NMS all on GPU.

### Three Performance Tiers

| Tier | Endpoint | Batching | Use Case |
|------|----------|----------|----------|
| **Streaming** | `small_gpu_e2e_streaming` | 0.1ms | Video streaming |
| **Balanced** | `small_gpu_e2e` | 0.5ms | General purpose |
| **Batch** | `small_gpu_e2e_batch` | 5ms | Offline processing |

### Example Usage

```bash
# Streaming (low latency)
curl -X POST http://localhost:4603/predict/small_gpu_e2e_streaming \
  -F "image=@test.jpg"

# Batch (max throughput)
curl -X POST http://localhost:4603/predict/small_gpu_e2e_batch \
  -F "image=@test.jpg"
```

### Performance

- **Latency:** 2-3ms (streaming) to 25-40ms (batch)
- **Throughput:** 1500-2500 images/sec
- **Speedup:** 10-15x faster than Track A

---

## Track E: Visual Search (MobileCLIP + OpenSearch)

**Router:** `routers/track_e.py`
**Services:** `services/embedding.py`, `services/visual_search.py`
**Client:** `clients/opensearch.py`

Visual search using MobileCLIP embeddings with OpenSearch k-NN.

### Endpoints

**Inference:**
```
POST /track_e/detect              # YOLO detection only
POST /track_e/predict             # Detection + global embedding
POST /track_e/predict_full        # Detection + global + per-box embeddings
POST /track_e/embed/image         # Image embedding only
POST /track_e/embed/text          # Text embedding only
```

**Search:**
```
POST /track_e/search/image        # Image-to-image similarity
POST /track_e/search/text         # Text-to-image search
POST /track_e/search/object       # Object-level search
```

**Index Management:**
```
POST /track_e/ingest              # Ingest image into index
POST /track_e/index/create        # Create/recreate index
DELETE /track_e/index             # Delete index
GET /track_e/index/stats          # Index statistics
```

### Example Usage

```bash
# Ingest an image
curl -X POST http://localhost:4603/track_e/ingest \
  -F "image=@photo.jpg" \
  -F "image_id=photo_001"

# Search by text
curl -X POST http://localhost:4603/track_e/search/text \
  -H "Content-Type: application/json" \
  -d '{"query": "red car", "k": 10}'

# Search by image
curl -X POST http://localhost:4603/track_e/search/image \
  -F "image=@query.jpg" \
  -F "k=10"
```

---

## Unified API Response Format

All detection tracks (A-D) return consistent response format:

```json
{
  "detections": [
    {
      "x1": 0.15,
      "y1": 0.22,
      "x2": 0.45,
      "y2": 0.68,
      "confidence": 0.92,
      "class_id": 0
    }
  ],
  "image": {"width": 1920, "height": 1080},
  "model": {"name": "yolov11_small", "backend": "triton"},
  "track": "C",
  "total_time_ms": 4.25
}
```

Coordinates are normalized to [0, 1] range.

---

## Configuration

Settings managed via `config/settings.py` with environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `TRITON_URL` | `triton-api:8001` | Triton gRPC address |
| `OPENSEARCH_URL` | `http://opensearch:9200` | OpenSearch REST address |
| `ENABLE_PYTORCH` | `true` | Enable Track A |
| `MAX_FILE_SIZE_MB` | `50` | Maximum upload size |
| `SLOW_REQUEST_THRESHOLD_MS` | `1000` | Log slow requests |

---

## Running the Service

### Via Docker Compose (Production)

```bash
# Start all services
docker compose up -d

# View logs
docker compose logs -f yolo-api

# Stop services
docker compose down
```

### Standalone (Development)

```bash
# Requires Triton server running
docker compose up -d triton-api opensearch

# Run unified service
uvicorn src.main:app \
  --host 0.0.0.0 \
  --port 4603 \
  --workers 2 \
  --reload
```

---

## Performance Comparison

| Metric | Track A | Track B | Track C | Track D |
|--------|---------|---------|---------|---------|
| **Latency (P50)** | 12-18ms | 8-12ms | 3-5ms | **2-3ms** |
| **Throughput** | 80-120 rps | 150-250 rps | 300-500 rps | **1500-2500 rps** |
| **Speedup** | 1.0x | 2.0x | 4.0x | **12.5x** |
| **Preprocessing** | CPU | CPU | CPU | **GPU (DALI)** |
| **NMS Location** | CPU | CPU | **GPU** | **GPU** |

**Processing 100,000 images:**
- Track A: ~14 minutes
- Track B: ~7 minutes
- Track C: ~3.5 minutes
- Track D: **~40 seconds**

---

## Related Documentation

- [../README.md](../README.md) - Project overview
- [../CLAUDE.md](../CLAUDE.md) - Development instructions
- [utils/README.md](utils/README.md) - Utilities documentation
- [../docs/TRACK_E_GUIDE.md](../docs/TRACK_E_GUIDE.md) - Track E deep dive
