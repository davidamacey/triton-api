# Unified YOLO Inference FastAPI Service

**Single service providing all 4 performance tracks** through one unified API.

## 🎯 Service Overview

**Unified Architecture:** All tracks run in a **single FastAPI service** at port **9600**.

| Track | Technology | Endpoint Pattern | Speedup |
|-------|-----------|------------------|---------|
| **A** | PyTorch Direct | `/pytorch/predict/{model}` | 1x (baseline) |
| **B** | TRT + CPU NMS | `/predict/{model}` | 2x |
| **C** | TRT + GPU NMS | `/predict/{model}_end2end` | 4x |
| **D** | DALI + TRT + GPU NMS | `/predict/{model}_gpu_e2e_*` | **10-15x** |

**Key Innovation:** Track A (PyTorch) is **embedded** in the same service as Tracks B/C/D, eliminating the need for separate containers and simplifying deployment.

---

## 📁 File Structure

```
src/
├── main.py                         # ⭐ UNIFIED SERVICE (all 4 tracks)
│   ├── Track A: PyTorch models (loaded at startup)
│   ├── Track B: Triton standard TRT (Ultralytics wrapper)
│   ├── Track C: Triton End2End TRT (direct client)
│   └── Track D: DALI + TRT ensembles (full GPU)
│
└── utils/                          # Shared utilities
    ├── __init__.py                 # Public API exports
    ├── models.py                   # Pydantic response models
    ├── image_processing.py         # Fast image decode & validation
    ├── pytorch_utils.py            # PyTorch inference helpers
    ├── triton_end2end_client.py    # Direct Triton gRPC client
    ├── ultralytics_patches/        # Custom End2End export patches
    │   ├── __init__.py
    │   └── end2end_export.py       # TensorRT NMS operators
    └── README.md                   # Utils documentation
```

---

## 🚀 Track A: PyTorch Baseline

**Implementation:** Embedded in [main.py](main.py) lines 44-94, 178-281

### Overview

Direct PyTorch inference using Ultralytics YOLO - provides performance baseline for comparison.

**Key Features:**
- Native `.pt` models (no conversion required)
- Models loaded at startup in `lifespan()` context manager
- Thread-safe via `@ThreadingLocked` decorator from ultralytics
- Shared model instances across requests (memory efficient)
- CPU NMS (baseline for comparison)

### Implementation Details

```python
# In main.py

# Models loaded at startup (lines 44-48)
MODEL_IDENTIFIERS_PYTORCH = {
    "small": "/app/pytorch_models/yolo11s.pt",
}
MODEL_INSTANCES_PYTORCH = {}  # Populated in lifespan()

# Lifespan manager loads models (lines 73-100)
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading PyTorch models for Track A...")
    for model_name, model_path in MODEL_IDENTIFIERS_PYTORCH.items():
        model = YOLO(model_path, task="detect")
        # Warmup with dummy image
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        _ = model(dummy_img, verbose=False)
        MODEL_INSTANCES_PYTORCH[model_name] = model
    yield
    MODEL_INSTANCES_PYTORCH.clear()

# Thread-safe inference using shared instances (lines 178-220)
@app.post("/pytorch/predict/{model_name}")
async def predict_pytorch(model_name: str, image: UploadFile):
    model = MODEL_INSTANCES_PYTORCH[model_name]
    img = decode_image(image_data, filename)
    detections = thread_safe_predict(model, img)  # @ThreadingLocked
    return format_detections(detections)
```

### Endpoints

**Base URL:** `http://localhost:9600`

```
GET  /                                     # Service info (all tracks)
GET  /health                               # Health check
POST /pytorch/predict/{model_name}         # Single image
POST /pytorch/predict_batch/{model_name}   # Batch inference
```

### Example Usage

```bash
# Single image
curl -X POST http://localhost:9600/pytorch/predict/small \
  -F "image=@test.jpg"

# Batch inference
curl -X POST http://localhost:9600/pytorch/predict_batch/small \
  -F "images=@img1.jpg" \
  -F "images=@img2.jpg"

# Health check
curl http://localhost:9600/health | jq
```

### Performance Characteristics

- **Latency:** ~12-18ms (small model)
- **Throughput:** 80-120 images/sec
- **Speedup:** 1.0x (baseline reference)
- **Bottleneck:** CPU NMS (~5-10ms per image)

**Why Shared Instances?**
PyTorch models load weights into GPU memory. Creating per-request instances would:
- Exhaust GPU memory (each instance ~2GB)
- Slow initialization (~2-5 seconds per load)
- Require complex cleanup

Solution: Shared instances + `@ThreadingLocked` decorator serializes access.

---

## ⚡ Track B: Standard TRT + CPU NMS

**Implementation:** [main.py](main.py) lines 286-407

### Overview

TensorRT-optimized inference via Triton server with Ultralytics YOLO client wrapper.

**Key Features:**
- TensorRT FP16 optimization
- gRPC communication with Triton
- Per-request model instances (via Ultralytics wrapper)
- CPU NMS (via Ultralytics)
- No thread contention (true parallelism)

### Implementation Details

```python
# In main.py

# Model URLs (lines 51-53)
MODEL_URLS_STANDARD = {
    "small": "grpc://triton-api:8001/yolov11_small_trt",
}

# Per-request instances (lines 366-398)
@app.post("/predict/{model_name}")
async def predict_triton(model_name: str, image: UploadFile):
    # ... routing logic ...

    # Track B: Create fresh instance per request
    model_url = MODEL_URLS_STANDARD[model_name]
    model = YOLO(model_url, task="detect")
    detections = await asyncio.to_thread(model, img, verbose=False)

    # Format response
    boxes = detections[0].boxes.xyxy.cpu().numpy()
    scores = detections[0].boxes.conf.cpu().numpy()
    classes = detections[0].boxes.cls.cpu().numpy()
    # ... format into JSON ...
```

### Endpoints

**Base URL:** `http://localhost:9600`

```
POST /predict/{model_name}         # Standard TRT: small
POST /predict_batch/{model_name}   # Batch inference
```

### Example Usage

```bash
# Single image (Track B)
curl -X POST http://localhost:9600/predict/small \
  -F "image=@test.jpg"

# Response shows CPU NMS
curl -X POST http://localhost:9600/predict/small \
  -F "image=@test.jpg" | jq '.nms_location'
# Output: "cpu"
```

### Performance Characteristics

- **Latency:** ~8-12ms (small model)
- **Throughput:** 150-250 images/sec
- **Speedup:** 1.5-2.5x faster than Track A
- **Bottleneck:** CPU NMS still required (~5-10ms)

**Why Per-Request Instances?**
Ultralytics YOLO with Triton backend creates lightweight gRPC connections. Creating per-request instances:
- Eliminates thread locking overhead
- Enables true parallel processing
- Minimal initialization cost (~0.1-0.5ms)

---

## 🚀 Track C: End2End TRT + GPU NMS

**Implementation:** [main.py](main.py) lines 342-363 + [utils/triton_end2end_client.py](utils/triton_end2end_client.py)

### Overview

Maximum performance with GPU NMS **compiled into TensorRT engine**. Eliminates all CPU-GPU data transfers for NMS.

**Key Features:**
- TensorRT with **compiled GPU NMS**
- Direct Triton gRPC client (bypasses Ultralytics wrapper)
- Zero CPU-GPU data transfers for NMS
- Custom YOLO preprocessing in client
- Per-request client instances

### Implementation Details

```python
# In main.py

# Model names (lines 55-57)
MODEL_NAMES_END2END = {
    "small": "yolov11_small_trt_end2end",
}

# Direct Triton client (lines 342-363)
@app.post("/predict/{model_name}")
async def predict_triton(model_name: str, image: UploadFile):
    # ... routing logic ...

    # Track C: Direct Triton client with GPU NMS
    base_model = model_name.replace("_end2end", "")
    triton_model_name = MODEL_NAMES_END2END[base_model]
    client = TritonEnd2EndClient(triton_url=TRITON_URL, model_name=triton_model_name)

    detections = await asyncio.to_thread(client.infer, img)
    results = await asyncio.to_thread(client.format_detections, detections)

    return {
        "detections": results,
        "track": "C",
        "nms_location": "gpu"
    }
```

**TritonEnd2EndClient** ([utils/triton_end2end_client.py](utils/triton_end2end_client.py)):
- Implements YOLO letterbox preprocessing
- Direct gRPC communication
- Handles End2End output format (num_dets, det_boxes, det_scores, det_classes)
- Vectorized coordinate transformation (100x faster than loops)

### Endpoints

**Base URL:** `http://localhost:9600`

**Naming Convention:** Append `_end2end` to model name

```
POST /predict/{model}_end2end         # Track C: GPU NMS
POST /predict_batch/{model}_end2end   # Batch inference
```

### Example Usage

```bash
# Single image (Track C - GPU NMS)
curl -X POST http://localhost:9600/predict/small_end2end \
  -F "image=@test.jpg"

# Batch inference
curl -X POST http://localhost:9600/predict_batch/small_end2end \
  -F "images=@img1.jpg" \
  -F "images=@img2.jpg"

# Verify GPU NMS
curl -X POST http://localhost:9600/predict/small_end2end \
  -F "image=@test.jpg" | jq '.nms_location'
# Output: "gpu"
```

### Performance Characteristics

- **Latency:** ~3-5ms (small model)
- **Throughput:** 300-500 images/sec
- **Speedup:** 3-5x faster than Track A, 2-3x faster than Track B
- **Bottleneck:** CPU preprocessing (~2-3ms)

### Why So Fast?

**Track B (Standard TRT):**
```
GPU (TRT inference) → CPU transfer → CPU (NMS) → Response
                    ↑ Bottleneck: 2x data transfers + CPU NMS
```

**Track C (End2End TRT):**
```
GPU (TRT inference + compiled NMS) → Response
↑ Zero transfers! NMS compiled into TensorRT engine
```

**Result:** Eliminates CPU-GPU data transfers + NMS runs on GPU = 2-3x speedup!

---

## 🔥 Track D: DALI + TRT + GPU NMS (Full GPU Pipeline)

**Implementation:** [main.py](main.py) lines 315-340 + DALI ensembles

### Overview

**Maximum throughput** with 100% GPU pipeline - preprocessing, inference, and NMS all on GPU.

**Key Features:**
- **GPU JPEG decode** via NVIDIA DALI (nvJPEG)
- **GPU letterbox resize** via DALI warp_affine
- **GPU inference + NMS** via TRT End2End
- **Zero CPU processing** except affine matrix calculation
- **3 performance tiers:** streaming, balanced, batch

### Implementation Details

```python
# In main.py

# Model names (lines 59-65)
MODEL_NAMES_GPU_E2E = {
    "small": "yolov11_small_gpu_e2e",
}

# Track D: 100% GPU pipeline (lines 315-340)
@app.post("/predict/{model_name}")
async def predict_triton(model_name: str, image: UploadFile):
    # ... routing logic ...

    # Track D: Send raw JPEG bytes to DALI ensemble
    base_model = model_name.replace("_gpu_e2e_streaming", "").replace("_gpu_e2e_batch", "")
    triton_model_name = MODEL_NAMES_GPU_E2E[base_model]

    client = TritonEnd2EndClient(triton_url=TRITON_URL, model_name=triton_model_name)
    # Send compressed JPEG bytes - DALI does GPU decode!
    detections = await asyncio.to_thread(client.infer_raw_bytes, image_data)
    results = await asyncio.to_thread(client.format_detections, detections)

    return {
        "detections": results,
        "track": "D",
        "preprocessing": "gpu_dali",
        "nms_location": "gpu"
    }
```

**DALI Pipeline** (created via [dali/create_dali_letterbox_pipeline.py](../dali/create_dali_letterbox_pipeline.py)):
1. GPU JPEG decode (nvJPEG - 5-10x faster than CPU)
2. GPU affine transformation (warp_affine with CPU-calculated matrix)
3. GPU normalization and CHW transpose
4. Ensemble routes to TRT End2End model

### Endpoints

**Base URL:** `http://localhost:9600`

**Three Performance Tiers:**

```
POST /predict/small_gpu_e2e_streaming  # Low latency (0.1ms batching)
POST /predict/small_gpu_e2e            # Balanced (0.5ms batching)
POST /predict/small_gpu_e2e_batch      # Max throughput (5ms batching)
```

### Example Usage

```bash
# Low latency tier (streaming)
curl -X POST http://localhost:9600/predict/small_gpu_e2e_streaming \
  -F "image=@test.jpg"

# Balanced tier (general purpose)
curl -X POST http://localhost:9600/predict/small_gpu_e2e \
  -F "image=@test.jpg"

# High throughput tier (batch processing)
curl -X POST http://localhost:9600/predict/small_gpu_e2e_batch \
  -F "image=@test.jpg"

# Verify GPU preprocessing
curl -X POST http://localhost:9600/predict/small_gpu_e2e_batch \
  -F "image=@test.jpg" | jq '.preprocessing'
# Output: "gpu_dali"
```

### Performance Characteristics

| Tier | Batching | Latency (P50) | Throughput | Use Case |
|------|----------|---------------|------------|----------|
| **Streaming** | 0.1ms | 15-20ms | 800-1200 rps | Video streaming |
| **Balanced** | 0.5ms | 20-30ms | 1500-2000 rps | General purpose |
| **Batch** | 5ms | 25-40ms | **2000-2500 rps** | Offline processing |

- **Speedup:** 10-15x faster than Track A
- **Bottleneck:** Minimal CPU overhead (~0.3-0.5ms for affine calculation)

### Why Track D is 10-15x Faster

**Track C (CPU Preprocessing):**
```
CPU (decode JPEG) → CPU (letterbox) → GPU (TRT+NMS) → Response
    ~2-3ms              ~1-2ms             ~2-3ms
```

**Track D (GPU Preprocessing):**
```
GPU (nvJPEG decode) → GPU (warp_affine) → GPU (TRT+NMS) → Response
    ~0.2-0.5ms            ~0.1-0.3ms           ~2-3ms
CPU (affine calc): ~0.1ms
```

**Result:**
- CPU preprocessing eliminated (2-3ms → 0.1ms)
- Higher concurrency (no CPU bottleneck)
- Better batching efficiency (longer queue times allowed)

---

## 🔌 Unified API Specification

All four tracks use the **same REST API format** for seamless switching.

### Request Format

**Single Image:**
```bash
POST /predict/{model_name}
Content-Type: multipart/form-data

image: <JPEG/PNG file>
```

**Batch:**
```bash
POST /predict_batch/{model_name}
Content-Type: multipart/form-data

images: <file1>
images: <file2>
images: <file3>
```

### Response Format

**Single Image:**
```json
{
  "detections": [
    {
      "x1": 100.5,
      "y1": 200.3,
      "x2": 300.8,
      "y2": 450.2,
      "confidence": 0.92,
      "class": 0
    }
  ],
  "status": "success",
  "track": "D",
  "preprocessing": "gpu_dali",
  "nms_location": "gpu"
}
```

**Batch:**
```json
{
  "total_images": 3,
  "processed_images": 3,
  "failed_images": 0,
  "results": [
    {
      "filename": "img1.jpg",
      "image_index": 0,
      "detections": [...],
      "status": "success",
      "track": "D"
    }
  ],
  "status": "success"
}
```

---

## 🖼️ Image Processing

All tracks use optimized utilities from [utils/image_processing.py](utils/image_processing.py):

### Supported Formats

✅ **Primary (cv2):** JPEG, PNG, BMP, TIFF
✅ **Fallback (PIL):** WebP, GIF, exotic formats

### Processing Pipeline

```python
# 1. Fast decode (zero-copy when possible)
img = decode_image(image_bytes, filename)
# - Fast path: cv2.imdecode (~0.5-2ms, 95% of images)
# - Fallback: PIL (~2-5ms, exotic formats)

# 2. Security validation (<0.01ms overhead)
validate_image(img, filename)
# - Prevent OOM attacks (max 16K resolution)
# - Minimal checks (YOLO handles format validation)

# 3. Track-specific preprocessing
# - Track A/B/C: YOLO letterbox (resize + pad)
# - Track D: DALI GPU letterbox (warp_affine)
```

### Philosophy

**We do:** Format decoding, security validation
**YOLO/DALI does:** Resize, pad, normalize, inference, NMS

**Result:** <2ms CPU overhead for 95% of images

---

## 🏗️ Architecture Patterns

### Shared Instances (Track A Only)

**Pattern:**
```python
# PyTorch models loaded at startup
MODEL_INSTANCES_PYTORCH = {}  # Shared across requests

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models once
    for name, path in MODEL_IDENTIFIERS_PYTORCH.items():
        MODEL_INSTANCES_PYTORCH[name] = YOLO(path)
    yield
    MODEL_INSTANCES_PYTORCH.clear()

# Thread-safe access
@app.post("/pytorch/predict/{model_name}")
async def predict_pytorch(model_name: str):
    model = MODEL_INSTANCES_PYTORCH[model_name]  # Shared
    detections = thread_safe_predict(model, img)  # @ThreadingLocked
```

**Why:** PyTorch models are large (~2GB in GPU memory), can't create per-request.

### Per-Request Instances (Tracks B/C/D)

**Pattern:**
```python
@app.post("/predict/{model_name}")
async def predict_triton(model_name: str):
    # Create fresh client per request
    if is_end2end:
        client = TritonEnd2EndClient(triton_url=TRITON_URL, model_name=model)
    else:
        model = YOLO(MODEL_URLS_STANDARD[model_name])

    detections = await asyncio.to_thread(model_or_client, img)
```

**Why:** Triton clients are lightweight gRPC connections, optimal for parallelism.

---

## 📊 Performance Comparison

| Metric | Track A | Track B | Track C | Track D |
|--------|---------|---------|---------|---------|
| **Latency (P50)** | 12-18ms | 8-12ms | 3-5ms | **2-3ms** |
| **Throughput** | 80-120 rps | 150-250 rps | 300-500 rps | **1500-2500 rps** |
| **Speedup** | 1.0x | 2.0x | 4.0x | **12.5x** 🚀 |
| **Preprocessing** | CPU | CPU | CPU | **GPU (DALI)** |
| **NMS Location** | CPU | CPU | **GPU** | **GPU** |
| **Model Format** | `.pt` | TRT `.plan` | TRT `.plan` + NMS | Ensemble |
| **Client** | PyTorch | Ultralytics | Direct Triton | Direct Triton |
| **Concurrency** | Thread-locked | True parallel | True parallel | True parallel |

**Processing 100,000 images:**
- Track A: ~14 minutes
- Track B: ~7 minutes
- Track C: ~3.5 minutes
- Track D: **~40 seconds** 🚀

---

## 🚀 Running the Service

### Via Docker Compose (Production)

```bash
# Start all services
docker compose up -d

# View logs for unified service
docker compose logs -f yolo-api

# View Triton logs
docker compose logs -f triton-api

# Stop services
docker compose down
```

### Standalone (Development)

```bash
# Requires Triton server running
docker compose up -d triton-api

# Run unified service (all tracks)
uvicorn src.main:app \
  --host 0.0.0.0 \
  --port 9600 \
  --workers 32 \
  --reload
```

### Testing All Tracks

```bash
# Test Track A (PyTorch)
curl -X POST http://localhost:9600/pytorch/predict/small \
  -F "image=@test.jpg" | jq

# Test Track B (Standard TRT)
curl -X POST http://localhost:9600/predict/small \
  -F "image=@test.jpg" | jq

# Test Track C (End2End TRT)
curl -X POST http://localhost:9600/predict/small_end2end \
  -F "image=@test.jpg" | jq

# Test Track D (DALI + TRT)
curl -X POST http://localhost:9600/predict/small_gpu_e2e_batch \
  -F "image=@test.jpg" | jq
```

---

## 🐛 Troubleshooting

### "Model not found" (Track A)

```bash
# Download PyTorch models
bash export/download_pytorch_models.sh

# Verify models exist
ls -lh pytorch_models/

# Check service loaded them
curl http://localhost:9600/health | jq '.tracks.track_a_pytorch'
```

### "Connection refused to triton-api" (Tracks B/C/D)

```bash
# Check Triton is running
docker compose ps triton-api

# Check Triton health
curl http://localhost:9500/v2/health/ready

# View Triton logs
docker compose logs triton-api | grep -i error

# Restart Triton
docker compose restart triton-api
```

### "Model not ready" (Track C/D)

```bash
# Verify models exported
ls -lh models/yolov11_small_trt_end2end/1/model.plan
ls -lh models/yolo_preprocess_dali/1/model.dali

# Check Triton loaded them
curl http://localhost:9500/v2/models/yolov11_small_trt_end2end/ready
curl http://localhost:9500/v2/models/yolo_preprocess_dali/ready

# List all loaded models
curl http://localhost:9500/v2/models | jq
```

### Performance not improved (Track D)

```bash
# Check batching is working (should see batch > 1)
docker compose logs triton-api | grep "batch size"

# Increase concurrency
cd benchmarks
./triton_bench --mode full --track D_batch --clients 256

# Check GPU utilization (should be 80-100%)
nvidia-smi -l 1
```

---

## 🔗 Related Documentation

- **[../README.md](../README.md)** - Project overview and quick start
- **[../docs/DEPLOYMENT_GUIDE.md](../docs/DEPLOYMENT_GUIDE.md)** - Deployment instructions
- **[../docs/Tracks/TRACK_D_COMPLETE.md](../docs/Tracks/TRACK_D_COMPLETE.md)** - Track D deep dive
- **[../docs/TESTING.md](../docs/TESTING.md)** - Testing procedures
- **[../benchmarks/README.md](../benchmarks/README.md)** - Benchmarking guide
- **[utils/README.md](utils/README.md)** - Utilities documentation

---

## 🎯 Track Selection Guide

**Choose your track based on requirements:**

| Requirement | Recommended Track |
|-------------|------------------|
| **Maximum throughput** | Track D (batch tier) |
| **Low latency** | Track D (streaming tier) or Track C |
| **Simplicity** | Track A (no export needed) |
| **GPU optimization** | Track C or D |
| **CPU-only deployment** | Track A only option |
| **Balanced** | Track C (GPU NMS, simple setup) |

**Typical progression:**
1. Start with **Track A** (baseline, verify workflow)
2. Export models for **Track B** (quick TRT speedup)
3. Enable **Track C** (GPU NMS, 4x faster)
4. Add **Track D** (full GPU, 12x faster for production)
