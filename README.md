# Triton YOLO Inference Server

**Process 100,000+ images with 10-15x speedup using NVIDIA Triton + YOLO11**

High-performance object detection and visual search with GPU-accelerated inference, dynamic batching, DALI preprocessing, and MobileCLIP embeddings.

---

## Quick Start (5 Minutes)

### 1. Start Services

```bash
cd /mnt/nvm/repos/triton-api

# Start everything
docker compose up -d

# Wait for models to load (2-3 minutes first time)
docker compose logs -f triton-api | grep "successfully loaded"
# Press Ctrl+C when models show as READY
```

### 2. Verify

```bash
# Check services
docker compose ps
curl http://localhost:4603/health

# Check GPU
nvidia-smi
```

### 3. Benchmark

```bash
cd benchmarks
./build.sh                    # Build benchmark tool
./triton_bench --mode quick   # Run 30-second test
```

**Done!** Results saved to `benchmarks/results/`

---

## Five Performance Tracks

| Track | Technology | Speedup | Best For |
|-------|-----------|---------|----------|
| **A** | PyTorch + CPU NMS | 1x (baseline) | Reference/debugging |
| **B** | TensorRT + CPU NMS | 2x | Standard acceleration |
| **C** | TensorRT + GPU NMS | 4x | Compiled NMS |
| **D** | DALI + TRT + GPU NMS | **10-15x** | Maximum throughput |
| **E** | YOLO + MobileCLIP + OpenSearch | N/A | Visual search |

**Track D has 3 variants**: streaming (low latency), balanced (general), batch (max throughput)

**Track E includes:**
- Visual search with MobileCLIP embeddings + OpenSearch k-NN
- Face detection (YOLO11-face or SCRFD) + ArcFace identity embeddings
- OCR text extraction (PP-OCRv5) with trigram search
- Batch ingestion endpoint (300+ RPS)

---

## Benchmarking

Single tool, 7 test modes:

```bash
cd benchmarks

# Quick tests
./triton_bench --mode single   # Test one image
./triton_bench --mode quick    # 30-second check
./triton_bench --mode full     # Full benchmark

# Advanced
./triton_bench --mode all          # Process all images
./triton_bench --mode sustained    # Find max throughput
./triton_bench --mode variable     # Variable load patterns
```

Common commands:

```bash
# Full test with 128 clients
./triton_bench --mode full --clients 128 --duration 60

# Test specific track
./triton_bench --mode full --track D_batch --clients 256

# Process your images
./triton_bench --mode all --images /path/to/images --clients 128
```

Results auto-saved with timestamps: `benchmarks/results/quick_concurrency_20250116_153215.json`

**For complete benchmark guide**: [benchmarks/README.md](benchmarks/README.md)

---

## Expected Performance

NVIDIA A100 GPU with 128-256 concurrent clients:

| Track | Throughput | P50 Latency | Speedup |
|-------|-----------|-------------|---------|
| A (PyTorch) | 150-200 rps | 40-60ms | 1.0x |
| B (TRT) | 300-400 rps | 20-30ms | 2.0x |
| C (End2End) | 600-800 rps | 10-15ms | 4.0x |
| D (Batch) | **1500-2500 rps** | 20-30ms | **12.5x** |

**100,000 images**: PyTorch ~9 min, Track D ~40 seconds 🚀

---

## Configuration

### FastAPI Workers (Concurrency)

Worker count configured in `docker-compose.yml`:
- **Development/Testing**: 2 workers (when ENABLE_PYTORCH=true to avoid GPU memory waste)
- **Production**: 64 workers × 512 concurrent = 32,768 total capacity

**Change** in `docker-compose.yml` uvicorn command:
```yaml
- --workers=64  # Production: 64 workers for max throughput
```

### Triton Batching

**Change** in `models/*/config.pbtxt`:
```protobuf
dynamic_batching {
  preferred_batch_size: [ 8, 16, 32, 64 ]
  max_queue_delay_microseconds: 5000
}
```

### GPU Selection

**Change** in `docker-compose.yml`:
```yaml
device_ids: [ '0', '2' ]  # Change GPU IDs here
```

---

## Monitoring

While running benchmarks:

**Terminal 1: GPU**
```bash
nvidia-smi -l 1
# Should show 80-100% utilization
```

**Terminal 2: Batching**
```bash
docker compose logs -f triton-api | grep "batch size"
# Should show: batch size: 8, 16, 32, etc.
```

**Terminal 3: Grafana Dashboard**
```bash
# Open http://localhost:4605 (admin/admin)
# Import dashboard: monitoring/triton-dashboard.json
```

---

## Troubleshooting

### No speed gains?

```bash
# Check workers (should be 3 or 65)
docker compose exec yolo-api ps aux | grep uvicorn | wc -l

# Check batching (should show size > 1)
docker compose logs triton-api | grep "batch size"

# Try higher concurrency
./triton_bench --mode full --clients 256
```

### Services won't start?

```bash
docker compose logs triton-api | grep -i error
docker compose restart
```

### High error rate?

```bash
docker stats                      # Check resources
docker compose logs triton-api    # Check errors
./triton_bench --mode full --clients 32  # Reduce load
```

---

## System Requirements

### Minimum
- 16 CPU cores, 32GB RAM
- NVIDIA GPU 8GB+ VRAM (Ampere+)
- Docker 24.0+, NVIDIA Container Toolkit

### Recommended
- 48+ CPU cores, 64GB+ RAM
- NVIDIA A100/A6000/RTX 4090 (16GB+)
- 100GB+ NVMe SSD

---

## Project Structure

```
triton-api/
├── README.md                       # This file
├── CLAUDE.md                       # AI assistant instructions
├── ATTRIBUTION.md                  # Third-party code attribution
├── Makefile                        # Development commands (60+ targets)
├── docker-compose.yml              # Services orchestration
├── Dockerfile                      # FastAPI service container
├── Dockerfile.triton               # Triton server with PyTorch backend
├── pyproject.toml                  # Python project config & linting
│
├── export/                         # Model export (all tracks)
│   ├── export_models.py            # Main YOLO export tool
│   ├── export_mobileclip_image_encoder.py  # Track E image encoder
│   ├── export_mobileclip_text_encoder.py   # Track E text encoder
│   └── download_pytorch_models.py  # Download .pt files
│
├── dali/                           # DALI preprocessing
│   ├── create_dali_letterbox_pipeline.py   # Track D DALI
│   ├── create_dual_dali_pipeline.py        # Track E triple-branch DALI
│   ├── create_ensembles.py                 # Track D ensembles
│   └── validate_*.py                       # Validation scripts
│
├── scripts/                        # Utilities
│   ├── check_services.sh           # Health check
│   └── track_e/                    # Track E setup scripts
│       ├── setup_mobileclip_env.sh # Clone reference repos
│       └── install_mobileclip_deps.sh
│
├── tests/                          # Testing & validation
│   ├── test_inference.sh           # Integration test (all tracks)
│   ├── compare_tracks.py           # Cross-track comparison
│   ├── test_track_e_*.py           # Track E test suite
│   └── test_*.py                   # Other test scripts
│
├── benchmarks/                     # Performance testing
│   ├── triton_bench.go             # Go benchmark tool
│   └── results/                    # Auto-generated results
│
├── models/                         # Triton model repository
│   ├── yolov11_small_trt/          # Track B
│   ├── yolov11_small_trt_end2end/  # Track C
│   ├── yolo_preprocess_dali_batch/ # Track D DALI preprocessing
│   ├── yolov11_small_gpu_e2e_batch/# Track D ensemble
│   ├── mobileclip2_s2_image_encoder/   # Track E image encoder
│   ├── mobileclip2_s2_text_encoder/    # Track E text encoder
│   ├── dual_preprocess_dali/           # Track E triple-branch DALI
│   ├── box_embedding_extractor/        # Track E per-box embeddings
│   └── yolo_mobileclip_ensemble/       # Track E full ensemble
│
├── monitoring/                     # Prometheus & Grafana
│   ├── prometheus.yml
│   ├── grafana-datasources.yml
│   └── dashboards/
│
├── docs/                           # Documentation
│   ├── MODEL_EXPORT_GUIDE.md
│   ├── TRACK_E_*.md                # Track E guides
│   └── Tracks/                     # Per-track documentation
│
├── src/                            # FastAPI service
│   ├── main.py                     # Application entry point
│   ├── routers/                    # API endpoints (health, track_a, triton, track_e)
│   ├── services/                   # Business logic (inference, visual_search)
│   ├── clients/                    # Triton & OpenSearch clients
│   ├── schemas/                    # Pydantic response models
│   ├── config/                     # Settings & configuration
│   └── utils/                      # Image processing, caching
│
└── reference_repos/                # External repos (cloned on setup)
    ├── ml-mobileclip/              # Apple MobileCLIP (Track E)
    └── open_clip/                  # OpenCLIP framework (Track E)
```

---

## API Usage

**All tracks available on single service at port 4603:**

```python
import requests

# Track A: PyTorch Baseline
files = {'image': open('image.jpg', 'rb')}
response = requests.post('http://localhost:4603/pytorch/predict/small', files=files)

# Track B: Standard TRT
response = requests.post('http://localhost:4603/predict/small', files=files)

# Track C: End2End TRT + GPU NMS
response = requests.post('http://localhost:4603/predict/small_end2end', files=files)

# Track D: DALI + TRT (Maximum Performance)
response = requests.post('http://localhost:4603/predict/small_gpu_e2e_batch', files=files)

# Track E: Visual Search - Detection + Embeddings
response = requests.post('http://localhost:4603/track_e/detect', files=files)
response = requests.post('http://localhost:4603/track_e/predict', files=files)
response = requests.post('http://localhost:4603/track_e/predict_full', files=files)

# Track E: Embedding Only
response = requests.post('http://localhost:4603/track_e/embed/image', files=files)
response = requests.post('http://localhost:4603/track_e/embed/text', json={'text': 'a red car'})

# Track E: Image Ingestion
response = requests.post('http://localhost:4603/track_e/ingest',
                        files=files, data={'image_id': 'img_001'})

# Track E: Image-to-Image Search
response = requests.post('http://localhost:4603/track_e/search/image', files=files)

# Track E: Text-to-Image Search
response = requests.post('http://localhost:4603/track_e/search/text',
                        json={'text': 'a red car', 'top_k': 10})

# Track E: Object-Level Search
response = requests.post('http://localhost:4603/track_e/search/object',
                        files=files, data={'box_index': 0, 'top_k': 10})

# Track E: Index Management
response = requests.get('http://localhost:4603/track_e/index/stats')
response = requests.post('http://localhost:4603/track_e/index/create')
response = requests.delete('http://localhost:4603/track_e/index')

# Track E: Batch Ingestion (300+ RPS)
files = [('files', open(f, 'rb')) for f in image_paths]
response = requests.post('http://localhost:4603/track_e/ingest_batch', files=files)

# Track E: Face Detection (YOLO11-face or SCRFD)
response = requests.post('http://localhost:4603/track_e/faces/detect',
                        files=files, params={'detector': 'yolo11'})

# Track E: Face Recognition (detection + ArcFace embeddings)
response = requests.post('http://localhost:4603/track_e/faces/recognize', files=files)

# Track E: Face Search
response = requests.post('http://localhost:4603/track_e/faces/search',
                        files=files, params={'top_k': 10})

# Track E: OCR Text Extraction
response = requests.post('http://localhost:4603/track_e/ocr/predict', files=files)

# Track E: Search by OCR Text
response = requests.post('http://localhost:4603/track_e/search/ocr',
                        json={'query': 'STOP', 'top_k': 10})

print(response.json())
```

Response format (Tracks A-D):
```json
{
  "detections": [
    {"x1": 0.245, "y1": 0.123, "x2": 0.456, "y2": 0.389,
     "confidence": 0.94, "class_id": 0}
  ],
  "image": {"width": 1920, "height": 1080},
  "model": {"name": "yolov11_small", "backend": "triton"},
  "track": "D",
  "total_time_ms": 12.5
}
```

Response format (Track E search):
```json
{
  "results": [
    {"image_id": "img_001", "score": 0.95, "image_path": "/path/to/image.jpg"}
  ],
  "total_results": 10,
  "search_time_ms": 15.2
}
```

Response format (Track E faces/detect):
```json
{
  "num_faces": 2,
  "faces": [
    {"box": [0.1, 0.2, 0.3, 0.4], "confidence": 0.98, "landmarks": [...]}
  ],
  "detector": "yolo11",
  "total_time_ms": 25.3
}
```

Response format (Track E ocr/predict):
```json
{
  "status": "success",
  "num_texts": 3,
  "texts": ["STOP", "ONE WAY", "EXIT"],
  "boxes_normalized": [[0.1, 0.2, 0.3, 0.25], ...],
  "det_scores": [0.95, 0.92, 0.89],
  "rec_scores": [0.98, 0.95, 0.91]
}
```

---

## Documentation

- **This README**: Overview and quick start (YOU ARE HERE)
- **[CLAUDE.md](CLAUDE.md)**: AI assistant instructions and architecture overview
- **[ATTRIBUTION.md](ATTRIBUTION.md)**: Third-party code attribution and licensing
- **[Makefile](Makefile)**: 60+ development commands (`make help` for list)
- **[benchmarks/README.md](benchmarks/README.md)**: Benchmark tool documentation
- **[docs/](docs/)**: Technical reference documents
  - [docs/MODEL_EXPORT_GUIDE.md](docs/MODEL_EXPORT_GUIDE.md): Complete export guide
  - [docs/TRACK_E_GUIDE.md](docs/TRACK_E_GUIDE.md): Visual search setup and usage

---

## Architecture

**Unified Single-Service Design:**
- One FastAPI service (`yolo-api`) handles all 5 tracks
- PyTorch models loaded at startup (Track A, when enabled)
- Triton models accessed via gRPC per-request (Tracks B/C/D/E)
- OpenSearch for vector similarity search (Track E)
- Direct .plan files (no warmup needed)

**Key Improvements:**
1. **Simplified Deployment**: All services orchestrated via Docker Compose
2. **Unified Endpoints**: All tracks on port 4603
3. **Visual Search**: Track E adds MobileCLIP embeddings + OpenSearch k-NN
4. **Production-Ready**: Direct TensorRT .plan files, instant startup

**Services:**
- `yolo-api`: FastAPI service (port 4603)
- `triton-api`: Triton Inference Server (ports 4600-4602)
- `opensearch`: Vector database for Track E (port 4607)
- `prometheus`/`grafana`: Monitoring (ports 4604/4605)

---

## Production Deployment

- Change Grafana password (default: admin/admin)
- Add TLS/SSL with reverse proxy
- Configure resource limits in docker-compose.yml
- Set up Prometheus alerts
- Use multiple GPUs: `device_ids: ['0', '1', '2']`
- **Track E**: Enable OpenSearch security plugin (`DISABLE_SECURITY_PLUGIN=false`)
- **Track E**: Configure multi-node OpenSearch cluster for high availability
- Increase workers: Set `--workers=64` in docker-compose.yml for production

---

## Resources

- **NVIDIA Triton**: https://docs.nvidia.com/deeplearning/triton-inference-server/
- **Ultralytics**: https://docs.ultralytics.com/
- **NVIDIA DALI**: https://docs.nvidia.com/deeplearning/dali/
- **OpenSearch**: https://opensearch.org/docs/latest/
- **Apple MobileCLIP**: https://github.com/apple/ml-mobileclip

---

## Attribution

This project uses code from the [levipereira/ultralytics](https://github.com/levipereira/ultralytics) fork for end2end YOLO export with GPU-accelerated NMS. This enables **Track C** (4x speedup) and **Track D** (10-15x speedup) by embedding TensorRT EfficientNMS into the model.

**Track E** uses:
- [Apple MobileCLIP](https://github.com/apple/ml-mobileclip) for visual embeddings
- [OpenSearch](https://opensearch.org/) for k-NN vector similarity search
- [YOLO11-face](https://github.com/akanametov/yolo-face) for face detection (alternative to SCRFD)
- [ArcFace](https://github.com/deepinsight/insightface) for face identity embeddings
- [PP-OCRv5](https://github.com/PaddlePaddle/PaddleOCR) for text detection and recognition

See [ATTRIBUTION.md](ATTRIBUTION.md) for complete third-party code attribution and licensing information.

---

**Built for maximum throughput** 🚀 *100K+ images in minutes, visual search in milliseconds*
