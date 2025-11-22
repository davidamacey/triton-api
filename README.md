# Triton YOLO Inference Server

**Process 100,000+ images with 10-15x speedup using NVIDIA Triton + YOLO11**

High-performance object detection with GPU-accelerated inference, dynamic batching, and DALI preprocessing.

---

## Quick Start (5 Minutes)

### 1. Start Services

```bash
cd /mnt/nvm/repos/triton-api

# Start everything
docker compose up -d

# Wait for models to load (2-3 minutes first time)
docker compose logs -f triton-api | grep "successfully loaded"
# Press Ctrl+C when all 6 models show as READY
```

### 2. Verify

```bash
# Check services
docker compose ps
curl http://localhost:9600/health

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

## Four Performance Tracks

| Track | Technology | Speedup | Best For |
|-------|-----------|---------|----------|
| **A** | PyTorch + CPU NMS | 1x (baseline) | Reference |
| **B** | TensorRT + CPU NMS | 2x | Standard acceleration |
| **C** | TensorRT + GPU NMS | 4x | Compiled NMS |
| **D** | DALI + TRT + GPU NMS | **10-15x** | Maximum throughput |

**Track D has 3 variants**: streaming (low latency), balanced (general), batch (max throughput)

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

32 workers handling 512 requests each = 16,384 total concurrent capacity

**Change** in [Dockerfile](Dockerfile):
```dockerfile
CMD ["uvicorn", "src.main:app", "--workers", "32", ...]
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
device_ids: [ '0' ]  # Change GPU ID here
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
# Open http://localhost:3000 (admin/admin)
# Import dashboard: monitoring/triton-dashboard.json
```

---

## Troubleshooting

### No speed gains?

```bash
# Check workers (should be 33)
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
├── AUTOMATION.md                   # Complete automation guide
├── docker-compose.yml              # Services orchestration
├── Dockerfile                      # Unified service (yolo-api, 32 workers)
│
├── export/                         # Model export (all tracks)
│   ├── export_models.py            # Main export tool
│   ├── export_small_only.sh        # Quick export wrapper
│   ├── download_pytorch_models.sh  # Download .pt files
│   └── cleanup_for_reexport.sh     # Clean re-export
│
├── dali/                           # DALI preprocessing (Track D)
│   ├── create_dali_letterbox_pipeline.py
│   ├── validate_dali_letterbox.py
│   └── create_ensembles.py
│
├── scripts/                        # Core utilities
│   └── check_services.sh           # Health check
│
├── tests/                          # Testing & validation
│   ├── test_inference.sh           # Integration test
│   └── test_*.py                   # Test scripts
│
├── benchmarks/                     # Performance testing
│   ├── triton_bench.go             # Master benchmark tool
│   ├── build.sh                    # Build script
│   └── results/                    # Auto-generated results
│
├── models/                         # Triton model repository
│   ├── yolov11_small_trt/          # Track B
│   ├── yolov11_small_trt_end2end/  # Track C
│   ├── yolo_preprocess_dali/       # Track D preprocessing
│   ├── yolov11_small_gpu_e2e/      # Track D balanced
│   ├── yolov11_small_gpu_e2e_streaming/  # Track D streaming
│   └── yolov11_small_gpu_e2e_batch/      # Track D batch
│
├── monitoring/                     # Prometheus & Grafana
│   ├── prometheus.yml
│   ├── grafana-datasources.yml
│   └── triton-dashboard.json
│
├── docs/                           # Documentation
│   ├── MODEL_EXPORT_GUIDE.md
│   ├── TRACK_D_COMPLETE.md
│   └── ...
│
└── src/                            # FastAPI service (unified)
    └── main.py                     # All 4 tracks (A/B/C/D)
```

---

## API Usage

**All tracks available on single service at port 9600:**

```python
import requests

# Track A: PyTorch Baseline
files = {'image': open('image.jpg', 'rb')}
response = requests.post('http://localhost:9600/pytorch/predict/small', files=files)

# Track B: Standard TRT
response = requests.post('http://localhost:9600/predict/small', files=files)

# Track C: End2End TRT + GPU NMS
response = requests.post('http://localhost:9600/predict/small_end2end', files=files)

# Track D: DALI + TRT (Maximum Performance)
response = requests.post('http://localhost:9600/predict/small_gpu_e2e_batch', files=files)

print(response.json())
```

Response format:
```json
{
  "detections": [
    {"x1": 245.3, "y1": 123.7, "x2": 456.2, "y2": 389.1,
     "confidence": 0.94, "class": 0}
  ],
  "status": "success",
  "track": "D",
  "preprocessing": "gpu_dali",
  "nms_location": "gpu"
}
```

---

## Documentation

- **This README**: Overview and quick start (YOU ARE HERE)
- **[AUTOMATION.md](AUTOMATION.md)**: Complete automation scripts guide
- **[benchmarks/README.md](benchmarks/README.md)**: Benchmark tool documentation
- **[docs/](docs/)**: Technical reference documents

---

## Architecture

**Unified Single-Service Design:**
- One FastAPI service (`yolo-api`) handles all 4 tracks
- PyTorch models loaded at startup (Track A)
- Triton models accessed via gRPC per-request (Tracks B/C/D)
- 32 workers × 512 concurrent requests = **16,384 total capacity**
- Direct .plan files (no warmup needed)

**Key Improvements:**
1. **Simplified Deployment**: One Docker service vs two
2. **Unified Endpoints**: All tracks on port 9600
3. **No Configuration Flags**: All tracks always available
4. **Production-Ready**: Direct TensorRT .plan files, instant startup

---

## Production Deployment

- Change Grafana password (default: admin/admin)
- Add TLS/SSL with reverse proxy
- Configure resource limits in docker-compose.yml
- Set up Prometheus alerts
- Use multiple GPUs: `device_ids: ['0', '1', '2']`

---

## Resources

- **NVIDIA Triton**: https://docs.nvidia.com/deeplearning/triton-inference-server/
- **Ultralytics**: https://docs.ultralytics.com/
- **NVIDIA DALI**: https://docs.nvidia.com/deeplearning/dali/

---

## Attribution

This project uses code from the [levipereira/ultralytics](https://github.com/levipereira/ultralytics) fork for end2end YOLO export with GPU-accelerated NMS. This enables **Track C** (4x speedup) by embedding TensorRT EfficientNMS into the model.

See [ATTRIBUTION.md](ATTRIBUTION.md) for complete third-party code attribution and licensing information.

---

**Built for maximum throughput** 🚀 *100K+ images in minutes*
