# Testing Guide

Complete testing guide for the Triton YOLO Inference System.

## Quick Start

```bash
# 1. Start services
docker compose up -d

# 2. Wait for models to load (2-3 minutes first time)
docker compose logs -f triton-api | grep "successfully loaded"

# 3. Check services
bash scripts/check_services.sh

# 4. Run integration tests
bash tests/test_inference.sh

# 5. Run benchmarks
cd benchmarks
./build.sh
./triton_bench --mode quick
```

---

## Architecture Overview

### Unified Single-Service Design

All 4 inference tracks are accessible via a single FastAPI service on **port 9600**:

```
┌─────────────────────────────────────────┐
│         yolo-api (port 9600)            │
├─────────────────────────────────────────┤
│  Track A: PyTorch baseline              │
│  Track B: TRT + CPU NMS                 │
│  Track C: TRT End2End + GPU NMS         │
│  Track D: DALI + TRT End2End (3 modes)  │
└─────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│    triton-api (internal: port 8000)     │
│    NVIDIA Triton Inference Server       │
└─────────────────────────────────────────┘
```

### Service Endpoints

**All tracks on port 9600:**

```bash
# Health check
curl http://localhost:9600/health

# Track A: PyTorch baseline
curl -X POST http://localhost:9600/pytorch/predict/small \
  -F "image=@test.jpg"

# Track B: Standard TensorRT
curl -X POST http://localhost:9600/predict/small \
  -F "image=@test.jpg"

# Track C: End2End TRT + GPU NMS
curl -X POST http://localhost:9600/predict/small_end2end \
  -F "image=@test.jpg"

# Track D: DALI + TRT (3 performance tiers)
curl -X POST http://localhost:9600/predict/small_gpu_e2e_streaming \
  -F "image=@test.jpg"

curl -X POST http://localhost:9600/predict/small_gpu_e2e \
  -F "image=@test.jpg"

curl -X POST http://localhost:9600/predict/small_gpu_e2e_batch \
  -F "image=@test.jpg"
```

---

## Testing Levels

### 1. Service Health Check

**Purpose:** Verify services are running and models are loaded

**Command:**
```bash
bash scripts/check_services.sh
```

**What it checks:**
- Triton server health and model status
- FastAPI service health
- Model readiness for all tracks

**Expected output:**
```
✓ Triton server is healthy
✓ All 6 models loaded successfully
✓ FastAPI service is healthy
```

---

### 2. Integration Testing

**Purpose:** Quick functional test with sample images

**Command:**
```bash
bash tests/test_inference.sh
```

**What it does:**
- Sends 10 test images through all available tracks
- Verifies successful inference responses
- Checks detection format and validity
- No dependencies required (uses docker exec)

**Expected output:**
```
Testing Track A (PyTorch)...
✓ Image 1/10: 3 detections
✓ Image 2/10: 5 detections
...

Testing Track B (TRT)...
...

All tests passed!
```

---

### 3. Performance Benchmarking

**Purpose:** Measure throughput and latency under load

**Tool:** Go-based benchmark tool (`triton_bench`)

**Setup:**
```bash
cd benchmarks
./build.sh
```

**Test Modes:**

| Mode | Description | Duration | Best For |
|------|-------------|----------|----------|
| `single` | Single image test | Instant | Quick validation |
| `quick` | 30-second concurrency test | 30s | Development checks |
| `full` | Comprehensive benchmark | 60s | Performance analysis |
| `all` | Process all test images | Variable | Accuracy testing |
| `sustained` | Max throughput test | 5 min | Capacity planning |
| `variable` | Variable load patterns | 10 min | Real-world simulation |

**Examples:**
```bash
# Quick test (30 seconds)
./triton_bench --mode quick

# Full benchmark all tracks
./triton_bench --mode full

# Test specific track with high concurrency
./triton_bench --mode full --track D_batch --clients 256

# Process large image set
./triton_bench --mode all --images /path/to/images --clients 128

# Sustained max throughput test
./triton_bench --mode sustained --duration 300
```

**Output:**
- JSON results saved to `benchmarks/results/`
- Metrics: throughput (rps), latency (p50/p95/p99), error rate
- Per-track comparison

**See:** [benchmarks/README.md](../benchmarks/README.md)

---

### 4. Load Testing

**Purpose:** Stress test with realistic production patterns

**Tool:** Shell script wrapper for comprehensive benchmarks

**Commands:**
```bash
cd benchmarks

# Quick validation (30s, 16 clients)
./triton_bench --mode quick

# Full benchmark (60s, 128 clients)
./triton_bench --mode full --clients 128 --duration 60

# Sustained throughput test
./triton_bench --mode sustained
```

**What it tests:**
- Concurrent client handling (up to 512 concurrent requests)
- Dynamic batching effectiveness
- GPU utilization under load
- Memory stability over time

---

## Test Image Sets

### Built-in Test Images

Location: `test_images/`

Common test images:
- `bus.jpg` - Multi-object scene (bus, people, cars)
- `zidane.jpg` - Person detection
- Various aspect ratios for preprocessing validation

### Custom Image Sets

For benchmarking with your own data:

```bash
./triton_bench --mode all --images /path/to/your/images --clients 128
```

Supported formats: JPG, PNG, BMP, TIFF

---

## Validation Tests

### ONNX End2End Model

Test the exported End2End model locally:

```bash
docker compose exec yolo-api python /app/tests/test_onnx_end2end.py
```

### DALI Pipeline Validation

Verify DALI preprocessing accuracy:

```bash
docker compose exec yolo-api python /app/dali/validate_dali_letterbox.py
```

Expected: <0.5% pixel difference vs. CPU preprocessing

### End2End Patch Verification

Verify Ultralytics end2end export patch:

```bash
docker compose exec yolo-api python /app/tests/test_end2end_patch.py
```

---

## Monitoring During Tests

### Terminal 1: GPU Utilization
```bash
nvidia-smi -l 1
# Should show 80-100% GPU utilization during load
```

### Terminal 2: Triton Batching
```bash
docker compose logs -f triton-api | grep "batch size"
# Should show dynamic batching: 8, 16, 32, 64
```

### Terminal 3: FastAPI Logs
```bash
docker compose logs -f yolo-api
# Monitor request flow and any errors
```

### Terminal 4: Grafana Dashboard
```
http://localhost:3000 (admin/admin)
```

Import dashboard: `monitoring/triton-dashboard.json`

---

## Expected Performance

NVIDIA A100 GPU, 128-256 concurrent clients:

| Track | Technology | Throughput | P50 Latency | Speedup |
|-------|-----------|------------|-------------|---------|
| A | PyTorch + CPU NMS | 150-200 rps | 40-60ms | 1.0x |
| B | TRT + CPU NMS | 300-400 rps | 20-30ms | 2.0x |
| C | TRT End2End + GPU NMS | 600-800 rps | 10-15ms | 4.0x |
| D (batch) | DALI + TRT + GPU NMS | **1500-2500 rps** | 20-30ms | **12.5x** |

**100,000 images:**
- Track A (PyTorch): ~9 minutes
- Track D (DALI): ~40 seconds

---

## Troubleshooting

### Services Won't Start

```bash
# Check logs
docker compose logs triton-api | grep -i error
docker compose logs yolo-api | grep -i error

# Restart services
docker compose restart

# Full rebuild
docker compose down
docker compose up -d --build
```

### No Performance Gains

```bash
# Check worker count (should be 32+1 master = 33 processes)
docker compose exec yolo-api ps aux | grep uvicorn | wc -l

# Check Triton batching (should see batch size > 1)
docker compose logs triton-api | grep "batch size"

# Try higher concurrency
./triton_bench --mode full --clients 256

# Check GPU usage
nvidia-smi
```

### High Error Rate

```bash
# Check resource usage
docker stats

# Reduce concurrency
./triton_bench --mode full --clients 32

# Check for OOM errors
docker compose logs | grep -i "out of memory"
```

### Models Not Loading

```bash
# Check Triton model status
curl http://localhost:9500/v2/models/yolov11_small_trt/ready

# Check if models exist
ls -la models/yolov11_small_trt/
ls -la models/yolov11_small_trt_end2end/

# Re-export models
docker compose exec yolo-api python /app/export/export_models.py \
    --models small --formats trt trt_end2end

# Restart Triton
docker compose restart triton-api
```

---

## Continuous Integration

### Basic CI Pipeline

```bash
#!/bin/bash
# ci-test.sh

set -e

echo "Starting services..."
docker compose up -d

echo "Waiting for services..."
sleep 30

echo "Running health checks..."
bash scripts/check_services.sh

echo "Running integration tests..."
bash tests/test_inference.sh

echo "Running quick benchmark..."
cd benchmarks && ./triton_bench --mode quick

echo "All tests passed!"
```

---

## Development Workflow

### Test-Driven Development

```bash
# 1. Make code changes
# 2. Rebuild container
docker compose up -d --build yolo-api

# 3. Quick validation
bash tests/test_inference.sh

# 4. Performance check
cd benchmarks && ./triton_bench --mode quick

# 5. If good, full benchmark
./triton_bench --mode full
```

### Debugging Failed Tests

```bash
# Interactive shell in container
docker compose exec yolo-api bash

# Manual inference test
python /app/tests/test_onnx_end2end.py

# Check model files
ls -la /app/models/yolov11_small_trt/

# Test single endpoint manually
curl -X POST http://localhost:9600/predict/small \
  -F "image=@/app/test_images/bus.jpg" | jq
```

---

## Common Test Scenarios

### Scenario 1: Validate New Model Export

```bash
# Export new model size
docker compose exec yolo-api python /app/export/export_models.py \
    --models medium --formats trt trt_end2end

# Test it
bash tests/test_inference.sh

# Benchmark it
./triton_bench --mode quick
```

### Scenario 2: Compare Preprocessing Methods

```bash
# Benchmark Track C (CPU preprocessing)
./triton_bench --mode full --track C

# Benchmark Track D (GPU preprocessing)
./triton_bench --mode full --track D_batch

# Compare results in results/
```

### Scenario 3: Capacity Planning

```bash
# Find max throughput
./triton_bench --mode sustained --duration 300 --clients 256

# Variable load simulation
./triton_bench --mode variable

# Check GPU memory headroom
nvidia-smi
```

---

## Documentation

- **[AUTOMATION.md](../AUTOMATION.md)** - Complete automation guide
- **[README.md](../README.md)** - Main project documentation
- **[benchmarks/README.md](../benchmarks/README.md)** - Benchmark tool guide
- **[MODEL_EXPORT_GUIDE.md](MODEL_EXPORT_GUIDE.md)** - Model export reference

---

**Last Updated:** November 2025
