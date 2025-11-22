# Triton YOLO Inference Server - Complete Deployment Guide

**Optimized for Maximum Throughput on High-Performance Hardware**

This guide covers the complete deployment and benchmarking of a production-ready NVIDIA Triton Inference Server setup for YOLO11 models with four performance tracks, optimized for processing 100K+ images with maximum concurrency.

**Note:** This guide assumes models are already exported. For model building and export instructions, see [MODEL_EXPORT_GUIDE.md](MODEL_EXPORT_GUIDE.md).

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [System Requirements](#system-requirements)
3. [Quick Start](#quick-start)
4. [Four-Track Performance Architecture](#four-track-performance-architecture)
5. [Configuration Deep Dive](#configuration-deep-dive)
6. [Benchmarking](#benchmarking)
7. [Monitoring & Optimization](#monitoring--optimization)
8. [Troubleshooting](#troubleshooting)
9. [Production Deployment Best Practices](#production-deployment-best-practices)

---

## Architecture Overview

This deployment features a **four-track inference architecture** designed for comprehensive performance testing and optimization:

```
┌────────────────────────────────────────────────────────┐
│           Client Requests (HTTP/REST)                   │
└─────────────────────┬──────────────────────────────────┘
                      │
              ┌───────▼──────────┐
              │   YOLO API       │
              │ (All 4 Tracks)   │
              │   Port 9600      │
              │  32 workers      │
              └───────┬──────────┘
                      │
         ┌────────────┼────────────┐
         │            │            │
    Track A      ┌───▼────┐   Tracks B/C/D
   (PyTorch)     │ Triton │   (via gRPC)
  loaded at      │ Server │
   startup       │ GPU    │
                 │ 9500   │
                 └────┬───┘
                      │
         ┌────────────▼─────────────┐
         │     GPU 0 (NVIDIA)        │
         │  - Dynamic Batching       │
         │  - TensorRT Optimization  │
         │  - DALI GPU Preprocessing │
         │  - 16,384 concurrent cap  │
         └───────────────────────────┘
```

### Service Components

| Service | Port | Purpose | Workers |
|---------|------|---------|---------|
| **triton-api** | 9500 (HTTP), 9501 (gRPC), 9502 (metrics) | NVIDIA Triton Inference Server | GPU-based |
| **yolo-api** | 9600 | Unified FastAPI service (ALL 4 tracks) | 32 workers |
| **prometheus** | 9090 | Metrics collection | - |
| **grafana** | 3000 | Monitoring dashboard | - |

---

## System Requirements

### Minimum Requirements
- **CPU**: 16 cores
- **RAM**: 32 GB
- **GPU**: NVIDIA GPU with 8GB+ VRAM (Ampere or newer recommended)
- **Storage**: 50 GB free space
- **OS**: Ubuntu 20.04+ or similar Linux distribution

### Recommended (for 100K+ image processing)
- **CPU**: 48+ cores (as optimized in this deployment)
- **RAM**: 64+ GB
- **GPU**: NVIDIA A100, A6000, RTX 4090, or similar (16GB+ VRAM)
- **Storage**: 100+ GB NVMe SSD
- **Network**: 10 Gbps for distributed deployments

### Software Requirements
- Docker 24.0+
- Docker Compose 2.20+
- NVIDIA Container Toolkit
- CUDA 12.0+
- (Optional) Go 1.23+ for benchmark tool

---

## Quick Start

### 1. Clone and Navigate
```bash
cd /mnt/nvm/repos/triton-api
```

### 2. Build and Start All Services
```bash
# Stop any existing containers
docker compose down

# Rebuild images with new optimizations
docker compose build

# Start all services
docker compose up -d

# Watch startup logs
docker compose logs -f
```

### 3. Verify Services
```bash
# Check all containers are running
docker compose ps

# Check Triton server health
curl http://localhost:9500/v2/health/ready

# Check FastAPI gateways
curl http://localhost:9600/health  # Triton gateway (Tracks B+C+D)
curl http://localhost:9600/health  # PyTorch gateway (Track A)

# Check loaded models
curl http://localhost:9500/v2/models
```

### 4. Wait for Model Warmup (IMPORTANT!)

The first startup requires TensorRT engine compilation, which can take 2-5 minutes:

```bash
# Monitor Triton logs for warmup completion
docker compose logs -f triton-api | grep -i "successfully loaded"

# You should see:
# - yolov11_small_trt: 1 READY
# - yolov11_small_trt_end2end: 1 READY
# - yolo_preprocess_dali: 1 READY
# - yolov11_small_gpu_e2e: 1 READY
# - yolov11_small_gpu_e2e_streaming: 1 READY
# - yolov11_small_gpu_e2e_batch: 1 READY
```

### 5. Optional: Trigger FastAPI Warmup

```bash
# This runs inference on all models to finalize TensorRT engines
curl -X POST http://localhost:9600/warmup

# Check warmup status
curl http://localhost:9600/warmup_status
```

---

## Four-Track Performance Architecture

### Track A: PyTorch Baseline
- **Purpose**: Baseline performance reference
- **Technology**: Native PyTorch + CPU NMS
- **Endpoint**: `http://localhost:9600/pytorch/predict/small`
- **Expected Performance**: ~100-200 req/sec (depending on hardware)

### Track B: Triton Standard TRT
- **Purpose**: TensorRT acceleration with CPU NMS
- **Technology**: TensorRT FP16 engines + Ultralytics CPU NMS
- **Endpoint**: `http://localhost:9600/predict/small`
- **Expected Speedup**: 1.5-2.5x vs Track A
- **Configuration**:
  - Max batch size: 64
  - Instance count: 2
  - Dynamic batching: preferred [4, 8, 16, 32], 5ms delay
  - Precision: FP16

### Track C: Triton End2End TRT
- **Purpose**: Full GPU acceleration with compiled NMS
- **Technology**: TensorRT FP32 + GPU NMS (TRT::EfficientNMS_TRT)
- **Endpoint**: `http://localhost:9600/predict/small_end2end`
- **Expected Speedup**: 3-5x vs Track A
- **Configuration**:
  - Max batch size: 64
  - Instance count: 2
  - Dynamic batching: preferred [4, 8, 16, 32], 5ms delay
  - GPU NMS: Eliminates CPU bottleneck

### Track D: DALI + TRT End2End (Full GPU Pipeline)
- **Purpose**: Maximum performance with GPU preprocessing
- **Technology**: NVIDIA DALI + TensorRT + GPU NMS
- **Three Performance Tiers**:

#### D1: Streaming (Low Latency)
- **Endpoint**: `http://localhost:9600/predict/small_gpu_e2e_streaming`
- **Use Case**: Real-time video streaming
- **Configuration**:
  - Max batch size: 8
  - Batching window: 0.1ms
  - Instance count: 3
  - Preserve ordering: true

#### D2: Balanced (General Purpose)
- **Endpoint**: `http://localhost:9600/predict/small_gpu_e2e`
- **Use Case**: Mixed workloads
- **Configuration**:
  - Max batch size: 64
  - Batching window: 0.5ms
  - Instance count: 2

#### D3: Batch (Maximum Throughput)
- **Endpoint**: `http://localhost:9600/predict/small_gpu_e2e_batch`
- **Use Case**: Offline batch processing (100K+ images)
- **Configuration**:
  - Max batch size: 128
  - Batching window: 5ms
  - Instance count: 1
  - **Expected Speedup**: 5-10x vs Track A

**Track D Pipeline**:
```
Raw JPEG bytes → DALI GPU decode (nvJPEG)
                → DALI GPU letterbox (warp_affine)
                → DALI GPU normalize
                → TensorRT FP32 inference
                → GPU NMS (TRT::EfficientNMS_TRT)
                → Results
```

---

## Configuration Deep Dive

### FastAPI Worker Configuration

Both FastAPI services are optimized for maximum concurrent throughput on a 48-core system:

**File**: [`Dockerfile.triton-api`](Dockerfile.triton-api:79-89)
```dockerfile
CMD ["uvicorn", "src.main:app", \
     "--host", "0.0.0.0", \
     "--port", "9600", \
     "--workers", "32", \          # 32 workers for 48 CPU cores
     "--loop", "uvloop", \          # Fastest async event loop
     "--http", "httptools", \       # Ultra-fast HTTP parser
     "--backlog", "4096", \         # Handle burst traffic
     "--limit-concurrency", "512", \ # 512 concurrent requests per worker
     "--timeout-keep-alive", "75"]   # Standard keepalive
```

**Theoretical Capacity**: 32 workers × 512 concurrent/worker = **16,384 concurrent requests**

### Triton Dynamic Batching Configuration

**File**: [`models/yolov11_small_trt_end2end/config.pbtxt`](models/yolov11_small_trt_end2end/config.pbtxt:53-64)
```protobuf
dynamic_batching {
  preferred_batch_size: [ 4, 8, 16, 32 ]
  max_queue_delay_microseconds: 5000  # 5ms - enough time to fill batches
  preserve_ordering: false
  default_queue_policy {
    timeout_action: REJECT
    default_timeout_microseconds: 1000000  # 1 second
    max_queue_size: 128
  }
}
```

**Key Parameters**:
- **preferred_batch_size**: Triton tries to create these batch sizes for optimal GPU utilization
- **max_queue_delay**: Maximum time to wait for batch to fill (5ms = throughput-optimized)
- **preserve_ordering**: false = higher throughput (requests may return out of order)
- **max_queue_size**: Maximum requests in queue before rejection

### Optimal Concurrency Formula (NVIDIA Best Practice)

```
Optimal Concurrency = 2 × max_batch_size × instance_count
```

**For Track C (End2End TRT)**:
```
2 × 64 × 2 = 256 concurrent requests
```

To see dynamic batching in action, you need **at least 64-128 concurrent clients**.

---

## Benchmarking

### Option 1: Go Benchmark Tool (Recommended - True Concurrency)

**Setup**:
```bash
# Install Go (if not already installed)
cd /tmp
wget https://go.dev/dl/go1.23.4.linux-amd64.tar.gz
sudo tar -C /usr/local -xzf go1.23.4.linux-amd64.tar.gz
export PATH=$PATH:/usr/local/go/bin

# Build benchmark tool
cd /mnt/nvm/repos/triton-api/benchmarks
go build -o yolo_benchmark yolo_benchmark.go
```

**Run Benchmarks**:
```bash
# Test all tracks with 64 clients for 60 seconds
./yolo_benchmark --clients 64 --duration 60 --track all

# Test specific track with high concurrency
./yolo_benchmark --clients 256 --duration 120 --track D_batch

# Progressive load test
for clients in 16 32 64 128 256; do
  echo "Testing with $clients clients..."
  ./yolo_benchmark --clients $clients --duration 60 --output "results/benchmark_${clients}clients.json"
  sleep 10
done
```

**See detailed instructions**: [benchmarks/GO_SETUP.md](benchmarks/GO_SETUP.md)

### Option 2: Async Python Benchmark (Proper aiohttp)

```bash
# Test all tracks with 64 concurrent clients
python3 benchmarks/async_benchmark.py --clients 64 --duration 60 --track all

# Test specific track
python3 benchmarks/async_benchmark.py --clients 128 --duration 120 --track D_batch
```

### Option 3: Quick Concurrency Test (Fast Validation)

```bash
# Quick 30-second test to verify batching is working
python3 benchmarks/quick_concurrency_test.py
```

### Interpreting Results

**Expected Performance Ranges** (single NVIDIA A100 GPU):

| Track | Throughput (req/sec) | P50 Latency (ms) | Speedup vs A |
|-------|---------------------|------------------|--------------|
| A (PyTorch) | 150-200 | 40-60 | 1.0x (baseline) |
| B (TRT) | 300-400 | 20-30 | 2.0-2.5x |
| C (End2End TRT) | 600-800 | 10-15 | 4.0-5.0x |
| D (DALI Streaming) | 400-600 | 8-12 | 3.0-4.0x |
| D (DALI Balanced) | 800-1200 | 12-18 | 6.0-8.0x |
| D (DALI Batch) | 1500-2500 | 20-30 | 10.0-15.0x |

**Note**: Actual performance depends on your GPU model, CPU, and image resolution.

### Signs of Successful Dynamic Batching

1. **Check Triton Logs**:
```bash
docker compose logs triton-api | grep -i "batch size"

# You should see lines like:
# batch size: 8
# batch size: 16
# batch size: 32
```

2. **Monitor GPU Utilization**:
```bash
nvidia-smi -l 1

# GPU utilization should be 80-100% during benchmarks
```

3. **Performance Scaling**:
- Going from 1 → 16 clients should show significant throughput increase
- Going from 16 → 64 clients should show continued improvement
- Above 256 clients, throughput should plateau (GPU saturated)

---

## Monitoring & Optimization

### Grafana Dashboard

Access: http://localhost:3000 (admin/admin)

**Key Metrics to Monitor**:
1. **Inference Request Rate**: Should match your benchmark concurrency
2. **Batch Size Distribution**: Should show batches of 8, 16, 32, 64
3. **GPU Utilization**: Should be 80-100% during load
4. **Queue Time**: Should be under 5ms for most requests
5. **Inference Time**: Model execution time only

### Prometheus Metrics

Access: http://localhost:9090

**Useful Queries**:
```promql
# Request rate
rate(nv_inference_request_success{model="yolov11_small_trt_end2end"}[1m])

# Batch size
nv_inference_exec_count{model="yolov11_small_trt_end2end"}

# Queue time
nv_inference_queue_duration_us{model="yolov11_small_trt_end2end"}

# Compute time
nv_inference_compute_infer_duration_us{model="yolov11_small_trt_end2end"}
```

### Real-Time Monitoring Commands

```bash
# Watch GPU utilization
nvidia-smi -l 1

# Watch Triton logs (show batch sizes)
docker compose logs -f triton-api | grep -i batch

# Watch FastAPI logs
docker compose logs -f yolo-api
docker compose logs -f yolo-api

# Monitor Docker stats
docker stats
```

### Performance Tuning

#### If Throughput is Low:

1. **Increase Concurrency**:
   - Try 128, 256, or 512 concurrent clients
   - Formula: 2 × max_batch_size × instance_count

2. **Check GPU Utilization**:
   - Should be 80-100% during load
   - If lower, increase concurrency or batch sizes

3. **Adjust Batching Delay**:
   - Edit `config.pbtxt` → `max_queue_delay_microseconds`
   - Higher delay (5000-10000μs) = larger batches = higher throughput
   - Lower delay (100-1000μs) = smaller batches = lower latency

4. **Increase Instance Count**:
   - Edit `config.pbtxt` → `instance_group.count`
   - For Track C: Try `count: 3` or `count: 4`

#### If Latency is High:

1. **Reduce Batching Delay**:
   - Lower `max_queue_delay_microseconds` to 100-1000μs

2. **Use Streaming Tier** (Track D):
   - `small_gpu_e2e_streaming` optimized for low latency

3. **Reduce Concurrency**:
   - Lower concurrent clients to 16-32

---

## Troubleshooting

### Models Not Loading

**Symptom**: Triton shows model errors or models stuck in "loading" state

**Solutions**:
```bash
# Check Triton logs
docker compose logs triton-api | grep -i error

# Verify model files exist
ls -la models/yolov11_small_trt/1/
ls -la models/yolov11_small_trt_end2end/1/

# Restart Triton
docker compose restart triton-api
```

### No Speed Gains from Concurrency

**Symptom**: 16 clients same performance as 1 client

**Root Causes**:
1. ✓ **FIXED**: FastAPI workers = 1 (now = 32)
2. ✓ **FIXED**: Track B model not loaded (now loaded)
3. Check GPU is being used: `nvidia-smi`

**Verify Fix**:
```bash
# Check worker count in container
docker compose exec yolo-api ps aux | grep uvicorn

# Should show 32 worker processes
```

### High Error Rate

**Symptom**: 50%+ of requests fail

**Solutions**:
```bash
# Check service health
curl http://localhost:9600/health

# Check Docker resources
docker stats

# Increase timeout if needed (src/main.py)
# Or increase Docker memory limits (docker-compose.yml)

# Check for OOM errors
docker compose logs triton-api | grep -i "out of memory"
```

### GPU Out of Memory

**Symptom**: CUDA OOM errors in logs

**Solutions**:
1. **Reduce max_batch_size**: Edit `config.pbtxt` → `max_batch_size: 32`
2. **Reduce instance count**: Edit `config.pbtxt` → `instance_group.count: 1`
3. **Use smaller model**: Use `nano` instead of `small`
4. **Check GPU memory**: `nvidia-smi`

---

## Production Deployment Best Practices

### Security

1. **Remove default credentials**: Change Grafana admin password
2. **Use TLS/SSL**: Add HTTPS reverse proxy (nginx/traefik)
3. **Restrict ports**: Use firewall to limit external access
4. **Non-root containers**: Already configured (user: appuser)
5. **Read-only filesystem**: Add to docker-compose.yml

### Scalability

1. **Horizontal Scaling**: Deploy multiple Triton instances with load balancer
2. **GPU Pools**: Use multiple GPUs with `device_ids: ['0', '1', '2']`
3. **Model Routing**: Direct different clients to different tiers
4. **Caching**: Add Redis for response caching

### High Availability

1. **Health Checks**: Already configured in docker-compose.yml
2. **Auto-Restart**: `restart: always` configured
3. **Monitoring**: Prometheus + Grafana + alerts
4. **Backup Models**: Store model files in external storage

### Resource Limits

Add to docker-compose.yml:
```yaml
services:
  triton-api:
    deploy:
      resources:
        limits:
          cpus: '24'
          memory: 32G
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']
              capabilities: [gpu]
```

### Logging

```yaml
services:
  triton-api:
    logging:
      driver: "json-file"
      options:
        max-size: "100m"
        max-file: "5"
```

---

## Next Steps

1. **Run Initial Benchmark**: `./yolo_benchmark --clients 64 --duration 60`
2. **Analyze Results**: Check throughput and latency metrics
3. **Optimize**: Adjust batching configuration based on workload
4. **Scale**: Add more GPUs or instances as needed
5. **Monitor**: Set up alerts in Grafana for production

---

## Support & Resources

- **NVIDIA Triton Docs**: https://docs.nvidia.com/deeplearning/triton-inference-server/
- **Ultralytics Docs**: https://docs.ultralytics.com/
- **FastAPI Docs**: https://fastapi.tiangolo.com/
- **NVIDIA DALI**: https://docs.nvidia.com/deeplearning/dali/

---

## Summary of Optimizations Applied

### Critical Fixes
1. ✅ **Restored all four tracks** from archive
2. ✅ **Increased FastAPI workers** from 1 → 32 (2560% increase in capacity)
3. ✅ **Fixed ensemble configs** to include affine_matrices input
4. ✅ **Optimized connection pooling** with uvloop and httptools
5. ✅ **Fixed benchmarking** to use true async concurrency

### Performance Improvements
- **Theoretical capacity**: 1 concurrent request → 16,384 concurrent requests
- **Expected throughput gain**: 10-15x on Track D with proper concurrency
- **Latency improvements**: P95 latency should drop 50-70% with GPU NMS

### New Tools
- **Go benchmark tool**: Professional concurrency testing
- **Async Python benchmark**: Proper aiohttp implementation
- **Comprehensive monitoring**: Prometheus + Grafana dashboards

**Your system is now ready for 100K+ image processing with maximum throughput!** 🚀

---

## Related Documentation

- **[MODEL_EXPORT_GUIDE.md](MODEL_EXPORT_GUIDE.md)** - Model building and export for all tracks
- **[Tracks/BENCHMARKING_GUIDE.md](Tracks/BENCHMARKING_GUIDE.md)** - Comprehensive benchmarking methodology
- **[Tracks/TRACK_D_COMPLETE.md](Tracks/TRACK_D_COMPLETE.md)** - Complete Track D guide
- **[Technical/TRITON_BEST_PRACTICES.md](Technical/TRITON_BEST_PRACTICES.md)** - Triton optimization
- **[Attribution/END2END_ANALYSIS.md](Attribution/END2END_ANALYSIS.md)** - Fork attribution and analysis
- **[TESTING.md](TESTING.md)** - Testing methodology
- **[../ATTRIBUTION.md](../ATTRIBUTION.md)** - Third-party code attribution
