# Triton YOLO Benchmarking Guide

Comprehensive guide for benchmarking all 4 performance tracks.

---

## Overview

This guide covers benchmarking methodology, best practices, and analysis for the Triton YOLO deployment system. The project is optimized to use a **single model size (small)** across all tracks for clean, comparable benchmarks.

### Track Definitions

| Track | Description | Preprocessing | Inference | NMS | Expected Speedup |
|-------|-------------|---------------|-----------|-----|------------------|
| **A** | PyTorch Direct | CPU (Python) | GPU (PyTorch) | CPU (Python) | 1.0x (baseline) |
| **B** | Triton Standard TRT | CPU (Python) | GPU (TensorRT) | CPU (Ultralytics) | 1.5-2.5x |
| **C** | Triton End2End TRT | CPU (Python) | GPU (TensorRT) | GPU (EfficientNMS) | 3-5x |
| **D** | DALI + TRT End2End | GPU (DALI) | GPU (TensorRT) | GPU (EfficientNMS) | 4-8x |

### Architecture Benefits

**Track D (Full GPU Pipeline)** delivers the highest performance by:
- GPU JPEG decoding (nvJPEG)
- GPU preprocessing (DALI with letterbox)
- Pipelined operations (no CPU bottleneck)
- Smaller data transfers (JPEG bytes vs raw tensors)

---

## Best Practices

### Simplified Approach: Single Model Size

The project uses **only the small model** across all tracks for clean benchmarking.

#### GPU Memory Comparison

**Before (all 3 model sizes)**:

| Track | Models Loaded | Approx. Memory |
|-------|---------------|----------------|
| Track B (Standard TRT) | nano, small, medium | ~1.5 GB |
| Track C (End2End TRT) | nano, small, medium | ~2.0 GB |
| Track D (DALI) | 1 preprocessing model | ~0.2 GB |
| Track D (Ensembles) | 9 configs (3 sizes × 3 tiers) | 0 GB |
| **Total** | **7 TRT engines + 1 DALI** | **~3.7 GB** |

**After (small model only)**:

| Track | Models Loaded | Approx. Memory |
|-------|---------------|----------------|
| Track B (Standard TRT) | small only | ~0.5 GB |
| Track C (End2End TRT) | small only | ~0.7 GB |
| Track D (DALI) | 1 preprocessing model | ~0.2 GB |
| Track D (Ensembles) | 3 configs (1 size × 3 tiers) | 0 GB |
| **Total** | **2 TRT engines + 1 DALI + 3 ensembles** | **~1.4 GB** |

**Result**: 62% reduction in GPU memory usage

#### Benefits of Single Model Size

1. **No Memory Pressure**: Minimal GPU memory footprint
2. **No Resource Contention**: Only active models loaded
3. **Clean Benchmarks**: Isolated performance measurements
4. **Faster Deployment**: Fewer models to load and manage

### Checking GPU Memory Usage

Before benchmarking, verify GPU memory:

```bash
nvidia-smi
# Or use Grafana dashboard at http://localhost:3000
```

**Expected output** (with small models only):

```
================================================================================
GPU Memory Analysis
================================================================================

Current GPU Memory Usage:

  GPU 0 (NVIDIA RTX A6000):
    Memory: 1420 MB / 49140 MB (2.9% used)
    Free: 47720 MB
    Utilization: 8%

================================================================================
Triton Model Memory Breakdown
================================================================================

  ✓ yolov11_small_trt
  ✓ yolov11_small_trt_end2end
  ✓ yolo_preprocess_dali
  ✓ yolov11_small_gpu_e2e_streaming
  ✓ yolov11_small_gpu_e2e
  ✓ yolov11_small_gpu_e2e_batch

================================================================================
Recommendation
================================================================================

✓ GPU memory usage is healthy: 2.9% (1420 MB / 49140 MB)

Memory pressure is low. Benchmarks should be clean.
```

### Configuration

The main `docker-compose.yml` is optimized for both development and benchmarking:

```yaml
command:
  - tritonserver
  - --model-store=/models
  # Track B: Standard TRT (CPU NMS)
  - --load-model=yolov11_small_trt
  # Track C: End2End TRT (GPU NMS, CPU preprocess)
  - --load-model=yolov11_small_trt_end2end
  # Track D: DALI preprocessing
  - --load-model=yolo_preprocess_dali
  # Track D: Streaming ensemble (0.1ms batching)
  - --load-model=yolov11_small_gpu_e2e_streaming
  # Track D: Balanced ensemble (0.5ms batching)
  - --load-model=yolov11_small_gpu_e2e
  # Track D: Batch ensemble (5ms batching)
  - --load-model=yolov11_small_gpu_e2e_batch
```

### Testing Different Model Sizes

If you need to test **nano** or **medium** models:

| Model Size | Use Case | Speed | Accuracy |
|------------|----------|-------|----------|
| **nano** | Real-time video, high FPS | Fastest | Good |
| **small** | Balanced workloads (current) | Fast | Better |
| **medium** | Accuracy-critical | Moderate | Best |

Steps to switch model sizes:

1. Export the desired model size:
   ```bash
   docker compose exec yolo-api python /app/scripts/export_models.py \
     --models nano --formats trt trt_end2end
   ```

2. Create ensembles for that size:
   ```bash
   docker compose exec yolo-api python /app/scripts/create_ensembles.py --models nano
   ```

3. Update `docker-compose.yml` to load the new size instead of small

4. Restart services:
   ```bash
   docker compose down && docker compose up -d
   ```

---

## Test Methodology

### Benchmark Scenarios

#### Scenario 1: Single-Image Latency (Cold Start)

**Purpose**: Measure end-to-end latency for a single request (worst case).

**Method**:
1. Restart service (clear Triton model cache)
2. Send single image to endpoint
3. Measure total time from request to response

**Expected Results**:
| Track | Latency (cold) | Notes |
|-------|----------------|-------|
| A (PyTorch) | 15-25ms | PyTorch overhead |
| B (TRT) | 10-15ms | TRT faster, CPU NMS |
| C (End2End) | 8-12ms | GPU NMS saves 2-3ms |
| D (GPU E2E) | **6-8ms** | GPU preprocessing saves 3-5ms |

**Target**: Track D should be **30-40% faster** than Track C.

---

#### Scenario 2: Single-Image Latency (Warm)

**Purpose**: Measure steady-state latency after warmup.

**Method**:
1. Send 10 warmup requests to each endpoint
2. Send 100 single images sequentially
3. Measure P50, P95, P99 latency

**Expected Results**:
| Track | P50 Latency | P95 Latency | P99 Latency |
|-------|-------------|-------------|-------------|
| A | 12-15ms | 18-22ms | 25-30ms |
| B | 8-10ms | 12-15ms | 18-22ms |
| C | 6-8ms | 10-12ms | 15-18ms |
| D | **4-6ms** | **7-9ms** | **10-12ms** |

**Success Criteria**: Track D P99 < 10ms (suitable for 60 FPS streaming).

---

#### Scenario 3: Batch Throughput (Fixed Batch Size)

**Purpose**: Measure maximum throughput with optimal batching.

**Method**:
1. Send batches of N images (N = 1, 4, 8, 16, 32, 64, 128)
2. Measure images/second

**Expected Results** (batch=8):
| Track | Throughput (img/sec) | GPU Util | CPU Util |
|-------|----------------------|----------|----------|
| A | 80-120 | 60-70% | 30-40% |
| B | 150-250 | 75-85% | 40-60% (NMS) |
| C | 250-400 | 80-90% | 30-50% (preprocess) |
| D | **400-600** | **85-95%** | **<20%** |

**Expected Results** (batch=128):
| Track | Throughput (img/sec) | Notes |
|-------|----------------------|-------|
| A | N/A | OOM (PyTorch memory) |
| B | 300-500 | CPU NMS bottleneck |
| C | 500-800 | CPU preprocess bottleneck |
| D | **1000-1500** | Full GPU utilization |

**Success Criteria**: Track D achieves **2-3x throughput** vs Track C at batch=128.

---

#### Scenario 4: Concurrent Requests (Load Test)

**Purpose**: Simulate real-world API load with multiple concurrent clients.

**Method**:
1. Use concurrent clients with N workers (N = 1, 4, 8, 16, 32)
2. Each client sends images continuously for 60 seconds
3. Measure total throughput and latency distribution

**Expected Results** (16 concurrent clients):
| Track | Total Throughput | P50 Latency | P95 Latency | CPU Saturation |
|-------|------------------|-------------|-------------|----------------|
| A | 100-150 img/sec | 80-120ms | 150-200ms | Yes (>90%) |
| B | 200-300 img/sec | 40-60ms | 80-120ms | Yes (>80%) |
| C | 300-500 img/sec | 25-40ms | 60-80ms | Yes (>70%) |
| D | **600-1000 img/sec** | **15-25ms** | **40-60ms** | **No (<30%)** |

**Key Insight**: Track D should scale linearly with concurrency (no CPU bottleneck).

---

#### Scenario 5: Video Frame Processing (Sequential)

**Purpose**: Simulate video file processing (sequential frames, temporal order).

**Method**:
1. Extract 1000 frames from test video (1080p → resize to test image)
2. Send frames sequentially (maintain order)
3. Measure total processing time and FPS

**Expected Results**:
| Track | Processing Time | Avg FPS | Suitable for Real-Time? |
|-------|----------------|---------|-------------------------|
| A | 18-25 sec | 40-55 FPS | No (< 60 FPS) |
| B | 12-18 sec | 55-85 FPS | Marginal |
| C | 8-12 sec | 85-125 FPS | Yes |
| D | **5-8 sec** | **125-200 FPS** | Yes (2x headroom) |

**Success Criteria**: Track D processes 1000 frames in <8 seconds (>125 FPS).

---

#### Scenario 6: Streaming Latency (Real-Time)

**Purpose**: Measure frame-to-frame latency for streaming video (60 FPS target).

**Method**:
1. Simulate 60 FPS stream (send frame every 16.7ms)
2. Measure per-frame latency
3. Check for frame drops (latency > 16.7ms)

**Expected Results**:
| Track | P50 Latency | P99 Latency | Frame Drops (60 FPS) |
|-------|-------------|-------------|----------------------|
| A | 12-15ms | 22-28ms | ~15-25% |
| B | 8-10ms | 15-20ms | ~5-10% |
| C | 6-8ms | 12-16ms | ~1-3% |
| D | **4-6ms** | **8-10ms** | **<0.5%** |

**Success Criteria**: Track D maintains <1% frame drop rate at 60 FPS.

---

### Resource Monitoring During Benchmarks

**Monitor GPU utilization**:
```bash
# Terminal 1: Start benchmark
./triton_bench --mode full

# Terminal 2: Monitor GPU
nvidia-smi dmon -s ucm -d 1
```

**Expected GPU Utilization**:
| Track | GPU Util (single) | GPU Util (concurrent) | Bottleneck |
|-------|-------------------|-----------------------|------------|
| A | 40-50% | 60-70% | PyTorch overhead |
| B | 60-70% | 70-80% | CPU NMS |
| C | 70-80% | 75-85% | CPU preprocess |
| D | **80-90%** | **90-95%** | **None (GPU-bound)** |

**Success Criteria**: Track D should show **90%+ GPU utilization** under load.

---

## Running Benchmarks

### Prerequisites

```bash
# 1. Export TRT models (~30-40 minutes first time)
docker compose exec yolo-api python /app/scripts/export_models.py \
  --models small --formats trt trt_end2end

# 2. Create ensembles
docker compose exec yolo-api python /app/scripts/create_ensembles.py --models small

# 3. Start all services
docker compose up -d

# 4. Verify services
curl http://localhost:9600/health | jq
```

### Using the triton_bench Tool

All benchmarking is now done using the comprehensive `triton_bench` Go tool.

See [/mnt/nvm/repos/triton-api/benchmarks/README.md](../../benchmarks/README.md) for complete documentation.

#### Quick Start

```bash
# Build the tool
cd /mnt/nvm/repos/triton-api/benchmarks
go build -o triton_bench triton_bench.go

# Run quick test (30 seconds, all tracks)
./triton_bench --mode quick
```

#### Test Modes

| Mode | Command | Description |
|------|---------|-------------|
| **Single Image** | `./triton_bench --mode single` | One image through all tracks (10 iterations) |
| **Image Set** | `./triton_bench --mode set --limit 50` | Process N images sequentially |
| **Quick Concurrency** | `./triton_bench --mode quick` | 30-second test with 16 clients |
| **Full Concurrency** | `./triton_bench --mode full --clients 128 --duration 60` | Comprehensive concurrent load test |
| **All Images** | `./triton_bench --mode all --clients 64` | Process entire directory with concurrency |
| **Sustained Throughput** | `./triton_bench --mode sustained` | Find optimal client count + 5-min stress test |
| **Variable Load** | `./triton_bench --mode variable --load-pattern burst` | Test with changing load patterns |

#### Common Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `quick` | Test mode |
| `--images` | `/mnt/nvm/KILLBOY_SAMPLE_PICTURES` | Image directory |
| `--track` | `all` | Track filter (A, B, C, D_streaming, D_balanced, D_batch, all) |
| `--clients` | `64` | Concurrent clients |
| `--duration` | `60` | Test duration (seconds) |
| `--limit` | `100` | Max images to load |
| `--warmup` | `10` | Warmup requests |
| `--quiet` | `false` | Minimal output |
| `--json` | `false` | JSON output only |

#### Example Workflows

**Initial Validation**:
```bash
./triton_bench --mode single       # Verify correctness
./triton_bench --mode quick        # Check batching
./triton_bench --mode full         # Full benchmark
```

**Find Maximum Throughput**:
```bash
./triton_bench --mode sustained
```

**Process Production Dataset**:
```bash
./triton_bench --mode all --images /path/to/data --clients 128
```

**Load Testing**:
```bash
# Progressive load
for clients in 16 32 64 128 256; do
  ./triton_bench --mode full --clients $clients --duration 60
done
```

#### Output Files

Results automatically saved to `benchmarks/results/` with timestamps:

```
benchmarks/results/
├── single_image_20250116_153045.json
├── quick_concurrency_20250116_153215.json
├── full_concurrency_20250116_154330.json
└── sustained_throughput_20250116_160102.json
```

### Standard Workflow

```bash
# 1. Check current memory usage
nvidia-smi

# 2. Ensure services are running
docker compose up -d

# 3. Wait for models to load (~30s)
docker compose logs -f triton-api | grep "Successfully loaded"

# 4. Run benchmarks
cd benchmarks
./triton_bench --mode full
```

### Testing Endpoints Manually

```bash
# Test Track A (PyTorch)
curl http://localhost:9600/pytorch/predict/small -F "image=@test.jpg"

# Test Track B (Standard TRT)
curl http://localhost:9600/predict/small -F "image=@test.jpg"

# Test Track C (End2End TRT)
curl http://localhost:9600/predict/small_end2end -F "image=@test.jpg"

# Test Track D (DALI + End2End TRT)
curl http://localhost:9600/predict/small_gpu_e2e_streaming -F "image=@test.jpg"
```

### Monitoring

While running benchmarks, monitor in separate terminals:

```bash
# Terminal 1: GPU utilization
nvidia-smi -l 1

# Terminal 2: Check batching
docker compose logs -f triton-api | grep "batch size"

# Terminal 3: Grafana
# http://localhost:3000
```

---

## Results & Analysis

### Performance Targets

**Track D Goals** (vs Track C):
- 30-40% faster single-image latency
- 2-3x higher batch throughput
- 2-4x higher concurrent throughput
- <10ms P99 latency (suitable for 60 FPS streaming)
- 90%+ GPU utilization (no CPU bottleneck)

### Validation

**If Track D doesn't meet goals**:
- Investigate DALI pipeline (letterbox overhead? decode issues?)
- Check dynamic batching configuration
- Verify GPU utilization patterns

**If Track D meets goals**:
- Publish results
- Update documentation
- Deploy to production

### Expected Performance Summary

**Latency (Single Image)**:

| Track | nano | small | medium |
|-------|------|-------|--------|
| A (PyTorch) | ~12ms | ~18ms | ~28ms |
| B (Standard TRT) | ~8ms | ~12ms | ~18ms |
| C (End2End TRT) | ~3ms | ~5ms | ~8ms |
| D (GPU E2E) | **~2ms** | **~4ms** | **~6ms** |

**Throughput (Concurrent Requests)**:

| Track | Concurrency 1 | Concurrency 8 | Concurrency 16 |
|-------|---------------|---------------|----------------|
| A (PyTorch) | ~80 img/sec | ~85 img/sec | ~90 img/sec |
| B (Standard TRT) | ~120 img/sec | ~140 img/sec | ~150 img/sec |
| C (End2End TRT) | ~320 img/sec | ~400 img/sec | ~450 img/sec |
| D (GPU E2E) | **~500 img/sec** | **~800 img/sec** | **~1000 img/sec** |

**Speedup Summary**:

| Comparison | Speedup |
|------------|---------|
| Track B vs Track A | 1.5-2.0x |
| Track C vs Track A | 3.0-4.5x |
| Track D vs Track A | 4.0-8.0x |
| Track D vs Track C | 1.5-2.5x |

### Impact of Simplification

**Benefits observed with single model size**:

| Metric | Before (3 sizes) | After (small only) | Improvement |
|--------|------------------|-------------------|-------------|
| GPU Memory | 3.7 GB | 1.4 GB | **-62%** |
| Model Load Time | 45-60s | 20-25s | **-50%** |
| Memory Fragmentation | High | Low | ✓ |
| Benchmark Consistency | Variable | Stable | ✓ |

**Takeaway**: Loading only the model you're testing gives **cleaner, more accurate, and faster results**.

---

## Advanced Topics

### Interpreting Metrics

#### Latency

**Lower is better** - Important for real-time applications

- **Mean/Median**: Typical performance
- **p95/p99**: Worst-case scenarios
- **Std Dev**: Consistency (lower = more predictable)

**Good**: Low median, low std dev, small p99-median gap
**Bad**: High median, high std dev, large p99-median gap

#### Throughput

**Higher is better** - Important for batch processing

- **Images/sec**: Processing capacity
- **Scaling**: Should increase with concurrency up to GPU saturation

**Good**: Throughput scales linearly with concurrency
**Bad**: Throughput plateaus early or decreases

#### Detection Accuracy

**Compare to Track A (ground truth)**

- **Precision**: % of predictions that are correct (should be >95%)
- **Recall**: % of ground truth found (should be >95%)
- **IoU**: Overlap quality (should be >0.85)

**Good**: Precision & recall >95%, IoU >0.85
**Bad**: Precision or recall <90%, IoU <0.75

### Custom Test Scenarios

#### Using Triton perf_analyzer

For direct Triton server-side performance (bypasses FastAPI overhead):

```bash
# Run analysis for Track C (direct Triton)
docker compose exec triton-api perf_analyzer \
  -m yolov11_small_trt_end2end \
  --percentile=95 \
  --concurrency-range 1:16:1 \
  --input-data random \
  --measurement-interval 10000

# Run analysis for Track D (ensemble)
docker compose exec triton-api perf_analyzer \
  -m yolov11_small_gpu_e2e_streaming \
  --percentile=95 \
  --concurrency-range 1:16:1 \
  --input-data random \
  --measurement-interval 10000
```

### Integration with Monitoring

**Prometheus + Grafana** integration is available for production monitoring:

1. Metrics exposed at `http://localhost:8002/metrics`
2. Grafana dashboard at `http://localhost:3000`
3. Custom dashboards in `triton-dashboard.json`

See [DEPLOYMENT_GUIDE.md](../DEPLOYMENT_GUIDE.md) for monitoring setup.

### Production Load Testing

**Burst Pattern**:
```bash
./triton_bench --mode variable --load-pattern burst --burst-interval 10
```

**Ramp Pattern**:
```bash
./triton_bench --mode variable --load-pattern ramp --ramp-step 32
```

**Sustained Load**:
```bash
./triton_bench --mode sustained
# Finds optimal client count and runs 5-minute stress test
```

---

## Troubleshooting

### No Speed Gains

**Verify workers**:
```bash
docker compose exec yolo-api ps aux | grep uvicorn | wc -l
# Should show 33 (1 master + 32 workers)
```

**Check batching**:
```bash
docker compose logs triton-api | grep "batch size"
# Should show batch sizes > 1
```

### Track C Not Faster Than Track B

**Check GPU NMS is compiled**:
```bash
curl http://localhost:9500/v2/models/yolov11_small_trt_end2end/config | jq '.output'
# Should show: num_dets, det_boxes, det_scores, det_classes
```

**Verify correct endpoint**:
```bash
# ✓ Correct - uses _end2end suffix
curl -X POST http://localhost:9600/predict/small_end2end -F "image=@test.jpg"

# ✗ Wrong - this is Track B, not C
curl -X POST http://localhost:9600/predict/small -F "image=@test.jpg"
```

### High Error Rate

**Check service logs**:
```bash
docker compose logs yolo-api | tail -100
docker compose logs yolo-api | tail -100
```

**Reduce concurrency**:
```bash
# If errors occur at concurrency 32, try 16
./triton_bench --mode full --clients 16
```

### Inconsistent Results

**Run longer tests**:
```bash
./triton_bench --mode full --duration 120
```

**Check for background processes**:
```bash
nvidia-smi  # Check GPU usage
htop        # Check CPU usage
```

**Ensure warmup completed**:
```bash
# Let services warm up
sleep 30
# Then run benchmark
./triton_bench --mode quick
```

### Services Not Responding

**Verify services**:
```bash
docker compose ps
curl http://localhost:9600/health
```

**Check GPU availability**:
```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

---

## Summary

**Current Setup**:
- ✅ Single model size (small) across all tracks
- ✅ Optimized for both development and benchmarking
- ✅ 62% reduction in GPU memory usage
- ✅ Faster deployment and testing
- ✅ Clean, reproducible benchmark results

**Quick Start**:
```bash
# Deploy and benchmark (all in one)
docker compose up -d
cd benchmarks
./triton_bench --mode full
```

**Production Recommendations**:

**Choose Track D if**:
- You need maximum performance (4-8x faster than baseline)
- Latency is critical (<5ms target)
- You have high concurrent load
- Full GPU pipeline is supported

**Choose Track C if**:
- You want strong performance (3-5x faster)
- GPU NMS is beneficial but GPU preprocessing isn't needed
- Balanced approach between performance and complexity

**Choose Track B if**:
- You want good performance (1.5-2.5x faster)
- Wider model compatibility needed
- Stability is more important than max speed

**Choose Track A if**:
- Simplicity is priority
- Development/debugging phase
- Low request volume
- No TensorRT conversion needed

---

## Related Documentation

- [QUICK_START.md](../QUICK_START.md) - Get started with all tracks
- [DEPLOYMENT_GUIDE.md](../DEPLOYMENT_GUIDE.md) - Production deployment
- [benchmarks/README.md](../../benchmarks/README.md) - triton_bench Go tool documentation
- [END2END_ANALYSIS.md](../END2END_ANALYSIS.md) - Why Track C is faster
- [TRITON_BEST_PRACTICES.md](../TRITON_BEST_PRACTICES.md) - Production optimization
- [ARCHITECTURE.md](../ARCHITECTURE.md) - System architecture details

---

**Ready to benchmark?** Start with quick mode to validate performance! 🚀
