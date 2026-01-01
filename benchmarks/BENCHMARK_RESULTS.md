# Triton API Benchmark Results

**Date:** 2025-12-31
**Test Configuration:**
- Images: 1138 from `/mnt/nvm/KILLBOY_SAMPLE_PICTURES`
- Duration: 60 seconds per concurrency level
- Mode: Matrix stress test (high concurrency)
- Isolated: Each track tested with ONLY its required models loaded
- Queue delay: 100ms (all models)
- Queue size: 2048 (stress testing capacity)

## Hardware
- GPU 0: NVIDIA (49GB VRAM)
- FastAPI Workers: 4 (for Track A PyTorch)
- TRT Instances: 4 per model
- DALI Instances: 6-8 per model

## Results Summary

| Track | Peak RPS | @ Clients | Speedup | Avg Batch | Description |
|-------|----------|-----------|---------|-----------|-------------|
| **A** | 56.9 | 32 | 1.00x | N/A | PyTorch baseline |
| **B** | 65.6 | 64 | 1.15x | 12.18 | TensorRT + CPU NMS |
| **C** | 52.3 | 128 | 0.92x | 6.69 | TensorRT + GPU NMS |
| **D_batch** | 103.8 | 256 | 1.82x | 1.65 | DALI + TRT End2End |
| **E** | **123.3** | 64 | **2.17x** | 11.61 | YOLO + MobileCLIP |

---

### Track A - PyTorch Baseline
**Models loaded:** None (PyTorch in FastAPI workers)
**Workers:** 4 uvicorn workers
**GPU:** GPU 0

| Clients | Throughput | P50 (ms) | P95 (ms) | P99 (ms) | Success |
|---------|------------|----------|----------|----------|---------|
| 16 | 26.2 RPS | 298 | 587 | 7,038 | 100% |
| 32 | **56.9 RPS** | 451 | 555 | 3,692 | 100% |
| 64 | 36.5 RPS | 847 | 7,417 | 22,781 | 99.7% |
| 128 | 1.5 RPS | 20,275 | 29,745 | 29,891 | 38.6% |

**Peak:** 56.9 RPS @ 32 clients
**Note:** Falls apart at 64+ clients (only 4 workers)

---

### Track B - TensorRT + CPU NMS
**Models loaded:** yolov11_small_trt (4 instances)
**GPU:** GPU 0
**Avg Batch Size:** 12.18

| Clients | Throughput | P50 (ms) | P95 (ms) | P99 (ms) | Success |
|---------|------------|----------|----------|----------|---------|
| 64 | **65.6 RPS** | 952 | 1,249 | 1,443 | 100% |
| 128 | 56.2 RPS | 2,225 | 2,653 | 2,828 | 100% |
| 256 | 56.0 RPS | 4,351 | 6,010 | 6,503 | 100% |
| 512 | 57.9 RPS | 8,335 | 10,278 | 11,387 | 100% |

**Peak:** 65.6 RPS @ 64 clients
**Speedup vs A:** 1.15x

---

### Track C - TensorRT + GPU NMS (End2End)
**Models loaded:** yolov11_small_trt_end2end (4 instances)
**GPU:** GPU 0
**Avg Batch Size:** 6.69

| Clients | Throughput | P50 (ms) | P95 (ms) | P99 (ms) | Success |
|---------|------------|----------|----------|----------|---------|
| 64 | 48.9 RPS | 1,278 | 1,687 | 2,016 | 100% |
| 128 | **52.3 RPS** | 2,429 | 3,200 | 3,535 | 100% |
| 256 | 50.3 RPS | 4,852 | 5,973 | 6,553 | 100% |
| 512 | 52.3 RPS | 9,265 | 10,675 | 11,303 | 100% |

**Peak:** 52.3 RPS @ 128 clients
**Speedup vs A:** 0.92x (slower due to GPU NMS overhead in this config)

---

### Track D_batch - DALI + TensorRT (Full GPU Pipeline)
**Models loaded:** yolo_preprocess_dali_batch (6 instances) + yolov11_small_trt_end2end_batch (4 instances)
**GPU:** GPU 0
**Avg Batch Size:** DALI 1.65, Ensemble 1.0 (not batching)

| Clients | Throughput | P50 (ms) | P95 (ms) | P99 (ms) | Success |
|---------|------------|----------|----------|----------|---------|
| 64 | 66.6 RPS | 878 | 1,563 | 1,812 | 99.9% |
| 128 | 56.0 RPS | 2,178 | 3,441 | 3,786 | 99.5% |
| 256 | **103.8 RPS** | 2,307 | 3,530 | 4,079 | 100% |
| 512 | 95.9 RPS | 4,961 | 7,078 | 8,386 | 100% |

**Peak:** 103.8 RPS @ 256 clients
**Speedup vs A:** 1.82x
**Note:** Ensemble scheduler processes requests individually (batch=1), child models batch internally

---

### Track E - Visual Search (YOLO + MobileCLIP)
**Models loaded:** yolo_clip_ensemble + yolo_clip_preprocess_dali (8 instances) + yolov11_small_trt_end2end (4 instances) + mobileclip2_s2_image_encoder (4 instances)
**GPU:** GPU 0
**Avg Batch Size:** YOLO 11.61, MobileCLIP 11.61

| Clients | Throughput | P50 (ms) | P95 (ms) | P99 (ms) | Success |
|---------|------------|----------|----------|----------|---------|
| 64 | **123.3 RPS** | 489 | 744 | 964 | 100% |
| 128 | 117.9 RPS | 1,038 | 1,448 | 2,091 | 100% |
| 256 | 101.8 RPS | 2,354 | 3,349 | 4,489 | 99.5% |
| 512 | 62.8 RPS | 6,704 | 11,898 | 16,603 | 93.5% |

**Peak:** 123.3 RPS @ 64 clients
**Speedup vs A:** 2.17x
**Note:** Best performer! YOLO + CLIP extraction with excellent batching (11.61 avg)

---

## Key Observations

1. **Track E (YOLO+CLIP) is fastest:** 123.3 RPS with 2.17x speedup. Excellent batching (11.61 avg) on child models makes this the best performer.

2. **Track D_batch (DALI+TRT):** 103.8 RPS with 1.82x speedup. Full GPU pipeline works well at high concurrency (256+ clients).

3. **Track B (TRT + CPU NMS):** 65.6 RPS with 1.15x speedup. Good batching (12.18 avg) but CPU NMS limits scalability.

4. **Track C (TRT + GPU NMS):** 52.3 RPS, actually slower than baseline. GPU NMS adds overhead in this configuration.

5. **Track A (PyTorch):** Baseline at 56.9 RPS. Collapses at 64+ clients due to 4 worker limit.

6. **Batching Architecture:**
   - Ensemble scheduler processes requests individually (batch=1)
   - Child models (YOLO TRT, MobileCLIP) batch internally via dynamic batching
   - Best results when child models have high batch utilization

## Configuration Changes Made

```bash
# All models updated with:
max_queue_delay_microseconds: 100000  # 100ms
max_queue_size: 2048
preferred_batch_size: [16, 32, 64]
timeout: 30 seconds
```

## Test Commands

```bash
# Unload all models
bash unload_all.sh

# Load specific track models
curl -X POST "http://localhost:4600/v2/repository/models/MODEL_NAME/load"

# Run stress test with all images
./triton_bench --mode matrix --track TRACK --limit 1138 --matrix-clients 64,128,256,512 --duration 60
```
