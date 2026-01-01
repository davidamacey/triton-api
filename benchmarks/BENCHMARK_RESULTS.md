# Triton API Benchmark Results

**Date:** 2025-12-31
**Test Configuration:**
- Images: 100 from `/mnt/nvm/KILLBOY_SAMPLE_PICTURES`
- Duration: 30 seconds per concurrency level
- Mode: Matrix (multiple concurrency levels)
- Isolated: Each track tested with ONLY its required models loaded

## Hardware
- GPU 0: NVIDIA (49GB VRAM)
- GPU 2: NVIDIA (49GB VRAM)
- FastAPI Workers: 4 (for Track A PyTorch)

## Results Summary

### Track A - PyTorch Baseline
**Models loaded:** None (PyTorch in FastAPI workers)
**Workers:** 4 uvicorn workers
**GPU:** GPU 0

| Clients | Throughput | P50 (ms) | P95 (ms) | P99 (ms) | Success |
|---------|------------|----------|----------|----------|---------|
| 4 | 16.7 RPS | 231 | 261 | 313 | 100% |
| 8 | 26.5 RPS | 242 | 287 | 3,194 | 100% |
| 16 | **45.6 RPS** | 261 | 317 | 3,568 | 100% |
| 32 | 43.6 RPS | 362 | 554 | 13,603 | 100% |

**Peak:** 45.6 RPS @ 16 clients

---

### Track B - TensorRT + CPU NMS
**Models loaded:** yolov11_small_trt only (4 instances)
**GPU:** GPU 0

| Clients | Throughput | P50 (ms) | P95 (ms) | P99 (ms) | Success |
|---------|------------|----------|----------|----------|---------|
| 16 | 36.4 RPS | 428 | 536 | 638 | 100% |
| 32 | **57.8 RPS** | 538 | 720 | 850 | 100% |
| 64 | 51.5 RPS | 1,192 | 1,564 | 1,727 | 100% |
| 128 | 52.3 RPS | 2,536 | 3,014 | 3,180 | 100% |

**Peak:** 57.8 RPS @ 32 clients
**Speedup vs A:** 1.27x

---

### Track C - TensorRT + GPU NMS (End2End)
**Models loaded:** yolov11_small_trt_end2end only (4 instances)
**GPU:** GPU 0

| Clients | Throughput | P50 (ms) | P95 (ms) | P99 (ms) | Success |
|---------|------------|----------|----------|----------|---------|
| 16 | 32.5 RPS | 481 | 597 | 665 | 100% |
| 32 | 40.7 RPS | 775 | 1,005 | 1,196 | 100% |
| 64 | 46.8 RPS | 1,317 | 1,728 | 1,962 | 100% |
| 128 | **50.7 RPS** | 2,405 | 3,305 | 3,815 | 100% |

**Peak:** 50.7 RPS @ 128 clients
**Speedup vs A:** 1.11x

---

### Track D_batch - DALI + TensorRT (Full GPU Pipeline)
**Models loaded:** yolo_preprocess_dali_batch (6 instances) + yolov11_small_trt_end2end_batch (4 instances) + yolov11_small_gpu_e2e_batch
**GPU:** GPU 0

| Clients | Throughput | P50 (ms) | P95 (ms) | P99 (ms) | Success |
|---------|------------|----------|----------|----------|---------|
| 16 | 32.2 RPS | 463 | 645 | 1,193 | 100% |
| 32 | 90.7 RPS | 332 | 511 | 707 | 100% |
| 64 | 93.7 RPS | 661 | 1,001 | 1,374 | 100% |
| 128 | **98.0 RPS** | 1,247 | 1,744 | 2,364 | 100% |

**Peak:** 98.0 RPS @ 128 clients
**Speedup vs A:** 2.15x

---

### Track E - Visual Search (YOLO + MobileCLIP)
**Models loaded:** yolo_clip_ensemble + yolo_clip_preprocess_dali + yolov11_small_trt_end2end + mobileclip2_s2_image_encoder
**GPU:** GPU 0

| Clients | Throughput | P50 (ms) | P95 (ms) | P99 (ms) | Success |
|---------|------------|----------|----------|----------|---------|
| 16 | 52.4 RPS | 286 | 429 | 661 | 99.9% |
| 32 | **77.4 RPS** | 368 | 705 | 1,010 | 100% |
| 64 | 51.0 RPS | 1,203 | 1,895 | 2,126 | 99.1% |
| 128 | 73.3 RPS | 1,664 | 2,586 | 3,210 | 100% |

**Peak:** 77.4 RPS @ 32 clients
**Speedup vs A:** 1.70x
**Note:** Track E performs YOLO detection + MobileCLIP embedding extraction (more compute than detection-only tracks)

---

## Key Observations

1. **Track A (PyTorch):** Baseline at 45.6 RPS with 4 workers. Optimal at 16 clients.

2. **Track B (TRT + CPU NMS):** 1.27x speedup. Optimal at 32 clients. CPU NMS parallelism helps at mid-concurrency.

3. **Track C (TRT + GPU NMS):** 1.11x speedup. Scales to high concurrency (peak at 128 clients). GPU NMS reduces CPU overhead.

4. **Track D_batch (DALI + TRT):** **2.15x speedup** - Best performer! Full GPU pipeline eliminates CPU preprocessing bottleneck. DALI batching (6 instances) combined with TRT (4 instances) achieves 98 RPS.

5. **Track E (Visual Search):** 1.70x speedup despite extra CLIP embedding computation. Peak at 32 clients.

6. **100% Success Rate:** All tracks achieved ~100% success at all concurrency levels when properly isolated.

## Test Commands

```bash
# Unload all models
bash unload_all.sh

# Load specific model for isolated test
curl -X POST "http://localhost:4600/v2/repository/models/MODEL_NAME/load"

# Run matrix benchmark
./triton_bench --mode matrix --track TRACK --matrix-clients 16,32,64,128 --duration 30
```
