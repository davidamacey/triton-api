# Streaming & Real-Time Optimization for Track D

## Workload Requirements

**Target Use Cases**:
1. **Static Images**: Batch inference on photos (low latency priority)
2. **Video Frames**: Sequential frames from video files (moderate latency, high throughput)
3. **Streaming Video**: Real-time camera feeds, live streams (CRITICAL low latency)

**Key Constraint**: Streaming video is latency-sensitive. Cannot sacrifice real-time performance for batching efficiency.

---

## Batching Strategy for Multi-Modal Workloads

### The Challenge

Different workloads have conflicting requirements:

| Workload | Priority | Ideal Batch Size | Max Tolerable Delay |
|----------|----------|------------------|---------------------|
| **Streaming video** | Latency | 1-4 | 1-2ms (60 FPS = 16.7ms/frame) |
| **Video frames** | Throughput | 8-32 | 5-10ms (offline processing) |
| **Batch images** | Throughput | 32-128 | 10-50ms (not time-critical) |

**Triton's dynamic batching** cannot differentiate between workload types automatically.

### Solution: Multi-Tier Batching Configuration

Use **separate Triton instances** or **multiple ensemble variants** for different latency requirements:

---

## Configuration Tier 1: Streaming-Optimized (CRITICAL)

**Model Name**: `yolov11_nano_gpu_e2e_streaming`

**Purpose**: Real-time video streams (30-60 FPS), webcams, live RTSP feeds.

**Batching Config**:
```protobuf
# models/yolov11_nano_gpu_e2e_streaming/config.pbtxt

name: "yolov11_nano_gpu_e2e_streaming"
platform: "ensemble"
max_batch_size: 8  # Small batches for low latency

# ... input/output definitions same as standard ensemble ...

ensemble_scheduling {
  # Same DALI → TRT pipeline
  # ...
}

# CRITICAL: Aggressive low-latency batching
dynamic_batching {
  preferred_batch_size: [ 1, 2, 4 ]  # Small batches only
  max_queue_delay_microseconds: 100   # 0.1ms max delay
  preserve_ordering: true             # Maintain frame order for video

  default_queue_policy {
    timeout_action: REJECT
    default_timeout_microseconds: 5000  # 5ms timeout (strict)
    allow_timeout_override: false
    max_queue_size: 16                  # Small queue (avoid backlog)
  }

  # Priority queue for streaming
  priority_levels: 2
  default_priority_level: 1
  priority_queue_policy: {
    key: 0
    value: {
      timeout_action: DELAY
      default_timeout_microseconds: 2000  # High-priority: 2ms max
      max_queue_size: 4
    }
  }
}

# Multiple instances for parallel streams
instance_group [
  {
    count: 3    # Handle 3 parallel streams
    kind: KIND_GPU
    gpus: [ 0 ]
  }
]
```

**Key Settings**:
- ✅ `max_queue_delay_microseconds: 100` → 0.1ms batching window (near-instant)
- ✅ `preferred_batch_size: [1, 2, 4]` → Tiny batches (prioritize latency)
- ✅ `preserve_ordering: true` → Essential for video frame sequences
- ✅ `max_queue_size: 16` → Prevent backlog accumulation
- ✅ `instance_group.count: 3` → Handle multiple streams concurrently

**Expected Performance**:
- Latency: 6-8ms end-to-end (decode + infer + NMS)
- Throughput: ~120-150 FPS per stream (limited by batch=1 most of the time)
- **REAL-TIME**: ✅ Can handle 60 FPS with headroom

---

## Configuration Tier 2: Balanced (DEFAULT)

**Model Name**: `yolov11_nano_gpu_e2e`

**Purpose**: General-purpose inference, video file processing, moderate concurrency.

**Batching Config**:
```protobuf
# models/yolov11_nano_gpu_e2e/config.pbtxt

dynamic_batching {
  preferred_batch_size: [ 4, 8, 16, 32 ]  # Balanced batch sizes
  max_queue_delay_microseconds: 500       # 0.5ms batching window

  default_queue_policy {
    timeout_action: REJECT
    default_timeout_microseconds: 50000   # 50ms timeout
    max_queue_size: 64
  }
}

instance_group [
  {
    count: 2
    kind: KIND_GPU
    gpus: [ 0 ]
  }
]
```

**Expected Performance**:
- Latency: 8-12ms (allows more batching)
- Throughput: 300-500 FPS (batch-8 to batch-16 typical)

---

## Configuration Tier 3: Throughput-Optimized

**Model Name**: `yolov11_nano_gpu_e2e_batch`

**Purpose**: Offline batch processing, large video files, maximum throughput.

**Batching Config**:
```protobuf
# models/yolov11_nano_gpu_e2e_batch/config.pbtxt

max_batch_size: 128  # Large batches

dynamic_batching {
  preferred_batch_size: [ 32, 64, 128 ]   # Large batches
  max_queue_delay_microseconds: 5000      # 5ms batching window

  default_queue_policy {
    timeout_action: DELAY
    default_timeout_microseconds: 100000  # 100ms timeout (very lenient)
    max_queue_size: 256
  }
}

instance_group [
  {
    count: 1    # Single instance with large batches
    kind: KIND_GPU
    gpus: [ 0 ]
  }
]
```

**Expected Performance**:
- Latency: 15-25ms (aggressive batching)
- Throughput: 800-1200 FPS (batch-64+ typical)

---

## Client-Side Model Selection

**FastAPI Gateway** routes requests to appropriate ensemble:

```python
# src/main.py

MODEL_NAMES_GPU_E2E = {
    # Streaming-optimized (default for single images/frames)
    "nano": "yolov11_nano_gpu_e2e_streaming",
    "small": "yolov11_small_gpu_e2e_streaming",
    "medium": "yolov11_medium_gpu_e2e_streaming",
}

MODEL_NAMES_GPU_E2E_BATCH = {
    # Batch-optimized (explicit batch endpoint)
    "nano": "yolov11_nano_gpu_e2e_batch",
    "small": "yolov11_small_gpu_e2e_batch",
    "medium": "yolov11_medium_gpu_e2e_batch",
}

@app.post("/predict/{model_name}")
async def predict(model_name: str, image: UploadFile):
    """Single image/frame - use streaming-optimized ensemble"""
    # Route to streaming config
    triton_model = MODEL_NAMES_GPU_E2E[model_name]
    # ...

@app.post("/predict_batch/{model_name}")
async def predict_batch(model_name: str, images: list[UploadFile]):
    """Batch inference - use throughput-optimized ensemble"""
    # Route to batch config
    triton_model = MODEL_NAMES_GPU_E2E_BATCH[model_name]
    # ...

@app.post("/predict_stream/{model_name}")
async def predict_stream(model_name: str, stream_url: str):
    """
    Video stream processing (future).

    Opens video stream, extracts frames, sends to streaming ensemble
    with strict latency requirements.
    """
    # Route to streaming config with priority queue
    # ...
```

---

## Video Streaming Architecture (Future)

For real-time video streams (RTSP, WebRTC, etc.):

```python
# Future: src/stream_processor.py

import cv2
import asyncio
from src.utils import TritonEnd2EndClient

class VideoStreamProcessor:
    """
    Real-time video stream processor with frame-level latency control.
    """
    def __init__(self, stream_url: str, model_name: str):
        self.stream = cv2.VideoCapture(stream_url)
        self.client = TritonEnd2EndClient(
            triton_url=TRITON_URL,
            model_name=f"{model_name}_streaming"  # Use streaming config
        )
        self.frame_queue = asyncio.Queue(maxsize=2)  # Tiny queue (drop frames if backed up)

    async def process_stream(self):
        """
        Read frames → JPEG encode → Triton → yield detections.

        Maintains <16ms latency for 60 FPS streams.
        """
        while True:
            ret, frame = self.stream.read()
            if not ret:
                break

            # JPEG encode (on CPU, but fast ~1-2ms)
            _, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            jpeg_bytes = encoded.tobytes()

            # Send to Triton (non-blocking)
            detections = await asyncio.to_thread(
                self.client.infer_raw_bytes,
                jpeg_bytes
            )

            # Yield detections (websocket, etc.)
            yield detections

            # FPS throttle (if needed)
            await asyncio.sleep(1/60)  # 60 FPS max
```

**Key Design Points**:
- ✅ Tiny frame queue (maxsize=2) → drop frames if GPU can't keep up (better than backlog)
- ✅ Non-blocking inference → continue reading frames while GPU processes
- ✅ JPEG encode on CPU acceptable (1-2ms, minimal overhead)
- ✅ Triton does JPEG decode on GPU (nvJPEG, 0.5-1ms)

---

## Benchmarking Streaming Latency

**Critical Metrics for Streaming**:

1. **P50/P95/P99 Latency**: Must be <16ms for 60 FPS (ideally <10ms)
2. **Latency Variance**: Low jitter (std dev <2ms)
3. **Frame Drop Rate**: <1% at target FPS

**Benchmark Script** (`benchmarks/test_streaming_latency.py`):

```python
import time
import numpy as np
from src.utils import TritonEnd2EndClient

def benchmark_streaming_latency(model_name: str, num_frames=1000):
    """
    Simulate streaming workload: sequential single-frame inference.
    """
    client = TritonEnd2EndClient(
        triton_url="triton-api:8001",
        model_name=f"{model_name}_streaming"
    )

    # Generate test frame (640x640 JPEG)
    test_frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    _, encoded = cv2.imencode('.jpg', test_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    jpeg_bytes = encoded.tobytes()

    latencies = []

    for i in range(num_frames):
        start = time.perf_counter()
        detections = client.infer_raw_bytes(jpeg_bytes)
        latency = (time.perf_counter() - start) * 1000  # ms
        latencies.append(latency)

        # Simulate 60 FPS (16.7ms per frame)
        time.sleep(max(0, 0.0167 - (latency / 1000)))

    # Compute stats
    latencies = np.array(latencies)
    print(f"\n{'='*60}")
    print(f"Streaming Latency Benchmark: {model_name}")
    print(f"{'='*60}")
    print(f"Frames processed: {num_frames}")
    print(f"P50 latency: {np.percentile(latencies, 50):.2f}ms")
    print(f"P95 latency: {np.percentile(latencies, 95):.2f}ms")
    print(f"P99 latency: {np.percentile(latencies, 99):.2f}ms")
    print(f"Std dev: {np.std(latencies):.2f}ms")
    print(f"Max latency: {np.max(latencies):.2f}ms")

    # Check if suitable for 60 FPS
    p99 = np.percentile(latencies, 99)
    if p99 < 10:
        print(f"✓ EXCELLENT for 60 FPS streaming (P99 < 10ms)")
    elif p99 < 16:
        print(f"✓ Suitable for 60 FPS streaming (P99 < 16ms)")
    else:
        print(f"✗ NOT suitable for 60 FPS (P99 > 16ms)")

    return latencies

# Run benchmark
latencies = benchmark_streaming_latency("yolov11_nano_gpu_e2e", num_frames=1000)
```

**Target Results for Track D (Streaming Config)**:
```
========================================================
Streaming Latency Benchmark: yolov11_nano_gpu_e2e
========================================================
Frames processed: 1000
P50 latency: 6.2ms
P95 latency: 7.8ms
P99 latency: 9.1ms
Std dev: 0.8ms
Max latency: 11.3ms
✓ EXCELLENT for 60 FPS streaming (P99 < 10ms)
```

---

## GPU Memory Considerations

**Streaming workload** = many small batches → less GPU memory per batch but more overhead.

**Memory Usage** (estimated for RTX A6000):

| Configuration | Batch Size | GPU Memory (per instance) | Instances | Total GPU Memory |
|---------------|------------|---------------------------|-----------|------------------|
| Streaming | 1-4 | 500-800 MB | 3 | ~2.4 GB |
| Balanced | 8-16 | 1-1.5 GB | 2 | ~3 GB |
| Batch | 64-128 | 3-4 GB | 1 | ~4 GB |

**Total for all configs**: ~10 GB (fits on A6000's 48 GB easily).

**Recommendation**: Run all three tiers simultaneously on same GPU (different model names).

---

## Frame Ordering for Video

**CRITICAL for video streams**: Preserve temporal ordering.

**Triton Config**:
```protobuf
dynamic_batching {
  preserve_ordering: true  # ← ESSENTIAL
}
```

**Why this matters**:
- Video frames must be processed in sequence
- Dynamic batching can reorder requests for efficiency
- `preserve_ordering: true` disables reordering (slight throughput cost, necessary for video)

**Trade-off**:
- ✅ Temporal consistency (frame N always processed before frame N+1)
- ❌ Slightly lower throughput (~5-10% penalty)

For **static images**: Set `preserve_ordering: false` (no ordering requirement).

---

## Comparison: Track C vs Track D for Streaming

**Track C (CPU Preprocessing)**:
```
Per-frame latency:
- JPEG decode (CPU): 1-2ms
- Resize/letterbox (CPU): 1-2ms
- Normalize (CPU): 0.5-1ms
- gRPC transfer: 0.2-0.5ms
- TRT + NMS (GPU): 3-5ms
Total: 6-10.5ms (barely meets 60 FPS)

Problem: Python GIL → serialized preprocessing → jitter
```

**Track D (GPU Preprocessing)**:
```
Per-frame latency:
- gRPC transfer (JPEG bytes): 0.1-0.2ms
- DALI decode (GPU): 0.5-1ms
- DALI letterbox (GPU): 0.3-0.5ms
- TRT + NMS (GPU): 3-5ms
Total: 3.9-6.7ms (comfortable headroom for 60 FPS)

Benefit: Pipelined GPU ops → consistent latency → low jitter
```

**Expected improvement**: 35-40% latency reduction + lower variance.

---

## Recommendations

**For your use case** (images + video frames + streaming):

1. **Deploy all three configs**:
   - `*_streaming` for real-time streams (0.1ms batching window)
   - `*_gpu_e2e` for general use (0.5ms batching window)
   - `*_batch` for offline processing (5ms batching window)

2. **Default to streaming config** for `/predict` endpoint
   - Optimizes for lowest latency (most common use case)
   - Still gets batching benefits when concurrent requests arrive

3. **Add explicit `/predict_batch` endpoint** for offline workloads
   - Routes to throughput-optimized config
   - Achieves maximum FPS for video file processing

4. **Future-proof for streaming** with `preserve_ordering: true`
   - Minimal overhead now
   - Essential when you add video stream support

---

## Summary: Streaming Optimization Checklist

- [x] Use `max_queue_delay_microseconds: 100` for streaming config (0.1ms)
- [x] Set `preferred_batch_size: [1, 2, 4]` (small batches)
- [x] Enable `preserve_ordering: true` for video
- [x] Use multiple instances (count=3) for concurrent streams
- [x] Keep queue small (`max_queue_size: 16`) to prevent backlog
- [x] Target P99 latency <10ms for 60 FPS support
- [x] Benchmark with sequential single-frame workload (not concurrent batches)

---

**Next Step**: Implement streaming config in ensemble creation (Phase 2).
