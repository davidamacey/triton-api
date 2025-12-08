# DALI Preprocessing Pipelines

GPU-accelerated preprocessing pipelines using NVIDIA DALI for Triton Inference Server.

## Overview

This folder contains scripts for creating, validating, and managing DALI preprocessing pipelines that provide significant performance improvements over CPU-based preprocessing:

| Track | Pipeline | Speedup | Use Case |
|-------|----------|---------|----------|
| **D** | YOLO letterbox only | 10-15x | High-throughput object detection |
| **E Full** | YOLO + CLIP + HD cropping | 8-12x | Visual search with per-object embeddings |
| **E Simple** | YOLO + CLIP only | 10-15x | Fast visual search (global embeddings only) |

## Architecture

### Track D: YOLO Preprocessing

```
Client                    Triton Ensemble                    Output
  │                            │                               │
  ├─ JPEG bytes ──────────────>│                               │
  ├─ affine_matrices ─────────>│                               │
  │                            │                               │
  │                      DALI Pipeline (GPU)                   │
  │                      ├─ nvJPEG decode                      │
  │                      ├─ warp_affine (letterbox)            │
  │                      └─ normalize + CHW                    │
  │                            │                               │
  │                      YOLO TRT End2End (GPU)                │
  │                      ├─ TensorRT inference                 │
  │                      └─ EfficientNMS                       │
  │                            │                               │
  │<────────── detections ─────┘                               │
```

### Track E: Dual/Triple-Branch Preprocessing

```
encoded_jpeg
     │
     ▼
nvJPEG Decode (GPU)
     │
┌────┴────────────┬────────────┐
▼                 ▼            ▼
YOLO Branch    CLIP Branch   HD Branch (full only)
640x640        256x256       max 1920px
letterbox      center crop   preserve aspect
     │              │             │
     ▼              ▼             ▼
[yolo_images]  [clip_images]  [original_images]
```

## Files

### Configuration

| File | Description |
|------|-------------|
| `config.py` | Shared constants and configuration dataclasses |
| `utils.py` | Utility functions (affine calculation, test helpers) |
| `__init__.py` | Package exports |

### Pipeline Creation Scripts

| File | Track | Output Model | Description |
|------|-------|--------------|-------------|
| `create_dali_letterbox_pipeline.py` | D | `yolo_preprocess_dali` | YOLO letterbox preprocessing |
| `create_dual_dali_pipeline.py` | E Full | `dual_preprocess_dali` | Triple-branch (YOLO + CLIP + HD) |
| `create_yolo_clip_dali_pipeline.py` | E Simple | `yolo_clip_preprocess_dali` | Dual-branch (YOLO + CLIP) |
| `create_ensembles.py` | D | `yolov11_*_gpu_e2e_*` | Ensemble configs for 3 tiers |

### Validation Scripts

| File | Track | Description |
|------|-------|-------------|
| `validate_dali_letterbox.py` | D | Tests DALI pipeline standalone and via Triton |
| `validate_dual_dali_preprocessing.py` | E | Compares DALI vs PyTorch reference |

## Quick Start

### Track D Setup

```bash
# Create DALI pipeline and ensembles
make create-dali

# Validate
make validate-dali
```

### Track E Setup

```bash
# Create triple-branch pipeline (with HD cropping for per-object embeddings)
make create-dali-dual

# Or create simple dual-branch pipeline (faster, global embeddings only)
make create-dali-simple

# Validate
make validate-dali-dual
```

## Makefile Targets

| Target | Description |
|--------|-------------|
| `make create-dali` | Create Track D pipeline + ensembles + restart Triton |
| `make create-ensembles` | Create Track D ensemble configs only |
| `make create-dali-dual` | Create Track E triple-branch pipeline |
| `make create-dali-simple` | Create Track E dual-branch pipeline (faster) |
| `make rebuild-dali` | Rebuild Track E pipeline (alias for create-dali-dual) |
| `make validate-dali` | Validate Track D pipeline |
| `make validate-dali-dual` | Validate Track E pipeline vs PyTorch |

## Configuration

All configuration is centralized in `config.py`:

```python
# Image sizes
YOLO_SIZE = 640          # YOLO input size
CLIP_SIZE = 256          # MobileCLIP input size
YOLO_PAD_VALUE = 114     # Gray padding color

# Pipeline settings
MAX_BATCH_SIZE = 128     # Maximum batch size
NUM_THREADS = 4          # CPU threads for orchestration
HW_DECODER_LOAD = 0.65   # Hardware decoder offload (Ampere+)

# Performance targets
CORRECTNESS_THRESHOLD = 0.01   # Max acceptable mean diff
PERFORMANCE_TARGET_MS = 5.0    # Target preprocessing latency
```

## Ensemble Tiers (Track D)

Three tiers optimize for different workloads:

| Tier | Suffix | Batching | Instances | Use Case |
|------|--------|----------|-----------|----------|
| **Streaming** | `_gpu_e2e_streaming` | 0.1ms | 3 | Real-time video |
| **Balanced** | `_gpu_e2e` | 0.5ms | 2 | General purpose |
| **Batch** | `_gpu_e2e_batch` | 5ms | 1 | Offline processing |

## Output Formats

### Track D (`yolo_preprocess_dali`)

| Output | Shape | Type | Range |
|--------|-------|------|-------|
| `preprocessed_images` | `[N, 3, 640, 640]` | FP32 | [0, 1] |

### Track E Triple-Branch (`dual_preprocess_dali`)

| Output | Shape | Type | Description |
|--------|-------|------|-------------|
| `yolo_images` | `[N, 3, 640, 640]` | FP32 | Letterboxed for detection |
| `clip_images` | `[N, 3, 256, 256]` | FP32 | Center-cropped for embedding |
| `original_images` | `[N, 3, H, W]` | FP32 | HD (max 1920px) for cropping |

### Track E Dual-Branch (`yolo_clip_preprocess_dali`)

| Output | Shape | Type | Description |
|--------|-------|------|-------------|
| `yolo_images` | `[N, 3, 640, 640]` | FP32 | Letterboxed for detection |
| `clip_images` | `[N, 3, 256, 256]` | FP32 | Center-cropped for embedding |

## Affine Matrix Calculation

All pipelines require CPU-calculated affine transformation matrices:

```python
from dali.utils import calculate_letterbox_affine

# Get affine matrix for letterbox transformation
affine_matrix, scale, pad_x, pad_y = calculate_letterbox_affine(
    orig_w=1920,
    orig_h=1080,
    target_size=640,
)
# affine_matrix shape: [2, 3]
# Format: [[scale_x, 0, offset_x], [0, scale_y, offset_y]]
```

**CPU overhead**: ~0.5-1.0ms per image (negligible for batch processing).

## Performance Characteristics

### Latency (single image)

| Component | Latency |
|-----------|---------|
| Affine calculation (CPU) | ~0.5-1.0ms |
| nvJPEG decode (GPU) | ~0.5-1.0ms |
| warp_affine (GPU) | ~0.2ms |
| Normalize + transpose (GPU) | ~0.1ms |
| **Total preprocessing** | **~1.5-2.5ms** |

### Throughput (batch=128)

| Track | Images/sec | vs CPU |
|-------|------------|--------|
| D (YOLO only) | 3000-5000 | 10-15x |
| E Full (triple-branch) | 2000-3500 | 8-12x |
| E Simple (dual-branch) | 3000-4500 | 10-14x |

## Troubleshooting

### "DALI not installed" error

Run scripts from the **yolo-api** container:

```bash
docker compose exec yolo-api python /app/dali/<script>.py
```

### "Model not ready" error

Check Triton has loaded the models:

```bash
docker compose exec triton-api curl localhost:8000/v2/models/yolo_preprocess_dali
```

### "Test image not found" error

Ensure test images exist:

```bash
docker compose exec yolo-api ls /app/test_images/
```

### Validation failure

If DALI outputs don't match PyTorch reference:

1. Check image format (RGB vs BGR)
2. Verify affine matrix calculation
3. Check normalization (should be /255 for both)
4. Run with verbose: `python script.py -v`

## NVIDIA Best Practices

1. **Instance count = 1**: Multiple DALI instances cause high memory usage
2. **device="mixed"**: Use for image decoder (CPU decode -> GPU output via nvJPEG)
3. **hw_decoder_load=0.65**: Optimal for Ampere+ hardware decoder offload
4. **Batch size**: Match to Triton ensemble max_batch_size (typically 128)

## References

- [NVIDIA DALI Documentation](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/)
- [Triton DALI Backend](https://github.com/triton-inference-server/dali_backend)
- [YOLO Letterbox Preprocessing](https://docs.ultralytics.com/modes/predict/)
