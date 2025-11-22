# DALI Preprocessing Pipeline

This folder contains scripts for creating and validating GPU-accelerated YOLO preprocessing pipelines using NVIDIA DALI (Data Loading Library).

## Overview

**NVIDIA DALI** is a GPU-accelerated library for data loading and preprocessing. It enables:
- GPU-accelerated JPEG decoding (nvJPEG)
- GPU-based image transformations
- Optimized data pipelines for inference
- Integration with NVIDIA Triton Inference Server

This implementation provides **Track D** performance optimization: Full GPU pipeline with DALI preprocessing + TensorRT inference + GPU NMS.

## Architecture

### Current Implementation (Affine Transformation)

```
Client → [Triton Ensemble] → Detections
         ├─ DALI Preprocessing (GPU)
         │  ├─ nvJPEG decode (GPU)
         │  ├─ warp_affine with CPU-calculated matrix (GPU)
         │  └─ Normalize + HWC→CHW (GPU)
         └─ YOLO TRT End2End (GPU)
            ├─ TensorRT inference (GPU)
            └─ EfficientNMS (GPU)
```

**CPU Overhead**: Client calculates affine transformation matrix (~0.5-1ms)
**GPU Pipeline**: Decode → Transform → Normalize → Inference → NMS

## Files

### 1. create_dali_letterbox_pipeline.py

**Purpose**: Creates and serializes the DALI preprocessing pipeline for Triton deployment.

**What it does**:
- Defines a DALI pipeline with letterbox preprocessing using affine transformations
- Builds and serializes the pipeline to `models/yolo_preprocess_dali/1/model.dali`
- Generates a Triton `config.pbtxt` for the DALI backend
- Tests the pipeline with dummy data to verify correctness

**Pipeline operations**:
1. Decode JPEG using nvJPEG (GPU-accelerated)
2. Apply affine transformation via `warp_affine` (GPU)
3. Normalize to [0, 1] and convert HWC → CHW (GPU)

**Usage**:
```bash
# Run from yolo-api container (has DALI installed)
docker compose exec yolo-api python /app/dali/create_dali_letterbox_pipeline.py
```

**Inputs**:
- `encoded_images`: Raw JPEG/PNG bytes (variable length)
- `affine_matrices`: Pre-calculated affine transformation matrices [N, 2, 3] FP32

**Output**:
- `preprocessed_images`: [N, 3, 640, 640] FP32 tensor, normalized [0, 1]

**Key configuration**:
- Batch size: 128 (matches Triton ensemble configs)
- Device: GPU 0
- Threads: 4 (for CPU orchestration)
- Instance count: 1 (NVIDIA best practice for DALI)

---

### 2. create_ensembles.py

**Purpose**: Generates three-tier ensemble model configurations that chain DALI preprocessing with YOLO TRT End2End models.

**What it does**:
- Creates ensemble `config.pbtxt` files for different workload patterns
- Supports multiple YOLO model sizes (nano, small, medium)
- Optimizes dynamic batching parameters per tier
- Sets up proper input/output mappings between DALI and TRT models

**Three ensemble tiers**:

| Tier | Use Case | Max Batch | Batching Window | Preserve Order | Instance Count |
|------|----------|-----------|-----------------|----------------|----------------|
| **Streaming** | Real-time video | 8 | 0.1ms | ✓ | 3 |
| **Balanced** | General purpose | 64 | 0.5ms | ✗ | 2 |
| **Batch** | Offline processing | 128 | 5ms | ✗ | 1 |

**Ensemble naming convention**:
- `yolov11_small_gpu_e2e_streaming` → Real-time streaming (0.1ms batching)
- `yolov11_small_gpu_e2e` → General purpose (0.5ms batching)
- `yolov11_small_gpu_e2e_batch` → Offline batch (5ms batching)

**Usage**:
```bash
# Create ensembles for specific model sizes
python dali/create_ensembles.py --models small
python dali/create_ensembles.py --models nano small medium

# Create ensembles for all sizes
python dali/create_ensembles.py --models all
```

**Generated output**:
```
models/
├── yolov11_small_gpu_e2e/
│   ├── 1/  (empty for ensemble)
│   └── config.pbtxt
├── yolov11_small_gpu_e2e_streaming/
└── yolov11_small_gpu_e2e_batch/
```

---

### 3. validate_dali_letterbox.py

**Purpose**: Validates the DALI pipeline and ensemble models through comprehensive testing.

**What it does**:
- Tests DALI pipeline in standalone mode (deserialized from `model.dali`)
- Tests DALI model serving through Triton gRPC API
- Tests full ensemble pipeline (DALI + TRT End2End)
- Validates output shapes, dtypes, and value ranges
- Confirms affine transformation calculations

**Three validation tests**:

1. **Test 1: DALI Standalone Pipeline**
   - Deserializes `model.dali` and runs it directly
   - Feeds test image JPEG bytes + affine matrix
   - Validates output format: (3, 640, 640) FP32 [0, 1]

2. **Test 2: DALI Model via Triton**
   - Calls `yolo_preprocess_dali` model through Triton gRPC
   - Tests Triton serving of DALI backend
   - Validates preprocessing output

3. **Test 3: Full Ensemble (Track D)**
   - Calls `yolov11_small_gpu_e2e` ensemble
   - Tests complete pipeline: DALI → TRT → NMS
   - Validates detection outputs (boxes, scores, classes)

**Usage**:
```bash
# Run from yolo-api container
docker compose exec yolo-api python /app/dali/validate_dali_letterbox.py
```

**Prerequisites**:
- DALI pipeline must be created first
- Triton server must be running
- Test image at `/app/test_images/bus.jpg`

## Workflow

### Initial Setup

1. **Create DALI pipeline**:
   ```bash
   docker compose exec yolo-api python /app/dali/create_dali_letterbox_pipeline.py
   ```
   This generates:
   - `models/yolo_preprocess_dali/1/model.dali`
   - `models/yolo_preprocess_dali/config.pbtxt`

2. **Create ensemble models**:
   ```bash
   docker compose exec yolo-api python /app/dali/create_ensembles.py --models small
   ```
   This generates ensemble configs like:
   - `models/yolov11_small_gpu_e2e/config.pbtxt`
   - `models/yolov11_small_gpu_e2e_streaming/config.pbtxt`
   - `models/yolov11_small_gpu_e2e_batch/config.pbtxt`

3. **Restart Triton to load new models**:
   ```bash
   docker compose restart triton-api
   ```

4. **Validate the pipeline**:
   ```bash
   docker compose exec yolo-api python /app/dali/validate_dali_letterbox.py
   ```

### Making Changes

If you modify the DALI pipeline:
1. Re-run `create_dali_letterbox_pipeline.py`
2. Restart Triton
3. Re-validate with `validate_dali_letterbox.py`

If you modify ensemble configurations:
1. Re-run `create_ensembles.py`
2. Restart Triton
3. Test with inference scripts or benchmarks

## Key Implementation Details

### Affine Transformation Matrix

The current implementation requires CPU calculation of letterbox affine matrices:

```python
def calculate_letterbox_affine(orig_w, orig_h, target_size=640):
    scale = min(target_size / orig_h, target_size / orig_w)
    if scale > 1.0:
        scale = 1.0

    new_h = int(round(orig_h * scale))
    new_w = int(round(orig_w * scale))

    pad_x = (target_size - new_w) / 2.0
    pad_y = (target_size - new_h) / 2.0

    # Affine matrix format: [[scale_x, 0, offset_x], [0, scale_y, offset_y]]
    affine_matrix = np.array([
        [scale, 0.0, pad_x],
        [0.0, scale, pad_y]
    ], dtype=np.float32)

    return affine_matrix
```

**CPU overhead**: ~0.5-1.0ms per image for matrix calculation
**Trade-off**: Simpler implementation vs. pure GPU pipeline

### DALI Best Practices

From NVIDIA documentation and testing:

1. **Instance count = 1**: Multiple DALI instances cause unnaturally high memory usage
2. **device="mixed"**: Use for image decoder (CPU decode → GPU output via nvJPEG)
3. **hw_decoder_load=0.65**: Optimal for Ampere+ hardware decoder offload
4. **Batch size**: Match to Triton ensemble max_batch_size (typically 128)

### Ensemble Configuration

Ensembles combine models without copying data:
- Input: Raw JPEG bytes
- Step 1: DALI preprocessing → `preprocessed_images`
- Step 2: TRT End2End uses `preprocessed_images` → detections
- Output: Detections (boxes, scores, classes)

All data stays on GPU between steps (zero-copy).

## Performance Characteristics

### Track D (DALI + TRT End2End)

**Advantages**:
- GPU-accelerated JPEG decoding (nvJPEG)
- GPU-based image transformations
- No CPU→GPU data transfer for preprocessing
- Optimized batch processing

**CPU overhead**:
- Affine matrix calculation: ~0.5-1.0ms per image
- Negligible for batch processing (calculated once per image)

**GPU pipeline** (fully pipelined):
- nvJPEG decode: ~0.5-1ms
- warp_affine: ~0.2ms
- Normalize + transpose: ~0.1ms
- TensorRT inference: ~2-5ms (model dependent)
- EfficientNMS: ~0.3ms

**Total latency** (batch=1): ~3-7ms (vs. 5-10ms for CPU preprocessing)
**Throughput** (batch=128): ~10-20x faster than CPU preprocessing

## Testing

### Quick validation:
```bash
docker compose exec yolo-api python /app/dali/validate_dali_letterbox.py
```

### Benchmark comparisons:
```bash
# Compare all tracks (Track A/B/C/D)
python benchmarks/four_track_comparison.py --image test_images/bus.jpg

# Compare ensemble tiers
python benchmarks/compare_ensemble_tiers.py
```

## Troubleshooting

### "DALI not installed" error
Run scripts from the **yolo-api** container (not triton-api):
```bash
docker compose exec yolo-api python /app/dali/<script>.py
```

### "Model not ready" error
Ensure Triton has loaded the models:
```bash
docker compose exec triton-api curl localhost:8000/v2/models/yolo_preprocess_dali
```

### "Test image not found" error
Make sure test images exist:
```bash
docker compose exec yolo-api ls /app/test_images/
```

## References

- [NVIDIA DALI Documentation](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/)
- [Triton DALI Backend](https://github.com/triton-inference-server/dali_backend)
- [YOLO Letterbox Preprocessing](https://docs.ultralytics.com/modes/predict/#image-and-video-formats)
- [Track D Implementation Details](../ATTRIBUTION.md)

## Future Improvements

Potential optimizations (not yet implemented):

1. **Pure GPU letterbox**: Calculate transformation matrix on GPU using DALI custom operators
2. **DALI resize+pad**: Use native DALI ops instead of affine transformation
3. **Batched matrix calculation**: Calculate affine matrices in parallel on GPU
4. **Adaptive batching**: Dynamic batch size based on image size distribution

These would eliminate the ~0.5-1ms CPU overhead per image.
