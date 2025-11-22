# YOLO Model Export Scripts

This folder contains scripts for downloading and exporting YOLO models in multiple formats optimized for NVIDIA Triton Inference Server deployment.

## Overview

The export process transforms PyTorch YOLO models (`.pt` files) into optimized formats for GPU inference. This repository supports **four performance tracks** (A/B/C/D) with increasing optimization levels:

| Track | Format | Preprocessing | NMS | Performance |
|-------|--------|---------------|-----|-------------|
| **A** | ONNX | CPU | CPU | Baseline |
| **B** | TensorRT | CPU | CPU | 2x faster |
| **C** | TRT End2End | CPU | GPU | 4x faster |
| **D** | DALI + TRT End2End | GPU | GPU | 10-15x faster |

This folder handles **Tracks A, B, and C**. Track D (DALI preprocessing) is managed in the [dali](../dali/) folder.

## Architecture

### Export Pipeline

```
PyTorch Model (.pt)
    ├─→ [1] Standard ONNX → Track A
    ├─→ [2] TensorRT Engine (from ONNX) → Track B
    ├─→ [3] ONNX End2End (with NMS operators) → Track C (intermediate)
    └─→ [4] TRT End2End (compiled NMS) → Track C (final)
```

### Model Naming Convention

For each YOLO model size (nano/small/medium), exports create:

```
models/
├── yolov11_small/                    # Track A: Standard ONNX
│   ├── 1/model.onnx
│   └── config.pbtxt
├── yolov11_small_trt/                # Track B: TensorRT Engine
│   ├── 1/model.plan
│   └── config.pbtxt
├── yolov11_small_end2end/            # Track C: ONNX End2End (intermediate)
│   ├── 1/model.onnx
│   └── config.pbtxt
└── yolov11_small_trt_end2end/        # Track C: TRT End2End (fastest)
    ├── 1/model.plan
    └── config.pbtxt
```

## Files

### 1. export_models.py

**Purpose**: Unified export script that generates YOLO models in all four formats.

**What it does**:
- Loads PyTorch YOLO models (`.pt` files)
- Exports to 4 different formats optimized for different use cases
- Handles dynamic batching configuration
- Applies custom End2End patches for GPU NMS support
- Generates Triton-compatible model files and configs
- Validates exports and provides detailed summaries

**Four export formats**:

#### Format 1: Standard ONNX (Track A)
- **Output**: `models/{model_name}/1/model.onnx`
- **Output format**: `[84, 8400]` (raw detections, no NMS)
- **NMS**: CPU post-processing required
- **Use case**: Maximum compatibility, baseline performance
- **Speed**: Baseline inference + 5-10ms CPU NMS

```python
# Exports with dynamic batching enabled
model.export(
    format="onnx",
    dynamic=True,    # Variable batch/height/width
    simplify=True,   # Optimize for TensorRT
    half=True        # FP16 precision
)
```

#### Format 2: TensorRT Engine (Track B)
- **Output**: `models/{model_name}_trt/1/model.plan`
- **Output format**: `[84, 8400]` (raw detections, no NMS)
- **NMS**: CPU post-processing required
- **Use case**: Faster inference than ONNX
- **Speed**: 1.5x faster inference + 5-10ms CPU NMS
- **Build time**: 5-10 minutes (TensorRT optimization)

```python
# Built using TensorRT Python API
# - Dynamic batching: min=1, opt=max_batch//2, max=max_batch
# - FP16 precision
# - 4GB workspace
# - Optimized for A6000 GPU
```

#### Format 3: ONNX End2End (Track C - Intermediate)
- **Output**: `models/{model_name}_end2end/1/model.onnx`
- **Output format**: `num_dets, det_boxes, det_scores, det_classes`
- **NMS**: GPU (TensorRT EfficientNMS operators)
- **Use case**: Testing ONNX with GPU NMS (via TensorRT Execution Provider)
- **Speed**: 2-3x faster than Track A

```python
# Uses custom Ultralytics patch (export_onnx_trt)
# - Embeds TRT::EfficientNMS_TRT operators
# - topk_all=100: Max detections per image
# - iou_thres=0.45: NMS IoU threshold
# - conf_thres=0.25: Confidence threshold
```

#### Format 4: TRT End2End (Track C - Final)
- **Output**: `models/{model_name}_trt_end2end/1/model.plan`
- **Output format**: `num_dets, det_boxes, det_scores, det_classes`
- **NMS**: Compiled GPU NMS (fastest)
- **Use case**: Maximum performance for CPU preprocessing workflows
- **Speed**: 3-5x faster than Track A
- **Build time**: 5-10 minutes (compiles NMS into TensorRT engine)

**Usage**:

```bash
# Export all formats for all models (default)
docker compose exec yolo-api python /app/export/export_models.py

# Export only TRT End2End (Track C) for small model
docker compose exec yolo-api python /app/export/export_models.py \
    --models small \
    --formats trt_end2end

# Export standard formats only (Tracks A & B)
docker compose exec yolo-api python /app/export/export_models.py \
    --formats onnx trt

# Export end2end formats (Track C)
docker compose exec yolo-api python /app/export/export_models.py \
    --formats onnx_end2end trt_end2end
```

**Command-line arguments**:
- `--models`: Model sizes to export (`nano`, `small`, `medium`, or `all`)
- `--formats`: Export formats (`onnx`, `trt`, `onnx_end2end`, `trt_end2end`, or `all`)

**Model configurations**:

```python
MODELS = {
    "nano": {
        "max_batch": 128,  # Nano is small, can handle large batches
        "topk": 100        # Max detections per image
    },
    "small": {
        "max_batch": 64,
        "topk": 100
    },
    "medium": {
        "max_batch": 32,
        "topk": 100
    }
}
```

**Export settings**:
- Image size: 640x640
- Precision: FP16 (half precision)
- Device: GPU 0
- NMS thresholds: IoU=0.45, Confidence=0.25

---

### 2. download_pytorch_models.sh

**Purpose**: Downloads pre-trained PyTorch YOLO models from Ultralytics GitHub releases.

**What it does**:
- Creates `pytorch_models/` directory
- Downloads YOLO `.pt` files from official Ultralytics releases
- Skips already-downloaded models
- Verifies file sizes

**Current configuration**:
Downloads only the **small** model for focused benchmarking:
- `yolo11s.pt` (Small model, ~40MB)

**Usage**:

```bash
# Run from repository root
bash export/download_pytorch_models.sh

# Or from anywhere
cd /path/to/triton-api
./export/download_pytorch_models.sh
```

**Output**:
```
pytorch_models/
└── yolo11s.pt
```

**Extending to other models**:

To download additional models, edit the `MODELS` array in the script:

```bash
declare -A MODELS=(
    ["yolo11n.pt"]="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
    ["yolo11s.pt"]="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt"
    ["yolo11m.pt"]="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt"
)
```

---

### 3. cleanup_for_reexport.sh

**Purpose**: Prepares the environment for a clean re-export by removing old model files while preserving configurations.

**What it does**:
1. **Backs up configs**: Copies all `config.pbtxt` files to timestamped backup directory
2. **Removes old exports**: Deletes `.onnx` and `.plan` files from model directories
3. **Archives old scripts**: Moves deprecated export scripts to `scripts/archived/`
4. **Clears TRT cache**: Removes TensorRT engine cache files

**Usage**:

```bash
# Run from repository root
bash export/cleanup_for_reexport.sh
```

**What gets removed**:
- `models/yolov11_*/1/model.onnx` (and `.onnx.old` backups)
- `models/yolov11_*_trt/1/model.plan` (and `.plan.old` backups)
- `trt_cache/*` (TensorRT build cache)

**What gets preserved**:
- `models/yolov11_*/config.pbtxt` (backed up to `models/backup_configs_*/`)
- Directory structure (all `models/*/1/` directories remain)

**Use cases**:
- Re-exporting models after Ultralytics updates
- Testing different export settings
- Troubleshooting failed exports
- Cleaning up disk space before fresh export

---

### 4. export_small_only.sh

**Purpose**: Convenience script that exports only the `small` model in optimized formats (Tracks B & C).

**What it does**:
- Exports `yolo11s` (small) model only
- Exports both TRT and TRT End2End formats
- Skips ONNX formats (assumes you don't need them for production)
- Provides next-steps guidance for Track D setup

**Usage**:

```bash
# Run from repository root
bash export/export_small_only.sh
```

**Equivalent to**:
```bash
docker compose exec yolo-api python /app/export/export_models.py \
    --models small \
    --formats trt trt_end2end
```

**What gets exported**:
- `models/yolov11_small_trt/1/model.plan` (Track B)
- `models/yolov11_small_trt_end2end/1/model.plan` (Track C)

**Next steps after running** (printed by script):
1. Create DALI pipeline for Track D
2. Create ensemble models for Track D
3. Restart Triton to load new models

**Use case**: Simplified workflow for benchmarking focused on the small model only.

---

## Workflow

### Initial Setup (First Time)

1. **Download PyTorch models**:
   ```bash
   bash export/download_pytorch_models.sh
   ```

2. **Export all formats** (recommended for testing):
   ```bash
   docker compose exec yolo-api python /app/export/export_models.py
   ```

3. **Restart Triton** to load exported models:
   ```bash
   docker compose restart triton-api
   ```

4. **Verify models loaded**:
   ```bash
   docker compose exec triton-api curl localhost:8000/v2/models
   ```

### Production Workflow (Small Model Only)

For focused benchmarking with just the small model:

1. **Download model**:
   ```bash
   bash export/download_pytorch_models.sh
   ```

2. **Export optimized formats**:
   ```bash
   bash export/export_small_only.sh
   ```

3. **Create Track D components** (see [dali/README.md](../dali/README.md)):
   ```bash
   docker compose exec yolo-api python /app/dali/create_dali_letterbox_pipeline.py
   docker compose exec yolo-api python /app/dali/create_ensembles.py --models small
   ```

4. **Restart Triton**:
   ```bash
   docker compose restart triton-api
   ```

### Re-exporting Models

If you need to re-export (e.g., after updating Ultralytics or changing settings):

1. **Clean old exports**:
   ```bash
   bash export/cleanup_for_reexport.sh
   ```

2. **Re-export**:
   ```bash
   docker compose exec yolo-api python /app/export/export_models.py
   ```

3. **Restart Triton**:
   ```bash
   docker compose restart triton-api
   ```

## Key Implementation Details

### End2End Export (GPU NMS)

The End2End export uses a **custom Ultralytics patch** that adds the `export_onnx_trt()` method:

```python
# Located in: src/ultralytics_patches/
from ultralytics_patches import apply_end2end_patch
apply_end2end_patch()

# This enables:
exporter.export_onnx_trt(prefix="ONNX TRT:")
```

**What the patch does**:
1. Wraps the YOLO model with `End2End_TRT` class
2. Adds `TRT_EfficientNMS` custom operators to ONNX graph
3. Configures NMS parameters (topk, iou_thres, conf_thres)
4. Exports ONNX with `TRT::EfficientNMS_TRT` operators

**Attribution**: ~600 lines of code from [levipereira/ultralytics](https://github.com/levipereira/ultralytics) fork. See [ATTRIBUTION.md](../ATTRIBUTION.md) for details.

### TensorRT Engine Building

Both standard and End2End TRT engines are built using **TensorRT Python API**:

```python
import tensorrt as trt

# Initialize plugins (required for EfficientNMS)
trt.init_libnvinfer_plugins(logger, '')

# Create builder and config
builder = trt.Builder(logger)
config = builder.create_builder_config()

# Set workspace (4GB)
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)

# Enable FP16
config.set_flag(trt.BuilderFlag.FP16)

# Dynamic batching profile
profile = builder.create_optimization_profile()
profile.set_shape("images",
    min=(1, 3, 640, 640),
    opt=(max_batch // 2, 3, 640, 640),
    max=(max_batch, 3, 640, 640)
)
config.add_optimization_profile(profile)

# Build engine (5-10 minutes)
engine = builder.build_serialized_network(network, config)
```

**Build time**: 5-10 minutes per model (one-time cost)

**Output**: Binary `.plan` file optimized for specific GPU architecture

### Dynamic Batching

All exports support **dynamic batching** with these characteristics:

| Model | Min Batch | Optimal Batch | Max Batch |
|-------|-----------|---------------|-----------|
| Nano | 1 | 64 | 128 |
| Small | 1 | 32 | 64 |
| Medium | 1 | 16 | 32 |

**Why different max batch sizes?**
- Nano: Smallest model, fits more in GPU memory
- Small: Balanced size
- Medium: Larger model, requires more memory per image

### NMS Configuration

For End2End exports (Tracks C & D):

```python
# NMS thresholds
IOU_THRESHOLD = 0.45    # IoU threshold for NMS
CONF_THRESHOLD = 0.25   # Confidence threshold

# Detection limits
TOPK = 100             # Max detections per image
```

**Output format**:
- `num_dets`: INT32 [1] - Number of detections (0-100)
- `det_boxes`: FP32 [100, 4] - Bounding boxes (x, y, w, h)
- `det_scores`: FP32 [100] - Confidence scores
- `det_classes`: INT32 [100] - Class IDs (0-79 for COCO)

## Performance Characteristics

### Export Comparison

| Format | Export Time | File Size | Inference (batch=1) | Inference (batch=64) |
|--------|-------------|-----------|---------------------|----------------------|
| ONNX Standard | ~30s | ~40 MB | ~5ms + CPU NMS | ~80ms + CPU NMS |
| TRT Standard | ~8min | ~100 MB | ~3ms + CPU NMS | ~50ms + CPU NMS |
| ONNX End2End | ~1min | ~45 MB | ~4ms (GPU NMS) | ~70ms (GPU NMS) |
| TRT End2End | ~10min | ~120 MB | ~2ms (GPU NMS) | ~35ms (GPU NMS) |

**Notes**:
- Export times are one-time costs
- TRT engines are GPU-architecture specific (must re-export for different GPUs)
- CPU NMS adds 5-10ms per image regardless of batch size

### Track Comparison (Small Model)

| Track | Format | Total Latency (batch=1) | Throughput (batch=64) |
|-------|--------|-------------------------|----------------------|
| **A** | ONNX | ~10-15ms | ~100 img/s |
| **B** | TRT | ~8-13ms | ~150 img/s |
| **C** | TRT End2End | ~5-7ms | ~350 img/s |
| **D** | DALI + TRT End2End | ~3-5ms | ~800 img/s |

**Bottlenecks**:
- Track A/B: CPU NMS (5-10ms per image)
- Track C: CPU preprocessing (3-5ms per image)
- Track D: Minimal CPU overhead (<1ms)

## File Locations

### Source Files (Inside Containers)

```
/app/
├── export/
│   ├── export_models.py              # Main export script
│   ├── download_pytorch_models.sh    # Download .pt files
│   ├── cleanup_for_reexport.sh       # Cleanup utility
│   └── export_small_only.sh          # Small-model convenience script
├── pytorch_models/
│   └── yolo11s.pt                    # PyTorch source model
└── models/
    ├── yolov11_small/                # Track A: ONNX
    ├── yolov11_small_trt/            # Track B: TRT
    ├── yolov11_small_end2end/        # Track C: ONNX End2End
    └── yolov11_small_trt_end2end/    # Track C: TRT End2End
```

### Host Paths (Mounted Volumes)

```
./export/                  → /app/export/
./pytorch_models/          → /app/pytorch_models/
./models/                  → /app/models/
```

## Troubleshooting

### "Model file not found" error

```bash
# Download PyTorch models first
bash export/download_pytorch_models.sh
```

### "Failed to build TensorRT engine"

**Possible causes**:
1. Out of GPU memory (reduce max_batch or use smaller model)
2. CUDA/TensorRT version mismatch (check container versions)
3. Corrupted ONNX file (re-export ONNX first)

**Solutions**:
```bash
# Check GPU memory
docker compose exec yolo-api nvidia-smi

# Reduce batch size in export_models.py (edit MODELS dict)
# Re-export ONNX first
docker compose exec yolo-api python /app/export/export_models.py --formats onnx
```

### "No NMS operators found" warning

This is expected for standard ONNX/TRT exports (Tracks A & B).

For End2End exports, verify the patch is applied:
```python
from ultralytics_patches import apply_end2end_patch
apply_end2end_patch()
```

### TensorRT build takes too long (>15 minutes)

**Normal behavior**: First build on new GPU architecture takes longer

**Speed up**:
1. Use smaller workspace: `workspace_gb = 2` (reduces memory but may lower performance)
2. Disable FP16: `HALF = False` (faster build, slower inference)
3. Reduce max_batch size

### "Triton can't load model" after export

**Common issues**:
1. Model file not in correct location: `models/{name}/1/model.{onnx,plan}`
2. Missing or incorrect `config.pbtxt`
3. Triton not restarted after export

**Solutions**:
```bash
# Verify file structure
ls -lh models/yolov11_small/1/
ls -lh models/yolov11_small/config.pbtxt

# Restart Triton
docker compose restart triton-api

# Check Triton logs
docker compose logs triton-api | grep ERROR
```

## Best Practices

1. **Start with small model**: Test export pipeline with small model first
2. **Export incrementally**: Export one format at a time for easier debugging
3. **Verify each export**: Check file sizes and ONNX operators before building TRT
4. **Backup configs**: Cleanup script backs up configs, but manual backups recommended
5. **Monitor GPU memory**: Watch `nvidia-smi` during TRT builds
6. **Version control exports**: Use git tags for successful export configurations

## References

- [Ultralytics YOLO Export Docs](https://docs.ultralytics.com/modes/export/)
- [TensorRT Python API](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/)
- [Triton Model Configuration](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html)
- [TensorRT EfficientNMS Plugin](https://github.com/NVIDIA/TensorRT/tree/main/plugin/efficientNMSPlugin)
- [Custom End2End Code Attribution](../ATTRIBUTION.md)

## Next Steps

After exporting models:

1. **Configure Triton**: Edit `config.pbtxt` files for dynamic batching, instance counts
2. **Test inference**: Use `tests/` scripts to validate model outputs
3. **Create Track D**: Follow [dali/README.md](../dali/README.md) for DALI preprocessing
4. **Run benchmarks**: Use `benchmarks/` scripts to compare all tracks
5. **Deploy production**: Update `docker-compose.yml` to load only needed models

## Future Improvements

Potential enhancements (not yet implemented):

1. **Automated config generation**: Generate `config.pbtxt` based on model metadata
2. **Multi-GPU export**: Parallel export on multiple GPUs
3. **INT8 quantization**: PTQ/QAT for 2x faster inference
4. **Model versioning**: Automatic version management for A/B testing
5. **Export validation**: Automated output comparison between formats
