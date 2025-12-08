# Model Export Scripts

This folder contains scripts for downloading and exporting models in multiple formats optimized for NVIDIA Triton Inference Server deployment.

## Overview

The export process transforms PyTorch models into optimized formats for GPU inference. This repository supports **five performance tracks**:

### YOLO Detection Tracks (A/B/C/D)

| Track | Format | Preprocessing | NMS | Performance |
|-------|--------|---------------|-----|-------------|
| **A** | PyTorch (.pt) | CPU | CPU | Baseline |
| **B** | TensorRT | CPU | CPU | 2x faster |
| **C** | TRT End2End | CPU | GPU | 4x faster |
| **D** | DALI + TRT End2End | GPU | GPU | 10-15x faster |

This folder handles **Tracks B and C** exports. Track A uses PyTorch models directly (no export needed). Track D (DALI preprocessing) is managed in the [dali](../dali/) folder.

### Track E: Visual Search with MobileCLIP

| Component | Format | Input | Output | Use Case |
|-----------|--------|-------|--------|----------|
| **Image Encoder** | TensorRT | 256x256 images | 512-dim embeddings | Image-to-image search |
| **Text Encoder** | TensorRT | 77 tokens | 512-dim embeddings | Text-to-image search |

Track E exports are managed in this folder using MobileCLIP2-S2 (recommended) or MobileCLIP2-B (higher accuracy).

## Architecture

### YOLO Export Pipeline

```
PyTorch Model (.pt)
    ├─→ [Track A] Used directly (no export needed)
    ├─→ [Track B] TensorRT Engine (from ONNX)
    ├─→ [Track C] ONNX End2End (with NMS operators) → TRT End2End (compiled NMS)
    └─→ [Track D] Uses Track C TRT End2End + DALI preprocessing
```

### MobileCLIP Export Pipeline (Track E)

```
MobileCLIP2 Model (.pt)
    ├─→ Image Encoder: PyTorch → ONNX → TensorRT (256x256 → 512-dim)
    └─→ Text Encoder:  PyTorch → ONNX → TensorRT (77 tokens → 512-dim)
```

### Model Naming Convention

For each YOLO model size (nano/small/medium), exports create:

```
pytorch_models/
└── yolo11s.pt                        # Track A: PyTorch model (used directly)

models/
├── yolov11_small_trt/                # Track B: TensorRT Engine
│   ├── 1/model.plan
│   └── config.pbtxt
├── yolov11_small_end2end/            # Track C: ONNX End2End (intermediate)
│   ├── 1/model.onnx
│   └── config.pbtxt
└── yolov11_small_trt_end2end/        # Track C/D: TRT End2End (fastest)
    ├── 1/model.plan
    └── config.pbtxt
```

## Files

### 1. export_models.py

**Purpose**: Unified export script that generates YOLO models in all four formats with normalized bounding boxes.

**What it does**:
- Loads PyTorch YOLO models (`.pt` files)
- Exports to 4 different formats optimized for different use cases
- Handles dynamic batching configuration
- Applies custom End2End patches for GPU NMS support
- Generates Triton-compatible model files and configs
- Validates exports and provides detailed summaries

**Export formats**:

#### Track A: PyTorch (No Export Needed)
- **Output**: Uses `pytorch_models/yolo11{n,s,m,l,x}.pt` directly
- **Output format**: Native Ultralytics results
- **NMS**: CPU post-processing (built into Ultralytics)
- **Use case**: Development, debugging, baseline reference
- **Speed**: Baseline inference

Track A loads PyTorch models at startup via Ultralytics SDK - no export required.

#### Track B: TensorRT Engine
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

#### Track C: ONNX End2End (Intermediate)
- **Output**: `models/{model_name}_end2end/1/model.onnx`
- **Output format**: `num_dets, det_boxes, det_scores, det_classes`
- **NMS**: GPU (TensorRT EfficientNMS operators)
- **Use case**: Testing ONNX with GPU NMS (via TensorRT Execution Provider)
- **Speed**: 2-3x faster than Track A

```python
# Uses custom Ultralytics patch (export_onnx_trt)
# - Embeds TRT::EfficientNMS_TRT operators
# - topk_all=300: Max detections per image
# - iou_thres=0.7: NMS IoU threshold
# - conf_thres=0.25: Confidence threshold
```

#### Track C/D: TRT End2End (Final)
- **Output**: `models/{model_name}_trt_end2end/1/model.plan`
- **Output format**: `num_dets, det_boxes, det_scores, det_classes`
- **NMS**: Compiled GPU NMS (fastest)
- **Use case**: Maximum performance for CPU preprocessing workflows
- **Speed**: 3-5x faster than Track A
- **Build time**: 5-10 minutes (compiles NMS into TensorRT engine)

**Usage via Makefile** (recommended):

```bash
# Export small model (TRT + End2End with normalized boxes)
make export-models

# Export all formats for all models
make export-all

# Export only End2End formats (Track C/D)
make export-end2end

# Export ONNX only
make export-onnx

# Check export status
make export-status

# Clean old exports before re-exporting
make clean-exports
```

**Direct usage** (advanced):

```bash
# Export all formats for all models
docker compose exec yolo-api python /app/export/export_models.py --normalize-boxes

# Export only TRT End2End (Track C) for small model
docker compose exec yolo-api python /app/export/export_models.py \
    --models small \
    --formats trt_end2end \
    --normalize-boxes
```

**Command-line arguments**:
- `--models`: Built-in model sizes (`nano`, `small`, `medium`, `large`, `xlarge`)
- `--custom-model`: Custom trained model path (see [Custom Model Export](#custom-model-export) below)
- `--config-file`: YAML configuration file for multiple models
- `--formats`: Export formats (`onnx`, `trt`, `onnx_end2end`, `trt_end2end`, or `all`)
- `--normalize-boxes`: Output normalized coordinates [0-1] instead of pixel coordinates (required for Triton)
- `--save-labels`: Auto-extract and save class names to `labels.txt`
- `--generate-config`: Auto-generate Triton `config.pbtxt` files
- `--list-models`: List available built-in models and exit

**Model configurations**:

```python
MODELS = {
    "nano": {
        "max_batch": 128,  # Nano is small, can handle large batches
        "topk": 300        # Max detections per image
    },
    "small": {
        "max_batch": 64,
        "topk": 300
    },
    "medium": {
        "max_batch": 32,
        "topk": 300
    },
    "large": {
        "max_batch": 16,
        "topk": 300
    },
    "xlarge": {
        "max_batch": 8,
        "topk": 300
    }
}
```

**Export settings**:
- Image size: 640x640
- Precision: FP16 (half precision)
- Device: GPU 0
- NMS thresholds: IoU=0.7, Confidence=0.25

---

### Custom Model Export

Export your own custom-trained YOLO11 models for deployment on Triton.

#### Quick Start (Makefile)

```bash
# Export a custom model (simplest)
make export-custom MODEL=/app/pytorch_models/my_detector.pt

# Export with custom name and batch size
make export-custom MODEL=/app/pytorch_models/my_detector.pt NAME=vehicle_detector BATCH=64

# List available built-in models
make export-list
```

#### Direct CLI Usage

```bash
# Basic: export custom model with auto-generated name
docker compose exec yolo-api python /app/export/export_models.py \
    --custom-model /app/pytorch_models/my_model.pt \
    --formats trt trt_end2end \
    --normalize-boxes \
    --save-labels \
    --generate-config

# Advanced: custom name and batch size (format: path:name:batch)
docker compose exec yolo-api python /app/export/export_models.py \
    --custom-model /app/pytorch_models/my_model.pt:vehicle_detector:64 \
    --formats trt trt_end2end \
    --normalize-boxes

# Multiple custom models at once
docker compose exec yolo-api python /app/export/export_models.py \
    --custom-model /app/pytorch_models/model1.pt:detector1 \
    --custom-model /app/pytorch_models/model2.pt:detector2:32 \
    --formats trt_end2end
```

#### YAML Configuration File

For complex deployments with multiple models, use a YAML configuration file:

```yaml
# models_config.yaml
models:
  vehicle_detector:
    pt_file: /app/pytorch_models/vehicle_yolo11s.pt
    triton_name: vehicle_detector_v1  # optional, auto-generated if omitted
    max_batch: 64                      # optional, default 32
    topk: 300                          # optional, default 300
    # num_classes: 5                   # optional, auto-detected from model
    # class_names:                     # optional, auto-detected from model
    #   - car
    #   - truck
    #   - bus
    #   - motorcycle
    #   - bicycle

  person_detector:
    pt_file: /app/pytorch_models/person_yolo11m.pt
    triton_name: person_detector_v1
    max_batch: 32

  defect_detector:
    pt_file: /app/pytorch_models/defect_yolo11n.pt
    # Uses auto-generated name: defect_yolo11n
    max_batch: 128  # Small model, can handle larger batches
```

Export using config file:

```bash
# Via Makefile
make export-config CONFIG=/app/export/models_config.yaml

# Direct CLI
docker compose exec yolo-api python /app/export/export_models.py \
    --config-file /app/export/models_config.yaml \
    --formats trt trt_end2end \
    --normalize-boxes \
    --save-labels \
    --generate-config
```

#### Auto-Generated Files

When exporting custom models with `--save-labels` and `--generate-config`, the script automatically creates:

```
models/
└── vehicle_detector_trt_end2end/
    ├── 1/
    │   └── model.plan          # TensorRT engine
    ├── config.pbtxt            # Auto-generated Triton config
    └── labels.txt              # Class names extracted from model
```

**labels.txt example** (auto-extracted from model):
```
car
truck
bus
motorcycle
bicycle
```

**config.pbtxt example** (auto-generated):
```
name: "vehicle_detector_trt_end2end"
platform: "tensorrt_plan"
max_batch_size: 64

input [
  {
    name: "images"
    data_type: TYPE_FP16
    dims: [ 3, 640, 640 ]
  }
]

output [
  {
    name: "num_dets"
    data_type: TYPE_INT32
    dims: [ 1 ]
  },
  ...
]

dynamic_batching {
  preferred_batch_size: [ 8, 16, 32, 64 ]
  max_queue_delay_microseconds: 5000
}
```

#### Custom Model Requirements

Your custom YOLO11 model must:
1. Be a valid Ultralytics YOLO11 `.pt` file (trained with `ultralytics` library)
2. Use 640x640 input size (standard YOLO11)
3. Have class names embedded (automatic if trained with Ultralytics)

**Supported model architectures**:
- YOLO11n, YOLO11s, YOLO11m, YOLO11l, YOLO11x (detection)
- Custom-trained variants based on above

**Not supported** (yet):
- YOLO11 segmentation models
- YOLO11 pose estimation models
- YOLO11 OBB (oriented bounding box) models
- Non-Ultralytics YOLO variants

#### Workflow for Custom Models

1. **Copy your model** to the container:
   ```bash
   # Option A: Mount volume in docker-compose.yml
   # volumes:
   #   - ./my_models:/app/custom_models

   # Option B: Copy directly
   docker cp my_detector.pt $(docker compose ps -q yolo-api):/app/pytorch_models/
   ```

2. **Export the model**:
   ```bash
   make export-custom MODEL=/app/pytorch_models/my_detector.pt NAME=my_detector BATCH=32
   ```

3. **Restart Triton** to load the new model:
   ```bash
   make restart-triton
   ```

4. **Test the model**:
   ```bash
   curl -X POST http://localhost:4603/predict/my_detector_end2end \
       -F "image=@test_images/test.jpg" | jq '.'
   ```

---

### 2. download_pytorch_models.py

**Purpose**: Downloads pre-trained PyTorch YOLO models using Ultralytics' built-in download functionality.

**What it does**:
- Uses Ultralytics' `attempt_download_asset()` for reliable downloads
- Automatic retry on failure with curl fallback
- Disk space validation before download
- Skips already-downloaded models
- Supports downloading specific models or all models

**Available models** (YOLO11 lineup):
- `nano` → `yolo11n.pt` (~6 MB) - Fastest, lowest accuracy
- `small` → `yolo11s.pt` (~18 MB) - Good balance
- `medium` → `yolo11m.pt` (~39 MB) - Better accuracy
- `large` → `yolo11l.pt` (~49 MB) - High accuracy
- `xlarge` → `yolo11x.pt` (~109 MB) - Highest accuracy, slowest

**Usage via Makefile** (recommended):

```bash
# Download small model (default)
make download-pytorch

# Download specific models
make download-pytorch MODELS="small medium large"

# Download all models (nano through xlarge)
make download-pytorch-all

# List available models
make download-pytorch-list
```

**Direct usage**:

```bash
# Download small model (default)
docker compose exec yolo-api python /app/export/download_pytorch_models.py

# Download specific models
docker compose exec yolo-api python /app/export/download_pytorch_models.py --models small medium

# Download all models
docker compose exec yolo-api python /app/export/download_pytorch_models.py --models all

# List available models
docker compose exec yolo-api python /app/export/download_pytorch_models.py --list
```

**Output**:
```
pytorch_models/
└── yolo11s.pt
```

---

### 3. export_mobileclip_image_encoder.py

**Purpose**: Export MobileCLIP2 image encoder for Track E visual search deployment.

**What it does**:
- Loads MobileCLIP2 model with proper configuration (image_mean=0, image_std=1)
- Reparameterizes model before export (merges train-time branches)
- Exports to ONNX with dynamic batch support
- Converts to TensorRT engine for maximum throughput
- Validates output matches PyTorch
- Benchmarks inference performance

**Supported models**:
- **MobileCLIP2-S2** (35.7M params) - Recommended, balanced speed/accuracy
- **MobileCLIP2-B** (86.3M params) - Maximum accuracy

**Model specifications**:
- Input: `images [B, 3, 256, 256]` FP32, normalized [0, 1]
- Output: `image_embeddings [B, 512]` FP32, L2-normalized
- Opset version: 17 (optimized for TensorRT 10.x)
- Dynamic batching: 1 to 128 (configurable)

**Usage via Makefile** (recommended):

```bash
# Export both image and text encoders (MobileCLIP2-S2)
make export-mobileclip
```

**Direct usage**:

```bash
# Export MobileCLIP2-S2 image encoder (default, recommended)
docker compose exec yolo-api python /app/export/export_mobileclip_image_encoder.py

# Export MobileCLIP2-B image encoder (higher accuracy)
docker compose exec yolo-api python /app/export/export_mobileclip_image_encoder.py --model B

# Export ONNX only (skip TensorRT conversion)
docker compose exec yolo-api python /app/export/export_mobileclip_image_encoder.py --skip-tensorrt

# Use FP32 precision instead of FP16
docker compose exec yolo-api python /app/export/export_mobileclip_image_encoder.py --fp32

# Customize max batch size
docker compose exec yolo-api python /app/export/export_mobileclip_image_encoder.py --max-batch-size 64
```

**Output files**:
```
pytorch_models/
└── mobileclip2_s2_image_encoder.onnx  # Intermediate ONNX model

models/
└── mobileclip2_s2_image_encoder/
    ├── 1/
    │   └── model.plan                 # TensorRT engine for Triton
    └── config.pbtxt                   # Triton configuration (manual)
```

**Key implementation notes**:
- **CRITICAL**: Must call `reparameterize_model()` before export or inference will be incorrect
- Uses simple ÷255 normalization (same as YOLO), compatible with DALI pipeline
- L2 normalization applied to outputs for cosine similarity search
- FP16 precision enabled by default for 2x speedup

---

### 4. export_mobileclip_text_encoder.py

**Purpose**: Export MobileCLIP2 text encoder for Track E text-to-image search.

**What it does**:
- Loads MobileCLIP2 text encoder (shared across S2/B variants)
- Reparameterizes model before export
- Exports to ONNX with dynamic batch support
- Converts to TensorRT engine for maximum throughput
- Validates with real text queries
- Tests embedding similarity

**Supported models**:
- **MobileCLIP2-S2** (63.4M text encoder) - Same encoder as S0/B
- **MobileCLIP2-B** (63.4M text encoder) - Same encoder as S0/S2

**Model specifications**:
- Input: `text_tokens [B, 77]` INT64 token IDs
- Output: `text_embeddings [B, 512]` FP32, L2-normalized
- Opset version: 17 (optimized for TensorRT 10.x)
- Dynamic batching: 1 to 64 (configurable)
- Context length: 77 tokens (max sequence length)

**Usage via Makefile** (recommended):

```bash
# Export both image and text encoders
make export-mobileclip
```

**Direct usage**:

```bash
# Export MobileCLIP2-S2 text encoder (default)
docker compose exec yolo-api python /app/export/export_mobileclip_text_encoder.py

# Export MobileCLIP2-B text encoder (identical to S2)
docker compose exec yolo-api python /app/export/export_mobileclip_text_encoder.py --model B

# Export ONNX only (skip TensorRT)
docker compose exec yolo-api python /app/export/export_mobileclip_text_encoder.py --skip-tensorrt

# Customize max batch size
docker compose exec yolo-api python /app/export/export_mobileclip_text_encoder.py --max-batch-size 32
```

**Output files**:
```
pytorch_models/
└── mobileclip2_s2_text_encoder.onnx   # Intermediate ONNX model

models/
└── mobileclip2_s2_text_encoder/
    ├── 1/
    │   └── model.plan                 # TensorRT engine for Triton
    └── config.pbtxt                   # Triton configuration (manual)
```

**Key implementation notes**:
- Text encoder is shared across MobileCLIP2-S0, S2, and B variants
- Handles token embeddings, positional encoding, transformer layers, and pooling
- L2 normalization ensures compatibility with image embeddings for similarity search
- Test queries included for validation: "a photo of a dog", "red car on highway", etc.

---

## Workflow

### Initial Setup (First Time)

1. **Download PyTorch models**:
   ```bash
   make download-pytorch
   ```

2. **Export all formats** (recommended for testing):
   ```bash
   make export-models
   ```

3. **Restart Triton** to load exported models:
   ```bash
   make restart-triton
   ```

4. **Verify models loaded**:
   ```bash
   make validate-exports
   ```

### Production Workflow (Small Model Only)

For focused benchmarking with just the small model:

1. **Download model**:
   ```bash
   make download-pytorch
   ```

2. **Export optimized formats**:
   ```bash
   make export-small
   ```

3. **Create Track D components** (see [dali/README.md](../dali/README.md)):
   ```bash
   make create-dali
   ```

4. **Verify all tracks working**:
   ```bash
   make test-all-tracks
   ```

### Track E Setup (Visual Search)

For Track E visual search with MobileCLIP:

1. **Download MobileCLIP models** (on host):
   ```bash
   bash scripts/track_e/setup_mobileclip_env.sh
   ```

2. **Export MobileCLIP encoders**:
   ```bash
   make export-mobileclip
   ```

3. **Create Track E pipeline** (see [dali/README.md](../dali/README.md)):
   ```bash
   make create-dali-dual
   ```

4. **Verify Track E working**:
   ```bash
   make test-track-e
   ```

### Re-exporting Models

If you need to re-export (e.g., after updating Ultralytics or changing settings):

1. **Clean old exports**:
   ```bash
   make clean-exports
   ```

2. **Check status** (verify files removed):
   ```bash
   make export-status
   ```

3. **Re-export**:
   ```bash
   make export-models
   ```

4. **Restart Triton**:
   ```bash
   make restart-triton
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
| Large | 1 | 8 | 16 |
| XLarge | 1 | 4 | 8 |

**Why different max batch sizes?**
- Nano: Smallest model, fits more in GPU memory
- Small: Good balance of speed and accuracy
- Medium: More accurate, moderate memory usage
- Large: High accuracy, more memory per image
- XLarge: Highest accuracy, largest memory footprint

### NMS Configuration

For End2End exports (Tracks C & D):

```python
# NMS thresholds
IOU_THRESHOLD = 0.7     # IoU threshold for NMS
CONF_THRESHOLD = 0.25   # Confidence threshold

# Detection limits
TOPK = 300              # Max detections per image
```

**Output format**:
- `num_dets`: INT32 [1] - Number of detections (0-300)
- `det_boxes`: FP32 [300, 4] - Bounding boxes (x, y, w, h)
- `det_scores`: FP32 [300] - Confidence scores
- `det_classes`: INT32 [300] - Class IDs (0-79 for COCO)

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
│   ├── export_models.py                        # YOLO export script
│   ├── download_pytorch_models.py              # Download YOLO .pt files
│   ├── export_mobileclip_image_encoder.py      # MobileCLIP image encoder export
│   ├── export_mobileclip_text_encoder.py       # MobileCLIP text encoder export
│   └── README.md                               # This documentation
├── pytorch_models/
│   ├── yolo11s.pt                              # YOLO PyTorch model
│   ├── mobileclip2_s2/                         # MobileCLIP checkpoint
│   ├── mobileclip2_s2_image_encoder.onnx       # Intermediate ONNX
│   └── mobileclip2_s2_text_encoder.onnx        # Intermediate ONNX
└── models/
    ├── yolov11_small/                          # Track A: ONNX
    ├── yolov11_small_trt/                      # Track B: TRT
    ├── yolov11_small_end2end/                  # Track C: ONNX End2End
    ├── yolov11_small_trt_end2end/              # Track C: TRT End2End
    ├── mobileclip2_s2_image_encoder/           # Track E: Image encoder
    └── mobileclip2_s2_text_encoder/            # Track E: Text encoder
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
make download-pytorch
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
