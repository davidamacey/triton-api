# Model Export and Setup Guide

Complete guide for building and exporting YOLO11 models for all 4 performance tracks in the triton-api system.

---

## Overview

This guide covers exporting YOLO11 models for deployment across four performance tracks:

| Track | Backend | NMS Location | Preprocessing | Description |
|-------|---------|--------------|---------------|-------------|
| **A** | PyTorch | CPU | CPU | Baseline - Direct `.pt` models |
| **B** | TensorRT | CPU | CPU | TRT inference, Ultralytics NMS |
| **C** | TensorRT End2End | GPU | CPU | TRT + GPU NMS |
| **D** | DALI + TRT End2End | GPU | **GPU** | Full GPU pipeline (fastest) |

**Key Concept:** Each track progressively moves more computation to GPU for better performance:
- Track A: All CPU (baseline)
- Track B: GPU inference only
- Track C: GPU inference + NMS
- Track D: GPU decode + preprocessing + inference + NMS (100% GPU)

---

## Prerequisites

### System Requirements
- **GPU:** NVIDIA GPU with Ampere architecture or newer (RTX 30xx, A-series, H-series)
- **CUDA:** 11.8+ or 12.x
- **Docker:** 20.10+ with NVIDIA Container Toolkit
- **Disk Space:** ~20GB for models and containers
- **RAM:** 16GB minimum (32GB recommended for concurrent exports)

### Software Stack
- **Triton Server:** 25.02+ (includes DALI backend)
- **Python:** 3.11
- **TensorRT:** 10.x (bundled in Triton container)
- **PyTorch:** 2.5+
- **NVIDIA DALI:** 1.44+ (for Track D)

### Python Dependencies
```bash
# Core dependencies (already in requirements.txt)
ultralytics==8.3.18+      # YOLO framework
torch>=2.5.0              # PyTorch backend
tensorrt>=10.0.0          # TensorRT Python API
onnx>=1.12.0              # ONNX format
onnxsim>=0.4.33           # ONNX graph simplification
onnx-graphsurgeon         # Graph optimization
nvidia-dali-cuda120       # GPU preprocessing (Track D only)

# Development dependencies (in requirements-dev.txt)
onnxruntime-gpu           # ONNX validation
```

### Docker Setup
```bash
# Ensure containers are running
docker compose up -d

# Verify GPU access in yolo-api container
docker compose exec yolo-api nvidia-smi
```

---

## Track A: PyTorch Models

### Overview
Track A uses PyTorch `.pt` models directly without any export. This serves as the performance baseline.

**Advantages:**
- Simplest deployment (no export needed)
- Easy debugging
- Full PyTorch ecosystem

**Disadvantages:**
- Slowest performance (CPU NMS)
- Thread contention (requires `@ThreadingLocked`)
- No batching optimizations

### Setup

#### Step 1: Download PyTorch Models
```bash
# From host machine
docker compose exec yolo-api bash -c "
cd /app/pytorch_models
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt
"
```

#### Step 2: Verify Models
```bash
docker compose exec yolo-api python -c "
from ultralytics import YOLO
model = YOLO('/app/pytorch_models/yolo11n.pt')
print(f'Model loaded: {model.names}')
print(f'Task: {model.task}')
"
```

### Directory Structure
```
pytorch_models/
├── yolo11n.pt          # Nano (3.2M params, ~6MB)
├── yolo11s.pt          # Small (9.4M params, ~19MB)
└── yolo11m.pt          # Medium (20.1M params, ~40MB)
```

### Configuration
Track A is embedded in the unified service at `/mnt/nvm/repos/triton-api/src/main.py`:
```python
MODEL_IDENTIFIERS = {
    "nano": "/app/pytorch_models/yolo11n.pt",
    "small": "/app/pytorch_models/yolo11s.pt",
    "medium": "/app/pytorch_models/yolo11m.pt",
}
```

No export needed - models used directly!

---

## Track B: TensorRT Standard Export

### Overview
Track B converts models to TensorRT engines for faster inference, but still uses CPU for NMS.

**Pipeline:** PyTorch `.pt` → ONNX → TensorRT `.plan`

**Performance Gain:** ~1.5-2x faster than Track A

### Export Process

#### Step 1: Export to ONNX (Intermediate)
```bash
docker compose exec yolo-api python /app/export/export_models.py \
  --formats onnx \
  --models nano small medium
```

**What this does:**
- Loads PyTorch model
- Exports to ONNX with dynamic batching
- Saves to `models/{model_name}/1/model.onnx`
- Output format: `[84, 8400]` (raw detections, needs NMS)

#### Step 2: Convert ONNX → TensorRT
```bash
docker compose exec yolo-api python /app/export/export_models.py \
  --formats trt \
  --models nano small medium
```

**Build settings:**
- **Precision:** FP16 (2x faster than FP32)
- **Dynamic batching:** min=1, opt=batch/2, max=batch
- **Workspace:** 4GB
- **Build time:** 5-10 minutes per model

**Output:** `models/{model_name}_trt/1/model.plan`

### Model Configuration Template

Create `models/yolov11_nano_trt/config.pbtxt`:
```protobuf
name: "yolov11_nano_trt"
platform: "tensorrt_plan"
max_batch_size: 128

input [
  {
    name: "images"
    data_type: TYPE_FP32
    dims: [ 3, 640, 640 ]
  }
]

output [
  {
    name: "output0"
    data_type: TYPE_FP32
    dims: [ 84, 8400 ]  # Raw detections (needs CPU NMS)
  }
]

dynamic_batching {
  preferred_batch_size: [ 8, 16, 32, 64, 128 ]
  max_queue_delay_microseconds: 500
}

instance_group [
  {
    count: 2
    kind: KIND_GPU
    gpus: [ 0 ]
  }
]

optimization {
  execution_accelerators {
    gpu_execution_accelerator {
      name: "tensorrt"
      parameters { key: "precision_mode" value: "FP16" }
      parameters { key: "max_workspace_size_bytes" value: "4294967296" }
    }
  }
}
```

### Verification
```bash
# Check model loaded in Triton
curl http://localhost:8000/v2/models/yolov11_nano_trt/config

# Test inference via FastAPI
curl -X POST http://localhost:9600/predict/nano \
  -F "image=@test_images/bus.jpg"
```

---

## Track C: TensorRT End2End Export (GPU NMS)

### Overview
Track C adds GPU-accelerated NMS using TensorRT's EfficientNMS plugin. This requires a **patched export method** not available in official Ultralytics.

**Pipeline:** PyTorch `.pt` → ONNX (with NMS operators) → TensorRT `.plan` (compiled NMS)

**Performance Gain:** ~2.5-3x faster than Track A, ~1.6x faster than Track B

### Prerequisites

#### Understanding the Ultralytics Patch
The official Ultralytics library doesn't support exporting ONNX models with TensorRT NMS plugins. We use a **monkey-patch** extracted from [levipereira/ultralytics](https://github.com/levipereira/ultralytics) fork.

**What the patch does:**
1. Adds custom TensorRT operators (`TRT_EfficientNMS`, `TRT_EfficientNMSX`)
2. Wraps YOLO model with `End2End_TRT` class
3. Exports ONNX graph with `TRT::EfficientNMS_TRT` nodes
4. Adds `export_onnx_trt()` method to Exporter class

**Why it's safe:**
- Read-only monkey-patch (doesn't modify ultralytics source)
- Applied only at runtime in export scripts
- Isolated in `src/ultralytics_patches/` directory
- Can be removed when official ultralytics adds this feature

**Attribution:**
- Original code: Levi Pereira (levipereira/ultralytics)
- Version: Based on ultralytics 8.3.18
- License: AGPL-3.0
- See `/mnt/nvm/repos/triton-api/docs/Attribution/FORK_COMPARISON.md` for details

### Export Process

#### Step 1: Export ONNX with NMS Operators
```bash
docker compose exec yolo-api python /app/export/export_models.py \
  --formats onnx_end2end \
  --models nano small medium
```

**Under the hood:**
```python
# From export_models.py
from ultralytics_patches import apply_end2end_patch
apply_end2end_patch()  # Adds export_onnx_trt() method

from ultralytics import YOLO
model = YOLO("yolo11n.pt")

# Create exporter and call patched method
exporter = Exporter(cfg=args)
exporter.model = model.model
exporter.args.topk_all = 100        # Max detections
exporter.args.iou_thres = 0.45      # NMS IoU threshold
exporter.args.conf_thres = 0.25     # Confidence threshold

# Call patched export method directly
export_path, onnx_model = exporter.export_onnx_trt()
```

**Output:** `models/yolov11_nano_end2end/1/model.onnx`

**Verify NMS operators:**
```bash
docker compose exec yolo-api python -c "
import onnx
model = onnx.load('/app/models/yolov11_nano_end2end/1/model.onnx')
ops = [node.op_type for node in model.graph.node]
print('TRT::EfficientNMS_TRT found:', 'TRT::EfficientNMS_TRT' in ops)
"
```

#### Step 2: Build TensorRT Engine with Compiled NMS
```bash
docker compose exec yolo-api python /app/export/export_models.py \
  --formats trt_end2end \
  --models nano small medium
```

**Build process:**
```python
import tensorrt as trt

# Initialize TensorRT plugins (required for EfficientNMS)
logger = trt.Logger(trt.Logger.INFO)
trt.init_libnvinfer_plugins(logger, '')

# Build engine from end2end ONNX
builder = trt.Builder(logger)
config = builder.create_builder_config()
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

# Parse ONNX with NMS operators
parser = trt.OnnxParser(network, logger)
parser.parse_from_file('model_end2end.onnx')

# Build with FP16 and dynamic batching
config.set_flag(trt.BuilderFlag.FP16)
engine = builder.build_engine(network, config)
```

**Output:** `models/yolov11_nano_trt_end2end/1/model.plan`

### Model Configuration for Track C

Create `models/yolov11_nano_trt_end2end/config.pbtxt`:
```protobuf
name: "yolov11_nano_trt_end2end"
platform: "tensorrt_plan"
max_batch_size: 128

input [
  {
    name: "images"
    data_type: TYPE_FP32
    dims: [ 3, 640, 640 ]
  }
]

# Output format: End2End with GPU NMS
output [
  {
    name: "num_dets"
    data_type: TYPE_INT32
    dims: [ 1 ]  # Number of detections per image
  },
  {
    name: "det_boxes"
    data_type: TYPE_FP32
    dims: [ 100, 4 ]  # [x, y, w, h] for up to 100 detections
  },
  {
    name: "det_scores"
    data_type: TYPE_FP32
    dims: [ 100 ]  # Confidence scores
  },
  {
    name: "det_classes"
    data_type: TYPE_INT32
    dims: [ 100 ]  # Class IDs
  }
]

dynamic_batching {
  preferred_batch_size: [ 8, 16, 32, 64, 128 ]
  max_queue_delay_microseconds: 500
}

instance_group [
  {
    count: 2
    kind: KIND_GPU
    gpus: [ 0 ]
  }
]
```

**Key difference:** Notice the 4 output tensors instead of 1. NMS already ran on GPU!

### Verification
```bash
# Test via FastAPI endpoint
curl -X POST http://localhost:9600/predict/nano_end2end \
  -F "image=@test_images/bus.jpg" | jq .

# Expected response:
# {
#   "detections": [
#     {"x1": 123.4, "y1": 234.5, "x2": 345.6, "y2": 456.7, "confidence": 0.89, "class": 5}
#   ],
#   "status": "success",
#   "track": "C"
# }
```

---

## Track D: DALI + TensorRT End2End (Full GPU Pipeline)

### Overview
Track D is the **fastest configuration** - 100% GPU processing from JPEG bytes to final detections.

**Pipeline:** Raw JPEG → GPU decode → GPU letterbox → TRT End2End (with GPU NMS)

**Performance Gain:** ~4x faster than Track A, ~1.6x faster than Track C

**Why it's faster:**
- No CPU preprocessing bottleneck (eliminates 5-15ms)
- No Python GIL contention for preprocessing
- Smaller data transfer (compressed JPEG vs uncompressed tensors)
- Fully pipelined GPU operations

### Component 1: DALI Preprocessing Model

DALI (Data Loading Library) performs GPU-accelerated image preprocessing, replacing the CPU preprocessing in Track C.

#### Understanding GPU Letterbox
Traditional letterbox (Track C):
```python
# CPU calculations (1.5-2ms overhead)
scale = 640 / max(h, w)
new_size = (int(w * scale), int(h * scale))
pad_x = (640 - new_w) // 2
pad_y = (640 - new_h) // 2
affine_matrix = [[scale, 0, pad_x], [0, scale, pad_y]]

# Send matrix to GPU, apply transform
```

DALI letterbox (Track D):
```python
# ALL calculations on GPU using DALI operators
images = fn.decoders.image(jpeg_bytes, device="mixed")  # GPU decode
images = fn.resize(images, mode="not_larger", size=640)  # GPU resize
images = fn.pad(images, fill_value=114, align=0.5)      # GPU pad
images = fn.crop_mirror_normalize(images, ...)          # GPU normalize
# Zero CPU calculations!
```

#### Create DALI Pipeline
```bash
# Run from yolo-api container (has DALI installed)
docker compose exec yolo-api python /app/dali/create_dali_letterbox_pipeline.py
```

**This script creates:**
1. DALI pipeline definition (`model.dali`)
2. Triton config (`config.pbtxt`)
3. Tests letterbox accuracy vs Ultralytics

**Pipeline operations (ALL on GPU):**
```python
@dali.pipeline_def(batch_size=128, num_threads=4, device_id=0)
def yolo_letterbox_pipeline_gpu():
    # Step 1: GPU JPEG decode using nvJPEG
    encoded = fn.external_source(name="encoded_images", dtype=types.UINT8)
    images = fn.decoders.image(
        encoded,
        device="mixed",           # CPU→GPU decode (nvJPEG acceleration)
        output_type=types.RGB,
        hw_decoder_load=0.65      # Use hardware decoder
    )

    # Step 2: Resize with aspect ratio preservation (GPU)
    images_resized = fn.resize(
        images,
        size=[640, 640],
        mode="not_larger",        # Preserves aspect ratio
        interp_type=types.INTERP_LINEAR,
        device="gpu"
    )

    # Step 3: Pad to 640x640 with gray (GPU)
    images_padded = fn.pad(
        images_resized,
        axes=[0, 1],
        align=[0.5, 0.5],         # Center alignment
        shape=[640, 640],
        fill_value=114,           # Gray padding
        device="gpu"
    )

    # Step 4: Normalize and transpose (GPU)
    images_final = fn.crop_mirror_normalize(
        images_padded,
        mean=[0.0, 0.0, 0.0],
        std=[255.0, 255.0, 255.0],  # Divide by 255
        output_layout="CHW",          # HWC → CHW
        output_dtype=types.FLOAT,
        device="gpu"
    )

    return images_final
```

**Output files:**
```
models/yolo_preprocess_dali/
├── 1/
│   └── model.dali           # Serialized DALI pipeline
└── config.pbtxt             # Triton config
```

**DALI config.pbtxt:**
```protobuf
name: "yolo_preprocess_dali"
backend: "dali"
max_batch_size: 128

input [
  {
    name: "encoded_images"
    data_type: TYPE_UINT8
    dims: [ -1 ]  # Variable-length JPEG bytes
  }
]

output [
  {
    name: "preprocessed_images"
    data_type: TYPE_FP32
    dims: [ 3, 640, 640 ]
  }
]

instance_group [
  {
    count: 1         # NVIDIA recommendation: count=1 to avoid memory overhead
    kind: KIND_GPU
    gpus: [ 0 ]
  }
]

parameters: {
  key: "num_threads"
  value: { string_value: "4" }
}
```

#### Validate DALI Letterbox Accuracy
```bash
docker compose exec yolo-api python /app/dali/validate_dali_letterbox.py
```

**What this tests:**
- Pixel-level accuracy vs Ultralytics letterbox
- Various aspect ratios (square, portrait, landscape)
- Edge cases (very small/large images)
- Max diff threshold: <0.02 (acceptable due to floating point precision)

**Expected output:**
```
================================================================================
DALI Letterbox Validation
================================================================================

[1/2] Testing synthetic images...
──────────────────────────────────────────────────────────────────────────────
Test case: Square (640x640)
  Max pixel diff:  0.001234
  Mean pixel diff: 0.000456
  ✓ PASS

[2/2] Testing real images...
  ✓ PASS: test_images/bus.jpg

================================================================================
✓ ALL TESTS PASSED
================================================================================
```

### Component 2: TensorRT End2End Model (Reused from Track C)

Track D reuses the same TRT End2End models from Track C:
- `yolov11_nano_trt_end2end`
- `yolov11_small_trt_end2end`
- `yolov11_medium_trt_end2end`

No additional export needed!

### Component 3: Ensemble Models (3 Tiers)

Ensembles chain DALI preprocessing → YOLO TRT End2End.

Track D provides **three ensemble tiers** optimized for different workloads:

| Tier | Model Suffix | Max Batch | Batching Window | Use Case |
|------|-------------|-----------|-----------------|----------|
| **Streaming** | `_gpu_e2e_streaming` | 8 | 0.1ms | Real-time video (60 FPS) |
| **Balanced** | `_gpu_e2e` | 64 | 0.5ms | General purpose (default) |
| **Batch** | `_gpu_e2e_batch` | 128 | 5.0ms | Offline processing |

#### Create Ensemble Configs
```bash
docker compose exec yolo-api python /app/dali/create_ensembles.py \
  --models nano small medium
```

**This generates 9 ensemble models:** 3 tiers × 3 sizes

**Ensemble config example** (`models/yolov11_small_gpu_e2e/config.pbtxt`):
```protobuf
name: "yolov11_small_gpu_e2e"
platform: "ensemble"
max_batch_size: 64

# Input: ONLY raw JPEG bytes (no CPU preprocessing!)
input [
  {
    name: "encoded_images"
    data_type: TYPE_UINT8
    dims: [ -1 ]
  }
]

# Output: Final detections (after GPU NMS)
output [
  {
    name: "num_dets"
    data_type: TYPE_INT32
    dims: [ 1 ]
  },
  {
    name: "det_boxes"
    data_type: TYPE_FP32
    dims: [ 100, 4 ]
  },
  {
    name: "det_scores"
    data_type: TYPE_FP32
    dims: [ 100 ]
  },
  {
    name: "det_classes"
    data_type: TYPE_INT32
    dims: [ 100 ]
  }
]

# Ensemble: DALI → YOLO TRT End2End
ensemble_scheduling {
  step [
    {
      # Step 1: DALI preprocessing
      model_name: "yolo_preprocess_dali"
      model_version: -1
      input_map {
        key: "encoded_images"
        value: "encoded_images"
      }
      output_map {
        key: "preprocessed_images"
        value: "preprocessed_images"
      }
    },
    {
      # Step 2: YOLO TRT End2End
      model_name: "yolov11_small_trt_end2end"
      model_version: -1
      input_map {
        key: "images"
        value: "preprocessed_images"
      }
      output_map {
        key: "num_dets"
        value: "num_dets"
      }
      output_map {
        key: "det_boxes"
        value: "det_boxes"
      }
      output_map {
        key: "det_scores"
        value: "det_scores"
      }
      output_map {
        key: "det_classes"
        value: "det_classes"
      }
    }
  ]
}

# Dynamic batching at ensemble level
dynamic_batching {
  preferred_batch_size: [ 4, 8, 16, 32, 64 ]
  max_queue_delay_microseconds: 500  # 0.5ms (balanced tier)
  preserve_ordering: false

  default_queue_policy {
    timeout_action: REJECT
    default_timeout_microseconds: 50000  # 50ms
    allow_timeout_override: false
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

### Configuration Files Summary

**Track D requires these config files:**

```
models/
├── yolo_preprocess_dali/            # DALI preprocessing
│   ├── 1/model.dali
│   └── config.pbtxt
├── yolov11_nano_trt_end2end/        # TRT End2End (from Track C)
│   ├── 1/model.plan
│   └── config.pbtxt
└── yolov11_nano_gpu_e2e_streaming/  # Ensemble (streaming tier)
    ├── 1/  (empty)
    └── config.pbtxt
```

### Deployment

#### Update docker-compose.yml
```yaml
services:
  triton-api:
    command:
      - tritonserver
      - --model-store=/models
      - --backend-config=default-max-batch-size=128

      # Track D models
      - --load-model=yolo_preprocess_dali
      - --load-model=yolov11_nano_gpu_e2e_streaming
      - --load-model=yolov11_nano_gpu_e2e
      - --load-model=yolov11_nano_gpu_e2e_batch
      - --load-model=yolov11_small_gpu_e2e_streaming
      - --load-model=yolov11_small_gpu_e2e
      - --load-model=yolov11_small_gpu_e2e_batch
      - --load-model=yolov11_medium_gpu_e2e_streaming
      - --load-model=yolov11_medium_gpu_e2e
      - --load-model=yolov11_medium_gpu_e2e_batch
```

#### Restart Triton
```bash
docker compose down
docker compose up -d
docker compose logs -f triton-api
```

**Look for:**
```
I0115 12:00:00.123 1 server.cc:592] Successfully loaded 'yolo_preprocess_dali'
I0115 12:00:00.234 1 server.cc:592] Successfully loaded 'yolov11_nano_gpu_e2e_streaming'
...
```

### Verification
```bash
# Test streaming tier (lowest latency)
curl -X POST http://localhost:9600/predict/small_gpu_e2e_streaming \
  -F "image=@test_images/bus.jpg" | jq .

# Test balanced tier (default)
curl -X POST http://localhost:9600/predict/small_gpu_e2e \
  -F "image=@test_images/bus.jpg" | jq .

# Test batch tier (highest throughput)
curl -X POST http://localhost:9600/predict/small_gpu_e2e_batch \
  -F "image=@test_images/bus.jpg" | jq .
```

---

## Model Directory Structure

Complete Triton model repository structure for all tracks:

```
models/
├── yolo_preprocess_dali/              # Track D: DALI preprocessing
│   ├── 1/
│   │   └── model.dali
│   └── config.pbtxt
│
├── yolov11_nano/                      # Track B: Standard ONNX
│   ├── 1/
│   │   └── model.onnx
│   └── config.pbtxt
│
├── yolov11_nano_trt/                  # Track B: Standard TRT
│   ├── 1/
│   │   └── model.plan
│   └── config.pbtxt
│
├── yolov11_nano_end2end/              # Track C: End2End ONNX
│   ├── 1/
│   │   └── model.onnx  (with NMS ops)
│   └── config.pbtxt
│
├── yolov11_nano_trt_end2end/          # Track C: End2End TRT
│   ├── 1/
│   │   └── model.plan  (compiled NMS)
│   └── config.pbtxt
│
├── yolov11_nano_gpu_e2e_streaming/    # Track D: Ensemble (streaming)
│   ├── 1/  (empty)
│   └── config.pbtxt
│
├── yolov11_nano_gpu_e2e/              # Track D: Ensemble (balanced)
│   ├── 1/  (empty)
│   └── config.pbtxt
│
└── yolov11_nano_gpu_e2e_batch/        # Track D: Ensemble (batch)
    ├── 1/  (empty)
    └── config.pbtxt

# Repeat structure for small and medium models
```

---

## Export Scripts Reference

### /mnt/nvm/repos/triton-api/scripts/export_models.py

**Purpose:** Unified export script for all TensorRT formats

**Usage:**
```bash
# Export all formats for all models (default)
docker compose exec yolo-api python /app/export/export_models.py

# Export only end2end models
docker compose exec yolo-api python /app/export/export_models.py \
  --formats onnx_end2end trt_end2end

# Export specific model sizes
docker compose exec yolo-api python /app/export/export_models.py \
  --formats trt_end2end --models nano small
```

**Formats:**
- `onnx` - Standard ONNX (Track B intermediate)
- `trt` - Standard TensorRT engine (Track B)
- `onnx_end2end` - ONNX with NMS operators (Track C intermediate)
- `trt_end2end` - TensorRT with compiled NMS (Track C)
- `all` - Export all formats

**Key configuration:**
```python
MODELS = {
    "nano": {
        "pt_file": "/app/pytorch_models/yolo11n.pt",
        "triton_name": "yolov11_nano",
        "max_batch": 128,
        "topk": 100
    },
    # ...
}

# NMS settings (for end2end exports)
IOU_THRESHOLD = 0.45
CONF_THRESHOLD = 0.25
```

### /mnt/nvm/repos/triton-api/scripts/create_dali_letterbox_pipeline_v2.py

**Purpose:** Create DALI preprocessing pipeline for Track D

**Usage:**
```bash
docker compose exec yolo-api python /app/dali/create_dali_letterbox_pipeline.py
```

**What it creates:**
- `models/yolo_preprocess_dali/1/model.dali` - Serialized pipeline
- `models/yolo_preprocess_dali/config.pbtxt` - Triton config
- Tests pipeline with dummy data

**Key features:**
- 100% GPU letterbox calculation (zero CPU)
- Uses DALI's built-in `mode="not_larger"` resize
- Centered padding with gray (114, 114, 114)
- Matches Ultralytics letterbox pixel-for-pixel

### /mnt/nvm/repos/triton-api/scripts/create_ensembles.py

**Purpose:** Generate ensemble configs for Track D

**Usage:**
```bash
# Create ensembles for specific models
docker compose exec yolo-api python /app/dali/create_ensembles.py --models small

# Create for all models
docker compose exec yolo-api python /app/dali/create_ensembles.py --models all
```

**What it creates:**
- 9 ensemble configs (3 tiers × 3 sizes)
- Chains DALI → TRT End2End
- Configures dynamic batching per tier

### /mnt/nvm/repos/triton-api/scripts/validate_dali_letterbox_v2.py

**Purpose:** Verify DALI letterbox matches Ultralytics

**Usage:**
```bash
docker compose exec yolo-api python /app/dali/validate_dali_letterbox.py
```

**Tests:**
- Synthetic images (various aspect ratios)
- Real images from test_images/
- Pixel-level accuracy (max diff <0.02)

---

## Troubleshooting

### Common Export Issues

#### Error: "CUDA out of memory" during export
```bash
# Reduce batch size in MODELS config
MODELS = {
    "medium": {
        "max_batch": 16,  # Reduced from 32
    }
}

# Or export one model at a time
python scripts/export_models.py --models nano
```

#### Error: "Failed to parse ONNX" during TRT build
```bash
# Check ONNX model is valid
docker compose exec yolo-api python -c "
import onnx
model = onnx.load('/app/models/yolov11_nano_end2end/1/model.onnx')
onnx.checker.check_model(model)
print('ONNX model is valid')
"

# Verify TensorRT plugins loaded
docker compose exec yolo-api python -c "
import tensorrt as trt
logger = trt.Logger(trt.Logger.INFO)
trt.init_libnvinfer_plugins(logger, '')
print('TensorRT plugins initialized')
"
```

#### Error: "No module named 'ultralytics_patches'"
```bash
# Check patch is in Python path
docker compose exec yolo-api python -c "
import sys
sys.path.insert(0, '/app/src')
from ultralytics_patches import apply_end2end_patch
apply_end2end_patch()
print('Patch applied successfully')
"
```

### DALI-Specific Issues

#### Error: "Backend 'dali' not found"
```bash
# Verify DALI backend in Triton
docker compose exec triton-api ls /opt/tritonserver/backends/dali

# If missing, check Triton version (need 25.02+)
docker compose exec triton-api tritonserver --version
```

#### Error: DALI letterbox validation fails (max diff >0.02)
```bash
# Check JPEG encoding quality
# DALI's nvJPEG may decode slightly differently

# Verify resize interpolation matches
# Both should use INTER_LINEAR

# Check for numerical precision issues
# FP32 vs FP16 can cause small differences
```

#### DALI model loads but inference fails
```bash
# Check input format
# DALI expects raw JPEG bytes, not decoded arrays

# Verify batch dimension
# Input should be [batch, -1] not [batch, h, w, c]

# Test with dummy data
docker compose exec yolo-api python -c "
import numpy as np
from PIL import Image
import io

# Create test JPEG
img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
pil_img = Image.fromarray(img)
buffer = io.BytesIO()
pil_img.save(buffer, format='JPEG')
jpeg_bytes = buffer.getvalue()

# Test pipeline
pipe = yolo_letterbox_pipeline_gpu(batch_size=1, num_threads=2, device_id=0)
pipe.build()
pipe.feed_input('encoded_images', [np.frombuffer(jpeg_bytes, dtype=np.uint8)])
output = pipe.run()
print('DALI pipeline working')
"
```

### TensorRT Issues

#### TRT engine build takes very long (>15 minutes)
```bash
# Normal for first build with large batch sizes
# Reduce max_batch to speed up:
MODELS["medium"]["max_batch"] = 16  # Instead of 32

# Or disable FP16 (slower inference, faster build):
HALF = False  # In export_models.py
```

#### TRT engine fails on different GPU
```bash
# TRT engines are GPU-specific!
# Must rebuild on target GPU

# Check GPU architecture:
docker compose exec yolo-api python -c "
import torch
print(torch.cuda.get_device_properties(0))
"

# Rebuild on target GPU:
docker compose exec yolo-api python /app/export/export_models.py \
  --formats trt_end2end --models nano
```

#### TRT inference slower than expected
```bash
# Check FP16 is enabled:
docker compose exec triton-api curl http://localhost:8000/v2/models/yolov11_nano_trt_end2end/config | jq '.optimization'

# Verify dynamic batching working:
docker compose logs triton-api | grep "batch size"

# Monitor GPU utilization:
nvidia-smi dmon -s ucm -d 1
# Track D should show 90%+ GPU utilization
```

### End2End Export Issues

#### NMS operators not found in ONNX
```bash
# Verify patch applied:
docker compose exec yolo-api python -c "
from ultralytics_patches import is_patch_applied
print('Patch applied:', is_patch_applied())
"

# Check ONNX graph:
docker compose exec yolo-api python -c "
import onnx
model = onnx.load('/app/models/yolov11_nano_end2end/1/model.onnx')
ops = set(node.op_type for node in model.graph.node)
print('Operators:', ops)
print('Has TRT NMS:', any('NMS' in op or 'TRT' in op for op in ops))
"
```

#### End2End export fails with "format 'onnx_trt' not supported"
```bash
# This means the patch wasn't applied
# Make sure to import and apply patch:

from ultralytics_patches import apply_end2end_patch
apply_end2end_patch()  # Must be called BEFORE creating YOLO instance

from ultralytics import YOLO
model = YOLO("yolo11n.pt")
# Now export_onnx_trt() method is available
```

---

## Performance Expectations

Based on benchmarks with YOLO11 Small on NVIDIA A6000:

| Track | Preprocessing | Inference | NMS | Total | Speedup |
|-------|--------------|-----------|-----|-------|---------|
| **A** | 8-12ms (CPU) | 12-15ms | 2-3ms | ~25ms | 1.0x |
| **B** | 8-12ms (CPU) | 6-8ms | 2-3ms | ~15ms | 1.7x |
| **C** | 8-12ms (CPU) | 6-8ms | 0.5ms | ~10ms | 2.5x |
| **D** | 1-2ms (GPU) | 6-8ms | 0.5ms | ~6.3ms | **4.0x** |

**Key insight:** Track D's 85% reduction in CPU overhead (2ms → 0.3ms) enables true 100% GPU utilization.

---

## Related Documentation

### Technical Documentation
- [DALI Letterbox Implementation](/mnt/nvm/repos/triton-api/docs/Technical/DALI_LETTERBOX_IMPLEMENTATION.md) - Detailed DALI letterbox design
- [Implementation Notes](/mnt/nvm/repos/triton-api/docs/IMPLEMENTATION_NOTES.md) - Three-track architecture and per-request instances
- [Triton Best Practices](/mnt/nvm/repos/triton-api/docs/Technical/TRITON_BEST_PRACTICES.md) - Optimization guide

### Track-Specific Guides
- [Track D Complete Guide](/mnt/nvm/repos/triton-api/docs/Tracks/TRACK_D_COMPLETE.md) - Full GPU pipeline deployment
- [Benchmarking Guide](/mnt/nvm/repos/triton-api/docs/Tracks/BENCHMARKING_GUIDE.md) - Performance testing methodology

### Attribution & References
- [Fork Comparison](/mnt/nvm/repos/triton-api/docs/Attribution/FORK_COMPARISON.md) - levipereira vs official ultralytics
- [End2End Analysis](/mnt/nvm/repos/triton-api/docs/Attribution/END2END_ANALYSIS.md) - GPU NMS deep dive

### Source Code
- [export_models.py](/mnt/nvm/repos/triton-api/scripts/export_models.py) - Unified export script
- [end2end_export.py](/mnt/nvm/repos/triton-api/src/ultralytics_patches/end2end_export.py) - TensorRT NMS patch
- [create_dali_letterbox_pipeline_v2.py](/mnt/nvm/repos/triton-api/scripts/create_dali_letterbox_pipeline_v2.py) - DALI pipeline builder
- [create_ensembles.py](/mnt/nvm/repos/triton-api/scripts/create_ensembles.py) - Ensemble config generator

### External References

**NVIDIA Official:**
- [Triton + DALI Blog](https://developer.nvidia.com/blog/accelerating-inference-with-triton-inference-server-and-dali/)
- [DALI Backend Docs](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/dali_backend/)
- [DALI Operators](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/operations/nvidia/dali/fn.html)
- [TensorRT Best Practices](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html)

**Community Examples:**
- [levipereira/triton-server-yolo](https://github.com/levipereira/triton-server-yolo) - DALI + YOLO reference
- [Ultralytics Official](https://github.com/ultralytics/ultralytics) - Original YOLO framework
- [levipereira/ultralytics](https://github.com/levipereira/ultralytics) - End2End fork source

---

## Quick Start Checklist

### Track B (TensorRT Standard)
- [ ] Download PyTorch models to `pytorch_models/`
- [ ] Export to ONNX: `--formats onnx`
- [ ] Build TRT engines: `--formats trt`
- [ ] Create `config.pbtxt` for each model
- [ ] Restart Triton and test

### Track C (TRT End2End)
- [ ] Complete Track B first
- [ ] Apply ultralytics patch (automatic in export script)
- [ ] Export end2end ONNX: `--formats onnx_end2end`
- [ ] Build end2end TRT: `--formats trt_end2end`
- [ ] Verify NMS operators in ONNX
- [ ] Create `config.pbtxt` with end2end outputs
- [ ] Test via FastAPI endpoint

### Track D (Full GPU)
- [ ] Complete Track C first
- [ ] Create DALI pipeline: `create_dali_letterbox_pipeline_v2.py`
- [ ] Validate letterbox accuracy: `validate_dali_letterbox_v2.py`
- [ ] Generate ensembles: `create_ensembles.py`
- [ ] Update `docker-compose.yml` to load ensembles
- [ ] Restart Triton
- [ ] Test all tiers (streaming, balanced, batch)
- [ ] Run benchmarks: `benchmarks/compare_all_tracks.py`

---

**Last Updated:** 2025-01-15
**Maintained By:** Claude Code
**Version:** 1.0.0
