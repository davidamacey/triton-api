# Model Folder Structure & Organization

## 📁 Folder Overview

```
triton-api/
├── pytorch_models/              # Source PyTorch models (.pt files)
│   ├── yolo11n.pt              # Nano - 5.4MB
│   ├── yolo11s.pt              # Small - 18MB
│   └── yolo11m.pt              # Medium - 40MB
│
├── models/                      # Triton model repository
│   ├── yolov11_nano_trt/       # Standard TRT (no NMS)
│   ├── yolov11_small_trt/      # Standard TRT (no NMS)
│   ├── yolov11_medium_trt/     # Standard TRT (no NMS)
│   ├── yolov11_nano_trt_end2end/    # TRT with NMS ⚡
│   ├── yolov11_small_trt_end2end/   # TRT with NMS ⚡
│   └── yolov11_medium_trt_end2end/  # TRT with NMS ⚡
│
└── trt_cache/                   # DEPRECATED - not needed for .plan files
    └── (can be removed)

```

## 🔍 Folder Purposes

### `pytorch_models/` - Source Models
**Purpose:** Original PyTorch model files used as input for export

**Contents:**
- `yolo11n.pt` - YOLOv11 Nano (smallest, fastest)
- `yolo11s.pt` - YOLOv11 Small
- `yolo11m.pt` - YOLOv11 Medium

**Usage:** Export script reads from here
```python
model = YOLO("/app/pytorch_models/yolo11n.pt")
model.export(format="engine", ...)
```

**Keep or Remove?** ✅ KEEP - needed for exports

---

### `models/` - Triton Model Repository
**Purpose:** Deployed models that Triton serves

**Structure for each model:**
```
models/yolov11_nano_trt/
├── config.pbtxt              # Triton configuration
└── 1/                        # Version 1
    └── model.plan            # TensorRT engine file
```

**Keep or Remove?** ✅ KEEP - this is what Triton loads

---

### `trt_cache/` - TensorRT Cache
**Purpose:** Caches TensorRT engines when using ONNX Runtime with TensorRT EP

**When Used:**
- With `platform: "onnxruntime_onnx"` + TensorRT EP
- Stores built `.engine` files to avoid rebuilding

**When NOT Used:**
- With `platform: "tensorrt_plan"` (native TRT engines)
- We're using native `.plan` files, so cache not needed

**Keep or Remove?** ❌ CAN REMOVE - not needed for native TRT engines

---

## 📊 Config Comparison: Standard vs End2End

### Standard TRT Config (`yolov11_nano_trt`)

```protobuf
name: "yolov11_nano_trt"
platform: "tensorrt_plan"          # Native TRT engine
max_batch_size: 128

input [
  { name: "images", data_type: TYPE_FP32, dims: [3, 640, 640] }
]

output [
  { name: "output0", data_type: TYPE_FP32, dims: [84, 8400] }  # Raw detections
]

# Output: [84, 8400] → Needs CPU Python NMS
# Speed: Fast inference, but +5-10ms CPU NMS
```

### End2End TRT Config (`yolov11_nano_trt_end2end`)

```protobuf
name: "yolov11_nano_trt_end2end"
platform: "tensorrt_plan"          # Native TRT engine with compiled NMS
max_batch_size: 128

input [
  { name: "images", data_type: TYPE_FP32, dims: [3, 640, 640] }
]

output [
  { name: "num_dets", data_type: TYPE_INT32, dims: [1] },
  { name: "det_boxes", data_type: TYPE_FP32, dims: [100, 4] },
  { name: "det_scores", data_type: TYPE_FP32, dims: [100] },
  { name: "det_classes", data_type: TYPE_INT32, dims: [100] }
]

# Output: Final detections (NMS already applied on GPU!)
# Speed: 3-5x faster total (no CPU NMS needed)
```

## 🎯 Key Differences

| Aspect | Standard TRT | End2End TRT |
|--------|--------------|-------------|
| **Platform** | `tensorrt_plan` | `tensorrt_plan` |
| **Source** | PyTorch .pt file | End2End ONNX (with NMS ops) |
| **Output** | `[84, 8400]` | `num_dets, det_boxes, det_scores, det_classes` |
| **NMS Location** | CPU Python | GPU (compiled in engine) |
| **Post-processing** | Required | None needed |
| **Total Latency** | Inference + 5-10ms | Inference only |
| **Performance** | Baseline | **3-5x faster** |

## 🚀 Deployment Strategy

### For Comparison Testing (Current Goal)

Load both types to compare:
```yaml
# docker-compose.yml
command:
  - tritonserver
  - --model-store=/models
  - --load-model=yolov11_nano_trt          # Standard
  - --load-model=yolov11_nano_trt_end2end  # With NMS
  - --load-model=yolov11_small_trt
  - --load-model=yolov11_small_trt_end2end
  - --load-model=yolov11_medium_trt
  - --load-model=yolov11_medium_trt_end2end
```

### For Production (After Testing)

Load only end2end (fastest):
```yaml
command:
  - tritonserver
  - --model-store=/models
  - --load-model=yolov11_nano_trt_end2end
  - --load-model=yolov11_small_trt_end2end
  - --load-model=yolov11_medium_trt_end2end
```

## 🧹 Cleanup Recommendations

### Can Be Removed
- ❌ `trt_cache/` - Not used with native TRT engines
- ❌ `models/yolov11_nano/` - ONNX models (not loading into Triton)
- ❌ `models/yolov11_small/` - ONNX models
- ❌ `models/yolov11_medium/` - ONNX models
- ❌ Old `.onnx.old`, `.plan.old` backup files

### Must Keep
- ✅ `pytorch_models/*.pt` - Source files for export
- ✅ `models/yolov11_*_trt/` - Standard TRT models (for comparison)
- ✅ `models/yolov11_*_trt_end2end/` - End2End TRT models (production)
- ✅ All `config.pbtxt` files

## 📋 Current Status

**Configs Created:**
- ✅ `yolov11_nano_trt/config.pbtxt` - Standard TRT
- ✅ `yolov11_small_trt/config.pbtxt` - Standard TRT
- ✅ `yolov11_medium_trt/config.pbtxt` - Standard TRT
- ✅ `yolov11_nano_trt_end2end/config.pbtxt` - TRT with NMS
- ✅ `yolov11_small_trt_end2end/config.pbtxt` - TRT with NMS
- ✅ `yolov11_medium_trt_end2end/config.pbtxt` - TRT with NMS

**Next Steps:**
1. Run cleanup script to remove old model files
2. Export TRT models: `--formats trt trt_end2end`
3. Update docker-compose.yml to load TRT models
4. Restart Triton and verify models load
5. Benchmark: Standard TRT vs End2End TRT

## 🔧 Export Command

```bash
# Export only TRT formats (what we need)
docker compose exec yolo-api python /app/scripts/export_models.py \
  --formats trt trt_end2end

# This creates:
# - yolov11_nano_trt/1/model.plan (standard)
# - yolov11_nano_trt_end2end/1/model.plan (with NMS)
# - yolov11_small_trt/1/model.plan
# - yolov11_small_trt_end2end/1/model.plan
# - yolov11_medium_trt/1/model.plan
# - yolov11_medium_trt_end2end/1/model.plan
```

Total: 6 TRT engines (3 standard + 3 end2end)
