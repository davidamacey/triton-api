# Ultralytics Fork Comparison - levipereira vs Official

## Version Information

| Repository | Version | Last Updated | Commits Behind |
|------------|---------|--------------|----------------|
| **Official** ultralytics/ultralytics | 8.3.228 | Nov 15, 2025 (today) | - |
| **Fork** levipereira/ultralytics | 8.3.18 | Oct 20, 2024 | ~210 versions |

## Key Differences

### ✅ Fork ADDS (What We Need)

**1. `export_onnx_trt()` method** - End2End NMS export
- Adds ~365 lines of code for TensorRT EfficientNMS plugin integration
- Includes custom torch operators: `TRT_EfficientNMS`, `TRT_EfficientNMS_85`, `TRT_EfficientNMSX`, etc.
- Adds `End2End_TRT` wrapper class
- This is the **ONLY** reason we need this fork

### ❌ Fork MISSING (Compared to Official)

The fork is missing these newer export formats:
- `export_executorch` - PyTorch mobile runtime
- `export_imx` - NXP i.MX processors
- `export_mnn` - Alibaba MNN framework
- `export_rknn` - Rockchip NPU

**Impact:** None for our use case. We don't need these formats.

## YOLOv11 Support

### ✅ Both Support YOLOv11

**Official has:**
```
models/11/
├── yolo11-cls-resnet18.yaml
├── yolo11-cls.yaml
├── yolo11-obb.yaml
├── yolo11-pose.yaml
├── yolo11-seg.yaml
├── yolo11.yaml
├── yoloe-11-seg.yaml
└── yoloe-11.yaml
```

**Fork has:**
```
models/11/
├── yolo11-cls.yaml
├── yolo11-obb.yaml
├── yolo11-pose.yaml
├── yolo11-seg.yaml
└── yolo11.yaml
```

**Conclusion:** Fork has core YOLOv11 detection model support. Missing yoloe-11 variants (newer efficient models).

## Risk Assessment

### 🟡 Medium Risk - Version Lag

**Concerns:**
1. **210 versions behind** - Significant gap (8.3.18 vs 8.3.228)
2. **1 month outdated** - Official repo is actively developed
3. **Missing features** - New export formats, optimizations, bug fixes
4. **Compatibility issues** - Potential conflicts with latest PyTorch/ONNX/TensorRT versions

**What could break:**
- Latest TensorRT versions might not work with older export code
- Bug fixes in official repo not in fork
- Performance optimizations missing

## Alternative Approaches

### Option 1: Use Fork As-Is ⚠️
**Pros:**
- Ready to use immediately
- Known to work with YOLOv8/v9

**Cons:**
- 210 versions behind
- No recent bug fixes
- May have compatibility issues

### Option 2: Extract End2End Code & Port to Official ✅ RECOMMENDED
**Pros:**
- Use latest official ultralytics
- Get all bug fixes and optimizations
- Future-proof

**Cons:**
- Requires manual integration
- Need to test thoroughly

**Approach:**
- Copy `export_onnx_trt()` method from fork
- Copy custom TRT operators (`TRT_EfficientNMS`, etc.)
- Copy `End2End_TRT` class
- Add to official ultralytics in our container
- ~600 lines of code to copy

### Option 3: Hybrid - Fork + Official Side-by-Side
**Pros:**
- Can use both
- No modifications needed

**Cons:**
- Two separate ultralytics installations
- Version confusion
- Complexity

**Approach:**
```python
# Use fork for end2end export
sys.path.insert(0, '/path/to/fork')
from ultralytics import YOLO as YOLO_Fork
model = YOLO_Fork("yolo11n.pt")
model.export(format="onnx_trt", ...)

# Use official for everything else
sys.path.insert(0, '/path/to/official')
from ultralytics import YOLO as YOLO_Official
```

## Testing Strategy

Before committing to the fork, test:

### Test 1: Load YOLOv11 Model
```bash
docker compose exec pytorch-api python -c "
from ultralytics import YOLO
model = YOLO('yolo11n.pt')
print(f'Model loaded: {model.model}')
"
```

### Test 2: Standard ONNX Export
```bash
docker compose exec pytorch-api python -c "
from ultralytics import YOLO
model = YOLO('yolo11n.pt')
model.export(format='onnx', imgsz=640, dynamic=True)
print('Standard ONNX export successful')
"
```

### Test 3: End2End Export
```bash
docker compose exec pytorch-api python -c "
from ultralytics import YOLO
model = YOLO('yolo11n.pt')
model.export(
    format='onnx_trt',
    imgsz=640,
    topk_all=100,
    iou_thres=0.45,
    conf_thres=0.25,
    dynamic=True,
    half=True
)
print('End2End export successful')
"
```

### Test 4: Verify ONNX Has NMS Plugin
```bash
docker compose exec pytorch-api python -c "
import onnx
model = onnx.load('yolo11n-trt.onnx')
ops = [node.op_type for node in model.graph.node]
print(f'TRT::EfficientNMS_TRT found: {\"TRT::EfficientNMS_TRT\" in ops}')
for node in model.graph.node:
    if 'NMS' in node.op_type or 'TRT' in node.op_type:
        print(f'Found: {node.op_type}')
"
```

## Recommendation

### 🎯 Recommended Approach: Extract & Integrate

**Step 1:** Create a local patch file with end2end export code

Create `scripts/ultralytics_end2end_patch.py`:
```python
"""
Monkey-patch ultralytics to add export_onnx_trt() method.
Based on levipereira/ultralytics fork.
"""

import torch
import torch.nn as nn

# Copy TRT_EfficientNMS classes from fork (lines 1355-1647)
class TRT_EfficientNMS(torch.autograd.Function):
    # ... (copy from fork)

class End2End_TRT(torch.nn.Module):
    # ... (copy from fork)

# Monkey-patch the Exporter class
from ultralytics.engine.exporter import Exporter

def export_onnx_trt(self, prefix="ONNX:"):
    # ... (copy from fork lines 460-592)

# Patch it in
Exporter.export_onnx_trt = export_onnx_trt
```

**Step 2:** Use it in export scripts
```python
import ultralytics_end2end_patch  # Apply patch
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
model.export(format="onnx_trt", ...)
```

**Benefits:**
- ✅ Use official ultralytics (latest version)
- ✅ Add end2end export capability
- ✅ Easy to update when official adds this feature
- ✅ Isolated changes, easy to remove later

**Step 3:** If it works well, submit PR to official ultralytics repo!

## Decision Matrix

| Criteria | Fork As-Is | Extract & Integrate | Use Both |
|----------|-----------|-------------------|----------|
| **Ease of use** | ⭐⭐⭐ | ⭐⭐ | ⭐ |
| **Future-proof** | ❌ | ✅ | ⭐⭐ |
| **Maintenance** | ❌ | ✅ | ⭐ |
| **Risk** | 🟡 Medium | 🟢 Low | 🟡 Medium |
| **Time to implement** | 5 min | 2 hours | 1 hour |

## Final Recommendation

**For Production:** Extract & Integrate (Option 2)
**For Quick Testing:** Fork As-Is (Option 1)

Start with fork to validate the approach works, then migrate to extracted code for production.
