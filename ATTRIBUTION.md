# Third-Party Code Attribution

This document provides attribution for third-party code and reference architectures used in this project.

---

## Ultralytics YOLO - End2End ONNX Export

This project uses modified code from the **levipereira/ultralytics** fork to enable GPU-accelerated end-to-end YOLO inference with TensorRT EfficientNMS plugin integration.

### Repository Information
- **Fork Repository:** https://github.com/levipereira/ultralytics
- **Original Repository:** https://github.com/ultralytics/ultralytics (official)
- **Fork Version:** 8.3.18 (October 20, 2024)
- **License:** AGPL-3.0 (same as official ultralytics)
- **Fork Author:** Levi Pereira (@levipereira)

### Code Used

**Approximately 600 lines of custom code** from the fork, specifically:

1. **`export_onnx_trt()` method** (~365 lines)
   - Location: `ultralytics/engine/exporter.py` (lines 460-592 in fork)
   - Purpose: Adds TensorRT EfficientNMS plugin integration to ONNX export graph
   - Enables GPU-accelerated Non-Maximum Suppression (NMS) embedded in model

2. **TensorRT Custom Operators** (~280 lines)
   - `TRT_EfficientNMS` class (torch.autograd.Function)
   - `TRT_EfficientNMS_85` variant (80 classes + 5 additional outputs)
   - `TRT_EfficientNMSX` variant (extended functionality)
   - Location: Lines 1355-1647 in fork
   - Purpose: PyTorch operators that map to TensorRT's EfficientNMS plugin

3. **`End2End_TRT` wrapper class**
   - Wraps YOLO model with NMS layer for end-to-end inference
   - Enables single-pass GPU inference without CPU post-processing

### What This Enables

**Track C (TensorRT + GPU NMS):**
- Embeds Non-Maximum Suppression directly into TensorRT engine
- Eliminates CPU post-processing bottleneck
- Achieves **4x performance improvement** over standard TensorRT (Track B)
- Achieves **2-5x speedup** by avoiding CPU↔GPU memory transfers

**Performance Impact:**
- Track B (TensorRT + CPU NMS): 300-400 rps
- Track C (TensorRT + GPU NMS): 600-800 rps

### Fork Status

As of November 2025:
- Fork is **210 versions behind** official ultralytics (8.3.18 vs 8.3.228)
- Fork provides critical functionality not available in official repository
- Custom operators stable and production-tested by fork maintainer

### Usage in This Project

The fork's end2end export functionality is used to generate:
- `models/yolov11_small_trt_end2end/` - Track C model with GPU NMS
- `models/yolov11_small_gpu_e2e*/` - Track D DALI+TRT models with GPU NMS

See [docs/Attribution/END2END_ANALYSIS.md](docs/Attribution/END2END_ANALYSIS.md) for detailed technical analysis.

---

## NVIDIA TensorRT EfficientNMS Plugin

The end2end models use NVIDIA's **TensorRT EfficientNMS plugin** for GPU-accelerated post-processing.

### Information
- **Provider:** NVIDIA Corporation
- **Documentation:** https://docs.nvidia.com/deeplearning/tensorrt/
- **Plugin:** EfficientNMS_TRT
- **Purpose:** GPU-accelerated Non-Maximum Suppression
- **License:** NVIDIA Deep Learning Software License

### Integration
- Embedded via levipereira/ultralytics fork (see above)
- Compiled into TensorRT engine at model build time
- Executes entirely on GPU, eliminating CPU bottleneck

---

## Reference Architectures

The following repositories were used as **reference only** (no code directly copied):

### 1. levipereira/triton-server-yolo
- **URL:** https://github.com/levipereira/triton-server-yolo
- **Usage:** Reference architecture for deploying end2end YOLO models on Triton Inference Server
- **What We Learned:**
  - Ensemble model configuration patterns
  - DALI preprocessing integration with Triton
  - Dynamic batching configuration for YOLO workloads
- **License:** Not specified in repository
- **Author:** Levi Pereira (@levipereira)

### 2. omarabid59/yolov8-triton
- **URL:** https://github.com/omarabid59/yolov8-triton
- **Usage:** Reference for Triton ensemble patterns and model repository structure
- **What We Learned:**
  - Triton model repository conventions
  - Ensemble preprocessing/inference/postprocessing patterns
- **License:** Not specified in repository

### 3. NVIDIA Triton Inference Server
- **URL:** https://github.com/triton-inference-server/server
- **Documentation:** https://docs.nvidia.com/deeplearning/triton-inference-server/
- **Usage:** Official Triton documentation and examples
- **License:** BSD 3-Clause License

### 4. NVIDIA DALI
- **URL:** https://github.com/NVIDIA/DALI
- **Documentation:** https://docs.nvidia.com/deeplearning/dali/
- **Usage:** GPU-accelerated preprocessing for Track D
- **License:** Apache License 2.0

---

## Acknowledgments

Special thanks to:

- **Levi Pereira (@levipereira)** - For the ultralytics fork with end2end TensorRT export and the triton-server-yolo reference architecture
- **Ultralytics Team** - For the YOLO models and official ultralytics library
- **NVIDIA Corporation** - For Triton Inference Server, TensorRT, and DALI
- **Omar Abid (@omarabid59)** - For the yolov8-triton reference implementation

---

## License Compliance

### This Project
This project's original code is licensed under **MIT License** (see [LICENSE](LICENSE)).

### Third-Party Components

| Component | License | Attribution Required |
|-----------|---------|---------------------|
| levipereira/ultralytics | AGPL-3.0 | ✓ Yes (this file) |
| Ultralytics YOLO | AGPL-3.0 | ✓ Yes (inherited) |
| NVIDIA Triton | BSD 3-Clause | ✓ Yes |
| NVIDIA TensorRT | NVIDIA DSLA | ✓ Yes |
| NVIDIA DALI | Apache 2.0 | ✓ Yes |

**Note:** The use of AGPL-3.0 licensed code (ultralytics fork) may impose obligations on derivative works. Consult the AGPL-3.0 license for details: https://www.gnu.org/licenses/agpl-3.0.en.html

---

## Contact and Questions

For questions about attribution or licensing:
1. Review the detailed analysis in [docs/Attribution/](docs/Attribution/)
2. Consult the original repository licenses linked above
3. For fork-specific questions, contact the fork maintainer: https://github.com/levipereira

---

**Last Updated:** November 2025
