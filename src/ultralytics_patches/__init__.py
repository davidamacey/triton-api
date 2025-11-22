"""
Ultralytics Patches
===================

Monkey-patches for extending Ultralytics functionality.

Current patches:
- End2End TensorRT NMS Export: Adds GPU-accelerated NMS to ONNX export

Usage:
    from ultralytics_patches import apply_end2end_patch
    apply_end2end_patch()

    from ultralytics import YOLO
    model = YOLO("yolo11n.pt")
    model.export(format="onnx_trt", topk_all=100, iou_thres=0.45, conf_thres=0.25)
"""

from .end2end_export import (
    apply_end2end_patch,
    is_patch_applied,
    export_onnx_trt,
    End2End_TRT,
    ONNX_EfficientNMS_TRT,
    ONNX_EfficientNMSX_TRT,
    ONNX_End2End_MASK_TRT,
    TRT_EfficientNMS,
    TRT_EfficientNMS_85,
    TRT_EfficientNMSX,
    TRT_EfficientNMSX_85,
    TRT_ROIAlign,
    __version__,
)

__all__ = [
    'apply_end2end_patch',
    'is_patch_applied',
    'export_onnx_trt',
    'End2End_TRT',
    'ONNX_EfficientNMS_TRT',
    'ONNX_EfficientNMSX_TRT',
    'ONNX_End2End_MASK_TRT',
    'TRT_EfficientNMS',
    'TRT_EfficientNMS_85',
    'TRT_EfficientNMSX',
    'TRT_EfficientNMSX_85',
    'TRT_ROIAlign',
]

# Auto-apply patch on import (optional, can be disabled)
AUTO_APPLY = True

if AUTO_APPLY:
    try:
        apply_end2end_patch()
    except ImportError:
        print("⚠️  Ultralytics not installed yet, patch will be applied when you import YOLO")
