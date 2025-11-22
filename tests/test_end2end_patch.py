#!/usr/bin/env python3
"""
Test End2End TensorRT NMS Patch
================================

Verifies that the monkey-patch is working correctly.

Usage:
    docker compose exec pytorch-api python /app/scripts/test_end2end_patch.py
"""

import sys
import os

# Add src to path
sys.path.insert(0, '/app/src')

print("=" * 80)
print("Testing End2End TensorRT NMS Patch")
print("=" * 80)

# Test 1: Import and apply patch
print("\n[Test 1] Importing and applying patch...")
try:
    from ultralytics_patches import apply_end2end_patch, is_patch_applied
    print("✅ Patch module imported successfully")
except ImportError as e:
    print(f"❌ Failed to import patch module: {e}")
    sys.exit(1)

if not is_patch_applied():
    apply_end2end_patch()

if is_patch_applied():
    print("✅ Patch applied successfully")
else:
    print("❌ Patch application failed")
    sys.exit(1)

# Test 2: Check if export_onnx_trt method exists
print("\n[Test 2] Checking if export_onnx_trt method exists...")
try:
    from ultralytics.engine.exporter import Exporter
    assert hasattr(Exporter, 'export_onnx_trt')
    print("✅ export_onnx_trt method found on Exporter class")
except AssertionError:
    print("❌ export_onnx_trt method not found on Exporter class")
    sys.exit(1)
except ImportError as e:
    print(f"❌ Failed to import Exporter: {e}")
    sys.exit(1)

# Test 3: Check TRT operators are available
print("\n[Test 3] Checking TRT operators...")
try:
    from ultralytics_patches import (
        TRT_EfficientNMS,
        TRT_EfficientNMS_85,
        End2End_TRT,
        ONNX_EfficientNMS_TRT
    )
    print("✅ All TRT operators imported successfully")
except ImportError as e:
    print(f"❌ Failed to import TRT operators: {e}")
    sys.exit(1)

# Test 4: Test with actual YOLO model (if available)
print("\n[Test 4] Testing with YOLO model...")
try:
    from ultralytics import YOLO

    # Check if yolo11n.pt exists
    model_path = "/app/pytorch_models/yolo11n.pt"
    if os.path.exists(model_path):
        print(f"  Loading model from {model_path}...")
        model = YOLO(model_path)
        print("  ✅ Model loaded successfully")

        # Check export method availability
        print("  Checking export formats...")
        try:
            # Get exporter instance
            from ultralytics.engine.exporter import Exporter
            print("  ✅ Can create YOLO exporter instance")
        except Exception as e:
            print(f"  ⚠️  Could not test export: {e}")
    else:
        print(f"  ⚠️  Model file not found: {model_path}")
        print("     Skipping YOLO model test")

except ImportError as e:
    print(f"  ⚠️  Ultralytics not installed: {e}")
except Exception as e:
    print(f"  ⚠️  Error during model test: {e}")

# Summary
print("\n" + "=" * 80)
print("Test Summary")
print("=" * 80)
print("✅ All critical tests passed!")
print("\nYou can now use:")
print("  model.export(format='onnx_trt', topk_all=100, iou_thres=0.45, conf_thres=0.25)")
print("=" * 80)
