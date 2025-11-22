#!/usr/bin/env python3
"""
Unified YOLO Model Export Script
=================================

Export YOLO models in multiple formats for Triton deployment.

Formats:
--------
1. onnx         - Standard ONNX (no NMS, needs CPU post-processing)
2. trt          - Native TensorRT engine (no NMS, needs CPU post-processing)
3. onnx_end2end - ONNX with TensorRT EfficientNMS operators (GPU NMS)
4. trt_end2end  - TRT engine with compiled NMS (fastest, built from onnx_end2end)
5. all          - Export all formats (default)

Usage:
------
# Export all formats (default)
docker compose exec yolo-api python /app/export/export_models.py

# Export only end2end models
docker compose exec yolo-api python /app/export/export_models.py --formats onnx_end2end trt_end2end

# Export specific models
docker compose exec yolo-api python /app/export/export_models.py --models nano small

# Export everything
docker compose exec yolo-api python /app/export/export_models.py --formats all
"""

import sys
sys.path.insert(0, '/app/src')

import argparse
from pathlib import Path
import shutil
import json
import os

# Apply end2end patch for onnx_trt format
from ultralytics_patches import apply_end2end_patch
apply_end2end_patch()

from ultralytics import YOLO
import torch

# ============================================================================
# Configuration
# ============================================================================

MODELS = {
    "nano": {
        "pt_file": "/app/pytorch_models/yolo11n.pt",
        "triton_name": "yolov11_nano",
        "max_batch": 128,  # Nano is small, A6000 can handle this
        "topk": 100
    },
    "small": {
        "pt_file": "/app/pytorch_models/yolo11s.pt",
        "triton_name": "yolov11_small",
        "max_batch": 64,
        "topk": 100
    },
    "medium": {
        "pt_file": "/app/pytorch_models/yolo11m.pt",
        "triton_name": "yolov11_medium",
        "max_batch": 32,
        "topk": 100
    },
}

# Export settings
IMG_SIZE = 640
DEVICE = 0  # GPU 0
HALF = True  # FP16 precision

# NMS settings (for end2end exports)
IOU_THRESHOLD = 0.45
CONF_THRESHOLD = 0.25

# ============================================================================
# Export Functions
# ============================================================================

def export_onnx_standard(model, config, model_id):
    """Export standard ONNX (no NMS)"""
    print(f"\n{'─' * 80}")
    print(f"[1/4] Standard ONNX Export (ONNX Runtime + TensorRT EP)")
    print(f"{'─' * 80}")

    try:
        print(f"  Exporting ONNX with dynamic batching...")
        print(f"  - dynamic=True: Variable batch, height, width")
        print(f"  - simplify=True: Optimizes graph for TensorRT")

        onnx_path = model.export(
            format="onnx",
            imgsz=IMG_SIZE,
            device=DEVICE,
            dynamic=True,
            simplify=True,
            half=HALF,
            verbose=False
        )

        # Move to Triton model repository
        triton_name = config["triton_name"]
        onnx_model_dir = Path(f"/app/models/{triton_name}/1")
        onnx_model_dir.mkdir(parents=True, exist_ok=True)

        onnx_dest = onnx_model_dir / "model.onnx"
        if onnx_dest.exists():
            shutil.move(onnx_dest, onnx_dest.with_suffix(".onnx.old"))

        shutil.copy2(onnx_path, onnx_dest)
        print(f"  ✓ ONNX saved to: {onnx_dest}")
        print(f"    Output: [84, 8400] - needs CPU NMS post-processing")

        return {
            "status": "success",
            "path": str(onnx_dest),
            "host_path": str(onnx_dest).replace("/app/models", "./models"),
            "output_format": "[84, 8400] - raw detections"
        }
    except Exception as e:
        print(f"  ✗ ONNX export failed: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}


def export_trt_standard(model, config, model_id):
    """Export native TensorRT engine from standard ONNX (no NMS)"""
    print(f"\n{'─' * 80}")
    print(f"[2/4] Standard TensorRT Engine Export (no NMS)")
    print(f"{'─' * 80}")

    try:
        import tensorrt as trt
        import gc

        triton_name = config["triton_name"]

        # First, check if we have standard ONNX, if not export it
        onnx_path = Path(f"/app/models/{triton_name}/1/model.onnx")
        if not onnx_path.exists():
            print(f"  Standard ONNX not found, exporting first...")
            result = export_onnx_standard(model, config, model_id)
            if result.get("status") != "success":
                return result

        print(f"  Using standard ONNX: {onnx_path}")
        print(f"  Building TensorRT engine using TensorRT Python API...")

        max_batch = config["max_batch"]
        workspace_gb = 4
        print(f"  - Dynamic batch: min=1, opt={max_batch//2}, max={max_batch}")
        print(f"  - Workspace: {workspace_gb}GB")
        print(f"  - Precision: {'FP16' if HALF else 'FP32'}")
        print(f"  This will take 5-10 minutes...")

        # Setup TensorRT builder
        logger = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(logger, '')
        builder = trt.Builder(logger)
        config_trt = builder.create_builder_config()

        # Set workspace size
        workspace_bytes = int(workspace_gb * (1 << 30))
        is_trt10 = int(trt.__version__.split(".")[0]) >= 10
        if is_trt10:
            config_trt.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
        else:
            config_trt.max_workspace_size = workspace_bytes

        # Create network
        flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(flag)

        # Parse ONNX
        print(f"\n  Parsing ONNX model...")
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(str(onnx_path)):
            print(f"  ✗ Failed to parse ONNX:")
            for i in range(parser.num_errors):
                print(f"    {parser.get_error(i)}")
            return {"status": "error", "error": "ONNX parsing failed"}

        # Set optimization profile
        print(f"\n  Setting optimization profile...")
        profile = builder.create_optimization_profile()
        min_shape = (1, 3, IMG_SIZE, IMG_SIZE)
        opt_shape = (max_batch // 2, 3, IMG_SIZE, IMG_SIZE)
        max_shape = (max_batch, 3, IMG_SIZE, IMG_SIZE)

        for i in range(network.num_inputs):
            inp = network.get_input(i)
            profile.set_shape(inp.name, min=min_shape, opt=opt_shape, max=max_shape)

        config_trt.add_optimization_profile(profile)

        # Set FP16
        if HALF and builder.platform_has_fast_fp16:
            print(f"  Enabling FP16 precision...")
            config_trt.set_flag(trt.BuilderFlag.FP16)

        # Free memory
        gc.collect()
        torch.cuda.empty_cache()

        # Build engine
        print(f"\n  Building TensorRT engine (this may take 5-10 minutes)...")
        trt_model_dir = Path(f"/app/models/{triton_name}_trt/1")
        trt_model_dir.mkdir(parents=True, exist_ok=True)
        trt_dest = trt_model_dir / "model.plan"

        if trt_dest.exists():
            backup = trt_dest.with_suffix(".plan.old")
            if backup.exists():
                backup.unlink()
            shutil.move(trt_dest, backup)

        # Build and serialize
        if is_trt10:
            serialized_engine = builder.build_serialized_network(network, config_trt)
            if serialized_engine is None:
                return {"status": "error", "error": "Failed to build engine"}

            # Write pure TensorRT engine (no metadata - Triton can't read it)
            with open(trt_dest, "wb") as f:
                f.write(serialized_engine)
        else:
            engine = builder.build_engine(network, config_trt)
            if engine is None:
                return {"status": "error", "error": "Failed to build engine"}

            # Write pure TensorRT engine (no metadata - Triton can't read it)
            with open(trt_dest, "wb") as f:
                f.write(engine.serialize())

        print(f"\n  ✓ TensorRT Standard engine saved to: {trt_dest}")
        print(f"    File size: {trt_dest.stat().st_size / (1024*1024):.2f} MB")
        print(f"    Output format: [84, 8400] - raw detections (CPU NMS needed)")

        return {
            "status": "success",
            "path": str(trt_dest),
            "host_path": str(trt_dest).replace("/app/models", "./models"),
            "output_format": "[84, 8400] - raw detections"
        }
    except Exception as e:
        print(f"  ✗ TensorRT export failed: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}


def export_onnx_end2end(model, config, model_id):
    """Export ONNX with TensorRT EfficientNMS operators (GPU NMS)"""
    print(f"\n{'─' * 80}")
    print(f"[3/4] ONNX End2End Export (with GPU NMS operators)")
    print(f"{'─' * 80}")

    try:
        from ultralytics.engine.exporter import Exporter
        from ultralytics.cfg import get_cfg

        topk = config["topk"]
        print(f"  Exporting ONNX with EfficientNMS plugin...")
        print(f"  - Calling export_onnx_trt() directly (bypasses format validation)")
        print(f"  - topk_all={topk}: Max detections")
        print(f"  - iou_thres={IOU_THRESHOLD}: NMS IoU threshold")
        print(f"  - conf_thres={CONF_THRESHOLD}: Confidence threshold")
        print(f"  This will take 1-2 minutes...")

        # Create export arguments with standard ONNX format
        # We'll call export_onnx_trt() directly instead of using the format system
        args = get_cfg(overrides={
            'format': 'onnx',  # Use standard format to pass validation
            'imgsz': IMG_SIZE,
            'dynamic': True,
            'simplify': True,
            'half': HALF,
            'device': DEVICE,
            'opset': 17  # ONNX opset version
        })

        # Manually add custom End2End arguments
        args.topk_all = topk
        args.iou_thres = IOU_THRESHOLD
        args.conf_thres = CONF_THRESHOLD
        args.class_agnostic = False
        args.mask_resolution = 56
        args.pooler_scale = 0.25
        args.sampling_ratio = 0

        # Create exporter
        exporter = Exporter(cfg=args, _callbacks=model.callbacks)
        exporter.args = args

        # Set up exporter (mimics what __call__ does)
        exporter.model = model.model.to(DEVICE)  # Move model to GPU
        if hasattr(model, 'overrides'):
            exporter.model.args = model.overrides

        # Dummy input - create input tensor (not model output!)
        exporter.im = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
        exporter.file = Path(config["pt_file"])

        # Call export_onnx_trt directly (bypasses format routing)
        print(f"  Calling patched export_onnx_trt() method...")
        export_path, onnx_model = exporter.export_onnx_trt(prefix="ONNX TRT:")

        # Move to Triton model repository
        triton_name = config["triton_name"]
        onnx_model_dir = Path(f"/app/models/{triton_name}_end2end/1")
        onnx_model_dir.mkdir(parents=True, exist_ok=True)

        onnx_dest = onnx_model_dir / "model.onnx"

        export_file = Path(export_path)
        if export_file.exists():
            if onnx_dest.exists():
                shutil.move(onnx_dest, onnx_dest.with_suffix(".onnx.old"))

            shutil.copy2(export_file, onnx_dest)
            print(f"  ✓ ONNX End2End saved to: {onnx_dest}")
            print(f"    Contains: TRT::EfficientNMS_TRT operators")
            print(f"    Output: num_dets, det_boxes, det_scores, det_classes")

            # Verify NMS plugin
            try:
                import onnx
                onnx_model = onnx.load(str(onnx_dest))
                ops = [node.op_type for node in onnx_model.graph.node]
                has_nms = any('NMS' in op or 'TRT' in op for op in ops)
                if has_nms:
                    print(f"  ✓ Verified: NMS plugin in ONNX graph")
                else:
                    print(f"  ⚠️  Warning: No NMS operators found!")
            except:
                pass

            return {
                "status": "success",
                "path": str(onnx_dest),
                "host_path": str(onnx_dest).replace("/app/models", "./models"),
                "has_nms": True,
                "output_format": "num_dets, det_boxes, det_scores, det_classes"
            }
        else:
            print(f"  ✗ Export failed, file not found: {export_file}")
            return {"status": "error", "error": "Export file not found"}

    except Exception as e:
        print(f"  ✗ ONNX End2End export failed: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}


def export_trt_end2end(model, config, model_id):
    """Export TensorRT engine from end2end ONNX (compiled GPU NMS)"""
    print(f"\n{'─' * 80}")
    print(f"[4/4] TensorRT End2End Export (compiled GPU NMS)")
    print(f"{'─' * 80}")

    try:
        import tensorrt as trt
        import gc

        # First, ensure we have the end2end ONNX
        triton_name = config["triton_name"]
        onnx_end2end_path = Path(f"/app/models/{triton_name}_end2end/1/model.onnx")

        if not onnx_end2end_path.exists():
            print(f"  ⚠️  End2End ONNX not found: {onnx_end2end_path}")
            print(f"     Run with --formats onnx_end2end first!")
            return {"status": "error", "error": "End2End ONNX not found"}

        print(f"  Using end2end ONNX: {onnx_end2end_path}")
        print(f"  Building TensorRT engine using TensorRT Python API...")

        max_batch = config["max_batch"]
        workspace_gb = 4
        print(f"  - Dynamic batch: min=1, opt={max_batch//2}, max={max_batch}")
        print(f"  - Workspace: {workspace_gb}GB")
        print(f"  - Precision: {'FP16' if HALF else 'FP32'}")
        print(f"  This will take 5-10 minutes...")

        # Setup TensorRT builder (following Ultralytics pattern from reference repo)
        logger = trt.Logger(trt.Logger.INFO)

        # Initialize TensorRT plugins (required for EfficientNMS)
        trt.init_libnvinfer_plugins(logger, '')
        print(f"  ✓ TensorRT plugins initialized (EfficientNMS available)")

        builder = trt.Builder(logger)
        config_trt = builder.create_builder_config()

        # Set workspace size (GB -> bytes)
        workspace_bytes = int(workspace_gb * (1 << 30))
        is_trt10 = int(trt.__version__.split(".")[0]) >= 10
        if is_trt10:
            config_trt.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
        else:
            config_trt.max_workspace_size = workspace_bytes

        # Create network with explicit batch flag
        flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(flag)

        # Parse ONNX file
        print(f"\n  Parsing ONNX model...")
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(str(onnx_end2end_path)):
            print(f"  ✗ Failed to parse ONNX:")
            for i in range(parser.num_errors):
                print(f"    {parser.get_error(i)}")
            return {"status": "error", "error": "ONNX parsing failed"}

        # Log inputs/outputs
        print(f"\n  Network inputs:")
        for i in range(network.num_inputs):
            inp = network.get_input(i)
            print(f"    - {inp.name}: shape={inp.shape}, dtype={inp.dtype}")

        print(f"\n  Network outputs:")
        for i in range(network.num_outputs):
            out = network.get_output(i)
            print(f"    - {out.name}: shape={out.shape}, dtype={out.dtype}")

        # Set optimization profile for dynamic batching
        print(f"\n  Setting optimization profile...")
        profile = builder.create_optimization_profile()

        # Dynamic batch axis: min=1, opt=max_batch//2, max=max_batch
        # Fixed spatial: 640x640
        min_shape = (1, 3, IMG_SIZE, IMG_SIZE)
        opt_shape = (max_batch // 2, 3, IMG_SIZE, IMG_SIZE)
        max_shape = (max_batch, 3, IMG_SIZE, IMG_SIZE)

        for i in range(network.num_inputs):
            inp = network.get_input(i)
            profile.set_shape(inp.name, min=min_shape, opt=opt_shape, max=max_shape)
            print(f"    {inp.name}: min={min_shape}, opt={opt_shape}, max={max_shape}")

        config_trt.add_optimization_profile(profile)

        # Set FP16 precision
        if HALF and builder.platform_has_fast_fp16:
            print(f"\n  Enabling FP16 precision...")
            config_trt.set_flag(trt.BuilderFlag.FP16)
        else:
            print(f"\n  Using FP32 precision (FP16 not available or disabled)")

        # Free memory before building
        gc.collect()
        torch.cuda.empty_cache()

        # Build engine
        print(f"\n  Building TensorRT engine (this may take 5-10 minutes)...")
        trt_model_dir = Path(f"/app/models/{triton_name}_trt_end2end/1")
        trt_model_dir.mkdir(parents=True, exist_ok=True)
        trt_dest = trt_model_dir / "model.plan"

        # Backup old engine if exists
        if trt_dest.exists():
            backup = trt_dest.with_suffix(".plan.old")
            if backup.exists():
                backup.unlink()
            shutil.move(trt_dest, backup)

        # Build and serialize
        if is_trt10:
            # TensorRT 10+
            serialized_engine = builder.build_serialized_network(network, config_trt)
            if serialized_engine is None:
                return {"status": "error", "error": "Failed to build engine"}

            # Write pure TensorRT engine (no metadata - Triton can't read it)
            with open(trt_dest, "wb") as f:
                f.write(serialized_engine)
        else:
            # TensorRT 7-9
            engine = builder.build_engine(network, config_trt)
            if engine is None:
                return {"status": "error", "error": "Failed to build engine"}

            # Write pure TensorRT engine (no metadata - Triton can't read it)
            with open(trt_dest, "wb") as f:
                f.write(engine.serialize())

        print(f"\n  ✓ TensorRT End2End engine saved to: {trt_dest}")
        print(f"    File size: {trt_dest.stat().st_size / (1024*1024):.2f} MB")
        print(f"    Contains: Compiled GPU NMS (EfficientNMS)")
        print(f"    Output format: num_dets, det_boxes, det_scores, det_classes")

        return {
            "status": "success",
            "path": str(trt_dest),
            "host_path": str(trt_dest).replace("/app/models", "./models"),
            "has_nms": True,
            "output_format": "num_dets, det_boxes, det_scores, det_classes",
            "source": "end2end_onnx"
        }
    except Exception as e:
        print(f"  ✗ TensorRT End2End export failed: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}


# ============================================================================
# Main Export Logic
# ============================================================================

def export_model(model_id, config, formats):
    """Export a single model in specified formats"""
    print(f"\n{'═' * 80}")
    print(f"Processing {model_id} → {config['triton_name']}")
    print(f"{'═' * 80}")

    pt_file = config["pt_file"]

    if not Path(pt_file).exists():
        print(f"✗ Error: Model file not found: {pt_file}")
        return {
            "model": model_id,
            "status": "error",
            "error": "Model file not found"
        }

    results = {
        "model": model_id,
        "triton_name": config["triton_name"],
        "max_batch": config["max_batch"],
        "topk": config["topk"]
    }

    # Load model once
    model = YOLO(pt_file)

    # Export in requested formats
    if "onnx" in formats or "all" in formats:
        results["onnx"] = export_onnx_standard(model, config, model_id)

    if "trt" in formats or "all" in formats:
        # Reload model for clean state
        model = YOLO(pt_file)
        results["trt"] = export_trt_standard(model, config, model_id)

    if "onnx_end2end" in formats or "all" in formats:
        # Reload model for clean state
        model = YOLO(pt_file)
        results["onnx_end2end"] = export_onnx_end2end(model, config, model_id)

    if "trt_end2end" in formats or "all" in formats:
        # This uses the end2end ONNX, doesn't need model reload
        results["trt_end2end"] = export_trt_end2end(model, config, model_id)

    return results


def print_summary(results, formats):
    """Print export summary"""
    print(f"\n{'═' * 80}")
    print("Export Summary")
    print(f"{'═' * 80}\n")

    for result in results:
        print(f"{result['model']} ({result['triton_name']}):")

        if "onnx" in result:
            status = "✓" if result["onnx"]["status"] == "success" else "✗"
            print(f"  {status} ONNX Standard: {result['onnx']['status']}")
            if result["onnx"]["status"] == "success":
                print(f"     Host: {result['onnx']['host_path']}")

        if "trt" in result:
            status = "✓" if result["trt"]["status"] == "success" else "✗"
            print(f"  {status} TRT Standard: {result['trt']['status']}")
            if result["trt"]["status"] == "success":
                print(f"     Host: {result['trt']['host_path']}")

        if "onnx_end2end" in result:
            status = "✓" if result["onnx_end2end"]["status"] == "success" else "✗"
            print(f"  {status} ONNX End2End: {result['onnx_end2end']['status']}")
            if result["onnx_end2end"]["status"] == "success":
                print(f"     Host: {result['onnx_end2end']['host_path']}")
                print(f"     NMS: GPU (TRT operators)")

        if "trt_end2end" in result:
            status = "✓" if result["trt_end2end"]["status"] == "success" else "✗"
            print(f"  {status} TRT End2End: {result['trt_end2end']['status']}")
            if result["trt_end2end"]["status"] == "success":
                print(f"     Host: {result['trt_end2end']['host_path']}")
                print(f"     NMS: Compiled (maximum performance!)")

        print()

    print(f"{'═' * 80}")
    print("Format Comparison")
    print(f"{'═' * 80}")
    print("\n1. ONNX Standard:")
    print("   Output: [84, 8400] → CPU NMS needed")
    print("   Speed: Baseline inference, +5-10ms NMS on CPU")

    print("\n2. TRT Standard:")
    print("   Output: [84, 8400] → CPU NMS needed")
    print("   Speed: 1.5x faster inference, +5-10ms NMS on CPU")

    print("\n3. ONNX End2End:")
    print("   Output: num_dets, det_boxes, det_scores, det_classes")
    print("   Speed: 2-3x faster (GPU NMS via TensorRT EP)")

    print("\n4. TRT End2End:")
    print("   Output: num_dets, det_boxes, det_scores, det_classes")
    print("   Speed: 3-5x faster (compiled GPU NMS in engine)")
    print(f"{'═' * 80}")


def main():
    parser = argparse.ArgumentParser(
        description="Export YOLO models in multiple formats for Triton",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export all formats (default)
  python export_models.py

  # Export only end2end models
  python export_models.py --formats onnx_end2end trt_end2end

  # Export specific models
  python export_models.py --models nano small

  # Export standard formats only
  python export_models.py --formats onnx trt
        """
    )

    parser.add_argument(
        '--formats',
        nargs='+',
        choices=['onnx', 'trt', 'onnx_end2end', 'trt_end2end', 'all'],
        default=['all'],
        help='Formats to export (default: all)'
    )

    parser.add_argument(
        '--models',
        nargs='+',
        choices=list(MODELS.keys()),
        default=list(MODELS.keys()),
        help='Models to export (default: all models)'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("YOLO11 Unified Export Script")
    print("=" * 80)
    print(f"Models: {', '.join(args.models)}")
    print(f"Formats: {', '.join(args.formats)}")
    print(f"Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"Device: cuda:{DEVICE}")
    print(f"Precision: FP16")
    if "onnx_end2end" in args.formats or "trt_end2end" in args.formats or "all" in args.formats:
        print(f"NMS IoU threshold: {IOU_THRESHOLD}")
        print(f"Confidence threshold: {CONF_THRESHOLD}")
    print("=" * 80)

    results = []

    for model_id in args.models:
        config = MODELS[model_id]
        result = export_model(model_id, config, args.formats)
        results.append(result)

    # Print summary
    print_summary(results, args.formats)

    # Save results
    results_file = Path("/app/scripts/export_results.json")
    with open(results_file, 'w') as f:
        json.dump({
            "formats": args.formats,
            "models": args.models,
            "results": results
        }, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    # Show current model structure
    print(f"\n{'═' * 80}")
    print("Current Model Repository")
    print(f"{'═' * 80}")
    models_dir = Path("/app/models")
    if models_dir.exists():
        for model_dir in sorted(models_dir.iterdir()):
            if model_dir.is_dir() and model_dir.name.startswith("yolov11"):
                has_model = (model_dir / "1" / "model.onnx").exists() or (model_dir / "1" / "model.plan").exists()
                has_config = (model_dir / "config.pbtxt").exists()
                status = "✓" if has_model else "✗"
                config_status = "✓" if has_config else "⚠️"
                print(f"{status} {model_dir.name:30} config: {config_status}")
    print(f"{'═' * 80}")


if __name__ == "__main__":
    main()
