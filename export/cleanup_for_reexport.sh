#!/bin/bash
# Cleanup Script - Prepare for Clean Re-Export
# ==============================================
#
# This script removes old model exports and archived scripts to prepare
# for a clean re-export using the new unified export_models.py script.
#
# What it does:
# 1. Backs up config.pbtxt files (we want to keep these)
# 2. Removes old ONNX and TRT model files
# 3. Archives old export scripts
# 4. Clears TRT cache
#
# Usage:
#   bash scripts/cleanup_for_reexport.sh

set -e  # Exit on error

echo "========================================================================"
echo "Cleanup for Re-Export"
echo "========================================================================"

# Check if we're in the right directory
if [ ! -d "models" ] || [ ! -d "scripts" ]; then
    echo "Error: Must run from repository root"
    exit 1
fi

echo ""
echo "Step 1: Backing up all config.pbtxt files..."
echo "------------------------------------------------------------------------"

mkdir -p models/backup_configs_$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="models/backup_configs_$(date +%Y%m%d_%H%M%S)"

for config in models/*/config.pbtxt; do
    if [ -f "$config" ]; then
        model_name=$(dirname "$config" | xargs basename)
        echo "  Backing up: $config → $BACKUP_DIR/${model_name}_config.pbtxt"
        cp "$config" "$BACKUP_DIR/${model_name}_config.pbtxt"
    fi
done

echo "✓ Configs backed up to: $BACKUP_DIR"

echo ""
echo "Step 2: Removing old model files (keeping configs)..."
echo "------------------------------------------------------------------------"

# Remove ONNX models (keep version directory structure)
for model_dir in models/yolov11_nano models/yolov11_small models/yolov11_medium; do
    if [ -d "$model_dir/1" ]; then
        echo "  Removing ONNX from: $model_dir/1/"
        rm -f "$model_dir/1/model.onnx"
        rm -f "$model_dir/1/model.onnx.old"
        ls -lh "$model_dir/1/" 2>/dev/null || echo "    (empty)"
    fi
done

# Remove TRT engine files
for model_dir in models/yolov11_nano_trt models/yolov11_small_trt models/yolov11_medium_trt; do
    if [ -d "$model_dir/1" ]; then
        echo "  Removing TRT engine from: $model_dir/1/"
        rm -f "$model_dir/1/model.plan"
        rm -f "$model_dir/1/model.plan.old"
        rm -f "$model_dir/1/model.engine"
        ls -lh "$model_dir/1/" 2>/dev/null || echo "    (empty)"
    fi
done

echo "✓ Old model files removed"

echo ""
echo "Step 3: Archiving old export scripts..."
echo "------------------------------------------------------------------------"

mkdir -p scripts/archived

if [ -f "scripts/export_all_formats.py" ]; then
    echo "  Moving: export_all_formats.py → scripts/archived/"
    mv scripts/export_all_formats.py scripts/archived/
fi

if [ -f "scripts/export_end2end_all.py" ]; then
    echo "  Moving: export_end2end_all.py → scripts/archived/"
    mv scripts/export_end2end_all.py scripts/archived/
fi

echo "✓ Old scripts archived"

echo ""
echo "Step 4: Clearing TRT cache..."
echo "------------------------------------------------------------------------"

if [ -d "trt_cache" ]; then
    echo "  Clearing: trt_cache/*"
    rm -rf trt_cache/*
    echo "✓ TRT cache cleared"
else
    echo "  (no trt_cache directory found)"
fi

echo ""
echo "========================================================================"
echo "Cleanup Summary"
echo "========================================================================"
echo ""
echo "✓ Config files backed up to: $BACKUP_DIR"
echo "✓ Old ONNX models removed from:"
echo "    - models/yolov11_nano/1/"
echo "    - models/yolov11_small/1/"
echo "    - models/yolov11_medium/1/"
echo ""
echo "✓ Old TRT engines removed from:"
echo "    - models/yolov11_nano_trt/1/"
echo "    - models/yolov11_small_trt/1/"
echo "    - models/yolov11_medium_trt/1/"
echo ""
echo "✓ Old export scripts archived to: scripts/archived/"
echo ""
echo "✓ TRT cache cleared"
echo ""
echo "========================================================================"
echo "Current Model Repository Structure"
echo "========================================================================"
echo ""

for dir in models/yolov11_*; do
    if [ -d "$dir" ]; then
        model_name=$(basename "$dir")
        has_onnx=""
        has_plan=""
        has_config=""

        [ -f "$dir/1/model.onnx" ] && has_onnx="✓" || has_onnx="✗"
        [ -f "$dir/1/model.plan" ] && has_plan="✓" || has_plan="✗"
        [ -f "$dir/config.pbtxt" ] && has_config="✓" || has_config="⚠️"

        printf "%-35s ONNX: %s  Plan: %s  Config: %s\n" "$model_name" "$has_onnx" "$has_plan" "$has_config"
    fi
done

echo ""
echo "========================================================================"
echo "Ready to Export!"
echo "========================================================================"
echo ""
echo "Now you can run a clean export:"
echo ""
echo "  # Export all formats (recommended for testing)"
echo "  docker compose exec yolo-api python /app/export/export_models.py"
echo ""
echo "  # Or export specific formats"
echo "  docker compose exec yolo-api python /app/export/export_models.py --formats onnx_end2end trt_end2end"
echo ""
echo "========================================================================"
