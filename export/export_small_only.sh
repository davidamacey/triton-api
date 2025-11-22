#!/bin/bash
# Export ONLY small models for all tracks
# This script simplifies the export process to focus on small model size

set -e

echo "================================================================================"
echo "Simplified Model Export - Small Models Only"
echo "================================================================================"
echo ""
echo "This script exports only yolo11s (small) in all formats:"
echo "  1. Standard TRT (Track B)"
echo "  2. End2End ONNX + TRT (Track C)"
echo "  3. DALI + Ensembles (Track D - created separately)"
echo ""

# Run export script with small model only
docker compose exec yolo-api python /app/export/export_models.py \
    --models small \
    --formats trt trt_end2end

echo ""
echo "================================================================================"
echo "✓ Small model export complete"
echo "================================================================================"
echo ""
echo "Exported models:"
echo "  ✓ yolov11_small_trt (Track B)"
echo "  ✓ yolov11_small_trt_end2end (Track C)"
echo ""
echo "Next steps:"
echo "  1. Create DALI pipeline (creates both model.dali and config.pbtxt):"
echo "     docker compose exec yolo-api python /app/dali/create_dali_letterbox_pipeline.py"
echo ""
echo "  2. Create ensembles (small only):"
echo "     docker compose exec yolo-api python /app/dali/create_ensembles.py --models small"
echo ""
echo "  3. Restart services:"
echo "     docker compose restart triton-api"
echo ""
