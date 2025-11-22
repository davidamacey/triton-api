#!/bin/bash
#
# Setup and Run Padding Comparison Test
#
# This script automates the setup and execution of the padding methods comparison:
# 1. Creates simple padding DALI pipeline
# 2. Creates ensemble model
# 3. Restarts Triton to load new models
# 4. Runs comparison test
#
# Usage:
#   bash scripts/setup_padding_comparison.sh
#

set -e  # Exit on error

echo "================================================================================"
echo "YOLO Padding Methods Comparison Setup"
echo "================================================================================"
echo ""

# Check if running from correct directory
if [ ! -f "docker-compose.yml" ]; then
    echo "Error: Must run from repository root directory"
    echo "Usage: bash scripts/setup_padding_comparison.sh"
    exit 1
fi

# Step 1: Create simple padding DALI pipeline
echo "Step 1/5: Creating simple padding DALI pipeline..."
echo "--------------------------------------------------------------------------------"
docker compose exec yolo-api python /app/dali/create_dali_simple_padding_pipeline.py
echo ""

# Step 2: Create ensemble model
echo "Step 2/5: Creating ensemble model..."
echo "--------------------------------------------------------------------------------"
docker compose exec yolo-api python /app/dali/create_simple_padding_ensemble.py
echo ""

# Step 3: Verify files were created
echo "Step 3/5: Verifying files..."
echo "--------------------------------------------------------------------------------"
if [ -f "models/yolo_preprocess_dali_simple/1/model.dali" ]; then
    echo "✓ DALI pipeline created: models/yolo_preprocess_dali_simple/1/model.dali"
else
    echo "✗ ERROR: DALI pipeline not found"
    exit 1
fi

if [ -f "models/yolov11_small_simple_padding/config.pbtxt" ]; then
    echo "✓ Ensemble config created: models/yolov11_small_simple_padding/config.pbtxt"
else
    echo "✗ ERROR: Ensemble config not found"
    exit 1
fi
echo ""

# Step 4: Restart Triton to load new models
echo "Step 4/5: Restarting Triton to load new models..."
echo "--------------------------------------------------------------------------------"
docker compose restart triton-api

# Wait for Triton to be ready
echo "Waiting for Triton to be ready..."
sleep 5

# Check if Triton is healthy
for i in {1..30}; do
    if docker compose exec triton-api curl -s http://localhost:8000/v2/health/ready > /dev/null 2>&1; then
        echo "✓ Triton is ready"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "✗ ERROR: Triton failed to become ready after 30 seconds"
        echo "Check logs: docker compose logs triton-api"
        exit 1
    fi
    sleep 1
done

# Verify models are loaded
echo ""
echo "Verifying models are loaded..."
if docker compose logs triton-api 2>&1 | grep -q "Successfully loaded 'yolov11_small_simple_padding'"; then
    echo "✓ Simple padding ensemble model loaded"
else
    echo "⚠ WARNING: Could not verify simple padding model loaded"
    echo "Check logs: docker compose logs triton-api | grep simple_padding"
fi

if docker compose logs triton-api 2>&1 | grep -q "Successfully loaded 'yolo_preprocess_dali_simple'"; then
    echo "✓ Simple padding DALI model loaded"
else
    echo "⚠ WARNING: Could not verify DALI model loaded"
fi
echo ""

# Step 5: Run comparison test
echo "Step 5/5: Running comparison test..."
echo "================================================================================"
echo ""

# Check if benchmark images exist
if [ ! -d "benchmarks/images" ] || [ -z "$(ls -A benchmarks/images/*.jpg 2>/dev/null)" ]; then
    echo "⚠ WARNING: No benchmark images found in benchmarks/images/"
    echo ""
    echo "Please add test images to benchmarks/images/ and run:"
    echo "  docker compose exec yolo-api python /app/tests/compare_padding_methods.py"
    echo ""
    echo "Setup complete!"
    exit 0
fi

# Run the comparison
docker compose exec yolo-api python /app/tests/compare_padding_methods.py

echo ""
echo "================================================================================"
echo "✓ Comparison complete!"
echo "================================================================================"
echo ""
echo "Next steps:"
echo "  1. Review the comparison results above"
echo "  2. Check F1 scores and performance metrics"
echo "  3. Decide which padding method to use:"
echo "     - F1 ≥ 0.99: Use simple padding (faster, simpler)"
echo "     - F1 < 0.95: Use center padding (more accurate)"
echo ""
echo "For detailed results, run with --output-dir:"
echo "  docker compose exec yolo-api python /app/tests/compare_padding_methods.py --output-dir /app/results"
echo ""
echo "For more information, see: docs/PADDING_COMPARISON_GUIDE.md"
echo ""
