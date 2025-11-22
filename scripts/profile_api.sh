#!/bin/bash
###############################################################################
# FastAPI Performance Profiling Script
#
# This script helps identify performance bottlenecks in the FastAPI service
# using py-spy, a sampling profiler for Python programs.
#
# Prerequisites:
#   - py-spy installed: pip install py-spy
#   - API service running
#
# Usage:
#   ./scripts/profile_api.sh [duration_seconds] [output_file]
#
# Examples:
#   ./scripts/profile_api.sh 60 profile.svg          # Profile for 60 seconds
#   ./scripts/profile_api.sh 30 profile.speedscope  # Speedscope format
###############################################################################

set -e

# Configuration
DURATION=${1:-30}
OUTPUT=${2:-"performance_profile.svg"}
CONTAINER_NAME="yolo-api"

echo "======================================"
echo "FastAPI Performance Profiling"
echo "======================================"
echo "Duration: ${DURATION} seconds"
echo "Output: ${OUTPUT}"
echo "Container: ${CONTAINER_NAME}"
echo "======================================"

# Check if container is running
if ! docker ps | grep -q "${CONTAINER_NAME}"; then
    echo "ERROR: Container '${CONTAINER_NAME}' is not running"
    echo "Start the service with: docker compose up -d"
    exit 1
fi

# Get the container's main Python process PID
echo ""
echo "Finding main process..."
CONTAINER_PID=$(docker exec ${CONTAINER_NAME} pgrep -f "uvicorn src.main:app" | head -1)

if [ -z "$CONTAINER_PID" ]; then
    echo "ERROR: Could not find uvicorn process in container"
    exit 1
fi

echo "Found process: PID ${CONTAINER_PID}"
echo ""

# Determine output format
if [[ $OUTPUT == *.svg ]]; then
    FORMAT="flamegraph"
    echo "Generating flamegraph visualization..."
elif [[ $OUTPUT == *.speedscope ]]; then
    FORMAT="speedscope"
    echo "Generating speedscope visualization..."
else
    FORMAT="flamegraph"
    OUTPUT="performance_profile.svg"
    echo "Using default flamegraph format..."
fi

echo ""
echo "Starting profiling (this will take ${DURATION} seconds)..."
echo "Tip: Generate some load during profiling for better insights!"
echo ""

# Run py-spy inside the container
docker exec ${CONTAINER_NAME} py-spy record \
    --pid ${CONTAINER_PID} \
    --duration ${DURATION} \
    --rate 100 \
    --format ${FORMAT} \
    --output /tmp/${OUTPUT} \
    --subprocesses

# Copy output from container to host
docker cp ${CONTAINER_NAME}:/tmp/${OUTPUT} ./${OUTPUT}

echo ""
echo "======================================"
echo "Profiling Complete!"
echo "======================================"
echo "Output saved to: ${OUTPUT}"

if [[ $FORMAT == "flamegraph" ]]; then
    echo ""
    echo "To view the flamegraph:"
    echo "  - Open ${OUTPUT} in a web browser"
    echo "  - Click on functions to zoom in"
    echo "  - Look for wide bars (expensive operations)"
elif [[ $FORMAT == "speedscope" ]]; then
    echo ""
    echo "To view the speedscope profile:"
    echo "  1. Go to https://www.speedscope.app/"
    echo "  2. Drag and drop ${OUTPUT} onto the page"
    echo "  3. Analyze the timeline and call graph"
fi

echo ""
echo "Performance Analysis Tips:"
echo "  1. Wide bars = functions consuming most CPU time"
echo "  2. Deep stacks = nested function calls (potential optimization)"
echo "  3. Look for unexpected bottlenecks outside inference"
echo "  4. Compare profiles before/after optimizations"
echo ""
