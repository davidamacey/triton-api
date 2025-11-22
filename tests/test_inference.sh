#!/bin/bash
# Comprehensive Inference Test Script
# Tests all 4 performance tracks with sample images

# Note: Not using 'set -e' to allow tracking partial successes across all tracks

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Configuration
IMAGE_DIR=${1:-"./test_images"}
MODEL_SIZE=${2:-"small"}
NUM_TESTS=${3:-10}
TRACK_D_VARIANT=${4:-""}  # Optional: streaming, batch, or empty for balanced

echo "============================================================"
echo "Comprehensive Inference Testing - All 4 Tracks"
echo "============================================================"
echo "Image directory: $IMAGE_DIR"
echo "Model size: $MODEL_SIZE"
echo "Number of tests per track: $NUM_TESTS"
echo ""

# Find test images
TEST_IMAGES=($(find "$IMAGE_DIR" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | head -$NUM_TESTS))

if [ ${#TEST_IMAGES[@]} -eq 0 ]; then
    echo -e "${RED}✗ No images found in $IMAGE_DIR${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Found ${#TEST_IMAGES[@]} test images${NC}"
echo ""

# Function to test an endpoint
test_endpoint() {
    local track_name=$1
    local endpoint=$2
    local description=$3

    echo "============================================================"
    echo "Testing $track_name: $description"
    echo "============================================================"
    echo "Endpoint: $endpoint"
    echo ""

    local success=0
    local fail=0
    local total_time=0

    for IMG in "${TEST_IMAGES[@]}"; do
        echo -n "  Testing $(basename $IMG)... "

        # Time the request
        START_TIME=$(date +%s%N)
        RESPONSE=$(curl -sf --max-time 60 -X POST "$endpoint" \
            -F "image=@$IMG" 2>&1)
        END_TIME=$(date +%s%N)

        if [ $? -eq 0 ]; then
            # Calculate time in ms
            TIME_MS=$(( (END_TIME - START_TIME) / 1000000 ))
            total_time=$((total_time + TIME_MS))

            DETECTIONS=$(echo "$RESPONSE" | jq -r '.detections | length' 2>/dev/null || echo "0")
            echo -e "${GREEN}✓${NC} ${DETECTIONS} detections (${TIME_MS}ms)"
            success=$((success + 1))
        else
            echo -e "${RED}✗ Failed${NC}"
            fail=$((fail + 1))
        fi
    done

    # Calculate average latency
    if [ $success -gt 0 ]; then
        avg_time=$((total_time / success))
    else
        avg_time=0
    fi

    echo ""
    echo -e "Results: ${GREEN}$success passed${NC}, ${RED}$fail failed${NC}"
    if [ $success -gt 0 ]; then
        echo -e "Average latency: ${CYAN}${avg_time}ms${NC}"
    fi
    echo ""

    # Return success count (safe now without set -e)
    return $success
}

# Track results
TRACK_A_SUCCESS=0
TRACK_B_SUCCESS=0
TRACK_C_SUCCESS=0
TRACK_D_SUCCESS=0

# Track A: PyTorch (baseline)
test_endpoint "Track A" \
    "http://localhost:9600/pytorch/predict/$MODEL_SIZE" \
    "PyTorch + CPU NMS (baseline)"
TRACK_A_SUCCESS=$?

# Track B: TensorRT + CPU NMS
test_endpoint "Track B" \
    "http://localhost:9600/predict/$MODEL_SIZE" \
    "TensorRT + CPU NMS (2x speedup)"
TRACK_B_SUCCESS=$?

# Track C: TensorRT End2End (GPU NMS)
test_endpoint "Track C" \
    "http://localhost:9600/predict/${MODEL_SIZE}_end2end" \
    "TensorRT + GPU NMS (4x speedup)"
TRACK_C_SUCCESS=$?

# Track D: DALI + TensorRT End2End (full GPU pipeline)
if [ -n "$TRACK_D_VARIANT" ]; then
    # Test specific variant
    test_endpoint "Track D ($TRACK_D_VARIANT)" \
        "http://localhost:9600/predict/${MODEL_SIZE}_gpu_e2e_${TRACK_D_VARIANT}" \
        "DALI + TRT + GPU NMS - $TRACK_D_VARIANT (10-15x speedup)"
    TRACK_D_SUCCESS=$?
else
    # Test balanced variant (default)
    test_endpoint "Track D (balanced)" \
        "http://localhost:9600/predict/${MODEL_SIZE}_gpu_e2e" \
        "DALI + TRT + GPU NMS - balanced (10-15x speedup)"
    TRACK_D_SUCCESS=$?
fi

# Final Summary
echo "============================================================"
echo "FINAL SUMMARY"
echo "============================================================"
echo "Total images tested: ${#TEST_IMAGES[@]}"
echo ""

# Track summaries with colors
if [ $TRACK_A_SUCCESS -eq ${#TEST_IMAGES[@]} ]; then
    TRACK_A_COLOR=$GREEN
else
    TRACK_A_COLOR=$YELLOW
fi

if [ $TRACK_B_SUCCESS -eq ${#TEST_IMAGES[@]} ]; then
    TRACK_B_COLOR=$GREEN
else
    TRACK_B_COLOR=$YELLOW
fi

if [ $TRACK_C_SUCCESS -eq ${#TEST_IMAGES[@]} ]; then
    TRACK_C_COLOR=$GREEN
else
    TRACK_C_COLOR=$YELLOW
fi

if [ $TRACK_D_SUCCESS -eq ${#TEST_IMAGES[@]} ]; then
    TRACK_D_COLOR=$GREEN
else
    TRACK_D_COLOR=$YELLOW
fi

echo -e "${TRACK_A_COLOR}Track A (PyTorch):${NC}        $TRACK_A_SUCCESS/${#TEST_IMAGES[@]} passed"
echo -e "${TRACK_B_COLOR}Track B (TRT):${NC}            $TRACK_B_SUCCESS/${#TEST_IMAGES[@]} passed"
echo -e "${TRACK_C_COLOR}Track C (TRT End2End):${NC}    $TRACK_C_SUCCESS/${#TEST_IMAGES[@]} passed"
echo -e "${TRACK_D_COLOR}Track D (DALI + TRT):${NC}     $TRACK_D_SUCCESS/${#TEST_IMAGES[@]} passed"
echo ""

# Overall status
TOTAL_PASS=$((TRACK_A_SUCCESS + TRACK_B_SUCCESS + TRACK_C_SUCCESS + TRACK_D_SUCCESS))
TOTAL_POSSIBLE=$((${#TEST_IMAGES[@]} * 4))

if [ $TOTAL_PASS -eq $TOTAL_POSSIBLE ]; then
    echo -e "${GREEN}✓ ALL TESTS PASSED!${NC} ($TOTAL_PASS/$TOTAL_POSSIBLE)"
    echo ""
    echo "All 4 tracks are working correctly!"
    exit 0
elif [ $TOTAL_PASS -gt 0 ]; then
    echo -e "${YELLOW}⚠ PARTIAL SUCCESS${NC} ($TOTAL_PASS/$TOTAL_POSSIBLE tests passed)"
    echo ""
    echo "Some tracks have issues. Check the results above."
    exit 1
else
    echo -e "${RED}✗ ALL TESTS FAILED${NC}"
    echo ""
    echo "No tracks are working. Check service status:"
    echo "  docker compose ps"
    echo "  docker compose logs"
    exit 1
fi
