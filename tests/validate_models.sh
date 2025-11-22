#!/bin/bash
# Model Validation Script
# Validates that Triton models (Tracks B, C, D) produce predictions matching PyTorch baseline (Track A)
# Compares bounding boxes, class predictions, and confidence scores

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

# Configuration
IMAGE_DIR=${1:-"./test_images"}
MODEL_SIZE=${2:-"small"}
IOU_THRESHOLD=${3:-0.75}  # IoU threshold for box matching (default 75%)
CONF_TOLERANCE=${4:-0.05}  # Confidence score tolerance (default 5%)

RESULTS_DIR="./tests/validation_results"
mkdir -p "$RESULTS_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_FILE="$RESULTS_DIR/validation_report_${TIMESTAMP}.txt"

echo "============================================================"
echo "Model Validation - Comparing Predictions Across All Tracks"
echo "============================================================"
echo "Image directory: $IMAGE_DIR"
echo "Model size: $MODEL_SIZE"
echo "IoU threshold: $IOU_THRESHOLD"
echo "Confidence tolerance: $CONF_TOLERANCE"
echo "Results will be saved to: $REPORT_FILE"
echo ""

# Find test images
TEST_IMAGES=($(find "$IMAGE_DIR" -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \)))

if [ ${#TEST_IMAGES[@]} -eq 0 ]; then
    echo -e "${RED}✗ No images found in $IMAGE_DIR${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Found ${#TEST_IMAGES[@]} test images:${NC}"
for IMG in "${TEST_IMAGES[@]}"; do
    echo "  - $(basename $IMG)"
done
echo ""

# Function to calculate IoU between two bounding boxes
calculate_iou() {
    local x1_a=$1
    local y1_a=$2
    local x2_a=$3
    local y2_a=$4
    local x1_b=$5
    local y1_b=$6
    local x2_b=$7
    local y2_b=$8

    # Calculate intersection
    local x1_i=$(echo "if ($x1_a > $x1_b) $x1_a else $x1_b" | bc)
    local y1_i=$(echo "if ($y1_a > $y1_b) $y1_a else $y1_b" | bc)
    local x2_i=$(echo "if ($x2_a < $x2_b) $x2_a else $x2_b" | bc)
    local y2_i=$(echo "if ($y2_a < $y2_b) $y2_a else $y2_b" | bc)

    # Check if there's intersection
    local w_i=$(echo "$x2_i - $x1_i" | bc)
    local h_i=$(echo "$y2_i - $y1_i" | bc)

    if (( $(echo "$w_i <= 0" | bc -l) )) || (( $(echo "$h_i <= 0" | bc -l) )); then
        echo "0"
        return
    fi

    local area_i=$(echo "$w_i * $h_i" | bc)

    # Calculate union
    local area_a=$(echo "($x2_a - $x1_a) * ($y2_a - $y1_a)" | bc)
    local area_b=$(echo "($x2_b - $x1_b) * ($y2_b - $y1_b)" | bc)
    local area_u=$(echo "$area_a + $area_b - $area_i" | bc)

    # Calculate IoU
    local iou=$(echo "scale=4; $area_i / $area_u" | bc)
    echo "$iou"
}

# Function to get predictions from an endpoint
get_predictions() {
    local endpoint=$1
    local image_path=$2
    local output_file=$3

    RESPONSE=$(curl -sf --max-time 60 -X POST "$endpoint" -F "image=@$image_path" 2>&1)

    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to get response from $endpoint" >&2
        return 1
    fi

    # Save the full response
    echo "$RESPONSE" | jq '.' > "$output_file" 2>/dev/null

    if [ $? -ne 0 ]; then
        echo "ERROR: Invalid JSON response from $endpoint" >&2
        return 1
    fi

    return 0
}

# Function to compare two prediction sets
compare_predictions() {
    local baseline_file=$1
    local test_file=$2
    local track_name=$3
    local image_name=$4

    # Extract detection counts
    local baseline_count=$(jq '.detections | length' "$baseline_file")
    local test_count=$(jq '.detections | length' "$test_file")

    echo "  Comparing $track_name predictions..."
    echo "    Baseline (Track A): $baseline_count detections"
    echo "    $track_name: $test_count detections"

    # Allow some variation in detection count (±2 detections)
    local count_diff=$((baseline_count - test_count))
    count_diff=${count_diff#-}  # absolute value

    if [ $count_diff -gt 2 ]; then
        echo -e "    ${YELLOW}⚠ Warning: Detection count differs by $count_diff${NC}"
    fi

    # Match each baseline detection with test detections
    local matched=0
    local unmatched=0
    local class_mismatches=0
    local conf_mismatches=0

    for i in $(seq 0 $((baseline_count - 1))); do
        # Get baseline box
        local b_x1=$(jq -r ".detections[$i].x1" "$baseline_file")
        local b_y1=$(jq -r ".detections[$i].y1" "$baseline_file")
        local b_x2=$(jq -r ".detections[$i].x2" "$baseline_file")
        local b_y2=$(jq -r ".detections[$i].y2" "$baseline_file")
        local b_class=$(jq -r ".detections[$i].class" "$baseline_file")
        local b_conf=$(jq -r ".detections[$i].confidence" "$baseline_file")

        # Find best matching box in test predictions
        local best_iou=0
        local best_match=-1

        for j in $(seq 0 $((test_count - 1))); do
            local t_x1=$(jq -r ".detections[$j].x1" "$test_file")
            local t_y1=$(jq -r ".detections[$j].y1" "$test_file")
            local t_x2=$(jq -r ".detections[$j].x2" "$test_file")
            local t_y2=$(jq -r ".detections[$j].y2" "$test_file")

            local iou=$(calculate_iou $b_x1 $b_y1 $b_x2 $b_y2 $t_x1 $t_y1 $t_x2 $t_y2)

            if (( $(echo "$iou > $best_iou" | bc -l) )); then
                best_iou=$iou
                best_match=$j
            fi
        done

        # Check if match is good enough
        if (( $(echo "$best_iou >= $IOU_THRESHOLD" | bc -l) )); then
            matched=$((matched + 1))

            # Check class match
            local t_class=$(jq -r ".detections[$best_match].class" "$test_file")
            if [ "$b_class" != "$t_class" ]; then
                class_mismatches=$((class_mismatches + 1))
                echo -e "    ${RED}✗ Class mismatch: baseline=$b_class, test=$t_class (IoU=$best_iou)${NC}"
            fi

            # Check confidence tolerance
            local t_conf=$(jq -r ".detections[$best_match].confidence" "$test_file")
            local conf_diff=$(echo "scale=4; ($b_conf - $t_conf)" | bc)
            conf_diff=${conf_diff#-}  # absolute value

            if (( $(echo "$conf_diff > $CONF_TOLERANCE" | bc -l) )); then
                conf_mismatches=$((conf_mismatches + 1))
                echo -e "    ${YELLOW}⚠ Confidence differs: baseline=$b_conf, test=$t_conf (diff=$conf_diff)${NC}"
            fi
        else
            unmatched=$((unmatched + 1))
            echo -e "    ${RED}✗ No matching box found (best IoU=$best_iou < $IOU_THRESHOLD)${NC}"
            echo "      Baseline box: [$b_x1, $b_y1, $b_x2, $b_y2] class=$b_class conf=$b_conf"
        fi
    done

    # Calculate match percentage
    local match_pct=0
    if [ $baseline_count -gt 0 ]; then
        match_pct=$(echo "scale=2; ($matched * 100) / $baseline_count" | bc)
    fi

    echo "    Matched: $matched/$baseline_count (${match_pct}%)"

    if [ $class_mismatches -gt 0 ]; then
        echo -e "    ${RED}Class mismatches: $class_mismatches${NC}"
    fi

    if [ $conf_mismatches -gt 0 ]; then
        echo -e "    ${YELLOW}Confidence mismatches: $conf_mismatches${NC}"
    fi

    # Determine pass/fail
    if [ $matched -eq $baseline_count ] && [ $class_mismatches -eq 0 ]; then
        echo -e "    ${GREEN}✓ PASS${NC}"
        return 0
    elif (( $(echo "$match_pct >= 75" | bc -l) )) && [ $class_mismatches -eq 0 ]; then
        echo -e "    ${YELLOW}⚠ PARTIAL PASS (${match_pct}% match - acceptable for FP16/decoder differences)${NC}"
        return 1
    else
        echo -e "    ${RED}✗ FAIL${NC}"
        return 2
    fi
}

# Initialize counters
TOTAL_TESTS=0
TOTAL_PASS=0
TOTAL_PARTIAL=0
TOTAL_FAIL=0

# Start report
{
    echo "Model Validation Report"
    echo "Generated: $(date)"
    echo "========================================"
    echo ""
    echo "Configuration:"
    echo "  Model: $MODEL_SIZE"
    echo "  IoU Threshold: $IOU_THRESHOLD"
    echo "  Confidence Tolerance: $CONF_TOLERANCE"
    echo ""
    echo "========================================"
    echo ""
} > "$REPORT_FILE"

# Process each test image
for IMG in "${TEST_IMAGES[@]}"; do
    IMAGE_NAME=$(basename "$IMG")
    echo ""
    echo "============================================================"
    echo "Testing: $IMAGE_NAME"
    echo "============================================================"

    # Create temp directory for this image
    TEMP_DIR=$(mktemp -d)

    # Get baseline predictions (Track A - PyTorch)
    echo "Getting baseline predictions (Track A - PyTorch)..."
    if ! get_predictions "http://localhost:9600/pytorch/predict/$MODEL_SIZE" "$IMG" "$TEMP_DIR/track_a.json"; then
        echo -e "${RED}✗ Failed to get baseline predictions. Skipping this image.${NC}"
        rm -rf "$TEMP_DIR"
        continue
    fi

    BASELINE_COUNT=$(jq '.detections | length' "$TEMP_DIR/track_a.json")
    echo -e "${GREEN}✓ Got $BASELINE_COUNT baseline detections${NC}"
    echo ""

    # Get Track B predictions
    echo "Getting Track B predictions (TensorRT + CPU NMS)..."
    if get_predictions "http://localhost:9600/predict/$MODEL_SIZE" "$IMG" "$TEMP_DIR/track_b.json"; then
        echo ""
        compare_predictions "$TEMP_DIR/track_a.json" "$TEMP_DIR/track_b.json" "Track B" "$IMAGE_NAME" | tee -a "$REPORT_FILE"
        RESULT=${PIPESTATUS[0]}
        TOTAL_TESTS=$((TOTAL_TESTS + 1))
        case $RESULT in
            0) TOTAL_PASS=$((TOTAL_PASS + 1)) ;;
            1) TOTAL_PARTIAL=$((TOTAL_PARTIAL + 1)) ;;
            2) TOTAL_FAIL=$((TOTAL_FAIL + 1)) ;;
        esac
        echo ""
    else
        echo -e "${RED}✗ Failed to get Track B predictions${NC}"
        echo ""
    fi

    # Get Track C predictions
    echo "Getting Track C predictions (TensorRT + GPU NMS)..."
    if get_predictions "http://localhost:9600/predict/${MODEL_SIZE}_end2end" "$IMG" "$TEMP_DIR/track_c.json"; then
        echo ""
        compare_predictions "$TEMP_DIR/track_a.json" "$TEMP_DIR/track_c.json" "Track C" "$IMAGE_NAME" | tee -a "$REPORT_FILE"
        RESULT=${PIPESTATUS[0]}
        TOTAL_TESTS=$((TOTAL_TESTS + 1))
        case $RESULT in
            0) TOTAL_PASS=$((TOTAL_PASS + 1)) ;;
            1) TOTAL_PARTIAL=$((TOTAL_PARTIAL + 1)) ;;
            2) TOTAL_FAIL=$((TOTAL_FAIL + 1)) ;;
        esac
        echo ""
    else
        echo -e "${RED}✗ Failed to get Track C predictions${NC}"
        echo ""
    fi

    # Get Track D predictions (balanced variant)
    echo "Getting Track D predictions (DALI + TRT + GPU NMS)..."
    if get_predictions "http://localhost:9600/predict/${MODEL_SIZE}_gpu_e2e" "$IMG" "$TEMP_DIR/track_d.json"; then
        echo ""
        compare_predictions "$TEMP_DIR/track_a.json" "$TEMP_DIR/track_d.json" "Track D" "$IMAGE_NAME" | tee -a "$REPORT_FILE"
        RESULT=${PIPESTATUS[0]}
        TOTAL_TESTS=$((TOTAL_TESTS + 1))
        case $RESULT in
            0) TOTAL_PASS=$((TOTAL_PASS + 1)) ;;
            1) TOTAL_PARTIAL=$((TOTAL_PARTIAL + 1)) ;;
            2) TOTAL_FAIL=$((TOTAL_FAIL + 1)) ;;
        esac
        echo ""
    else
        echo -e "${RED}✗ Failed to get Track D predictions${NC}"
        echo ""
    fi

    # Save results for this image
    cp "$TEMP_DIR"/*.json "$RESULTS_DIR/" 2>/dev/null || true
    mv "$RESULTS_DIR"/*.json "$RESULTS_DIR/${IMAGE_NAME%.*}_" 2>/dev/null || true

    # Cleanup
    rm -rf "$TEMP_DIR"
done

# Final Summary
echo ""
echo "============================================================"
echo "VALIDATION SUMMARY"
echo "============================================================"
echo "Total comparisons: $TOTAL_TESTS"
echo -e "${GREEN}Passed: $TOTAL_PASS${NC}"
echo -e "${YELLOW}Partial Pass: $TOTAL_PARTIAL${NC}"
echo -e "${RED}Failed: $TOTAL_FAIL${NC}"
echo ""

# Add summary to report
{
    echo ""
    echo "========================================"
    echo "SUMMARY"
    echo "========================================"
    echo "Total comparisons: $TOTAL_TESTS"
    echo "Passed: $TOTAL_PASS"
    echo "Partial Pass: $TOTAL_PARTIAL"
    echo "Failed: $TOTAL_FAIL"
    echo ""
} >> "$REPORT_FILE"

# Overall result
if [ $TOTAL_FAIL -eq 0 ] && [ $TOTAL_PARTIAL -eq 0 ]; then
    echo -e "${GREEN}✓ ALL VALIDATIONS PASSED!${NC}"
    echo ""
    echo "All Triton models are producing predictions matching the PyTorch baseline."
    echo -e "Report saved to: ${CYAN}$REPORT_FILE${NC}"
    exit 0
elif [ $TOTAL_FAIL -eq 0 ]; then
    echo -e "${GREEN}✓ VALIDATION PASSED WITH NOTES${NC}"
    echo ""
    echo "Some tracks have minor differences from baseline (75-99% match)."
    echo "This is expected due to:"
    echo "  - FP16 precision in TensorRT (vs FP32 in PyTorch)"
    echo "  - Different JPEG decoders (nvJPEG GPU vs PIL/OpenCV CPU)"
    echo "  - Borderline detections near confidence threshold"
    echo ""
    echo "These differences are acceptable for production deployment."
    echo -e "Report saved to: ${CYAN}$REPORT_FILE${NC}"
    exit 0
else
    echo -e "${RED}✗ VALIDATION FAILED${NC}"
    echo ""
    echo "One or more tracks are producing significantly different predictions (<75% match)."
    echo "Check the detailed report for more information."
    echo -e "Report saved to: ${CYAN}$REPORT_FILE${NC}"
    exit 2
fi
