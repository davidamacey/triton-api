#!/bin/bash
#
# Test Shared Client vs Per-Request Client Performance
# ====================================================
#
# This script compares the performance impact of:
# - Shared gRPC client (enables batching)
# - Per-request gRPC client (no batching)
#
# Usage: bash tests/test_shared_vs_per_request.sh

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Shared vs Per-Request Client Test${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if service is running
if ! curl -s http://localhost:9600/health > /dev/null; then
    echo -e "${RED}Error: yolo-api service not responding${NC}"
    echo "Run: docker compose up -d"
    exit 1
fi

# Get test image
TEST_IMAGE="test_images/bus.jpg"
if [ ! -f "$TEST_IMAGE" ]; then
    echo -e "${YELLOW}Test image not found, using first available image...${NC}"
    TEST_IMAGE=$(find test_images -name "*.jpg" -o -name "*.png" | head -1)
    if [ -z "$TEST_IMAGE" ]; then
        echo -e "${RED}Error: No test images found in test_images/${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}Using test image: $TEST_IMAGE${NC}"
echo ""

# ============================================================================
# 1. Check current connection pool status
# ============================================================================

echo -e "${BLUE}1. Current Connection Pool Status${NC}"
curl -s http://localhost:9600/connection_pool_info | python3 -m json.tool
echo ""
echo ""

# ============================================================================
# 2. Test Shared Client Mode (DEFAULT)
# ============================================================================

echo -e "${BLUE}2. Testing SHARED CLIENT mode (batching enabled)${NC}"
echo -e "${YELLOW}Sending 10 concurrent requests...${NC}"

SHARED_START=$(date +%s%N)

for i in {1..10}; do
    curl -s -X POST "http://localhost:9600/predict/small_end2end?shared_client=true" \
        -F "image=@${TEST_IMAGE}" > /dev/null &
done
wait

SHARED_END=$(date +%s%N)
SHARED_TIME=$(( ($SHARED_END - $SHARED_START) / 1000000 ))

echo -e "${GREEN}✓ Shared client mode: ${SHARED_TIME}ms for 10 requests${NC}"
echo ""

# Check for batching in Triton logs
echo -e "${YELLOW}Checking Triton logs for batch sizes...${NC}"
docker compose logs triton-api --tail 50 | grep "batch size" | tail -5
echo ""
echo ""

# ============================================================================
# 3. Test Per-Request Client Mode
# ============================================================================

echo -e "${BLUE}3. Testing PER-REQUEST CLIENT mode (batching disabled)${NC}"
echo -e "${YELLOW}Sending 10 concurrent requests...${NC}"

PER_REQ_START=$(date +%s%N)

for i in {1..10}; do
    curl -s -X POST "http://localhost:9600/predict/small_end2end?shared_client=false" \
        -F "image=@${TEST_IMAGE}" > /dev/null &
done
wait

PER_REQ_END=$(date +%s%N)
PER_REQ_TIME=$(( ($PER_REQ_END - $PER_REQ_START) / 1000000 ))

echo -e "${GREEN}✓ Per-request client mode: ${PER_REQ_TIME}ms for 10 requests${NC}"
echo ""

# Check for batching in Triton logs (should be batch_size=1)
echo -e "${YELLOW}Checking Triton logs for batch sizes...${NC}"
docker compose logs triton-api --tail 50 | grep "batch size" | tail -5
echo ""
echo ""

# ============================================================================
# 4. Comparison
# ============================================================================

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Results Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "Shared Client Mode:     ${SHARED_TIME}ms"
echo -e "Per-Request Client Mode: ${PER_REQ_TIME}ms"
echo ""

if [ $SHARED_TIME -lt $PER_REQ_TIME ]; then
    DIFF=$(( $PER_REQ_TIME - $SHARED_TIME ))
    SPEEDUP=$(awk "BEGIN {printf \"%.2f\", $PER_REQ_TIME / $SHARED_TIME}")
    echo -e "${GREEN}✓ Shared client is FASTER by ${DIFF}ms (${SPEEDUP}x speedup)${NC}"
else
    DIFF=$(( $SHARED_TIME - $PER_REQ_TIME ))
    echo -e "${YELLOW}⚠ Per-request client is faster by ${DIFF}ms${NC}"
    echo -e "${YELLOW}  This suggests DALI processing may be the bottleneck, not batching${NC}"
fi
echo ""

# ============================================================================
# 5. Recommendations
# ============================================================================

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Testing Recommendations${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "For comprehensive testing, use the Go benchmark tool:"
echo ""
echo -e "${YELLOW}# Test shared mode (default)${NC}"
echo "cd benchmarks"
echo "./triton_bench --mode full --track C --clients 128 --duration 60"
echo ""
echo -e "${YELLOW}# Then test per-request mode${NC}"
echo "# Edit src/main.py and change default: shared_client: bool = Query(False, ...)"
echo "# Restart: docker compose restart yolo-api"
echo "./triton_bench --mode full --track C --clients 128 --duration 60"
echo ""
echo -e "${YELLOW}# Check batch sizes during test${NC}"
echo "docker compose logs -f triton-api | grep 'batch size'"
echo ""
echo -e "${GREEN}Expected with shared client: batch_size = 4, 8, 16, 32${NC}"
echo -e "${YELLOW}Expected with per-request:   batch_size = 1 (always)${NC}"
echo ""
