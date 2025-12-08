#!/bin/bash
# Check Services Status
# Verifies all services and models are running and ready

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "=========================================="
echo "Service Status Check"
echo "=========================================="
echo ""

# Check Docker containers
echo "Docker Containers:"
echo "------------------"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "triton-api|yolo-api|NAMES"
echo ""

# Check Triton server
echo "Triton Server:"
echo "--------------"
TRITON_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:4600/v2/health/ready 2>/dev/null)
if [ "$TRITON_STATUS" = "200" ]; then
    echo -e "${GREEN}✓ Triton server is ready${NC}"

    # Check all 6 models (Tracks B/C/D)
    echo ""
    echo "Model Status:"
    MODELS=(
        "yolov11_small_trt"                     # Track B: Standard TRT
        "yolov11_small_trt_end2end"             # Track C: TRT + GPU NMS
        "yolo_preprocess_dali"                  # Track D: DALI preprocessing
        "yolov11_small_gpu_e2e_streaming"       # Track D: Low latency
        "yolov11_small_gpu_e2e"                 # Track D: Balanced
        "yolov11_small_gpu_e2e_batch"           # Track D: High throughput
    )

    for MODEL in "${MODELS[@]}"; do
        MODEL_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:4600/v2/models/$MODEL/ready" 2>/dev/null)
        if [ "$MODEL_STATUS" = "200" ]; then
            echo -e "  ${GREEN}✓${NC} $MODEL: READY"
        else
            echo -e "  ${RED}✗${NC} $MODEL: NOT READY"
        fi
    done
else
    echo -e "${RED}✗ Triton server is not ready${NC}"
    echo "  Check logs: docker compose logs triton-api"
fi
echo ""

# Check Unified YOLO API (all tracks)
echo "Unified YOLO API (All Tracks):"
echo "------------------------------"
if curl -sf http://localhost:4603/health 2>&1 | grep -q "healthy"; then
    echo -e "${GREEN}✓ Service is healthy (Tracks A/B/C/D/E available)${NC}"
    echo "  URL: http://localhost:4603"
    echo ""
    echo "  Track A: http://localhost:4603/pytorch/predict/small"
    echo "  Track B: http://localhost:4603/predict/small"
    echo "  Track C: http://localhost:4603/predict/small_end2end"
    echo "  Track D: http://localhost:4603/predict/small_gpu_e2e_batch"
    echo "  Track E: http://localhost:4603/track_e/detect"
else
    echo -e "${RED}✗ Service is not responding${NC}"
    echo "  Check logs: docker compose logs yolo-api"
fi
echo ""

# GPU Status
echo "GPU Status:"
echo "-----------"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
    awk -F', ' '{printf "GPU %s: %s (Util: %s%%, Mem: %s/%s MB)\n", $1, $2, $3, $4, $5}'
else
    echo -e "${YELLOW}⚠ nvidia-smi not available${NC}"
fi
echo ""

# Summary
echo "=========================================="
echo "Quick Commands:"
echo "=========================================="
echo ""
echo "Test inference (all tracks):"
echo "  bash tests/test_inference.sh"
echo ""
echo "Run benchmarks:"
echo "  cd benchmarks && ./triton_bench --mode quick"
echo ""
echo "View logs:"
echo "  docker compose logs -f triton-api"
echo "  docker compose logs -f yolo-api"
echo ""
echo "View Grafana dashboard:"
echo "  http://localhost:4605 (admin/admin)"
echo ""
echo "Monitor GPU:"
echo "  nvidia-smi -l 1"
echo ""
