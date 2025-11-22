#!/bin/bash
# Quick test to verify shared client is working

echo "=================================="
echo "Shared Client Integration Test"
echo "=================================="
echo ""

# 1. Restart yolo-api to load new code
echo "[1/5] Restarting yolo-api service..."
docker compose restart yolo-api
sleep 15

# 2. Check for import errors
echo ""
echo "[2/5] Checking for import errors..."
docker compose logs yolo-api --tail 50 | grep -iE "(error|exception|traceback)" && echo "⚠️  Errors detected!" || echo "✓ No errors found"

# 3. Check shared client initialization
echo ""
echo "[3/5] Verifying shared client pool..."
docker compose logs yolo-api --tail 100 | grep -iE "(shared.*client|batching.*enabled)" | head -5

# 4. Test basic endpoint
echo ""
echo "[4/5] Testing inference endpoint..."
curl -X POST "http://localhost:9600/health" -s | python3 -m json.tool || echo "✗ Health check failed"

# 5. Run quick benchmark
echo ""
echo "[5/5] Running quick benchmark (10 seconds)..."
cd benchmarks
./triton_bench --mode quick --track D_batch --clients 16 2>&1 | tail -15

echo ""
echo "=================================="
echo "✓ Test Complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Check Triton logs for batch sizes > 1:"
echo "   docker compose logs triton-api | grep 'batch size'"
echo ""
echo "2. Run full benchmark:"
echo "   cd benchmarks && ./triton_bench --mode full --clients 128"
echo ""
echo "3. If issues occur, rollback:"
echo "   bash backups/ROLLBACK.sh"
