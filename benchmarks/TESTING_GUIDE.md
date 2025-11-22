# Comprehensive Batching Test Guide

This guide provides test scenarios to verify dynamic batching works correctly with the new streaming/balanced/batch model variants.

## Quick Reference

### All Available Tracks:
- **Track A**: PyTorch Baseline (CPU NMS)
- **Track B**: Triton Standard TRT (TensorRT + CPU NMS)
- **Track C**: Triton End2End TRT (TensorRT + GPU NMS) - Kept for benchmarking
- **Track D_streaming**: DALI + TRT (1ms delay, max_batch=8, low latency)
- **Track D_balanced**: DALI + TRT (10ms delay, max_batch=64, balanced)
- **Track D_batch**: DALI + TRT (50ms delay, max_batch=64, max throughput)

### Test Images:
- Location: `/mnt/nvm/KILLBOY_SAMPLE_PICTURES`
- Count: 1000+ real-world motorcycle racing images
- Format: JPEG

---

## Test Scenarios

### 1. Quick Validation (Recommended First Test)

**Purpose**: Verify all tracks work and get initial performance baseline

```bash
cd /mnt/nvm/repos/triton-api/benchmarks
./triton_bench --mode quick
```

**What it does**:
- Tests all 6 tracks
- 16 concurrent clients
- 30 seconds duration
- Uses 100 images from test folder

**Expected results**:
- All tracks should show >95% success rate
- Track D variants should show highest throughput
- Track D_batch should achieve larger batch sizes (visible in Grafana)

---

### 2. Single Track Deep Dive

**Purpose**: Test each Track D variant in isolation to observe batching behavior

**Track D Streaming (1ms delay - should batch quickly with small batches):**
```bash
./triton_bench --mode full --track D_streaming --clients 128 --duration 60
```

**Track D Balanced (10ms delay - should achieve medium batches):**
```bash
./triton_bench --mode full --track D_balanced --clients 128 --duration 60
```

**Track D Batch (50ms delay - should achieve largest batches):**
```bash
./triton_bench --mode full --track D_batch --clients 256 --duration 60
```

**What to observe in Grafana**:
- **D_streaming**: Batch sizes 1-8, lower latency, higher P99
- **D_balanced**: Batch sizes 4-32, balanced latency/throughput
- **D_batch**: Batch sizes 16-64, highest throughput, higher mean latency

---

### 3. Sustained Throughput Test

**Purpose**: Find maximum sustained throughput for each variant

```bash
# Test D_streaming
./triton_bench --mode sustained --track D_streaming

# Test D_balanced
./triton_bench --mode sustained --track D_balanced

# Test D_batch
./triton_bench --mode sustained --track D_batch
```

**What it does**:
- Automatically finds optimal client count
- Runs 5-minute sustained test
- Reports maximum achievable throughput

**Expected results**:
- D_streaming: Lower throughput, lowest latency
- D_balanced: Medium throughput, balanced latency
- D_batch: Highest throughput, higher latency

---

### 4. Variable Load Pattern Tests

**Purpose**: Test how batching adapts to different load patterns

**Burst Pattern (simulates traffic spikes):**
```bash
./triton_bench --mode variable --track D_batch \
  --load-pattern burst --burst-interval 10 --clients 256 --duration 120
```

**Ramp Pattern (gradually increasing load):**
```bash
./triton_bench --mode variable --track D_batch \
  --load-pattern ramp --ramp-step 32 --clients 256 --duration 120
```

**What to observe**:
- How quickly batching adapts to load changes
- Queue depth changes in Grafana
- Latency stability during load transitions

---

### 5. Full Comparison Across All Tracks

**Purpose**: Compare all tracks side-by-side with realistic load

```bash
./triton_bench --mode full --track all --clients 128 --duration 90
```

**Expected speedup vs Track A (PyTorch baseline)**:
- Track B: ~2x faster
- Track C: ~4x faster
- Track D_streaming: ~6-8x faster
- Track D_balanced: ~10-12x faster
- Track D_batch: ~12-15x faster

---

### 6. Process All 1000+ Images

**Purpose**: Real-world batch processing scenario

```bash
# Remove limit to process ALL images
./triton_bench --mode all --track D_batch --clients 256 --limit 999999
```

**What it does**:
- Processes every image in /mnt/nvm/KILLBOY_SAMPLE_PICTURES
- Uses 256 concurrent workers
- Reports total processing time and throughput

**Expected results**:
- D_batch should process 1000+ images in under 60 seconds
- Grafana should show sustained batch sizes of 32-64

---

## Monitoring with Grafana

While tests run, watch these metrics in Grafana:

### Key Metrics to Monitor:

1. **Batch Size** (`nv_inference_exec_count`)
   - Streaming: Should see batches 1-8
   - Balanced: Should see batches 4-32
   - Batch: Should see batches 16-64

2. **Queue Time** (`nv_inference_queue_duration_us`)
   - Streaming: <1000μs (1ms)
   - Balanced: ~10000μs (10ms)
   - Batch: ~50000μs (50ms)

3. **Compute Time** (`nv_inference_compute_infer_duration_us`)
   - Should decrease as batch size increases (amortized cost)

4. **Throughput** (`nv_inference_request_success / time`)
   - Should increase: Streaming < Balanced < Batch

5. **Queue Depth** (`nv_inference_pending_request_count`)
   - Higher queue depth = more batching opportunity
   - D_batch should maintain higher queue depth

---

## Interpreting Results

### Success Criteria:

✅ **Batching Working Correctly**:
- D_streaming achieves batches of 2-8 with <2ms latency
- D_balanced achieves batches of 8-32 with ~20ms P95 latency
- D_batch achieves batches of 32-64 with highest throughput

✅ **Performance Targets**:
- Track D_batch: >500 img/sec with 256 clients
- Track D_balanced: >400 img/sec with 128 clients
- Track D_streaming: >300 img/sec with 64 clients

❌ **Warning Signs**:
- All variants showing same batch size (batching config not applied)
- Batch sizes always 1 (queue delay too short or no concurrent load)
- Success rate <95% (errors, timeouts, or capacity issues)
- Throughput decreasing with more clients (resource saturation)

---

## Advanced Testing

### Custom Test Matrix

Test specific client counts to find optimal configuration:

```bash
# Test different client counts for D_batch
for clients in 64 128 256 512; do
  echo "Testing with $clients clients..."
  ./triton_bench --mode full --track D_batch \
    --clients $clients --duration 60 \
    --output "benchmarks/results/batch_${clients}clients.json"
done
```

### Compare All Variants

```bash
# Quick comparison of all D variants
./triton_bench --mode quick --track D_streaming --output results/streaming.json
./triton_bench --mode quick --track D_balanced --output results/balanced.json
./triton_bench --mode quick --track D_batch --output results/batch.json
```

---

## Troubleshooting

### Batching Not Working (Batch Size = 1)

**Check**:
1. Grafana shows queue depth > 0
2. Enough concurrent clients (need at least 8 for batching)
3. Triton logs show dynamic batching enabled
4. Images arriving faster than delay window

**Fix**:
- Increase client count: `--clients 256`
- Reduce delay to verify: Edit `config.pbtxt`
- Check Triton logs: `docker compose logs triton-api | grep -i batch`

### Low Throughput

**Check**:
1. GPU utilization in Grafana (should be >80%)
2. Network bandwidth (1000+ images = lots of data)
3. FastAPI worker count (should be 32 workers)

**Fix**:
- Increase clients beyond saturation point
- Check for throttling: `docker compose logs yolo-api`
- Verify shared gRPC client: Check connection count

### Inconsistent Results

**Check**:
1. Warmup phase completed (first 10 requests discarded)
2. Other processes using GPU
3. Thermal throttling (long sustained tests)

**Fix**:
- Add warmup: `--warmup 20`
- Isolate GPU: Stop other containers
- Monitor temperature: `nvidia-smi dmon`

---

## Next Steps After Testing

1. **Document baseline performance** from initial tests
2. **Tune batching delays** if needed based on results
3. **Adjust instance counts** for optimal resource utilization
4. **Create monitoring dashboards** for production
5. **Set up alerting** for success rate, latency, throughput

---

## Example Test Session

Complete test session to validate everything:

```bash
cd /mnt/nvm/repos/triton-api/benchmarks

# 1. Quick validation (5 minutes)
./triton_bench --mode quick

# 2. Deep dive on batch variant (2 minutes per test)
./triton_bench --mode full --track D_streaming --clients 64 --duration 60
./triton_bench --mode full --track D_balanced --clients 128 --duration 60
./triton_bench --mode full --track D_batch --clients 256 --duration 60

# 3. Find max throughput (15 minutes)
./triton_bench --mode sustained --track D_batch

# 4. Real-world processing (1-2 minutes)
./triton_bench --mode all --track D_batch --clients 256 --limit 999999

echo "Testing complete! Check benchmarks/results/ for detailed JSON results"
```

**Total time**: ~25-30 minutes for comprehensive validation

All results are automatically saved to `benchmarks/results/benchmark_results.json` with timestamps.
