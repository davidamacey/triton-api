# Triton YOLO Benchmark Suite

**Version 1.0.0** - Professional benchmarking tool for NVIDIA Triton YOLO inference

Single comprehensive Go tool for all benchmarking scenarios - no Python required!

---

## Quick Start

### 1. Install Go (One-time setup)

```bash
# Download and install Go 1.25.5 (latest stable)
cd /tmp
wget https://go.dev/dl/go1.25.5.linux-amd64.tar.gz
sudo rm -rf /usr/local/go && sudo tar -C /usr/local -xzf go1.25.5.linux-amd64.tar.gz

# Add to PATH
export PATH=$PATH:/usr/local/go/bin

# Make permanent (add to ~/.bashrc)
echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc

# Verify installation
go version  # Should show: go version go1.25.5 linux/amd64
```

For other platforms, download from https://go.dev/dl/

### 2. Build the Benchmark Tool

```bash
cd /mnt/nvm/repos/triton-api

# Using Makefile (recommended)
make bench-build

# Or manually
cd benchmarks && go build -o triton_bench triton_bench.go
```

### 3. Run Your First Test

```bash
# Quick concurrency test (30 seconds, all tracks)
make bench-quick

# Or directly
cd benchmarks && ./triton_bench --mode quick

# Results are automatically saved to benchmarks/results/ with timestamps
```

---

## Test Modes

Run `./triton_bench --list-modes` to see all available modes.

### Mode 1: Single Image
```bash
./triton_bench --mode single
```
Tests one image through all tracks (10 iterations for accuracy)

### Mode 2: Image Set
```bash
./triton_bench --mode set --limit 50
```
Process N images sequentially through each track

### Mode 3: Quick Concurrency
```bash
./triton_bench --mode quick
make bench-quick  # Makefile shortcut
```
Fast 30-second test with 16 clients - verify batching works

### Mode 4: Full Concurrency
```bash
./triton_bench --mode full --clients 128 --duration 60
make bench-full  # Makefile shortcut
```
Comprehensive concurrent load test

### Mode 5: All Images
```bash
./triton_bench --mode all --clients 64
```
Process entire image directory with concurrency

### Mode 6: Sustained Throughput
```bash
./triton_bench --mode sustained
```
Finds optimal client count and runs 5-minute stress test

### Mode 7: Variable Load
```bash
# Burst pattern
./triton_bench --mode variable --load-pattern burst --burst-interval 10

# Ramp pattern
./triton_bench --mode variable --load-pattern ramp --ramp-step 32
```
Test with changing load patterns

### Mode 8: Matrix Concurrency Test
```bash
./triton_bench --mode matrix --matrix-clients 32,64,128,256,512,1024 --duration 30
make bench-matrix  # Makefile shortcut
```
Tests multiple concurrency levels sequentially to find optimal client count. Output shows throughput/latency at each level, identifying peak throughput and saturation point.

**Use Cases:**
- Finding saturation point for capacity planning
- Comparing tracks at different loads
- Determining optimal concurrency configuration

---

## Common Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `quick` | Test mode |
| `--images` | `./test_images` | Image directory |
| `--track` | `all` | Track filter (see below) |
| `--clients` | `64` | Concurrent clients |
| `--duration` | `60` | Test duration (seconds) |
| `--limit` | `100` | Max images to load |
| `--warmup` | `10` | Warmup requests |
| `--quiet` | `false` | Minimal output |
| `--json` | `false` | JSON output only |
| `--matrix-clients` | - | Comma-separated client counts for matrix mode |

---

## Track-Specific Testing

### All Available Tracks:
- **Track A**: PyTorch Baseline (CPU NMS)
- **Track B**: Triton Standard TRT (TensorRT + CPU NMS)
- **Track C**: Triton End2End TRT (TensorRT + GPU NMS)
- **Track D_streaming**: DALI + TRT (1ms delay, max_batch=8, low latency)
- **Track D_balanced**: DALI + TRT (10ms delay, max_batch=64, balanced)
- **Track D_batch**: DALI + TRT (50ms delay, max_batch=64, max throughput)
- **Track E**: Visual Search (YOLO + global image embedding)
- **Track E_full**: Visual Search (YOLO + global + per-box embeddings)

### Track A: PyTorch Baseline
```bash
./triton_bench --mode full --track A --clients 64 --duration 60
make bench-track-a  # Makefile shortcut
```
**Purpose:** Development/debugging reference, baseline for speedup comparisons

### Track B: TensorRT + CPU NMS
```bash
./triton_bench --mode full --track B --clients 128 --duration 60
make bench-track-b  # Makefile shortcut
```
**Expected:** ~2x faster than Track A

### Track C: TensorRT + GPU NMS
```bash
./triton_bench --mode full --track C --clients 128 --duration 60
make bench-track-c  # Makefile shortcut
```
**Expected:** ~4x faster than Track A (kept for benchmarking)

### Track D: DALI Variants

**CRITICAL: Batching Behavior**

Track D has three variants with different batching configurations designed for specific use cases:

#### D_streaming (Low Latency)
```bash
./triton_bench --mode full --track D_streaming --clients 128 --duration 60
make bench-track-d-streaming  # Makefile shortcut
```
- **Queue Delay:** 1ms
- **Max Batch:** 8
- **Batch Sizes:** 1-8
- **Use Case:** Video streaming, real-time inference
- **Expected:** ~6-8x faster than Track A, lowest P95 latency

#### D_balanced (Balanced)
```bash
./triton_bench --mode full --track D_balanced --clients 128 --duration 60
make bench-track-d-balanced  # Makefile shortcut
```
- **Queue Delay:** 10ms
- **Max Batch:** 64
- **Batch Sizes:** 4-32
- **Use Case:** General purpose inference
- **Expected:** ~10-12x faster than Track A, balanced latency/throughput

#### D_batch (Max Throughput)
```bash
./triton_bench --mode full --track D_batch --clients 256 --duration 60
make bench-track-d  # Makefile shortcut (alias: make bench-track-d-batch)
```
- **Queue Delay:** 50ms
- **Max Batch:** 64
- **Batch Sizes:** 16-64
- **Use Case:** Batch processing, maximum throughput
- **Expected:** ~12-15x faster than Track A, highest throughput

### Track E: Visual Search (YOLO + CLIP)

Track E combines YOLO detection with MobileCLIP embeddings for visual search.

#### E (Simple)
```bash
./triton_bench --mode full --track E --clients 128 --duration 60
make bench-track-e  # Makefile shortcut
```
- **Models:** YOLO + global image embedding
- **Use Case:** Basic image similarity search
- **Expected:** ~250-350 req/sec with 128 clients

#### E_full (Full)
```bash
./triton_bench --mode full --track E_full --clients 128 --duration 60
make bench-track-e-full  # Makefile shortcut
```
- **Models:** YOLO + global + per-box embeddings
- **Use Case:** Object-level visual search
- **Expected:** Slightly slower than E due to per-box embedding computation

**Monitoring Track E:**
- Check both YOLO and MobileCLIP model stats in Grafana
- Verify ensemble batching is occurring
- Monitor OpenSearch ingestion rate

---

## Understanding Results

### Output Files

Results automatically saved to `results/` with timestamps:

```
benchmarks/results/
├── benchmark_results.json                     # Latest (always overwritten)
├── benchmark_results_2025-12-07_143045.json   # Timestamped run 1
├── benchmark_results_2025-12-07_150230.json   # Timestamped run 2
└── benchmark_results_2025-12-07_163521.json   # Timestamped run 3
```

### Per-Track Metrics

- **Total requests**: Number of requests sent
- **Successful**: Successfully processed requests
- **Failed**: Failed requests (timeouts, errors)
- **Duration**: Actual test duration
- **Throughput**: Requests per second (req/sec)

### Latency Metrics (milliseconds)

- **Mean**: Average latency
- **Median (P50)**: 50th percentile
- **P95**: 95th percentile (95% of requests faster than this)
- **P99**: 99th percentile (99% of requests faster than this)
- **Min/Max**: Fastest and slowest requests

### Comparison Table

Shows all tracks side-by-side with **Speedup vs A** (how much faster compared to PyTorch baseline)

### Example Output Structure
```json
{
  "timestamp": "2025-12-07T15:30:00Z",
  "config": {
    "image_dir": "/mnt/nvm/KILLBOY_SAMPLE_PICTURES",
    "num_clients": 64,
    "duration": "60s",
    "warmup": 10
  },
  "results": {
    "A": {
      "track_id": "A",
      "track_name": "PyTorch Baseline",
      "total_requests": 12450,
      "throughput_rps": 207.5,
      "mean_latency_ms": 8.45,
      "p95_latency_ms": 12.3
    }
  }
}
```

---

## Monitoring & Verification

### Grafana Metrics

While tests run, monitor these metrics at http://localhost:4605 (admin/admin):

#### Key Metrics to Watch:

**1. Batch Size** (`nv_inference_exec_count`)
- **Streaming:** Should see batches 1-8
- **Balanced:** Should see batches 4-32
- **Batch:** Should see batches 16-64

**2. Queue Time** (`nv_inference_queue_duration_us`)
- **Streaming:** <1000μs (1ms)
- **Balanced:** ~10000μs (10ms)
- **Batch:** ~50000μs (50ms)

**3. Compute Time** (`nv_inference_compute_infer_duration_us`)
- Should decrease as batch size increases (amortized cost)

**4. Throughput** (`nv_inference_request_success / time`)
- Should increase: Streaming < Balanced < Batch

**5. Queue Depth** (`nv_inference_pending_request_count`)
- Higher queue depth = more batching opportunity
- D_batch should maintain higher queue depth

### Monitor in Terminal

```bash
# Terminal 1: GPU utilization
nvidia-smi -l 1

# Terminal 2: Check batching
docker compose logs -f triton-api | grep "batch size"

# Terminal 3: Grafana
# http://localhost:4605
```

### Success Criteria

**Batching Working Correctly:**
- D_streaming achieves batches of 2-8 with <2ms latency
- D_balanced achieves batches of 8-32 with ~20ms P95 latency
- D_batch achieves batches of 32-64 with highest throughput

**Performance Targets:**
- Track D_batch: >500 img/sec with 256 clients
- Track D_balanced: >400 img/sec with 128 clients
- Track D_streaming: >300 img/sec with 64 clients
- All tracks: >95% success rate

### Warning Signs

- All variants showing same batch size (batching config not applied)
- Batch sizes always 1 (queue delay too short or no concurrent load)
- Success rate <95% (errors, timeouts, or capacity issues)
- Throughput decreasing with more clients (resource saturation)

---

## Common Workflows

### Initial Validation
```bash
./triton_bench --mode single       # Verify correctness
./triton_bench --mode quick        # Check batching
./triton_bench --mode full         # Full benchmark
```

### Find Maximum Throughput
```bash
./triton_bench --mode sustained
make bench-track-d  # Test specific track
```

### Process Production Dataset
```bash
./triton_bench --mode all --images /path/to/data --clients 128
```

### Progressive Load Testing
```bash
# Test increasing concurrency
for clients in 16 32 64 128 256; do
  ./triton_bench --mode full --clients $clients --duration 60
done

# Or use matrix mode
./triton_bench --mode matrix --matrix-clients 16,32,64,128,256 --duration 30
make bench-matrix
```

### Compare All Track D Variants
```bash
./triton_bench --mode full --track D_streaming --clients 128 --duration 60
./triton_bench --mode full --track D_balanced --clients 128 --duration 60
./triton_bench --mode full --track D_batch --clients 256 --duration 60
```

### Complete Test Session (25-30 minutes)
```bash
cd /mnt/nvm/repos/triton-api/benchmarks

# 1. Quick validation (5 minutes)
./triton_bench --mode quick

# 2. Deep dive on batch variants (2 minutes per test)
./triton_bench --mode full --track D_streaming --clients 64 --duration 60
./triton_bench --mode full --track D_balanced --clients 128 --duration 60
./triton_bench --mode full --track D_batch --clients 256 --duration 60

# 3. Find max throughput (15 minutes)
./triton_bench --mode sustained --track D_batch

# 4. Real-world processing (1-2 minutes)
./triton_bench --mode all --track D_batch --clients 256 --limit 999999

echo "Testing complete! Check results/ for detailed JSON results"
```

---

## Troubleshooting

### No tracks available
```bash
docker compose ps                  # Check services
curl http://localhost:4603/health  # Test endpoint
```

### No speed gains / Batching not working
```bash
# Verify workers
docker compose exec yolo-api ps aux | grep uvicorn | wc -l
# Should show 3 or 65 (1 master + 2 or 64 workers)

# Check batching
docker compose logs triton-api | grep "batch size"
# Should show batch sizes > 1

# Verify enough concurrent clients (need at least 8 for batching)
# Increase client count: --clients 256
```

### High error rate
```bash
docker stats                      # Check resources
docker compose logs triton-api    # Check errors
docker compose logs yolo-api      # Check for throttling
```

### Inconsistent results
```bash
# Add warmup: --warmup 20
# Isolate GPU: stop other containers
# Monitor temperature: nvidia-smi dmon
# Check for other processes using GPU
```

### Low throughput
```bash
# Check GPU utilization (should be >80%)
nvidia-smi -l 1

# Check FastAPI worker count (should be 3 or 65)
docker compose exec yolo-api ps aux | grep uvicorn | wc -l

# Verify shared gRPC client: Check connection count
docker compose logs yolo-api | grep -i "grpc"
```

---

## Advanced Usage

### Using Makefile Targets

The repository includes convenient Makefile targets for common benchmarks:

```bash
# Build
make bench-build              # Build benchmark tool

# Quick tests
make bench-quick              # 30s, 16 clients
make bench-full               # 60s, 128 clients
make bench-matrix             # Multiple concurrency levels

# Track-specific tests
make bench-track-a            # Track A: PyTorch
make bench-track-b            # Track B: TensorRT
make bench-track-c            # Track C: TRT + GPU NMS
make bench-track-d            # Track D: DALI batch (alias: bench-track-d-batch)
make bench-track-d-streaming  # Track D: DALI streaming
make bench-track-d-balanced   # Track D: DALI balanced
make bench-track-e            # Track E: Visual Search
make bench-track-e-full       # Track E: Visual Search (full)

# Utilities
make bench-stress             # Stress test (512 clients, 120s)
make bench-results            # Show recent benchmark results
make lint-go                  # Run golangci-lint
make fmt-go                   # Format Go code
```

All Makefile targets use consistent durations and client counts for reproducibility.

### Variable Load Patterns

**Burst Pattern** (simulates traffic spikes):
```bash
./triton_bench --mode variable --track D_batch \
  --load-pattern burst --burst-interval 10 --clients 256 --duration 120
```

**Ramp Pattern** (gradually increasing load):
```bash
./triton_bench --mode variable --track D_batch \
  --load-pattern ramp --ramp-step 32 --clients 256 --duration 120
```

**What to observe:**
- How quickly batching adapts to load changes
- Queue depth changes in Grafana
- Latency stability during load transitions

### Custom Test Matrix

Test specific client counts to find optimal configuration:

```bash
# Test different client counts for D_batch
for clients in 64 128 256 512; do
  echo "Testing with $clients clients..."
  ./triton_bench --mode full --track D_batch \
    --clients $clients --duration 60 \
    --output "results/batch_${clients}clients.json"
done
```

### Building for Different Architectures

```bash
# Build for Linux AMD64 (default)
GOOS=linux GOARCH=amd64 go build -o triton_bench triton_bench.go

# Build for Linux ARM64
GOOS=linux GOARCH=arm64 go build -o triton_bench_arm64 triton_bench.go

# Build for macOS
GOOS=darwin GOARCH=amd64 go build -o triton_bench_mac triton_bench.go
```

---

## Tips for Best Results

1. **Start services first**: Make sure all Docker services are running
   ```bash
   cd /mnt/nvm/repos/triton-api
   docker compose up -d
   ```

2. **Wait for warmup**: The first few minutes after startup may show slower performance as TensorRT engines finalize

3. **Progressive testing**: Start with low concurrency and increase gradually

4. **Use matrix mode**: Automatically find optimal concurrency level
   ```bash
   ./triton_bench --mode matrix --track D_batch --duration 30
   ```

5. **Monitor with Grafana**: Open http://localhost:4605 to see real-time metrics

6. **Check batching**: Monitor batch sizes in Triton logs
   ```bash
   docker compose logs -f triton-api | grep -i batch
   ```

7. **Verify sufficient load**: Need at least 8-16 concurrent clients to observe batching benefits

---

For detailed project documentation, see:
- [DEPLOYMENT_GUIDE.md](../DEPLOYMENT_GUIDE.md) - Full deployment guide
- [CLAUDE.md](../CLAUDE.md) - Project overview and architecture

**Happy Benchmarking!**
