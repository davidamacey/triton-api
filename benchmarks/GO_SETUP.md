# Go Benchmark Tool - Setup Guide for Ubuntu

This guide will help you install Go and run the professional benchmark tool for true concurrency testing.

## Install Go on Ubuntu

```bash
# Remove any existing Go installation
sudo rm -rf /usr/local/go

# Download Go 1.23.4 (latest stable as of 2025)
cd /tmp
wget https://go.dev/dl/go1.23.4.linux-amd64.tar.gz

# Extract to /usr/local
sudo tar -C /usr/local -xzf go1.23.4.linux-amd64.tar.gz

# Add Go to your PATH (add to ~/.bashrc for permanence)
export PATH=$PATH:/usr/local/go/bin

# Verify installation
go version
```

## Quick Start - Run the Benchmark

### 1. Navigate to the benchmarks directory

```bash
cd /mnt/nvm/repos/triton-api/benchmarks
```

### 2. Build the benchmark tool

```bash
# Build the binary (or use build.sh script)
./build.sh

# Or manually build
go build -o triton_bench triton_bench.go

# View help
./triton_bench --help
```

### 3. Run benchmarks

#### Test all tracks with 64 concurrent clients for 60 seconds
```bash
./triton_bench \
  --images /mnt/nvm/KILLBOY_SAMPLE_PICTURES \
  --clients 64 \
  --duration 60 \
  --warmup 10 \
  --track all
```

#### Test specific track (e.g., Track D batch)
```bash
./triton_bench \
  --images /mnt/nvm/KILLBOY_SAMPLE_PICTURES \
  --clients 128 \
  --duration 60 \
  --track D_batch
```

#### Test with varying concurrency levels
```bash
# Low concurrency (16 clients)
./triton_bench --clients 16 --duration 30

# Medium concurrency (64 clients)
./triton_bench --clients 64 --duration 60

# High concurrency (256 clients)
./triton_bench --clients 256 --duration 120

# Maximum throughput test (512 clients)
./triton_bench --clients 512 --duration 180
```

## Command-Line Flags

```
--images string
    Directory containing test images (default: "/mnt/nvm/KILLBOY_SAMPLE_PICTURES")

--clients int
    Number of concurrent clients (default: 64)

--duration int
    Test duration in seconds (default: 60)

--warmup int
    Number of warmup requests per track (default: 10)

--output string
    Output JSON file path (default: "benchmarks/results/benchmark_results.json")

--track string
    Track to test: A, B, C, D_streaming, D_balanced, D_batch, or all (default: "all")
```

## Example Benchmark Scenarios

### Scenario 1: Quick Validation (30 seconds)
```bash
./triton_bench --mode quick
```

### Scenario 2: Standard Benchmark (60 seconds)
```bash
./triton_bench --mode full --clients 64 --duration 60 --track all
```

### Scenario 3: Maximum Throughput Test (180 seconds)
```bash
./triton_bench --mode full --clients 256 --duration 180 --track D_batch
```

### Scenario 4: Latency-Optimized Test
```bash
./triton_bench --mode full --clients 16 --duration 60 --track D_streaming
```

### Scenario 5: Progressive Load Test
```bash
# Run multiple tests with increasing concurrency
for clients in 16 32 64 128 256; do
  echo "Testing with $clients clients..."
  ./triton_bench --mode full --clients $clients --duration 60 --output "results/benchmark_${clients}clients.json"
  sleep 10
done
```

## Understanding the Results

The benchmark outputs:

### Per-Track Results
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
- Shows all tracks side-by-side
- **Speedup vs A**: How much faster compared to PyTorch baseline

## Output Files

Results are saved as JSON in `benchmarks/results/benchmark_results.json` by default.

Example output structure:
```json
{
  "timestamp": "2025-01-16T15:30:00Z",
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
    },
    ...
  }
}
```

## Tips for Best Results

1. **Start services first**: Make sure all Docker services are running
   ```bash
   cd /mnt/nvm/repos/triton-api
   docker compose up -d
   ```

2. **Wait for warmup**: The first few minutes after startup may show slower performance as TensorRT engines finalize

3. **Monitor GPU**: Use `nvidia-smi -l 1` in another terminal to watch GPU utilization

4. **Check logs**: Monitor batch sizes in Triton logs
   ```bash
   docker compose logs -f triton-api | grep -i batch
   ```

5. **Progressive testing**: Start with low concurrency and increase gradually

6. **Grafana monitoring**: Open http://localhost:3000 to see real-time metrics

## Troubleshooting

### "command not found: go"
- Make sure you added Go to your PATH: `export PATH=$PATH:/usr/local/go/bin`
- Add to ~/.bashrc for permanent: `echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc`

### "no JPEG images found"
- Check image directory exists: `ls /mnt/nvm/KILLBOY_SAMPLE_PICTURES`
- Use different directory: `--images /path/to/your/images`

### "Track not available"
- Check services are running: `docker compose ps`
- Check health endpoint: `curl http://localhost:9600/health`

### High error rate
- Increase timeout in code if needed
- Check Docker resource limits
- Monitor GPU memory: `nvidia-smi`

## Advanced: Building for Different Architectures

```bash
# Build for Linux AMD64 (default)
GOOS=linux GOARCH=amd64 go build -o triton_bench triton_bench.go

# Build for Linux ARM64
GOOS=linux GOARCH=arm64 go build -o triton_bench_arm64 triton_bench.go

# Build for macOS
GOOS=darwin GOARCH=amd64 go build -o triton_bench_mac triton_bench.go
```

## Next Steps

1. Run initial benchmark: `./triton_bench --mode quick`
2. Check results in `benchmarks/results/benchmark_results.json`
3. Monitor Grafana dashboard at http://localhost:3000
4. Experiment with different concurrency levels
5. Optimize based on results

Happy benchmarking! 🚀
