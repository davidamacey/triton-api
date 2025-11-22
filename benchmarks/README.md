# Triton YOLO Benchmark Suite

**Version 1.0.0** - Professional benchmarking tool for NVIDIA Triton YOLO inference

Single comprehensive Go tool for all benchmarking scenarios - no Python required!

---

## Quick Start

### 1. Install Go (One-time setup)

```bash
# Download and install Go 1.23.4
cd /tmp
wget https://go.dev/dl/go1.23.4.linux-amd64.tar.gz
sudo tar -C /usr/local -xzf go1.23.4.linux-amd64.tar.gz

# Add to PATH
export PATH=$PATH:/usr/local/go/bin

# Make permanent (add to ~/.bashrc)
echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc

# Verify installation
go version
```

### 2. Build the Benchmark Tool

```bash
cd /mnt/nvm/repos/triton-api/benchmarks
./build.sh
```

### 3. Run Your First Test

```bash
# Quick concurrency test (30 seconds, all tracks)
./triton_bench --mode quick

# That's it! Results are automatically saved to benchmarks/results/
```

---

## All Test Modes

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
```
Fast 30-second test with 16 clients - verify batching works

### Mode 4: Full Concurrency
```bash
./triton_bench --mode full --clients 128 --duration 60
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

---

## Common Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `quick` | Test mode |
| `--images` | `./test_images` | Image directory |
| `--track` | `all` | Track filter (A, B, C, D_streaming, D_balanced, D_batch, all) |
| `--clients` | `64` | Concurrent clients |
| `--duration` | `60` | Test duration (seconds) |
| `--limit` | `100` | Max images to load |
| `--warmup` | `10` | Warmup requests |
| `--quiet` | `false` | Minimal output |
| `--json` | `false` | JSON output only |

---

## Output Files

Results automatically saved to `benchmarks/results/` with timestamps:

```
benchmarks/results/
├── single_image_20250116_153045.json
├── quick_concurrency_20250116_153215.json
├── full_concurrency_20250116_154330.json
└── sustained_throughput_20250116_160102.json
```

---

## Example Workflows

### Initial Validation
```bash
./triton_bench --mode single       # Verify correctness
./triton_bench --mode quick        # Check batching
./triton_bench --mode full         # Full benchmark
```

### Find Maximum Throughput
```bash
./triton_bench --mode sustained
```

### Process Production Dataset
```bash
./triton_bench --mode all --images /path/to/data --clients 128
```

### Load Testing
```bash
# Progressive load
for clients in 16 32 64 128 256; do
  ./triton_bench --mode full --clients $clients --duration 60
done
```

---

## Monitoring

While running benchmarks, monitor in separate terminals:

```bash
# Terminal 1: GPU utilization
nvidia-smi -l 1

# Terminal 2: Check batching
docker compose logs -f triton-api | grep "batch size"

# Terminal 3: Grafana
# http://localhost:3000
```

---

## Troubleshooting

### No tracks available
```bash
docker compose ps                # Check services
curl http://localhost:9600/health  # Test endpoint
```

### No speed gains
```bash
# Verify workers
docker compose exec yolo-api ps aux | grep uvicorn | wc -l
# Should show 33 (1 master + 32 workers)

# Check batching
docker compose logs triton-api | grep "batch size"
# Should show batch sizes > 1
```

### High error rate
```bash
docker stats                      # Check resources
docker compose logs triton-api    # Check errors
```

---

## Migration from Python Scripts

All old Python scripts archived in `_archived_python_scripts/`:

- ~~simple_benchmark.py~~ → `triton_bench --mode set`
- ~~quick_concurrency_test.py~~ → `triton_bench --mode quick`
- ~~async_benchmark.py~~ → `triton_bench --mode full`
- ~~advanced_benchmark.py~~ → `triton_bench --mode sustained`

**Everything now in one tool!**

---

For detailed documentation, see [DEPLOYMENT_GUIDE.md](../DEPLOYMENT_GUIDE.md)

**Happy Benchmarking!** 🚀
