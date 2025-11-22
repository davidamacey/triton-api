# Scripts Reference Guide

Core deployment and utility scripts for the Triton YOLO inference system.

**Scripts in this folder: 1** (minimal, production-ready)

---

## What's in scripts/

This folder contains only essential deployment utilities. With direct TensorRT .plan files, we no longer need warmup or engine build scripts.

### check_services.sh - Service Health Check

**Purpose:** Comprehensive health check for all services and models

```bash
bash scripts/check_services.sh
```

**Checks:**
- Docker containers status
- Triton server health
- Model availability (all 6 models: Track B, C, D variants)
- FastAPI endpoints (Tracks A, B, C, D)
- GPU status and memory

**Output:**
- ✓ Green: Service healthy
- ✗ Red: Service down or error
- ⚠ Yellow: Warning or partial availability

---

## Removed Scripts (No Longer Needed)

These scripts were removed because we now have direct .plan files:

- ❌ **build_engines.sh** - Not needed (direct .plan files)
- ❌ **warmup_models.sh** - Not needed (no engine conversion required)
- ❌ **entrypoint_with_warmup.sh** - Simplified to direct uvicorn command
- ❌ **check_gpu_memory.sh** - Replaced by Grafana dashboard + nvtop

---

## Other Scripts (Organized by Folder)

Scripts have been organized into functional folders:

### Model Export → [../export/](../export/)
- `export_models.py` - Main export tool (all tracks)
- `export_small_only.sh` - Quick wrapper for small models
- `download_pytorch_models.sh` - Download PyTorch .pt files
- `cleanup_for_reexport.sh` - Clean before re-export

### Track D Setup → [../dali/](../dali/)
- `create_dali_letterbox_pipeline.py` - DALI GPU preprocessing
- `validate_dali_letterbox.py` - Validate DALI accuracy
- `create_ensembles.py` - DALI + TRT ensembles

### Testing → [../tests/](../tests/)
- `test_inference.sh` - Integration test (10 images)
- `test_onnx_end2end.py` - Test ONNX End2End locally
- `test_end2end_patch.py` - Verify ultralytics patch
- `create_test_images.py` - Generate test images

### Benchmarking → [../benchmarks/](../benchmarks/)
- `triton_bench.go` - Master benchmark tool (7 modes)
- `triton_bench` - Compiled binary
- `build.sh` - Build the benchmark tool

---

## Common Workflows

### First-Time Deployment

```bash
# 1. Check services
bash scripts/check_services.sh

# 2. Test inference
bash tests/test_inference.sh

# 3. Benchmark
cd benchmarks && ./triton_bench --mode quick
```

### Monitor System

```bash
# Service health
bash scripts/check_services.sh

# GPU utilization
nvidia-smi -l 1

# Grafana dashboard
open http://localhost:3000  # admin/admin

# View logs
docker compose logs -f triton-api
docker compose logs -f yolo-api
```

### Troubleshooting

```bash
# Check everything
bash scripts/check_services.sh

# View errors
docker compose logs triton-api | grep -i error
docker compose logs yolo-api | grep -i error

# Restart services
docker compose restart

# Check GPU
nvidia-smi
```

---

## Why So Few Scripts?

**Before (20 scripts):**
- Complex warmup orchestration
- Engine build automation
- GPU memory monitoring scripts
- Multiple entrypoints

**Now (1 script):**
- Direct .plan files (no engine building needed)
- Grafana for monitoring (no custom GPU scripts)
- Simple uvicorn command (no warmup entrypoint)
- Organized folders (export, dali, tests, benchmarks)

**Result:** Cleaner, more maintainable, production-ready.

---

## Documentation

- **[../AUTOMATION.md](../AUTOMATION.md)** - Complete automation guide
- **[../export/README.md](../export/README.md)** - Model export guide
- **[../tests/README.md](../tests/README.md)** - Testing guide
- **[../benchmarks/README.md](../benchmarks/README.md)** - Benchmark tool docs
- **[../docs/](../docs/)** - Technical reference documents

---

**Last Updated:** January 2025
