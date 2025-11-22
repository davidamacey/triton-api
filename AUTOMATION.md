# Automation Scripts Guide

Complete guide to all automation tools for the Triton YOLO Inference System.

**Organized by functional purpose** - find everything for a specific task in one folder.

---

## Quick Reference

| Task | Command | Location |
|------|---------|----------|
| **Export models** | `python export/export_models.py` | [export/](#export-model-export) |
| **Setup Track D** | `python dali/create_dali_letterbox_pipeline.py` | [dali/](#dali-track-d-setup) |
| **Run tests** | `bash tests/test_inference.sh` | [tests/](#tests-testing--validation) |
| **Benchmark** | `cd benchmarks && ./triton_bench --mode quick` | [benchmarks/](#benchmarks-performance-testing) |
| **Check services** | `bash scripts/check_services.sh` | [scripts/](#scripts-core-deployment) |

---

## Folder Structure

```
triton-api/
├── export/              # Model export for all tracks
├── dali/                # DALI preprocessing (Track D)
├── scripts/             # Core deployment & utilities
├── tests/               # Testing & validation
└── benchmarks/          # Performance benchmarking
```

---

## export/ (Model Export)

**Purpose:** Export YOLO models in all formats for Tracks A-C

| File | Description |
|------|-------------|
| `export_models.py` | **Main export tool** - Exports ONNX, TRT, End2End TRT |
| `export_small_only.sh` | Quick wrapper for small models only |
| `download_pytorch_models.sh` | Download .pt files for Track A |
| `cleanup_for_reexport.sh` | Clean old exports before re-export |

**Usage:**
```bash
# Export all formats for small model
docker compose exec yolo-api python /app/export/export_models.py \
    --models small \
    --formats trt trt_end2end

# Or use the wrapper
bash export/export_small_only.sh
```

**See:** [docs/MODEL_EXPORT_GUIDE.md](docs/MODEL_EXPORT_GUIDE.md)

---

## dali/ (Track D Setup)

**Purpose:** DALI GPU-accelerated preprocessing with affine transformation

| File | Description |
|------|-------------|
| `create_dali_letterbox_pipeline.py` | Create DALI preprocessing pipeline |
| `validate_dali_letterbox.py` | Validate DALI pipeline accuracy |
| `create_ensembles.py` | Create DALI + TRT End2End ensembles |
| `dali_validation_results.txt` | Validation test results |

**Key Features:**
- Uses affine transformation with **CPU-calculated matrices**
- GPU operations: decode (nvJPEG) + warp_affine + normalize
- 3 ensemble tiers: streaming (0.1ms), balanced (0.5ms), batch (5ms)

**Usage:**
```bash
# 1. Create DALI pipeline
docker compose exec yolo-api python /app/dali/create_dali_letterbox_pipeline.py

# 2. Validate it
docker compose exec yolo-api python /app/dali/validate_dali_letterbox.py

# 3. Create ensembles
docker compose exec yolo-api python /app/dali/create_ensembles.py --models small

# 4. Restart Triton
docker compose restart triton-api
```

**See:** [docs/Tracks/TRACK_D_COMPLETE.md](docs/Tracks/TRACK_D_COMPLETE.md)

---

## scripts/ (Core Utilities)

**Purpose:** Essential system utilities

| File | Description |
|------|-------------|
| `check_services.sh` | Comprehensive service health check |

**Note:** With direct .plan files, we no longer need warmup or engine build scripts.

**Usage:**
```bash
# Check if everything is running
bash scripts/check_services.sh
```

---

## tests/ (Testing & Validation)

**Purpose:** Integration tests and validation scripts

| File | Description |
|------|-------------|
| `test_inference.sh` | Quick integration test (10 images per track) |
| `test_onnx_end2end.py` | Test ONNX End2End model locally |
| `test_end2end_patch.py` | Verify ultralytics end2end patch |
| `create_test_images.py` | Generate test images (various aspect ratios) |

**Usage:**
```bash
# Quick integration test
bash tests/test_inference.sh

# Test ONNX End2End model
docker compose exec yolo-api python /app/tests/test_onnx_end2end.py

# Verify end2end patch is working
docker compose exec yolo-api python /app/tests/test_end2end_patch.py
```

---

## benchmarks/ (Performance Testing)

**Purpose:** Comprehensive performance testing with Go tool

| File | Description |
|------|-------------|
| `triton_bench.go` | Benchmark source code |
| `triton_bench` | Compiled Go binary (8.3MB) |
| `build.sh` | Build script |

**Usage:**
```bash
cd benchmarks

# Build tool once
./build.sh

# Quick test (30 seconds)
./triton_bench --mode quick

# Full benchmark
./triton_bench --mode full --clients 128 --duration 60

# Sustained throughput test
./triton_bench --mode sustained
```

**See:** [benchmarks/README.md](benchmarks/README.md)

---

## Common Workflows

### First-Time Setup

```bash
# 1. Download PyTorch models
bash export/download_pytorch_models.sh

# 2. Export TensorRT models
docker compose exec yolo-api python /app/export/export_models.py \
    --formats trt trt_end2end --models small

# 3. Setup Track D (DALI)
docker compose exec yolo-api python /app/dali/create_dali_letterbox_pipeline.py
docker compose exec yolo-api python /app/dali/create_ensembles.py --models small

# 4. Start services
docker compose up -d

# 5. Test
bash tests/test_inference.sh

# 6. Benchmark
cd benchmarks && ./triton_bench --mode quick
```

### Re-export Models

```bash
# Clean old exports
bash export/cleanup_for_reexport.sh

# Export fresh
docker compose exec yolo-api python /app/export/export_models.py \
    --formats trt trt_end2end --models small

# Restart
docker compose restart
```

### Deploy Track D

```bash
# Export end2end models
docker compose exec yolo-api python /app/export/export_models.py \
    --formats trt_end2end --models small

# Create DALI pipeline
docker compose exec yolo-api python /app/dali/create_dali_letterbox_pipeline.py

# Create ensembles
docker compose exec yolo-api python /app/dali/create_ensembles.py --models small

# Restart Triton
docker compose restart triton-api

# Validate
docker compose exec yolo-api python /app/dali/validate_dali_letterbox.py

# Check status
bash scripts/check_services.sh
```

---

## Container Paths

Scripts are mounted at different paths inside containers:

| Host Path | Container Path | Used By |
|-----------|---------------|---------|
| `export/` | `/app/export/` | yolo-api |
| `dali/` | `/app/dali/` | yolo-api |
| `scripts/` | `/app/scripts/` | All containers |
| `tests/` | `/app/tests/` | yolo-api |
| `benchmarks/` | `/app/benchmarks/` | yolo-api |

**Example:**
```bash
# From host
bash export/export_small_only.sh

# This runs inside container
docker compose exec yolo-api python /app/export/export_models.py ...
```

---

## Documentation

- **[MODEL_EXPORT_GUIDE.md](docs/MODEL_EXPORT_GUIDE.md)** - Complete model export guide
- **[TRACK_D_COMPLETE.md](docs/Tracks/TRACK_D_COMPLETE.md)** - Track D DALI guide
- **[DEPLOYMENT_GUIDE.md](docs/DEPLOYMENT_GUIDE.md)** - Deployment instructions
- **[TESTING.md](docs/TESTING.md)** - Testing procedures
- **[benchmarks/README.md](benchmarks/README.md)** - Benchmark tool docs

---

## Migration from Old Structure

**Old paths → New paths:**

| Old | New |
|-----|-----|
| `scripts/export_models.py` | `export/export_models.py` |
| `scripts/create_dali_letterbox_pipeline.py` | `dali/create_dali_letterbox_pipeline.py` |
| `scripts/create_ensembles.py` | `dali/create_ensembles.py` |
| `scripts/test_inference.sh` | `tests/test_inference.sh` |

**Note:** All v2 DALI scripts and `deploy_100pct_gpu_dali.sh` have been removed as they did not work.

---

**Last Updated:** November 2025
