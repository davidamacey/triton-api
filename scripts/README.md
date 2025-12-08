# Scripts

Utility scripts for deployment, maintenance, and image processing.

## Root-Level Scripts

### check_services.sh

Health check for all services and models.

```bash
bash scripts/check_services.sh
```

**Checks:**
- Docker container status
- Triton server health (port 4600)
- Model availability (Tracks B/C/D)
- FastAPI endpoints (port 4603)
- GPU status and memory

### resize_images.py

Batch image resizing utility with multiprocessing support.

```bash
# Resize images to 640px max dimension
python scripts/resize_images.py /path/to/images --size 640

# Custom output directory and worker count
python scripts/resize_images.py /path/to/images --size 1024 --output /path/to/output --workers 16
```

**Features:**
- Maintains aspect ratio
- 100% JPEG quality (no compression artifacts)
- Parallel processing via multiprocessing
- Progress bar with ETA

## Subfolders

### track_e/

Track E setup and configuration scripts (MobileCLIP + OpenSearch).

See [track_e/README.md](track_e/README.md) for complete documentation.

**Scripts:**
- `create_triton_configs.py` - Generate Triton model configurations
- `setup_mobileclip_env.sh` - Host-side MobileCLIP environment setup
- `install_mobileclip_deps.sh` - Container-side dependency installation

**Note:** Ingestion and search operations are now available via FastAPI endpoints (`/track_e/ingest`, `/track_e/search/*`).

## Makefile Operations

Many common operations are now available via Makefile:

```bash
make help          # List all available targets
make up            # Start all services
make down          # Stop all services
make logs          # View service logs
make test          # Run tests
```

## Related Folders

| Folder | Purpose |
|--------|---------|
| [export/](../export/) | Model export scripts (ONNX, TensorRT) |
| [tests/](../tests/) | Test scripts and utilities |
| [dali/](../dali/) | DALI pipeline creation and validation |
| [benchmarks/](../benchmarks/) | Go-based benchmarking tool |

## Port Reference

| Service | Port |
|---------|------|
| FastAPI (all tracks) | 4603 |
| Triton HTTP | 4600 |
| Triton gRPC | 4601 |
| Triton metrics | 4602 |
| Grafana | 4605 |
| OpenSearch | 4607 |
