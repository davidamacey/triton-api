# Track E Setup Scripts

Setup and configuration scripts for deploying the Track E visual search system.

## Overview

This directory contains **setup and configuration scripts only**.

**Note:** Ingestion and search operations are now available via FastAPI endpoints:
- `POST /track_e/ingest` - Ingest images to OpenSearch
- `POST /track_e/search/image` - Image-to-image search
- `POST /track_e/search/text` - Text-to-image search
- `POST /track_e/search/object` - Object-level search
- `GET /track_e/index/stats` - Index statistics
- `POST /track_e/index/create` - Create index
- `DELETE /track_e/index` - Delete index

## Scripts

### `setup_mobileclip_env.sh`

**Purpose:** Host-side MobileCLIP environment setup

**IMPORTANT: Run on HOST, not in container:**
```bash
# Run on host machine (not inside Docker container)
bash scripts/track_e/setup_mobileclip_env.sh
```

Creates conda environment, clones MobileCLIP, installs dependencies, downloads weights.

This script must run on the host because it:
- Clones reference repositories to `reference_repos/`
- Downloads model weights that are mounted into containers
- Sets up host environment for model export

---

### `install_mobileclip_deps.sh`

**Purpose:** Container-side dependency installation

```bash
docker compose exec yolo-api bash /app/scripts/track_e/install_mobileclip_deps.sh
```

Installs Python packages and configures the container environment.

---

### `create_triton_configs.py`

**Purpose:** Generate Triton model configurations

```bash
docker compose exec yolo-api python /app/scripts/track_e/create_triton_configs.py
```

Creates config.pbtxt files for all Track E models (encoders, ensemble).

---

## Initial Setup Workflow

```bash
# 1. Setup MobileCLIP environment (run on HOST, not in container)
bash scripts/track_e/setup_mobileclip_env.sh

# 2. Export MobileCLIP models (scripts in /app/export/)
make export-mobileclip

# 3. Create Triton configs
docker compose exec yolo-api python /app/scripts/track_e/create_triton_configs.py

# 4. Create DALI pipeline
make create-dali-dual

# 5. Restart Triton
make restart-triton

# 6. Create OpenSearch index
curl -X POST http://localhost:4603/track_e/index/create

# 7. Test endpoints
curl -X POST http://localhost:4603/track_e/predict -F "image=@test.jpg"
```

---

## API Endpoints Reference

### Ingestion

```bash
# Ingest single image
curl -X POST http://localhost:4603/track_e/ingest \
  -F "file=@image.jpg" \
  -F "image_id=my_image_001" \
  -F 'metadata={"source":"dataset_a"}'
```

### Search

```bash
# Image-to-image search
curl -X POST http://localhost:4603/track_e/search/image \
  -F "image=@query.jpg" \
  -F "top_k=10"

# Text-to-image search
curl -X POST http://localhost:4603/track_e/search/text \
  -H "Content-Type: application/json" \
  -d '{"text": "red car on highway"}'

# Object search (using first detected object)
curl -X POST http://localhost:4603/track_e/search/object \
  -F "image=@query.jpg" \
  -F "box_index=0" \
  -F "top_k=10"
```

### Index Management

```bash
# Get index stats
curl http://localhost:4603/track_e/index/stats

# Create index
curl -X POST http://localhost:4603/track_e/index/create

# Delete index
curl -X DELETE http://localhost:4603/track_e/index
```

---

## Related Locations

- **Export scripts:** `/app/export/export_mobileclip_*.py`
- **Test scripts:** `/app/tests/test_track_e_*.py`
- **DALI pipelines:** `/app/dali/`
- **OpenSearch client:** `/app/src/clients/opensearch.py`
- **Visual search service:** `/app/src/services/visual_search.py`
- **Track E router:** `/app/src/routers/track_e.py`

---

## Environment Variables

- `TRITON_URL`: Triton gRPC URL (default: localhost:4601)
- `OPENSEARCH_URL`: OpenSearch URL (default: http://localhost:4607)

---

## Makefile Commands

```bash
make setup-track-e      # Complete Track E setup
make export-mobileclip  # Export MobileCLIP models
make create-dali-dual   # Create DALI preprocessing pipeline
make test-track-e       # Run Track E tests
```
