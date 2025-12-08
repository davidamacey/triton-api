# Track E: Visual Search with MobileCLIP

Complete deployment and usage guide for the Track E visual search system.

## Overview

Track E is a production-grade visual search system that combines:
- **YOLO object detection** (GPU-accelerated with TensorRT)
- **MobileCLIP visual-language embeddings** (Apple's efficient CLIP variant)
- **OpenSearch k-NN vector search** (HNSW algorithm for fast similarity search)

### Key Features

- ✅ **Native Resolution Processing**: No arbitrary upscaling/downscaling
- ✅ **Normalized Bounding Boxes**: Output in [0, 1] range for any image size
- ✅ **Full GPU Pipeline**: DALI preprocessing + TensorRT inference
- ✅ **Multi-Modal Search**: Image-to-image, text-to-image, object-to-object
- ✅ **Production Ready**: Caching, monitoring, async API, comprehensive tests

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Track E Pipeline                         │
└─────────────────────────────────────────────────────────────────┘

Input Image (JPEG)
    │
    ▼
┌──────────────────────────────────┐
│  DALI Triple-Branch Preprocessing │
│  (GPU-accelerated)                │
└──────────────────────────────────┘
    │
    ├─► YOLO Branch (640×640 letterbox)
    ├─► MobileCLIP Branch (256×256 center crop)
    └─► Original Branch (native resolution, no resize!)
    │
    ▼
┌──────────────────────────────────┐
│  Parallel Inference               │
│  - YOLO: Object detection (TRT)   │
│  - MobileCLIP: Global embedding   │
└──────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────┐
│  Box Embedding Extractor          │
│  (Python Backend + BLS)           │
│  - Crops from native-res image    │
│  - Encodes per-object embeddings  │
│  - Normalizes boxes to [0, 1]     │
└──────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────┐
│  OpenSearch k-NN Indexing         │
│  - Global embedding (HNSW)        │
│  - Per-object embeddings (nested) │
│  - Normalized box coordinates     │
└──────────────────────────────────┘
    │
    ▼
Visual Search API
```

## Quick Start

### 1. Setup MobileCLIP (Host-Side, One-Time)

Run this on your **host machine** to clone repos and download the checkpoint:

```bash
# Clone repos and download MobileCLIP2-S2 checkpoint (~200MB)
bash scripts/track_e/setup_mobileclip_env.sh
```

This creates:
- `reference_repos/ml-mobileclip/` - Apple's MobileCLIP repository
- `reference_repos/open_clip/` - Patched OpenCLIP with MobileCLIP2 support
- `pytorch_models/mobileclip2_s2/mobileclip2_s2.pt` - Model checkpoint (~398 MB)

### 2. Deploy Services

```bash
# Build and start all services (Triton, OpenSearch, FastAPI)
docker compose up -d --build

# Wait for OpenSearch to be healthy
docker compose ps opensearch

# Check health
curl http://localhost:4603/health

# Verify OpenSearch is running (no auth needed - security disabled)
curl http://localhost:4607
```

### Port Reference

| Service | External Port | Internal Port | Description |
|---------|--------------|---------------|-------------|
| Triton HTTP | 4600 | 8000 | Triton REST API |
| Triton gRPC | 4601 | 8001 | Triton gRPC API |
| Triton Metrics | 4602 | 8002 | Triton Prometheus metrics |
| FastAPI (yolo-api) | 4603 | 8000 | YOLO + Track E API |
| Prometheus | 4604 | 9090 | Metrics collection |
| Grafana | 4605 | 3000 | Dashboards |
| Loki | 4606 | 3100 | Log aggregation |
| OpenSearch | 4607 | 9200 | Vector database |
| OpenSearch Dashboards | 4608 | 5601 | OpenSearch UI |

### 3. Install MobileCLIP Python Deps (Container-Side)

```bash
# Install OpenCLIP and mobileclip from mounted repos
docker compose exec yolo-api bash /app/scripts/track_e/install_mobileclip_deps.sh
```

### 4. Export MobileCLIP Models

```bash
# Export image encoder to ONNX/TensorRT
docker compose exec yolo-api python /app/scripts/track_e/export_mobileclip_image_encoder.py

# Export text encoder to ONNX/TensorRT
docker compose exec yolo-api python /app/scripts/track_e/export_mobileclip_text_encoder.py
```

### 5. Create DALI Pipeline

```bash
# Create triple-branch DALI preprocessing pipeline
docker compose exec yolo-api python /app/dali/create_dual_dali_pipeline.py
```

### 6. Create Triton Configs

```bash
# Generate all Triton model configs
docker compose exec yolo-api python /app/scripts/track_e/create_triton_configs.py
```

### 7. Restart Triton

```bash
# Load Track E models
docker compose restart triton-api

# Verify models loaded
docker compose logs triton-api | grep mobileclip
```

### 8. Create OpenSearch Index

```bash
# Create visual search index with k-NN
curl -X POST "http://localhost:9600/track_e/index/create?force_recreate=true"
```

### 9. Ingest Images

```bash
# Ingest directory of images
docker compose exec yolo-api python /app/scripts/track_e/ingest_to_opensearch.py \
    --image_dir /app/test_images \
    --batch_size 16
```

### 10. Test Search

```bash
# Image-to-image search
curl -X POST "http://localhost:9600/track_e/search/image" \
    -F "file=@/path/to/query.jpg" \
    -F "top_k=10"

# Text-to-image search
curl -X POST "http://localhost:9600/track_e/search/text" \
    -H "Content-Type: application/json" \
    -d '{"query_text": "a dog playing in the park"}'
```

## API Reference

### Ingestion

#### POST /track_e/ingest

Ingest single image into visual search index.

**Request:**
```bash
curl -X POST "http://localhost:9600/track_e/ingest" \
    -F "file=@image.jpg" \
    -F "image_id=my_image_001" \
    -F "metadata={\"category\":\"products\",\"tags\":[\"outdoor\"]}"
```

**Response:**
```json
{
  "status": "success",
  "image_id": "my_image_001",
  "message": "Image ingested successfully with 3 detections",
  "num_detections": 3,
  "global_embedding_norm": 1.0002
}
```

### Search

#### POST /track_e/search/image

Image-to-image similarity search.

**Request:**
```bash
curl -X POST "http://localhost:9600/track_e/search/image?top_k=5" \
    -F "file=@query.jpg"
```

**Response:**
```json
{
  "status": "success",
  "query_type": "image",
  "results": [
    {
      "image_id": "img_001",
      "image_path": "/path/to/img_001.jpg",
      "score": 0.9823,
      "num_detections": 5,
      "metadata": {"category": "products"}
    }
  ],
  "total_results": 5,
  "search_time_ms": 15.23
}
```

#### POST /track_e/search/text

Text-to-image search using MobileCLIP text encoder.

**Request:**
```bash
curl -X POST "http://localhost:9600/track_e/search/text?top_k=10" \
    -H "Content-Type: application/json" \
    -d '{"query_text": "red sports car on highway"}'
```

**Response:**
Same format as image search.

#### POST /track_e/search/object

Object-to-object search (searches within detected objects).

**Request:**
```bash
curl -X POST "http://localhost:9600/track_e/search/object?top_k=5&class_filter=0,2" \
    -F "file=@cropped_object.jpg"
```

**Response:**
```json
{
  "status": "success",
  "query_type": "object",
  "results": [
    {
      "image_id": "img_002",
      "image_path": "/path/to/img_002.jpg",
      "score": 0.9456,
      "matched_objects": [
        {
          "box": [0.123, 0.456, 0.789, 0.912],
          "class_id": 0,
          "score": 0.95,
          "similarity": 0.9456
        }
      ]
    }
  ],
  "total_results": 5,
  "search_time_ms": 18.67
}
```

### Index Management

#### GET /track_e/index/stats

Get index statistics.

**Response:**
```json
{
  "status": "success",
  "total_documents": 10000,
  "index_size_mb": 245.67
}
```

#### POST /track_e/index/create

Create or recreate visual search index.

**Request:**
```bash
curl -X POST "http://localhost:9600/track_e/index/create?force_recreate=true"
```

#### DELETE /track_e/index

Delete visual search index (WARNING: deletes all data).

### Cache Management

#### GET /track_e/cache/stats

Get embedding cache statistics.

**Response:**
```json
{
  "status": "success",
  "caches": {
    "image_cache": {
      "size": 423,
      "max_size": 1000,
      "hits": 8765,
      "misses": 1234,
      "hit_rate": 0.876,
      "ttl_seconds": 3600
    },
    "text_cache": {
      "size": 89,
      "max_size": 1000,
      "hits": 456,
      "misses": 123,
      "hit_rate": 0.787,
      "ttl_seconds": 3600
    }
  }
}
```

#### POST /track_e/cache/clear

Clear all embedding caches.

## Performance Tuning

### OpenSearch Configuration

Edit `docker-compose.yml` to adjust OpenSearch memory:

```yaml
environment:
  - "OPENSEARCH_JAVA_OPTS=-Xms4g -Xmx4g"  # Increase heap for large indices
```

**OpenSearch 3.x Notes:**
- Security plugin disabled by default (`DISABLE_SECURITY_PLUGIN=true`)
- No authentication required for development
- For production, enable security and SSL

### k-NN Algorithm Parameters

Adjust HNSW parameters in `src/opensearch_client.py`:

```python
"method": {
    "name": "hnsw",
    "space_type": "cosinesimil",
    "engine": "faiss",  # OpenSearch 3.x uses faiss (nmslib deprecated)
    "parameters": {
        "ef_construction": 512,  # Build quality (higher = better, slower)
        "m": 16,  # Graph connectivity (higher = better quality, more memory)
        "ef_search": 512  # Search quality
    }
}
```

**Recommendations:**
- Small datasets (<10K images): `ef_construction=256, m=8`
- Medium datasets (10K-100K): `ef_construction=512, m=16` (default)
- Large datasets (>100K): `ef_construction=1024, m=32`

### Embedding Cache

Adjust cache size in `src/cache_utils.py`:

```python
# Larger cache for frequently searched images
cache = EmbeddingCache(max_size=5000, ttl_seconds=7200)
```

### Batch Ingestion

For large-scale ingestion, use batch mode:

```bash
docker compose exec yolo-api python /app/scripts/track_e/ingest_to_opensearch.py \
    --image_dir /data/images \
    --batch_size 32 \
    --reset_index
```

## Monitoring

### Prometheus Metrics

Track E exposes metrics via Grafana (http://localhost:3000):

- **Search latency** (P50/P95/P99)
- **Cache hit rate**
- **Index size and growth**
- **Inference throughput**

### Performance Benchmarks

Run integration tests to measure performance:

```bash
docker compose exec yolo-api python /app/scripts/track_e/test_integration.py
```

Expected performance (RTX A6000):
- **Image ingestion**: ~50-80 images/sec
- **Search latency (P50)**: <20ms
- **Search latency (P95)**: <50ms
- **Cache hit rate**: >80% (after warmup)

## Troubleshooting

### Issue: "Failed to connect to OpenSearch"

**Solution:**
```bash
# Check OpenSearch is running
docker compose ps opensearch

# Check logs
docker compose logs opensearch

# Restart if needed
docker compose restart opensearch
```

### Issue: "Triton model not found: mobileclip2_s2_image_encoder"

**Solution:**
```bash
# Verify models exported
ls -la models/mobileclip2_s2_image_encoder/1/

# Re-export if needed
docker compose exec yolo-api python /app/scripts/track_e/export_mobileclip_image_encoder.py

# Restart Triton
docker compose restart triton-api
```

### Issue: "Search returns no results"

**Solution:**
```bash
# Check index exists and has documents
curl http://localhost:9600/track_e/index/stats

# Verify documents were ingested
curl http://localhost:9200/visual_search/_search?pretty

# Re-ingest if needed
```

### Issue: "Out of memory during ingestion"

**Solution:**
```bash
# Reduce batch size
python scripts/track_e/ingest_to_opensearch.py --batch_size 8

# Or increase OpenSearch heap
# Edit docker-compose.yml: OPENSEARCH_JAVA_OPTS=-Xms4g -Xmx4g
```

## Advanced Usage

### Custom Metadata Filtering

Index images with custom metadata:

```bash
curl -X POST "http://localhost:9600/track_e/ingest" \
    -F "file=@image.jpg" \
    -F "metadata={\"product_id\":\"SKU123\",\"brand\":\"Nike\",\"price\":99.99}"
```

Search with filters:

```python
# Coming soon: filter_metadata parameter in API
```

### Hybrid Search (Text + Visual)

Combine text and visual search for better results:

```python
# 1. Text search to filter candidates
text_results = search_by_text("red sports car")

# 2. Re-rank with image similarity
top_image = text_results[0]
visual_results = search_by_image(top_image, filter=text_results[:100])
```

### Production Deployment

For production, update `docker-compose.yml`:

```yaml
opensearch:
  environment:
    - plugins.security.ssl.http.enabled=true  # Enable SSL
  deploy:
    replicas: 3  # Multi-node cluster
```

Enable authentication in `src/opensearch_client.py`:

```python
OpenSearchClient(
    hosts=["https://opensearch:9200"],
    http_auth=("admin", os.getenv("OPENSEARCH_PASSWORD")),
    verify_certs=True
)
```

## File Reference

### Core Pipeline Files

- `dali/create_dual_dali_pipeline.py` - Triple-branch DALI preprocessing
- `models/box_embedding_extractor/1/model.py` - Python backend for object embeddings
- `scripts/track_e/create_triton_configs.py` - Triton config generator
- `scripts/track_e/export_mobileclip_*.py` - MobileCLIP model exporters

### API and Services

- `src/routers/track_e.py` - FastAPI router with all endpoints
- `src/opensearch_client.py` - Async OpenSearch client with k-NN
- `src/cache_utils.py` - Embedding cache for performance

### Utilities

- `scripts/track_e/ingest_to_opensearch.py` - Batch ingestion script
- `scripts/track_e/search_visual.py` - CLI search tool
- `scripts/track_e/test_integration.py` - Integration test suite
- `scripts/track_e/validate_mobileclip_triton.py` - Model validation

## Next Steps

1. **Scale to production**: Multi-node OpenSearch cluster, load balancing
2. **Add filters**: Metadata filtering in search API
3. **Optimize indices**: Tune HNSW parameters for your dataset size
4. **Monitor performance**: Set up Grafana dashboards for Track E metrics
5. **Custom embeddings**: Fine-tune MobileCLIP on your domain data

## Support

For issues and questions:
- Check logs: `docker compose logs triton-api opensearch yolo-api`
- Run tests: `python scripts/track_e/test_integration.py`
- Review status doc: `docs/TRACK_E_IMPLEMENTATION_STATUS.md`
