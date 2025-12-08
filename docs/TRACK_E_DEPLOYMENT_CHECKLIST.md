# Track E: Deployment Checklist

Step-by-step checklist for deploying the Track E visual search system.

## Prerequisites

- [ ] NVIDIA GPU with CUDA support (tested on RTX A6000)
- [ ] Docker and Docker Compose installed
- [ ] At least 16GB RAM (32GB recommended)
- [ ] At least 20GB free disk space
- [ ] NVIDIA Docker runtime configured

## Phase 1: Infrastructure Setup

### 1.1 Start Base Services

```bash
# Start Triton and FastAPI
docker compose up -d triton-api yolo-api

# Verify services are running
docker compose ps
```

- [ ] Triton server running on ports 9500-9502
- [ ] YOLO API running on port 9600
- [ ] Health check passes: `curl http://localhost:9600/health`

### 1.2 Start OpenSearch

```bash
# Start OpenSearch and dashboards
docker compose up -d opensearch opensearch-dashboards

# Wait for cluster to be healthy (may take 30-60 seconds)
sleep 30

# Check cluster health
curl http://localhost:9200/_cluster/health?pretty
```

- [ ] OpenSearch running on port 9200
- [ ] Dashboards running on port 5601
- [ ] Cluster status is "green" or "yellow"

**Troubleshooting:**
```bash
# If OpenSearch fails to start with memory errors:
# Increase Docker memory limit to at least 8GB

# Check logs
docker compose logs opensearch
```

## Phase 2: MobileCLIP Model Export

### 2.1 Setup MobileCLIP Environment

```bash
# Create conda environment with MobileCLIP
docker compose exec yolo-api bash /app/scripts/track_e/setup_mobileclip_env.sh
```

- [ ] Conda environment created
- [ ] MobileCLIP repository cloned
- [ ] Dependencies installed

**Expected output:**
```
✓ Conda environment created
✓ MobileCLIP installed
✓ Environment ready
```

### 2.2 Export Image Encoder

```bash
# Export MobileCLIP image encoder to ONNX → TensorRT
docker compose exec yolo-api python /app/scripts/track_e/export_mobileclip_image_encoder.py
```

- [ ] ONNX export successful
- [ ] TensorRT conversion complete
- [ ] Model file exists: `models/mobileclip2_s2_image_encoder/1/model.plan`
- [ ] Config file exists: `models/mobileclip2_s2_image_encoder/config.pbtxt`

**Verification:**
```bash
# Check model files
ls -lh models/mobileclip2_s2_image_encoder/1/
# Should see model.plan (~100-200 MB)
```

### 2.3 Export Text Encoder

```bash
# Export MobileCLIP text encoder to ONNX → TensorRT
docker compose exec yolo-api python /app/scripts/track_e/export_mobileclip_text_encoder.py
```

- [ ] ONNX export successful
- [ ] TensorRT conversion complete
- [ ] Model file exists: `models/mobileclip2_s2_text_encoder/1/model.plan`
- [ ] Config file exists: `models/mobileclip2_s2_text_encoder/config.pbtxt`

## Phase 3: DALI Pipeline

### 3.1 Create Triple-Branch DALI Pipeline

```bash
# Create DALI preprocessing pipeline
docker compose exec yolo-api python /app/dali/create_dual_dali_pipeline.py
```

- [ ] Pipeline built successfully
- [ ] Serialization complete
- [ ] Model file exists: `models/dual_preprocess_dali/1/model.dali`
- [ ] Config file exists: `models/dual_preprocess_dali/config.pbtxt`
- [ ] Test passed with validation

**Expected output:**
```
✓ YOLO output: Shape [3, 640, 640]
✓ MobileCLIP output: Shape [3, 256, 256]
✓ Original output: Shape [3, H, W]
✅ All assertions passed!
```

## Phase 4: Triton Configuration

### 4.1 Generate Model Configs

```bash
# Create all Triton model configs
docker compose exec yolo-api python /app/scripts/track_e/create_triton_configs.py
```

- [ ] Image encoder config created
- [ ] Text encoder config created
- [ ] Box embedding extractor config exists
- [ ] Ensemble config created

**Verification:**
```bash
# Check all configs exist
ls -l models/mobileclip2_s2_image_encoder/config.pbtxt
ls -l models/mobileclip2_s2_text_encoder/config.pbtxt
ls -l models/box_embedding_extractor/config.pbtxt
ls -l models/yolo_mobileclip_ensemble/config.pbtxt
```

### 4.2 Update Triton Load Models

Edit `docker-compose.yml` to add Track E models:

```yaml
triton-api:
  command:
    # ... existing models ...
    # Track E Models
    - --load-model=mobileclip2_s2_image_encoder
    - --load-model=mobileclip2_s2_text_encoder
    - --load-model=box_embedding_extractor
    - --load-model=dual_preprocess_dali
    - --load-model=yolo_mobileclip_ensemble
```

- [ ] Models added to Triton command
- [ ] File saved

### 4.3 Restart Triton

```bash
# Restart Triton to load Track E models
docker compose restart triton-api

# Wait for models to load (30-60 seconds)
sleep 45

# Check model status
docker compose logs triton-api | grep -E "(mobileclip|ensemble|box_embedding)"
```

- [ ] All 5 models loaded successfully
- [ ] No errors in logs
- [ ] Models in READY state

**Verification:**
```bash
# Query Triton for loaded models
curl localhost:9500/v2/models | jq '.[] | select(.name | contains("mobileclip"))'
```

## Phase 5: OpenSearch Index

### 5.1 Create Visual Search Index

```bash
# Create index with k-NN configuration
curl -X POST "http://localhost:9600/track_e/index/create?force_recreate=true"
```

- [ ] Index created successfully
- [ ] k-NN enabled
- [ ] HNSW algorithm configured

**Expected response:**
```json
{
  "status": "success",
  "message": "Index created successfully",
  "recreated": true
}
```

### 5.2 Verify Index

```bash
# Check index exists
curl "http://localhost:9600/track_e/index/stats" | jq
```

- [ ] Index exists with 0 documents
- [ ] Index size reported

## Phase 6: Test Pipeline

### 6.1 Test Ensemble Inference

```bash
# Test ensemble with sample image
docker compose exec yolo-api python /app/scripts/track_e/test_ensemble.py
```

- [ ] DALI preprocessing works
- [ ] YOLO detection works
- [ ] MobileCLIP encoding works
- [ ] Box embedding extraction works
- [ ] All outputs have correct shapes

**Expected output:**
```
✓ Ensemble output shapes correct
✓ Embeddings L2-normalized
✓ Boxes in [0, 1] range
✅ All tests passed!
```

### 6.2 Test Single Image Ingestion

```bash
# Find a test image
TEST_IMAGE=$(ls test_images/*.jpg | head -1)

# Ingest single image
curl -X POST "http://localhost:9600/track_e/ingest" \
    -F "file=@$TEST_IMAGE" \
    -F "image_id=test_001"
```

- [ ] Image ingested successfully
- [ ] Detections reported
- [ ] Embedding norm ~1.0

### 6.3 Test Image Search

```bash
# Search with same image (should return itself)
curl -X POST "http://localhost:9600/track_e/search/image?top_k=5" \
    -F "file=@$TEST_IMAGE" | jq
```

- [ ] Search returns results
- [ ] Top result is the ingested image
- [ ] Score is high (>0.99)
- [ ] Search time <100ms

## Phase 7: Batch Ingestion

### 7.1 Ingest Test Images

```bash
# Ingest all test images
docker compose exec yolo-api python /app/scripts/track_e/ingest_to_opensearch.py \
    --image_dir /app/test_images \
    --batch_size 16 \
    --reset_index
```

- [ ] All images ingested successfully
- [ ] No errors reported
- [ ] Index stats show correct document count

**Verification:**
```bash
# Check index stats
curl "http://localhost:9600/track_e/index/stats" | jq
```

### 7.2 Verify Search Quality

```bash
# Test various search modes
# Image search
curl -X POST "http://localhost:9600/track_e/search/image?top_k=10" \
    -F "file=@test_images/sample.jpg"

# Text search
curl -X POST "http://localhost:9600/track_e/search/text?top_k=10" \
    -H "Content-Type: application/json" \
    -d '{"query_text": "a person walking"}'

# Object search
curl -X POST "http://localhost:9600/track_e/search/object?top_k=10" \
    -F "file=@test_images/sample.jpg"
```

- [ ] Image search returns relevant results
- [ ] Text search works
- [ ] Object search returns results
- [ ] All searches complete in <100ms

## Phase 8: Integration Tests

### 8.1 Run Full Test Suite

```bash
# Run comprehensive integration tests
docker compose exec yolo-api python /app/scripts/track_e/test_integration.py
```

- [ ] Health check passes
- [ ] Index creation passes
- [ ] Single ingestion passes
- [ ] Batch ingestion passes
- [ ] Image search passes
- [ ] Text search passes
- [ ] Object search passes
- [ ] Index stats passes
- [ ] Performance benchmark passes (avg <500ms)

**Expected:** All tests pass (9/9 or 10/10)

### 8.2 Check Cache Performance

```bash
# Run searches multiple times to warm cache
for i in {1..10}; do
    curl -X POST "http://localhost:9600/track_e/search/image?top_k=5" \
        -F "file=@test_images/sample.jpg" > /dev/null 2>&1
done

# Check cache stats
curl "http://localhost:9600/track_e/cache/stats" | jq
```

- [ ] Cache hit rate >50% after warmup
- [ ] Cache size growing appropriately

## Phase 9: Production Readiness

### 9.1 Update Dependencies

```bash
# Ensure all Python packages installed
docker compose exec yolo-api pip install -r /app/requirements.txt
```

- [ ] `opensearch-py` installed
- [ ] `transformers` installed
- [ ] All dependencies up to date

### 9.2 Enable Monitoring

```bash
# Start Prometheus and Grafana
docker compose up -d prometheus grafana

# Access Grafana: http://localhost:3000 (admin/admin)
```

- [ ] Prometheus scraping metrics
- [ ] Grafana dashboards accessible
- [ ] Track E metrics visible

### 9.3 Security Configuration

For production deployment:

```yaml
# docker-compose.yml
opensearch:
  environment:
    - plugins.security.ssl.http.enabled=true
    - OPENSEARCH_INITIAL_ADMIN_PASSWORD=${OPENSEARCH_PASSWORD}
```

- [ ] SSL enabled for OpenSearch
- [ ] Strong admin password set
- [ ] Firewall rules configured
- [ ] API authentication enabled (if required)

### 9.4 Backup Configuration

```bash
# Create snapshot repository for OpenSearch
curl -X PUT "localhost:9200/_snapshot/backup_repo" -H 'Content-Type: application/json' -d'
{
  "type": "fs",
  "settings": {
    "location": "/backup/opensearch"
  }
}'
```

- [ ] Snapshot repository configured
- [ ] Backup schedule defined
- [ ] Restore procedure documented

## Phase 10: Documentation

### 10.1 Create Deployment Notes

Document your specific deployment:

- [ ] Server specifications
- [ ] Network configuration
- [ ] Custom modifications
- [ ] Performance benchmarks
- [ ] Known issues and solutions

### 10.2 Team Training

Prepare team for operations:

- [ ] API usage examples shared
- [ ] Monitoring dashboards explained
- [ ] Troubleshooting guide reviewed
- [ ] Backup/restore procedure tested

## Deployment Validation

Run final validation:

```bash
# Comprehensive test
docker compose exec yolo-api python /app/scripts/track_e/test_integration.py

# Performance test (10 searches)
for i in {1..10}; do
    curl -s -w "Time: %{time_total}s\n" \
        -X POST "http://localhost:9600/track_e/search/image?top_k=10" \
        -F "file=@test_images/sample.jpg" \
        -o /dev/null
done
```

**Success Criteria:**
- [ ] All integration tests pass
- [ ] Average search latency <100ms
- [ ] Cache hit rate >70%
- [ ] No errors in logs
- [ ] Index growing as expected

## Rollback Plan

If deployment fails:

```bash
# Stop Track E services
docker compose stop opensearch opensearch-dashboards

# Revert Triton config (remove Track E models from docker-compose.yml)
docker compose restart triton-api

# Check logs
docker compose logs --tail=100 opensearch triton-api yolo-api
```

## Post-Deployment

### Monitor These Metrics

1. **Search Performance**
   - P50/P95/P99 latency
   - Queries per second
   - Cache hit rate

2. **Index Growth**
   - Documents per day
   - Index size
   - Shard distribution

3. **Resource Usage**
   - OpenSearch heap usage
   - GPU memory (Triton)
   - API memory (yolo-api)

### Regular Maintenance

- [ ] Weekly: Review error logs
- [ ] Weekly: Check index stats and growth
- [ ] Monthly: Optimize index settings based on performance
- [ ] Monthly: Update cache size if needed
- [ ] Quarterly: Review and update MobileCLIP models

## Support Contacts

Internal:
- Infrastructure: [Your team]
- API Development: [Your team]
- ML/Model Updates: [Your team]

External:
- NVIDIA Triton: https://github.com/triton-inference-server/server
- OpenSearch: https://forum.opensearch.org/
- MobileCLIP: https://github.com/apple/ml-mobileclip

---

**Deployment Date:** _____________

**Deployed By:** _____________

**Sign-off:** _____________
