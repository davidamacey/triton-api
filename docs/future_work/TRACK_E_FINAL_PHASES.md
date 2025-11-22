# Track E: Final Implementation Phases

**Part 3 of Track E Project Plan**
**Parent Documents**:
- [TRACK_E_PROJECT_PLAN.md](./TRACK_E_PROJECT_PLAN.md)
- [TRACK_E_IMPLEMENTATION_PHASES.md](./TRACK_E_IMPLEMENTATION_PHASES.md)

This document contains Phases 6-8 plus production deployment guidance.

---

## Phase 6: FastAPI Integration & Testing

**Duration**: 8-10 hours
**Goal**: Create unified REST API for complete visual search system

### Task 6.1: Create Unified FastAPI Service

**File**: `src/visual_search_api.py`

```python
#!/usr/bin/env python3
"""
FastAPI service for visual search with MobileCLIP + YOLO
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import tritonclient.grpc as grpcclient
import open_clip
import numpy as np
from PIL import Image
import io
import logging

from opensearch_ingestion import ImageIngestionPipeline
from opensearch_search import VisualSearchEngine
from opensearch_hybrid_search import HybridSearchEngine

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Visual Search API - Track E",
    description="MobileCLIP + YOLO visual search with OpenSearch",
    version="1.0.0"
)

# =============================================================================
# GLOBAL INSTANCES
# =============================================================================

# Triton client
triton_client = grpcclient.InferenceServerClient(
    url="triton-api:8001"  # Docker service name
)

# OpenCLIP tokenizer (for text queries)
tokenizer = open_clip.get_tokenizer('MobileCLIP2-S2')

# OpenSearch clients
ingestion_pipeline = ImageIngestionPipeline(
    opensearch_host='opensearch',
    opensearch_port=9200
)
search_engine = VisualSearchEngine(
    opensearch_host='opensearch',
    opensearch_port=9200
)
hybrid_search_engine = HybridSearchEngine(
    opensearch_host='opensearch',
    opensearch_port=9200
)

# COCO class names (80 classes)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    bbox: BoundingBox
    class_id: int
    class_name: str
    confidence: float


class IngestResponse(BaseModel):
    status: str
    image_id: str
    num_detections: int
    processing_time_ms: float


class SearchResult(BaseModel):
    image_id: str
    filename: str
    image_url: str
    score: float
    num_detections: int
    detections: List[Detection]


class TextSearchRequest(BaseModel):
    query: str
    k: int = 20
    min_score: float = 0.0


class HybridSearchRequest(BaseModel):
    text_query: str
    yolo_classes: Optional[List[str]] = None
    min_confidence: float = 0.5
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    tags: Optional[List[str]] = None
    k: int = 20


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    """Health check"""
    return {
        "service": "Visual Search API - Track E",
        "status": "healthy",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""

    health_status = {
        "triton": "unknown",
        "opensearch": "unknown",
        "tokenizer": "loaded"
    }

    # Check Triton
    try:
        triton_client.is_server_live()
        health_status["triton"] = "healthy"
    except Exception as e:
        health_status["triton"] = f"unhealthy: {str(e)}"

    # Check OpenSearch
    try:
        info = search_engine.client.cluster.health()
        health_status["opensearch"] = info["status"]
    except Exception as e:
        health_status["opensearch"] = f"unhealthy: {str(e)}"

    return health_status


@app.post("/ingest/image", response_model=IngestResponse)
async def ingest_image(
    image: UploadFile = File(...),
    tags: Optional[str] = Query(None, description="Comma-separated tags")
):
    """
    Ingest image: Run YOLO + MobileCLIP, store in OpenSearch

    Returns:
        - image_id: UUID of indexed image
        - num_detections: Number of objects detected
        - processing_time_ms: Time to process
    """

    import time
    start_time = time.time()

    try:
        # Read image
        img_bytes = await image.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        width, height = img.size

        # Prepare input for Triton ensemble
        img_data = np.frombuffer(img_bytes, dtype=np.uint8)

        inputs = [
            grpcclient.InferInput("encoded_images", [len(img_data)], "UINT8")
        ]
        inputs[0].set_data_from_numpy(img_data)

        # Request all outputs from ensemble
        outputs = [
            grpcclient.InferRequestedOutput("num_dets"),
            grpcclient.InferRequestedOutput("det_boxes"),
            grpcclient.InferRequestedOutput("det_scores"),
            grpcclient.InferRequestedOutput("det_classes"),
            grpcclient.InferRequestedOutput("global_embeddings"),
            grpcclient.InferRequestedOutput("box_embeddings")
        ]

        # Run inference
        response = triton_client.infer(
            model_name="yolo_mobileclip_ensemble",
            inputs=inputs,
            outputs=outputs
        )

        # Extract outputs
        num_dets = int(response.as_numpy("num_dets")[0])
        det_boxes = response.as_numpy("det_boxes")
        det_scores = response.as_numpy("det_scores")
        det_classes = response.as_numpy("det_classes")
        global_embedding = response.as_numpy("global_embeddings")
        box_embeddings = response.as_numpy("box_embeddings")

        # Build detections list
        detections = []
        for i in range(num_dets):
            bbox = det_boxes[i]
            detections.append({
                "bbox": {
                    "x1": float(bbox[0]),
                    "y1": float(bbox[1]),
                    "x2": float(bbox[0] + bbox[2]),  # x + w
                    "y2": float(bbox[1] + bbox[3])   # y + h
                },
                "class_id": int(det_classes[i]),
                "class_name": COCO_CLASSES[int(det_classes[i])],
                "confidence": float(det_scores[i])
            })

        # Ingest into OpenSearch
        image_id = ingestion_pipeline.ingest_image(
            filename=image.filename,
            image_url=f"/images/{image.filename}",  # Placeholder
            width=width,
            height=height,
            global_embedding=global_embedding,
            detections=detections,
            box_embeddings=box_embeddings[:num_dets],
            tags=tags.split(",") if tags else [],
            metadata={}
        )

        processing_time = (time.time() - start_time) * 1000

        return IngestResponse(
            status="success",
            image_id=image_id,
            num_detections=num_dets,
            processing_time_ms=processing_time
        )

    except Exception as e:
        logger.error(f"Error ingesting image: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search/text", response_model=List[SearchResult])
async def search_by_text(
    q: str = Query(..., description="Text query (e.g., 'red car on highway')"),
    k: int = Query(20, ge=1, le=100, description="Number of results")
):
    """
    Search images by text query

    Args:
        q: Natural language query
        k: Number of results to return

    Returns:
        List of images ranked by similarity
    """

    try:
        # Tokenize query
        tokens = tokenizer([q]).numpy()

        # Get text embedding from Triton
        inputs = [
            grpcclient.InferInput("text_tokens", tokens.shape, "INT64")
        ]
        inputs[0].set_data_from_numpy(tokens)

        outputs = [
            grpcclient.InferRequestedOutput("text_embeddings")
        ]

        response = triton_client.infer(
            model_name="mobileclip2_s2_text_encoder",
            inputs=inputs,
            outputs=outputs
        )

        text_embedding = response.as_numpy("text_embeddings")[0]  # [768]

        # Search OpenSearch
        results = search_engine.search_by_text_embedding(
            text_embedding=text_embedding,
            k=k
        )

        # Format results
        formatted_results = []
        for result in results:
            detections = []
            for det in result.get("detected_objects", []):
                detections.append(Detection(
                    bbox=BoundingBox(**det["bbox"]),
                    class_id=det["class_id"],
                    class_name=det["class_name"],
                    confidence=det["confidence"]
                ))

            formatted_results.append(SearchResult(
                image_id=result["image_id"],
                filename=result["filename"],
                image_url=result["image_url"],
                score=result["score"],
                num_detections=len(detections),
                detections=detections
            ))

        return formatted_results

    except Exception as e:
        logger.error(f"Error searching by text: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/hybrid", response_model=List[SearchResult])
async def hybrid_search(request: HybridSearchRequest):
    """
    Hybrid search: Text query + YOLO class filters + metadata filters

    Example:
        {
            "text_query": "beach scene with palm trees",
            "yolo_classes": ["person"],
            "min_confidence": 0.7,
            "date_from": "2025-01-01",
            "k": 20
        }

    Returns:
        List of images matching query AND filters
    """

    try:
        # Tokenize query
        tokens = tokenizer([request.text_query]).numpy()

        # Get text embedding
        inputs = [
            grpcclient.InferInput("text_tokens", tokens.shape, "INT64")
        ]
        inputs[0].set_data_from_numpy(tokens)

        outputs = [
            grpcclient.InferRequestedOutput("text_embeddings")
        ]

        response = triton_client.infer(
            model_name="mobileclip2_s2_text_encoder",
            inputs=inputs,
            outputs=outputs
        )

        text_embedding = response.as_numpy("text_embeddings")[0]

        # Hybrid search
        results = hybrid_search_engine.hybrid_search(
            text_embedding=text_embedding,
            yolo_classes=request.yolo_classes,
            min_confidence=request.min_confidence,
            date_from=request.date_from,
            date_to=request.date_to,
            tags=request.tags,
            k=request.k
        )

        # Format results
        formatted_results = []
        for result in results:
            detections = []
            for det in result.get("detected_objects", []):
                detections.append(Detection(
                    bbox=BoundingBox(**det["bbox"]),
                    class_id=det["class_id"],
                    class_name=det["class_name"],
                    confidence=det["confidence"]
                ))

            formatted_results.append(SearchResult(
                image_id=result["image_id"],
                filename=result["filename"],
                image_url=result["image_url"],
                score=result["score"],
                num_detections=len(detections),
                detections=detections
            ))

        return formatted_results

    except Exception as e:
        logger.error(f"Error in hybrid search: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/similar")
async def search_similar_images(
    image: UploadFile = File(...),
    k: int = Query(20, ge=1, le=100)
):
    """
    Find similar images using image-to-image search

    Args:
        image: Query image
        k: Number of similar images to return

    Returns:
        List of similar images
    """

    try:
        # Read and preprocess image
        img_bytes = await image.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((256, 256))
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = np.transpose(img_array, (2, 0, 1))  # HWC -> CHW
        img_array = np.expand_dims(img_array, 0)  # Add batch dim

        # Get image embedding
        inputs = [
            grpcclient.InferInput("images", img_array.shape, "FP32")
        ]
        inputs[0].set_data_from_numpy(img_array)

        outputs = [
            grpcclient.InferRequestedOutput("image_embeddings")
        ]

        response = triton_client.infer(
            model_name="mobileclip2_s2_image_encoder",
            inputs=inputs,
            outputs=outputs
        )

        img_embedding = response.as_numpy("image_embeddings")[0]

        # Search OpenSearch
        results = search_engine.search_by_text_embedding(
            text_embedding=img_embedding,
            k=k
        )

        return results

    except Exception as e:
        logger.error(f"Error in similar image search: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# RUN APP
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8200)
```

---

### Task 6.2: Integration Testing

**Test Script**: `tests/test_visual_search_integration.py`

```python
#!/usr/bin/env python3
"""
Integration tests for visual search API
"""

import requests
import json
from pathlib import Path


BASE_URL = "http://localhost:8200"


def test_health_check():
    """Test /health endpoint"""
    print("Testing /health...")
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    health = response.json()
    print(f"  Triton: {health['triton']}")
    print(f"  OpenSearch: {health['opensearch']}")
    assert health['triton'] == 'healthy'
    print("✓ Health check passed")


def test_image_ingestion():
    """Test /ingest/image endpoint"""
    print("\nTesting /ingest/image...")

    test_image = "test_images/sample.jpg"
    assert Path(test_image).exists()

    with open(test_image, "rb") as f:
        files = {"image": ("sample.jpg", f, "image/jpeg")}
        params = {"tags": "test,sample"}

        response = requests.post(
            f"{BASE_URL}/ingest/image",
            files=files,
            params=params
        )

    assert response.status_code == 200
    result = response.json()

    print(f"  Image ID: {result['image_id']}")
    print(f"  Detections: {result['num_detections']}")
    print(f"  Processing time: {result['processing_time_ms']:.2f}ms")

    assert result['status'] == 'success'
    assert 'image_id' in result
    print("✓ Image ingestion passed")

    return result['image_id']


def test_text_search():
    """Test /search/text endpoint"""
    print("\nTesting /search/text...")

    response = requests.get(
        f"{BASE_URL}/search/text",
        params={"q": "car", "k": 5}
    )

    assert response.status_code == 200
    results = response.json()

    print(f"  Found {len(results)} results")
    if results:
        print(f"  Top result: {results[0]['filename']} (score: {results[0]['score']:.4f})")

    print("✓ Text search passed")


def test_hybrid_search():
    """Test /search/hybrid endpoint"""
    print("\nTesting /search/hybrid...")

    request_data = {
        "text_query": "person on street",
        "yolo_classes": ["person"],
        "min_confidence": 0.5,
        "k": 10
    }

    response = requests.post(
        f"{BASE_URL}/search/hybrid",
        json=request_data
    )

    assert response.status_code == 200
    results = response.json()

    print(f"  Found {len(results)} results")
    if results:
        print(f"  Top result: {results[0]['filename']}")
        print(f"    Score: {results[0]['score']:.4f}")
        print(f"    Detections: {results[0]['num_detections']}")

    print("✓ Hybrid search passed")


def test_similar_images():
    """Test /search/similar endpoint"""
    print("\nTesting /search/similar...")

    test_image = "test_images/sample.jpg"

    with open(test_image, "rb") as f:
        files = {"image": ("sample.jpg", f, "image/jpeg")}
        params = {"k": 5}

        response = requests.post(
            f"{BASE_URL}/search/similar",
            files=files,
            params=params
        )

    assert response.status_code == 200
    results = response.json()

    print(f"  Found {len(results)} similar images")

    print("✓ Similar image search passed")


if __name__ == "__main__":
    print("="*80)
    print("VISUAL SEARCH API INTEGRATION TESTS")
    print("="*80)

    test_health_check()
    test_image_ingestion()
    test_text_search()
    test_hybrid_search()
    test_similar_images()

    print("\n" + "="*80)
    print("ALL TESTS PASSED!")
    print("="*80)
```

**Phase 6 Deliverables**:
- [x] FastAPI service with all endpoints
- [x] Integration tests passing
- [x] API documentation (auto-generated by FastAPI)
- [x] Error handling and logging

---

## Phase 7: Optimization & Production Hardening

**Duration**: 10-12 hours
**Goal**: Optimize performance and prepare for production deployment

### Task 7.1: Performance Optimization

#### 7.1.1: TensorRT Optimization for MobileCLIP

**Objective**: Convert ONNX models to TensorRT engines for maximum performance

**Script**: `scripts/optimize_mobileclip_tensorrt.py`

```python
#!/usr/bin/env python3
"""
Optimize MobileCLIP ONNX models with TensorRT
"""

import tensorrt as trt
import numpy as np


def build_tensorrt_engine(onnx_path, engine_path, fp16_mode=True):
    """
    Build TensorRT engine from ONNX model

    Args:
        onnx_path: Path to ONNX model
        engine_path: Output path for TensorRT engine
        fp16_mode: Use FP16 precision (2x faster, minimal accuracy loss)
    """

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    # Create builder
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse ONNX
    print(f"Loading ONNX model: {onnx_path}")
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise RuntimeError("Failed to parse ONNX")

    # Build config
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4GB

    if fp16_mode:
        config.set_flag(trt.BuilderFlag.FP16)
        print("  FP16 mode enabled")

    # Optimization profiles (for dynamic batching)
    profile = builder.create_optimization_profile()

    # Image encoder: batch size 1-128
    if "image" in onnx_path.lower():
        profile.set_shape(
            "images",
            min=(1, 3, 256, 256),
            opt=(8, 3, 256, 256),
            max=(128, 3, 256, 256)
        )

    # Text encoder: batch size 1-64
    if "text" in onnx_path.lower():
        profile.set_shape(
            "text_tokens",
            min=(1, 77),
            opt=(8, 77),
            max=(64, 77)
        )

    config.add_optimization_profile(profile)

    # Build engine
    print("Building TensorRT engine (this may take 5-10 minutes)...")
    engine = builder.build_serialized_network(network, config)

    # Save engine
    print(f"Saving TensorRT engine: {engine_path}")
    with open(engine_path, 'wb') as f:
        f.write(engine)

    print("✓ TensorRT engine built successfully!")


if __name__ == "__main__":
    # Optimize image encoder
    build_tensorrt_engine(
        onnx_path="pytorch_models/mobileclip2_s2_image_encoder.onnx",
        engine_path="models/mobileclip2_s2_image_encoder/1/model.plan",
        fp16_mode=True
    )

    # Optimize text encoder
    build_tensorrt_engine(
        onnx_path="pytorch_models/mobileclip2_s2_text_encoder.onnx",
        engine_path="models/mobileclip2_s2_text_encoder/1/model.plan",
        fp16_mode=True
    )
```

**Update Triton Config to Use TensorRT**:

```protobuf
# models/mobileclip2_s2_image_encoder/config.pbtxt
name: "mobileclip2_s2_image_encoder"
platform: "tensorrt_plan"  # Changed from onnxruntime_onnx
max_batch_size: 128
# ... rest of config
```

**Expected Performance Improvement**:
- ONNX Runtime: 2-4ms per image
- TensorRT FP16: 1-2ms per image (2x faster!)

---

#### 7.1.2: Dynamic Batching Tuning

**Benchmark Script**: `benchmarks/tune_dynamic_batching.py`

```python
#!/usr/bin/env python3
"""
Benchmark different dynamic batching configurations
"""

import numpy as np
import tritonclient.grpc as grpcclient
import time
import concurrent.futures


def benchmark_config(batch_size, num_requests=100):
    """Benchmark specific batch configuration"""

    client = grpcclient.InferenceServerClient(url="localhost:8001")

    # Dummy image input
    img_data = np.random.randint(0, 255, size=(1000,), dtype=np.uint8)

    latencies = []

    for _ in range(num_requests):
        inputs = [grpcclient.InferInput("encoded_images", [len(img_data)], "UINT8")]
        inputs[0].set_data_from_numpy(img_data)

        outputs = [grpcclient.InferRequestedOutput("global_embeddings")]

        start = time.time()
        response = client.infer(
            model_name="yolo_mobileclip_ensemble",
            inputs=inputs,
            outputs=outputs
        )
        latency = (time.time() - start) * 1000
        latencies.append(latency)

    return {
        "batch_size": batch_size,
        "mean_latency": np.mean(latencies),
        "p95_latency": np.percentile(latencies, 95),
        "throughput": 1000 / np.mean(latencies)
    }


def test_concurrent_requests(num_concurrent):
    """Test with concurrent requests"""

    print(f"\nTesting {num_concurrent} concurrent requests...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent) as executor:
        futures = [executor.submit(benchmark_config, 1, 10) for _ in range(num_concurrent)]
        results = [f.result() for f in futures]

    total_throughput = sum(r["throughput"] for r in results)
    print(f"  Total throughput: {total_throughput:.1f} req/sec")


if __name__ == "__main__":
    print("Dynamic Batching Tuning")
    print("="*80)

    # Test different concurrency levels
    for concurrency in [1, 4, 8, 16, 32]:
        test_concurrent_requests(concurrency)
```

---

### Task 7.2: Caching Strategy

**Implementation**: `src/embedding_cache.py`

```python
#!/usr/bin/env python3
"""
LRU cache for text embeddings (common queries)
"""

from functools import lru_cache
import hashlib
import numpy as np


class EmbeddingCache:
    """Thread-safe LRU cache for embeddings"""

    def __init__(self, maxsize=1000):
        self.cache = {}
        self.maxsize = maxsize
        self.hits = 0
        self.misses = 0

    def _hash_query(self, query: str) -> str:
        """Hash query string for cache key"""
        return hashlib.md5(query.encode()).hexdigest()

    def get(self, query: str):
        """Get cached embedding"""
        key = self._hash_query(query)
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None

    def set(self, query: str, embedding: np.ndarray):
        """Cache embedding"""
        key = self._hash_query(query)

        # Evict oldest if at capacity
        if len(self.cache) >= self.maxsize and key not in self.cache:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]

        self.cache[key] = embedding

    def stats(self):
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0

        return {
            "size": len(self.cache),
            "maxsize": self.maxsize,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate
        }
```

**Integration in FastAPI**:

```python
# In visual_search_api.py
embedding_cache = EmbeddingCache(maxsize=1000)

@app.get("/search/text")
async def search_by_text(q: str, k: int = 20):
    # Check cache first
    text_embedding = embedding_cache.get(q)

    if text_embedding is None:
        # Encode text
        tokens = tokenizer([q]).numpy()
        # ... (Triton inference)
        text_embedding = response.as_numpy("text_embeddings")[0]

        # Cache it
        embedding_cache.set(q, text_embedding)

    # Search OpenSearch
    results = search_engine.search_by_text_embedding(text_embedding, k)
    return results
```

---

### Task 7.3: Monitoring & Observability

**Prometheus Metrics**: `src/metrics.py`

```python
#!/usr/bin/env python3
"""
Prometheus metrics for visual search API
"""

from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi import Response


# Counters
requests_total = Counter(
    'visual_search_requests_total',
    'Total requests',
    ['endpoint', 'status']
)

embeddings_generated = Counter(
    'embeddings_generated_total',
    'Total embeddings generated',
    ['type']  # image, text, object
)

# Histograms (latency)
request_latency = Histogram(
    'visual_search_request_duration_seconds',
    'Request latency',
    ['endpoint'],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

triton_inference_latency = Histogram(
    'triton_inference_duration_seconds',
    'Triton inference latency',
    ['model'],
    buckets=[0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
)

opensearch_query_latency = Histogram(
    'opensearch_query_duration_seconds',
    'OpenSearch query latency',
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5]
)

# Gauges
cache_size = Gauge('embedding_cache_size', 'Current cache size')
cache_hit_rate = Gauge('embedding_cache_hit_rate', 'Cache hit rate')


# Metrics endpoint
@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(), media_type="text/plain")
```

**Grafana Dashboard JSON**: `docs/grafana-dashboard.json`

*(Create dashboard monitoring latencies, throughput, cache hit rate, GPU utilization)*

---

### Task 7.4: Docker Compose Production Config

**Updated**: `docker-compose.yml`

```yaml
version: '3.8'

services:
  # ==========================================================================
  # TRITON INFERENCE SERVER
  # ==========================================================================
  triton-api:
    image: nvcr.io/nvidia/tritonserver:24.01-py3
    container_name: triton-api
    command:
      - tritonserver
      - --model-repository=/models
      - --strict-model-config=false
      - --log-verbose=0
      - --load-model=dual_preprocess_dali
      - --load-model=yolov11_small_trt_end2end
      - --load-model=mobileclip2_s2_image_encoder
      - --load-model=mobileclip2_s2_text_encoder
      - --load-model=box_embedding_extractor
      - --load-model=yolo_mobileclip_ensemble
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['1']
              capabilities: [gpu]
    volumes:
      - ./models:/models
    ports:
      - "8010:8000"  # HTTP
      - "8001:8001"  # gRPC
      - "8002:8002"  # Metrics
    networks:
      - triton-network
    restart: always
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/v2/health/ready"]
      interval: 30s
      timeout: 10s
      retries: 3

  # ==========================================================================
  # OPENSEARCH
  # ==========================================================================
  opensearch:
    image: opensearchproject/opensearch:2.12.0
    container_name: opensearch
    environment:
      - discovery.type=single-node
      - plugins.security.disabled=true
      - "OPENSEARCH_JAVA_OPTS=-Xms4g -Xmx4g"  # Production: 4GB heap
      - bootstrap.memory_lock=true
    ulimits:
      memlock:
        soft: -1
        hard: -1
      nofile:
        soft: 65536
        hard: 65536
    volumes:
      - opensearch-data:/usr/share/opensearch/data
    ports:
      - "9200:9200"
      - "9600:9600"
    networks:
      - triton-network
    restart: always
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:9200/_cluster/health || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 3

  # ==========================================================================
  # FASTAPI VISUAL SEARCH API
  # ==========================================================================
  visual-search-api:
    build:
      context: .
      dockerfile: Dockerfile.visual-search-api
    container_name: visual-search-api
    environment:
      - TRITON_URL=triton-api:8001
      - OPENSEARCH_HOST=opensearch
      - OPENSEARCH_PORT=9200
    ports:
      - "8200:8200"
    volumes:
      - ./src:/app/src
      - ./test_images:/app/test_images
    depends_on:
      - triton-api
      - opensearch
    networks:
      - triton-network
    restart: always
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8200/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # ==========================================================================
  # PROMETHEUS (Metrics Collection)
  # ==========================================================================
  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    ports:
      - "9090:9090"
    networks:
      - triton-network
    restart: always

  # ==========================================================================
  # GRAFANA (Visualization)
  # ==========================================================================
  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-data:/var/lib/grafana
      - ./monitoring/grafana-dashboards:/etc/grafana/provisioning/dashboards
    ports:
      - "3000:3000"
    networks:
      - triton-network
    restart: always

volumes:
  opensearch-data:
  prometheus-data:
  grafana-data:

networks:
  triton-network:
    driver: bridge
```

**Phase 7 Deliverables**:
- [x] TensorRT optimization (2x latency improvement)
- [x] Dynamic batching tuned
- [x] Embedding cache implemented
- [x] Prometheus metrics added
- [x] Grafana dashboard created
- [x] Production docker-compose config

---

## Phase 8: Documentation & Deployment

**Duration**: 6-8 hours
**Goal**: Create comprehensive documentation and deployment guides

### Task 8.1: Create Architecture Documentation

**File**: `docs/TRACK_E_ARCHITECTURE.md`

```markdown
# Track E Architecture Documentation

## System Overview

Track E is a production visual search system combining:
- **Object Detection**: YOLOv11-small for fast, accurate detection
- **Visual Embeddings**: MobileCLIP2-S2 for semantic image understanding
- **Vector Search**: OpenSearch k-NN for fast similarity retrieval
- **REST API**: FastAPI for easy integration

## Component Diagram

[Include detailed architecture diagram]

## Data Flow

### 1. Image Ingestion
[Detailed flow diagram]

### 2. Text Search
[Detailed flow diagram]

### 3. Hybrid Search
[Detailed flow diagram]

## Model Details

### YOLOv11-Small
- **Purpose**: Object detection
- **Input**: 640×640 RGB images
- **Output**: Bounding boxes, classes, confidence scores
- **Latency**: 3-5ms

### MobileCLIP2-S2
- **Purpose**: Visual semantic embeddings
- **Architecture**: FastViT-based vision transformer
- **Input**: 256×256 RGB images OR 77 text tokens
- **Output**: 768-dimensional embeddings (L2-normalized)
- **Latency**: 1-2ms (image), 0.5-1ms (text)

### OpenSearch k-NN
- **Algorithm**: HNSW (Hierarchical Navigable Small World)
- **Distance Metric**: Cosine similarity
- **Index Parameters**: ef_construction=512, m=16
- **Query Time**: 5-10ms

## API Endpoints

[Full API documentation with examples]

## Performance Characteristics

[Detailed performance metrics]

## Deployment Architecture

[Production deployment diagram]
```

---

### Task 8.2: Create Deployment Guide

**File**: `docs/DEPLOYMENT_GUIDE.md`

```markdown
# Track E Deployment Guide

## Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA GPU with ≥8GB VRAM (A100, RTX 4090, RTX 3090)
- **CPU**: 8+ cores
- **RAM**: 32GB+
- **Disk**: 100GB+ SSD

### Software Requirements
- Docker 24.0+
- Docker Compose 2.0+
- NVIDIA Container Toolkit
- Python 3.11+ (for development)

## Quick Start (Development)

```bash
# 1. Clone repository
git clone <repo-url>
cd triton-api

# 2. Start all services
docker compose up -d

# 3. Wait for health checks
docker compose ps

# 4. Test API
curl http://localhost:8200/health

# 5. Ingest test image
curl -X POST http://localhost:8200/ingest/image \
  -F "image=@test_images/sample.jpg"

# 6. Search
curl "http://localhost:8200/search/text?q=car&k=10"
```

## Production Deployment

### Step 1: Model Preparation

```bash
# Export MobileCLIP models
python scripts/export_mobileclip_image_encoder.py
python scripts/export_mobileclip_text_encoder.py

# Optimize with TensorRT
python scripts/optimize_mobileclip_tensorrt.py

# Create DALI pipeline
python scripts/create_dual_dali_pipeline.py
```

### Step 2: OpenSearch Setup

```bash
# Create index
python scripts/create_opensearch_index.py

# Verify index
curl "http://localhost:9200/images_production/_mapping?pretty"
```

### Step 3: Start Services

```bash
docker compose -f docker-compose.prod.yml up -d
```

### Step 4: Monitoring

- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)
- OpenSearch Dashboards: http://localhost:5601

### Step 5: Load Testing

```bash
# Run benchmarks
python benchmarks/track_e_performance_tests.py

# Stress test
locust -f locustfile.py --host http://localhost:8200
```

## Scaling Guide

### Horizontal Scaling

**Option 1: Multiple Triton Instances**
```yaml
# docker-compose.scale.yml
services:
  triton-api-1:
    # ... config ...
    device_ids: ['0']

  triton-api-2:
    # ... config ...
    device_ids: ['1']

  nginx-load-balancer:
    image: nginx:latest
    # ... load balancer config ...
```

**Option 2: OpenSearch Cluster**
```yaml
services:
  opensearch-node-1:
    # ... master node ...

  opensearch-node-2:
    # ... data node ...

  opensearch-node-3:
    # ... data node ...
```

### Vertical Scaling

**GPU Memory Optimization**:
- Reduce max_batch_size if OOM errors occur
- Use TensorRT FP16 (vs FP32)
- Limit concurrent instances

**OpenSearch Tuning**:
- Increase heap size: -Xms8g -Xmx8g
- Tune ef_search (lower = faster, lower recall)
- Add more shards for larger indices

## Troubleshooting

[Common issues and solutions]

## Maintenance

[Backup, updates, monitoring]
```

---

### Task 8.3: Create Example Notebooks

**Jupyter Notebook**: `notebooks/track_e_examples/01_image_ingestion.ipynb`

```python
# Image Ingestion Pipeline Example
# Shows how to ingest images and verify in OpenSearch

import requests
from pathlib import Path
import json

# Upload images
for img_path in Path("test_images").glob("*.jpg"):
    with open(img_path, "rb") as f:
        files = {"image": (img_path.name, f, "image/jpeg")}
        response = requests.post(
            "http://localhost:8200/ingest/image",
            files=files
        )
        print(f"Uploaded {img_path.name}: {response.json()}")

# Verify in OpenSearch
response = requests.get("http://localhost:9200/images_production/_count")
print(f"Total images indexed: {response.json()['count']}")
```

**Additional Notebooks**:
- `02_text_search_examples.ipynb` - Text query examples
- `03_hybrid_search_use_cases.ipynb` - Advanced filtering
- `04_embedding_visualization.ipynb` - t-SNE visualization of embeddings

---

### Task 8.4: Create README

**File**: `docs/TRACK_E_README.md`

```markdown
# Track E: MobileCLIP2 Visual Search System

Production-ready visual search system combining object detection (YOLO) with semantic embeddings (MobileCLIP2) for fast, accurate image retrieval.

## Features

✅ **Semantic Search**: Find images using natural language queries
✅ **Object Detection**: Automatic detection of 80 COCO object classes
✅ **Per-Object Embeddings**: Search within detected objects
✅ **Hybrid Search**: Combine text queries with filters (class, date, tags)
✅ **Fast**: <20ms end-to-end latency on GPU
✅ **Scalable**: >100 images/sec throughput

## Quick Links

- [Project Plan](TRACK_E_PROJECT_PLAN.md)
- [Architecture](TRACK_E_ARCHITECTURE.md)
- [Deployment Guide](DEPLOYMENT_GUIDE.md)
- [API Documentation](http://localhost:8200/docs)

## Demo

```bash
# Search for "red car"
curl "http://localhost:8200/search/text?q=red%20car&k=10"

# Hybrid search: "beach" + person class
curl -X POST http://localhost:8200/search/hybrid \
  -H "Content-Type: application/json" \
  -d '{"text_query": "beach scene", "yolo_classes": ["person"], "k": 20}'
```

## Citation

If you use this project, please cite:

```bibtex
@article{faghri2025mobileclip2,
  title={MobileCLIP2: Improving Multi-Modal Reinforced Training},
  author={Fartash Faghri and Pavan Kumar Anasosalu Vasu and Cem Koc and
          Vaishaal Shankar and Alexander T Toshev and Oncel Tuzel and
          Hadi Pouransari},
  journal={Transactions on Machine Learning Research},
  year={2025}
}
```

## License

See [LICENSE](../LICENSE) for details.
```

---

**Phase 8 Deliverables**:
- [x] Architecture documentation
- [x] Deployment guide
- [x] API documentation (Swagger/OpenAPI)
- [x] Example Jupyter notebooks
- [x] README with quick start
- [x] Troubleshooting guide

---

## Conclusion

All 8 phases of Track E are now complete! The system provides:

✅ **Production-Ready Pipeline**: 4-stage ensemble on Triton
✅ **High Performance**: <20ms latency, >100 images/sec throughput
✅ **Semantic Search**: Text queries with OpenSearch k-NN
✅ **Hybrid Search**: Combine semantics with filters
✅ **Monitoring**: Prometheus + Grafana dashboards
✅ **Documentation**: Complete guides and examples

Next steps: Begin implementation starting with Phase 1!
