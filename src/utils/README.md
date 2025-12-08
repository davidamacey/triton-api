# Utilities Module

Shared utility functions for image processing, affine transformations, caching, and PyTorch inference.

## Modules

### `affine.py` - Affine Transformation Utilities

GPU-accelerated image preprocessing requires affine transformation matrices calculated on CPU.

```python
from src.utils.affine import calculate_letterbox_affine, calculate_inverse_affine

# Calculate letterbox transformation for YOLO preprocessing
affine_matrix, scale, pad_x, pad_y = calculate_letterbox_affine(
    orig_w=1920,
    orig_h=1080,
    target_size=640
)

# Inverse transformation for mapping detections back to original coordinates
inv_matrix = calculate_inverse_affine(affine_matrix)
```

**Functions:**
- `calculate_letterbox_affine(orig_w, orig_h, target_size)` - Calculate affine matrix for letterbox preprocessing
- `calculate_inverse_affine(affine_matrix)` - Calculate inverse affine matrix for coordinate mapping
- `apply_affine_to_boxes(boxes, inv_matrix)` - Transform detection boxes to original image coordinates

**Used by:** Track D (DALI preprocessing), Track E (visual search)

---

### `cache.py` - Caching Utilities

LRU caching for embeddings, model artifacts, and frequently accessed data.

```python
from src.utils.cache import EmbeddingCache, ModelCache

# Cache embeddings with TTL
embedding_cache = EmbeddingCache(max_size=10000, ttl_seconds=3600)
embedding_cache.set("image_001", embedding_vector)
cached = embedding_cache.get("image_001")

# Cache model artifacts
model_cache = ModelCache(max_size=100)
```

**Classes:**
- `EmbeddingCache` - LRU cache for embedding vectors with TTL support
- `ModelCache` - Cache for model artifacts and configurations
- `ImageHashCache` - Cache for image hashes (deduplication)

**Features:**
- Thread-safe operations
- Configurable TTL (time-to-live)
- Memory-efficient LRU eviction
- Hit/miss statistics

**Used by:** Track E (embedding caching for search)

---

### `image_processing.py` - Image Decoding and Validation

Optimized image decoding with robust format handling and security validation.

```python
from src.utils.image_processing import decode_image, validate_image

# Fast, secure image decoding
image_bytes = await file.read()
img = decode_image(image_bytes, filename="test.jpg")

# Security validation (OOM prevention)
validate_image(img, filename="test.jpg")
```

**Functions:**

#### `decode_image(image_bytes, filename="unknown")`

Decodes image bytes to numpy array with robust format handling.

**Performance:**
- Fast path (95% of cases): cv2.imdecode (~0.5-2ms)
- Fallback (edge cases): PIL (~2-5ms)

**Supported formats:**
- Primary (cv2): JPEG, PNG, BMP, TIFF
- Fallback (PIL): WebP, GIF, exotic formats

**Returns:**
- BGR numpy array (H, W, 3), uint8
- Ready for YOLO inference (no further preprocessing needed)

#### `validate_image(img, filename="unknown", max_dimension=16384, min_dimension=16)`

Fast security validation to prevent resource attacks.

**What we check:**
- Image exists and is valid
- Dimensions within safe bounds (16px to 16K)

**Security:**
- Default max: 16K resolution (prevents OOM attacks)
- Default min: 16 pixels (sanity check)

**Used by:** All tracks (A/B/C/D/E)

---

### `pytorch_utils.py` - PyTorch Inference Helpers

Utilities for Track A PyTorch inference including model loading and result formatting.

```python
from src.utils.pytorch_utils import load_yolo_model, format_detections

# Load YOLO model with caching
model = load_yolo_model("small", device="cuda:0")

# Format YOLO results to API response
detections = format_detections(results, normalize=True)
```

**Functions:**
- `load_yolo_model(model_size, device)` - Load and cache YOLO PyTorch model
- `format_detections(results, normalize)` - Convert YOLO results to detection dicts
- `run_inference(model, image)` - Run inference with proper error handling
- `get_model_info(model)` - Extract model metadata (classes, input size)

**Features:**
- Model caching (loads once, reuses)
- Automatic device placement
- Normalized coordinate output
- Batch inference support

**Used by:** Track A (PyTorch baseline)

---

## Design Philosophy

### Fast Path Optimization

```
95% of requests: cv2.imdecode (fast C++ backend)
 |
 5% of requests: PIL fallback (exotic formats)
```

### Minimal Validation

```
Our validation:  Security checks only (OOM prevention)
YOLO validation: Format, channels, dtype, etc.
                 |
                 No duplicate work!
```

### Industry Best Practices

1. **Fast by default** - Optimize for the common case
2. **Secure** - Prevent resource exhaustion attacks
3. **Clear errors** - Help users fix issues quickly
4. **No duplication** - Don't redo YOLO's work

---

## File Structure

```
src/utils/
├── __init__.py              # Public exports
├── affine.py                # Affine transformation calculations
├── cache.py                 # LRU caching utilities
├── image_processing.py      # Image decode/validate
├── pytorch_utils.py         # PyTorch inference helpers
└── README.md                # This file
```

---

## Usage Example

```python
from fastapi import FastAPI, File, UploadFile
from src.utils import decode_image, validate_image
from src.utils.pytorch_utils import load_yolo_model, format_detections

app = FastAPI()

# Load model once at startup
model = load_yolo_model("small", device="cuda:0")

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    # 1. Decode image (fast, robust)
    image_bytes = await image.read()
    img = decode_image(image_bytes, image.filename)

    # 2. Validate (security)
    validate_image(img, image.filename)

    # 3. Inference
    results = model(img)

    # 4. Format response
    detections = format_detections(results, normalize=True)
    return {"detections": detections}
```

---

## Performance Characteristics

### Image Decoding

| Format | Method | Typical Time | Notes |
|--------|--------|--------------|-------|
| JPEG | cv2.imdecode | 0.5-2ms | Hardware-accelerated on some platforms |
| PNG | cv2.imdecode | 1-3ms | Optimized C++ backend |
| WebP | PIL fallback | 2-5ms | Slower but handles edge cases |
| TIFF | cv2/PIL | 2-10ms | Depends on compression |

### Validation

- Null checks: ~0.001ms
- Dimension checks: ~0.001ms
- **Total overhead: < 0.01ms** (negligible)

### Total Image Processing

```
decode_image:   0.5-5ms   (depends on format)
validate_image: <0.01ms   (always fast)
-----------------------------------
Total:          0.5-5ms   (< 10% of inference time)
```

---

## Related Documentation

- [../README.md](../README.md) - FastAPI service overview
- [../../dali/README.md](../../dali/README.md) - DALI preprocessing (uses affine.py)
- [../../CLAUDE.md](../../CLAUDE.md) - Project overview
