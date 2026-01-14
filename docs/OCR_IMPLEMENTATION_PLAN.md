# OCR Implementation Plan for Ralph Wiggum Loop

> **Purpose**: This document provides the complete implementation plan for adding OCR capabilities to triton-api, following Immich's approach but with 100% Triton GPU acceleration.

---

## Architecture Overview

### Model Components to Create

```
models/
├── paddleocr_det_trt/              # PP-OCRv5 Detection (TensorRT)
│   ├── 1/model.plan
│   └── config.pbtxt
├── paddleocr_rec_trt/              # PP-OCRv5 Recognition (TensorRT)
│   ├── 1/model.plan
│   └── config.pbtxt
├── ocr_pipeline/                    # Python BLS Orchestrator (like face_pipeline)
│   ├── 1/model.py
│   └── config.pbtxt
└── penta_preprocess_dali/          # 5-branch DALI (extends quad)
    ├── 1/model.dali                # YOLO + CLIP + SCRFD + HD + OCR
    └── config.pbtxt
```

### Files to Create

| File | Purpose | Pattern Reference |
|------|---------|-------------------|
| `export/download_paddleocr.py` | Download PP-OCRv5 ONNX models | `export/download_face_models.py` |
| `export/export_paddleocr_det.py` | Export detection TRT | `export/export_mobileclip_image_encoder.py` |
| `export/export_paddleocr_rec.py` | Export recognition TRT | `export/export_mobileclip_image_encoder.py` |
| `dali/create_penta_dali_pipeline.py` | 5-branch DALI pipeline | `dali/create_quad_dali_pipeline.py` |
| `models/ocr_pipeline/1/model.py` | Python BLS for OCR | `models/face_pipeline/1/model.py` |
| `models/ocr_pipeline/config.pbtxt` | Triton config | `models/face_pipeline/config.pbtxt` |
| `src/services/ocr_service.py` | OCR service wrapper | `src/services/duplicate_detection.py` |

### Files to Modify

| File | Changes |
|------|---------|
| `src/clients/opensearch.py` | Add `IndexName.OCR`, create OCR index schema |
| `src/routers/track_e.py` | Add OCR endpoints, update `/ingest` with `enable_ocr` |
| `src/services/visual_search.py` | Integrate OCR into ingestion pipeline |
| `src/services/inference.py` | Add `infer_ocr()` method |
| `dali/config.py` | Add OCR_SIZE constant (736x736 or 960x960) |
| `CLAUDE.md` | Document OCR endpoints |

---

## Implementation Details

### 1. PP-OCRv5 Model Specifications

**Detection Model (DB++):**
- Input: `[B, 3, H, W]` where H,W multiples of 32, max 960
- Output: Text region probability map
- Preprocessing: BGR, normalize `(x - 0.5) / 0.5`

**Recognition Model (SVTR):**
- Input: `[N, 3, 48, 320]` (batch of text crops)
- Output: Text strings + confidence scores
- Preprocessing: BGR, normalize `(x - 0.5) / 0.5`

### 2. DALI Preprocessing for OCR

OCR preprocessing differs from YOLO:
- Max dimension: 960px (vs 640 for YOLO)
- Pad to 32-pixel boundary (not letterbox)
- BGR color space (not RGB)
- Normalize: `(x / 255 - 0.5) / 0.5 = x / 127.5 - 1`

```python
# Branch 5: OCR preprocessing
ocr_images = fn.resize(
    images,
    resize_longer=960,  # Max 960px longest edge
    interp_type=types.INTERP_LINEAR,
    device='gpu',
)

# Pad to 32-pixel boundary
ocr_images = fn.pad(
    ocr_images,
    axis_names="HW",
    align=[32, 32],
    fill_value=0,
)

# Convert RGB -> BGR and normalize for PP-OCR
ocr_images = fn.crop_mirror_normalize(
    ocr_images,
    mean=[127.5, 127.5, 127.5],
    std=[127.5, 127.5, 127.5],  # (x - 127.5) / 127.5
    output_layout='CHW',
    output_dtype=types.FLOAT,
    device='gpu',
)

# Channel swap RGB -> BGR happens via indexing in Python BLS
```

### 3. OCR Pipeline (Python BLS)

Following `face_pipeline` pattern:

```python
class TritonPythonModel:
    """OCR Pipeline: Detection + Recognition via BLS"""

    def initialize(self, args):
        self.det_model = "paddleocr_det_trt"
        self.rec_model = "paddleocr_rec_trt"
        self.db_postprocess = DBPostProcess(...)

    def execute(self, requests):
        # 1. Call detection model via BLS
        det_outputs = self._call_detection(ocr_images)

        # 2. Post-process: threshold + box expansion + NMS
        boxes, scores = self.db_postprocess(det_outputs)

        # 3. Sort boxes in reading order
        boxes = self._sort_boxes(boxes)

        # 4. Crop text regions with perspective transform
        crops = self._get_text_crops(original_image, boxes)

        # 5. Call recognition model via BLS
        texts, text_scores = self._call_recognition(crops)

        return boxes, texts, scores
```

### 4. OpenSearch OCR Index

```python
class IndexName(str, Enum):
    GLOBAL = 'visual_search_global'
    VEHICLES = 'visual_search_vehicles'
    PEOPLE = 'visual_search_people'
    FACES = 'visual_search_faces'
    OCR = 'visual_search_ocr'  # NEW

# OCR Index Schema
OCR_INDEX_SETTINGS = {
    "settings": {
        "analysis": {
            "analyzer": {
                "trigram_analyzer": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase", "trigram_filter"]
                }
            },
            "filter": {
                "trigram_filter": {"type": "ngram", "min_gram": 3, "max_gram": 3}
            }
        }
    },
    "mappings": {
        "properties": {
            "ocr_id": {"type": "keyword"},
            "image_id": {"type": "keyword"},
            "image_path": {"type": "keyword"},
            "text": {"type": "text", "analyzer": "trigram_analyzer"},
            "text_raw": {"type": "keyword"},  # For exact match
            "confidence": {"type": "float"},
            "box": {"type": "float"},  # 8 coords (quad)
            "box_normalized": {"type": "float"},  # 4 coords (rect, normalized)
            "language": {"type": "keyword"},
            "created_at": {"type": "date"}
        }
    }
}
```

### 5. API Endpoints

```python
# OCR-specific endpoints
POST /track_e/ocr/predict          # Extract text + boxes from image
POST /track_e/ocr/predict_batch    # Batch OCR processing
POST /track_e/search/ocr           # Search images by text content
GET  /track_e/ocr/{image_id}       # Get all OCR results for an image

# Updated ingestion endpoint
POST /track_e/ingest
    ?enable_ocr=true               # Default: True
    ?enable_faces=true             # Default: True
    ?skip_duplicates=true          # Default: True
```

---

## Task Breakdown

### Phase 1: Model Export (Day 1)

1. **Create `export/download_paddleocr.py`**
   - Download PP-OCRv5 Mobile detection model
   - Download PP-OCRv5 Mobile recognition model
   - Download character dictionaries
   - Store in `/app/pytorch_models/paddleocr/`

2. **Create `export/export_paddleocr_det.py`**
   - Load PP-OCRv5 detection model
   - Export to ONNX with dynamic shapes
   - Convert to TensorRT with FP16
   - Validate output matches reference

3. **Create `export/export_paddleocr_rec.py`**
   - Load PP-OCRv5 recognition model
   - Export to ONNX with dynamic batch
   - Convert to TensorRT with FP16
   - Validate text output

### Phase 2: Triton Integration (Day 2)

4. **Create `dali/create_penta_dali_pipeline.py`**
   - Extend quad_preprocess_dali with OCR branch
   - Add OCR-specific normalization
   - Test with sample images
   - Serialize and create config.pbtxt

5. **Create `models/ocr_pipeline/1/model.py`**
   - Follow face_pipeline pattern
   - Implement DBPostProcess for detection
   - Implement box sorting (reading order)
   - Implement perspective cropping
   - Call detection and recognition via BLS

6. **Create `models/ocr_pipeline/config.pbtxt`**
   - Define inputs/outputs
   - Configure dynamic batching
   - Set instance count

### Phase 3: API Integration (Day 3)

7. **Extend `src/clients/opensearch.py`**
   - Add IndexName.OCR
   - Add OCR index schema with trigram analyzer
   - Add `create_ocr_index()` method
   - Add `index_ocr_results()` method
   - Add `search_by_text()` method

8. **Create `src/services/ocr_service.py`**
   - Wrapper for OCR inference
   - Text post-processing
   - Box normalization

9. **Extend `src/services/inference.py`**
   - Add `infer_ocr()` method
   - Add `infer_ocr_batch()` method

10. **Extend `src/routers/track_e.py`**
    - Add `/track_e/ocr/predict` endpoint
    - Add `/track_e/ocr/predict_batch` endpoint
    - Add `/track_e/search/ocr` endpoint
    - Add `/track_e/ocr/{image_id}` endpoint
    - Update `/track_e/ingest` with `enable_ocr` parameter

### Phase 4: Unified Ingestion (Day 4)

11. **Extend `src/services/visual_search.py`**
    - Integrate OCR into `ingest_image()`
    - Run OCR alongside YOLO + CLIP + faces
    - Index OCR results to visual_search_ocr

12. **Update `dali/config.py`**
    - Add OCR_SIZE constant
    - Add OCR normalization constants

13. **Testing**
    - Add `make test-ocr` target
    - Create OCR test script
    - Verify end-to-end pipeline

14. **Documentation**
    - Update CLAUDE.md with OCR endpoints
    - Update docs/FEATURE_ROADMAP.md status

---

## Success Criteria

The implementation is complete when:

1. **Model Export**: `paddleocr_det_trt` and `paddleocr_rec_trt` load in Triton
2. **DALI Pipeline**: `penta_preprocess_dali` outputs 5 branches correctly
3. **OCR Pipeline**: `ocr_pipeline` returns text + boxes for test images
4. **API Works**: `/track_e/ocr/predict` returns valid OCR results
5. **Search Works**: `/track_e/search/ocr` finds images by text content
6. **Ingestion Works**: `/track_e/ingest` with `enable_ocr=true` indexes text
7. **Tests Pass**: `make test-ocr` and all existing tests pass

---

## Reference Files

Study these existing files for patterns:

| File | Pattern |
|------|---------|
| `export/export_mobileclip_image_encoder.py` | TensorRT export pattern |
| `export/export_face_detection.py` | ONNX → TRT conversion |
| `dali/create_quad_dali_pipeline.py` | Multi-branch DALI |
| `models/face_pipeline/1/model.py` | Python BLS with GPU ops |
| `models/face_pipeline/config.pbtxt` | Triton config |
| `src/services/duplicate_detection.py` | Service pattern |
| `src/routers/track_e.py` | API endpoint pattern |
| `reference_repos/immich/machine-learning/immich_ml/models/ocr/detection.py` | Immich OCR detection |
| `reference_repos/immich/machine-learning/immich_ml/models/ocr/recognition.py` | Immich OCR recognition |

---

## Notes for Implementation

1. **PP-OCRv5 vs v4**: Use v5 Mobile for best speed/accuracy tradeoff
2. **BGR vs RGB**: PP-OCR expects BGR input, DALI outputs RGB - handle in Python BLS
3. **Reading Order**: Sort boxes top-to-bottom, left-to-right (Immich `sorted_boxes`)
4. **Perspective Transform**: Use batched SVD for efficiency (Immich pattern)
5. **Text Confidence**: Filter low-confidence text (min_score=0.9 like Immich)
6. **Trigram Search**: Enable fuzzy text matching in OpenSearch
