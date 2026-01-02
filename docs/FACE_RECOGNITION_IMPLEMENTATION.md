# Track E Face Detection & Recognition Implementation

> **Status:** Sprint 1 Complete | **Started:** 2026-01-01 | **Target:** Production Ready

## Quick Links

- [Implementation Checklist](#implementation-checklist)
- [Architecture](#architecture)
- [Sprint Progress](#sprint-progress)
- [Technical Specifications](#technical-specifications)

---

## Executive Summary

Extend Track E visual search with face detection (SCRFD-10G) and recognition (ArcFace w600k_r50) following existing GPU-accelerated patterns. Enables identity-based face search in OpenSearch alongside current YOLO+MobileCLIP pipeline.

### Selected Models

| Component | Model | Accuracy | Speed | Output | Size |
|-----------|-------|----------|-------|--------|------|
| Detection | SCRFD-10G-GNKPS | 95.2%/93.9%/83.1% WiderFace | ~5ms | Boxes + 5 landmarks | 17MB |
| Recognition | ArcFace w600k_r50 | 99.8% LFW | ~1.9ms | 512-dim embeddings | 166MB |

### Architecture Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Recognition Model | w600k_r50 (ResNet-50) | Best accuracy/speed balance |
| Pipeline | Parallel with YOLO | Single DALI decode, parallel inference |
| GPU | GPU 0 only | Simpler, less memory fragmentation |

---

## Implementation Checklist

### Sprint 1: Foundation ✅
- [x] **1.1** Download test datasets
  - [x] LFW Deep Funneled (13,233 images, 5,749 people) ✅
  - [x] WIDER Face Train (12,880 images, 61 event categories) ✅
  - [x] Create test subset script (`scripts/setup_face_test_data.sh`)
- [x] **1.2** Download pre-trained models
  - [x] `scrfd_10g_bnkps.onnx` from HuggingFace (16.1 MB)
  - [x] `arcface_w600k_r50.onnx` from HuggingFace (166.3 MB)
- [x] **1.3** Export SCRFD to TensorRT
  - [x] Validate ONNX with ONNX Runtime
  - [x] Convert to TensorRT FP16 (11.7 MB)
  - [x] Create Triton config.pbtxt
- [x] **1.4** Export ArcFace to TensorRT
  - [x] Validate ONNX with ONNX Runtime
  - [x] Convert to TensorRT FP16 (86.1 MB)
  - [x] Create Triton config.pbtxt
- [x] **1.5** Test standalone models in Triton
  - [x] Verify SCRFD outputs correct boxes/landmarks
  - [x] Verify ArcFace outputs 512-dim embeddings

### Sprint 2: Triton Integration
- [ ] **2.1** Create quad-branch DALI pipeline
  - [ ] Extend dual pipeline with SCRFD branch
  - [ ] Test all 4 outputs correct
- [ ] **2.2** Create face alignment utility
  - [ ] Implement landmark → affine matrix
  - [ ] Test alignment accuracy
- [ ] **2.3** Create face embedding extractor (Python backend)
  - [ ] Implement face cropping from landmarks
  - [ ] Implement ArcFace BLS call
  - [ ] Test end-to-end face embeddings
- [ ] **2.4** Create unified ensemble config
  - [ ] Wire quad DALI → YOLO/CLIP/SCRFD/HD
  - [ ] Wire box extractor + face extractor
  - [ ] Test full pipeline output

### Sprint 3: API & Search
- [ ] **3.1** Extend TritonClient
  - [ ] Add `infer_face_detect()` method
  - [ ] Add `infer_face_recognize()` method
  - [ ] Add `infer_faces_full()` method
- [ ] **3.2** Extend OpenSearch
  - [ ] Add `IndexName.FACES` enum
  - [ ] Implement `create_faces_index()`
  - [ ] Test index creation
- [ ] **3.3** Extend VisualSearchService
  - [ ] Implement `ingest_faces()`
  - [ ] Implement `search_faces()`
  - [ ] Test ingestion and search
- [ ] **3.4** Add API endpoints
  - [ ] `POST /track_e/faces/detect`
  - [ ] `POST /track_e/faces/recognize`
  - [ ] `POST /track_e/faces/ingest`
  - [ ] `POST /track_e/faces/search`
  - [ ] `POST /track_e/faces/identify`
  - [ ] `POST /track_e/faces/verify`
- [ ] **3.5** Add Makefile targets
  - [ ] `download-face-models`
  - [ ] `export-face-detection`
  - [ ] `export-face-recognition`
  - [ ] `setup-face-pipeline`
  - [ ] `test-track-e-faces`

### Sprint 4: Testing & Polish
- [ ] **4.1** Create validation test suite
  - [ ] Detection accuracy on LFW
  - [ ] Embedding quality (same-person similarity)
  - [ ] Search accuracy (identity retrieval)
- [ ] **4.2** Performance benchmarks
  - [ ] Latency per operation
  - [ ] Throughput (RPS)
  - [ ] GPU utilization
- [ ] **4.3** Add face tracks to triton_bench
  - [ ] Track_Face_Detect
  - [ ] Track_Face_Full
- [ ] **4.4** Documentation
  - [ ] Update CLAUDE.md with face endpoints
  - [ ] Add face API examples
  - [ ] Update README
- [ ] **4.5** Edge cases & robustness
  - [ ] No faces detected
  - [ ] Multiple faces
  - [ ] Low quality images
  - [ ] Very small/large faces

---

## Architecture

### Pipeline Flow (Parallel)

```
Input Image
    │
    ▼
DALI Quad Preprocess (single GPU decode)
    │
    ├──────────────────┬───────────────────┬─────────────────┐
    ▼                  ▼                   ▼                 ▼
YOLO 640x640      CLIP 256x256      SCRFD 640x640      HD Original
(detection)       (global embed)    (face detect)      (box crops)
    │                  │                   │                 │
    ▼                  ▼                   ▼                 ▼
det_boxes         global_embed       face_boxes         ┬────────┐
det_scores        [512-dim]          face_landmarks     │        │
det_classes                          face_scores        ▼        ▼
    │                                     │         Box      Face
    │                                     │         Extractor Extractor
    │                                     │             │        │
    │                                     │             ▼        ▼
    │                                     │        box_embed  face_embed
    │                                     │        [512-dim]  [512-dim]
    │                                     │             │        │
    └─────────────────────────────────────┴─────────────┴────────┘
                                │
                                ▼
                    OpenSearch Indexes
            ┌───────────┬───────────┬───────────┐
            ▼           ▼           ▼           ▼
        global      vehicles     people      faces
        (CLIP)      (CLIP)      (CLIP)     (ArcFace)
```

### Ensemble Configuration

```
yolo_face_clip_ensemble
├── Step 1: quad_preprocess_dali
│   └── Outputs: _yolo_images, _clip_images, _face_images, _original_images
├── Step 2a: yolov11_small_trt_end2end (parallel)
│   └── Outputs: det_boxes, det_scores, det_classes, num_dets
├── Step 2b: mobileclip2_s2_image_encoder (parallel)
│   └── Outputs: global_embeddings
├── Step 2c: scrfd_10g_face_detect (parallel)
│   └── Outputs: face_boxes, face_landmarks, face_scores, num_faces
├── Step 3a: box_embedding_extractor (after Step 2a)
│   └── Outputs: box_embeddings, normalized_boxes
└── Step 3b: face_embedding_extractor (after Step 2c)
    └── Outputs: face_embeddings, aligned_face_boxes
```

---

## Technical Specifications

### SCRFD Face Detection

**Model:** scrfd_10g_gnkps (Group Normalization, with Keypoints)

| Property | Value |
|----------|-------|
| Input | `[B, 3, 640, 640]` FP32, RGB, [0,1] normalized |
| Output: num_dets | `[B, 1]` INT32 |
| Output: det_boxes | `[B, 128, 4]` FP32 (x1,y1,x2,y2 pixels) |
| Output: det_landmarks | `[B, 128, 10]` FP32 (5 points × 2 coords) |
| Output: det_scores | `[B, 128]` FP32 |
| Max Batch | 64 |
| TensorRT | FP16, GPU 0 |

**Landmark Order:**
1. Left eye center
2. Right eye center
3. Nose tip
4. Left mouth corner
5. Right mouth corner

### ArcFace Recognition

**Model:** w600k_r50 (WebFace600K trained ResNet-50)

| Property | Value |
|----------|-------|
| Input | `[B, 3, 112, 112]` FP32, RGB, (x-127.5)/128 |
| Output | `[B, 512]` FP32, L2-normalized |
| Max Batch | 128 |
| TensorRT | FP16, GPU 0 |

**Reference Landmarks (112x112 aligned face):**
```python
ARCFACE_REF_LANDMARKS = [
    [38.2946, 51.6963],   # Left eye
    [73.5318, 51.5014],   # Right eye
    [56.0252, 71.7366],   # Nose
    [41.5493, 92.3655],   # Left mouth
    [70.7299, 92.2041],   # Right mouth
]
```

### OpenSearch Faces Index

**Index:** `visual_search_faces`

| Field | Type | Description |
|-------|------|-------------|
| face_id | keyword | Unique face detection ID |
| person_id | keyword | Identity cluster (same person) |
| image_id | keyword | Source image ID |
| image_path | keyword | Source image path |
| embedding | knn_vector[512] | ArcFace embedding |
| cluster_id | integer | FAISS IVF cluster |
| cluster_distance | float | Distance to centroid |
| box | float[4] | [x1,y1,x2,y2] normalized |
| landmarks | object | 5-point facial landmarks |
| confidence | float | Detection confidence |
| quality_score | float | Face quality metric |
| person_name | keyword | Optional identity label |
| is_reference | boolean | Reference face for clustering |

**HNSW Parameters:**
- ef_construction: 1024 (highest quality)
- m: 32 (most connections)
- ef_search: 512

---

## File Structure

### New Files to Create

```
export/
├── download_face_models.py           # Download SCRFD + ArcFace ONNX
├── export_face_detection.py          # SCRFD → TensorRT
└── export_face_recognition.py        # ArcFace → TensorRT

dali/
└── create_quad_dali_pipeline.py      # 4-branch preprocessing

models/
├── scrfd_10g_face_detect/
│   ├── config.pbtxt
│   └── 1/model.plan
├── arcface_w600k_r50/
│   ├── config.pbtxt
│   └── 1/model.plan
├── quad_preprocess_dali/
│   ├── config.pbtxt
│   └── 1/model.dali
├── face_embedding_extractor/
│   ├── config.pbtxt
│   └── 1/model.py
└── yolo_face_clip_ensemble/
    └── config.pbtxt

src/
├── utils/
│   └── face_alignment.py             # Landmark → affine matrix
└── schemas/
    └── face.py                       # Face response schemas

scripts/
├── setup_face_test_data.sh           # Download LFW dataset
└── test_face_pipeline.py             # Validation tests
```

### Files to Modify

```
src/clients/triton_client.py          # Add face inference methods
src/clients/opensearch.py             # Add faces index
src/services/visual_search.py         # Add face ingestion/search
src/routers/track_e.py                # Add face endpoints
src/services/clustering.py            # Add FACES cluster config
dali/config.py                        # Add FACE_SIZE, ARCFACE_SIZE
Makefile                              # Add face targets
CLAUDE.md                             # Update documentation
```

---

## Test Datasets

### LFW Deep Funneled (Primary)

| Property | Value |
|----------|-------|
| Images | 13,233 |
| People | 5,749 |
| Size | 111MB |
| Format | JPEG |
| Source | http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz |
| Local Path | `test_images/faces/lfw-deepfunneled/` |

**Download:**
```bash
mkdir -p test_images/faces
cd test_images/faces
wget http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz
tar -xzf lfw-deepfunneled.tgz
rm lfw-deepfunneled.tgz
```

### Test Subset (Quick Validation)

| Property | Value |
|----------|-------|
| Images | 100 |
| People | 20 |
| Size | ~2MB |
| Local Path | `test_images/faces/test_subset/` |

---

## Makefile Targets

```makefile
# Face Detection & Recognition
download-face-models      # Download SCRFD and ArcFace ONNX models
export-face-detection     # Export SCRFD to TensorRT
export-face-recognition   # Export ArcFace to TensorRT
create-face-dali          # Create quad-branch DALI pipeline
setup-face-pipeline       # Complete face pipeline setup
test-track-e-faces        # Test face detection and recognition
download-face-test-data   # Download LFW test dataset
```

---

## API Endpoints

### Face Detection

```
POST /track_e/faces/detect
```
Detect faces in image using SCRFD. Returns face bounding boxes, landmarks, and confidence scores.

### Face Recognition

```
POST /track_e/faces/recognize
```
Detect faces and extract ArcFace identity embeddings. Full GPU pipeline.

### Face Ingestion

```
POST /track_e/faces/ingest
  ?image_id=<optional>
  &person_name=<optional>
```
Ingest faces from image into identity database.

### Face Search

```
POST /track_e/faces/search
  ?face_index=0
  &top_k=10
  &min_score=0.6
```
Find matching identities for a face in the image.

### Face Identification (1:N)

```
POST /track_e/faces/identify
```
Identify all people in image against the face database.

### Face Verification (1:1)

```
POST /track_e/faces/verify
  ?threshold=0.6
```
Verify if two images contain the same person.

---

## Sprint Progress

### Sprint 1: Foundation
**Status:** ✅ Complete
**Completed:** 2026-01-01

| Task | Status | Notes |
|------|--------|-------|
| Download LFW dataset | ✅ | 13,233 images, 5,749 people (from Kaggle) |
| Download WIDER Face | ✅ | 12,880 images, 61 event categories (group photos, etc.) |
| Download SCRFD ONNX | ✅ | `scrfd_10g_bnkps.onnx` (16.1 MB) from HuggingFace |
| Download ArcFace ONNX | ✅ | `arcface_w600k_r50.onnx` (166.3 MB) from HuggingFace |
| Export SCRFD to TRT | ✅ | `model.plan` (11.7 MB), FP16, batch 1-64 |
| Export ArcFace to TRT | ✅ | `model.plan` (86.1 MB), FP16, batch 1-128 |
| Test standalone models | ✅ | Both models verified in Triton |
| Add Makefile targets | ✅ | `setup-face-pipeline`, `test-track-e-faces`, etc. |

**Files Created:**
- `export/download_face_models.py` - Model download utility
- `export/export_face_detection.py` - SCRFD → TensorRT
- `export/export_face_recognition.py` - ArcFace → TensorRT
- `scripts/setup_face_test_data.sh` - LFW dataset setup
- `models/scrfd_10g_face_detect/config.pbtxt` - Triton config
- `models/arcface_w600k_r50/config.pbtxt` - Triton config

**Notes:**
- Using `bnkps` (Batch Normalization) variant instead of `gnkps` (Group Normalization)
- ArcFace outputs not L2-normalized; will normalize in Python backend
- SCRFD has 9 outputs (3 strides × 3 tensors: scores, boxes, landmarks)

### Sprint 2: Triton Integration
**Status:** Not Started

### Sprint 3: API & Search
**Status:** Not Started

### Sprint 4: Testing & Polish
**Status:** Not Started

---

## References

### Model Sources
- [InsightFace GitHub](https://github.com/deepinsight/insightface)
- [SCRFD Documentation](https://github.com/deepinsight/insightface/tree/master/detection/scrfd)
- [InsightFace Model Zoo](https://github.com/deepinsight/insightface/blob/master/model_zoo/README.md)
- [InsightFace-REST (TensorRT deployment)](https://github.com/SthPhoenix/InsightFace-REST)

### Datasets
- [LFW Official](https://vis-www.cs.umass.edu/lfw/)
- [LFW on Kaggle](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset)

### Technical References
- [ArcFace TensorRT Guide](https://medium.com/@penolove15/face-recognition-with-arcface-with-tensorrt-abb544738e39)
- [Face Alignment Library](https://github.com/1adrianb/face-alignment)
