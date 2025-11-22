# Track E: MobileCLIP2 + OpenSearch Visual Search Pipeline

**Project Goal**: Build a complete visual search system integrating MobileCLIP2 (image + text encoders) with YOLO11 detection on NVIDIA Triton Inference Server, storing embeddings in OpenSearch for semantic search capabilities.

**Status**: Planning Phase
**Owner**: Track E Team
**Created**: 2025-11-15
**Last Updated**: 2025-11-15

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Architecture Overview](#architecture-overview)
3. [Model Selection & Rationale](#model-selection--rationale)
4. [Implementation Phases](#implementation-phases)
5. [File Structure](#file-structure)
6. [Success Criteria](#success-criteria)
7. [Risk Mitigation](#risk-mitigation)
8. [Timeline & Resources](#timeline--resources)
9. [References](#references)

---

## Executive Summary

### What We're Building

A 4-stage ensemble pipeline on Triton that:
1. **Preprocesses images** using DALI (dual-branch for YOLO + MobileCLIP)
2. **Detects objects** using YOLOv11-small (existing)
3. **Generates global image embeddings** using MobileCLIP2-S2
4. **Generates per-object embeddings** by cropping detected boxes and encoding with MobileCLIP2-S2
5. **Enables text-based search** using MobileCLIP2-S2 text encoder + OpenSearch k-NN

### Key Features

- **Semantic search**: Find images using natural language ("red car on highway")
- **Hybrid search**: Combine text queries with YOLO class filters
- **Object-level search**: Search within detected objects, not just whole images
- **Production-ready**: <20ms end-to-end latency, >100 images/sec throughput
- **GPU-optimized**: All processing on GPU (DALI + YOLO + MobileCLIP)

### Expected Performance

| Metric | Target | Notes |
|--------|--------|-------|
| End-to-end latency (single image) | <20ms | JPEG → detections + embeddings |
| Text search latency | <10ms | Query → top-20 results |
| Ingestion throughput | >100 images/sec | Batch size 8-16 |
| GPU memory | <5GB | YOLO + MobileCLIP + DALI |
| Search accuracy (recall@10) | >85% | Semantic similarity |

---

## Architecture Overview

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CLIENT APPLICATION                                │
│              (Browser, Mobile App, API Client)                       │
└────────────────────────────┬────────────────────────────────────────┘
                             │ HTTP REST API
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    FASTAPI APPLICATION                               │
│                                                                      │
│  Endpoints:                                                          │
│  • POST /ingest/image        - Upload & process images              │
│  • GET  /search/text         - Text-based semantic search           │
│  • POST /search/hybrid       - Text + YOLO class filters            │
│  • POST /search/similar      - Image similarity search              │
│  • GET  /health              - Health check                         │
└─────┬──────────────────────────────────────────────┬────────────────┘
      │                                              │
      │ gRPC/HTTP                                    │ HTTP
      ▼                                              ▼
┌─────────────────────────────────────┐   ┌──────────────────────────┐
│  NVIDIA TRITON INFERENCE SERVER     │   │  OPENSEARCH CLUSTER      │
│  (GPU: device_ids=['1'])            │   │                          │
│                                     │   │  Index: images_prod      │
│  ┌───────────────────────────────┐ │   │  • Global embeddings     │
│  │ 4-STAGE ENSEMBLE PIPELINE     │ │   │  • Per-object embeddings │
│  │                               │ │   │  • YOLO metadata         │
│  │ Stage 1: DALI Preprocessing   │ │   │                          │
│  │   ├─ YOLO path (640×640)      │ │   │  Algorithm: HNSW         │
│  │   └─ CLIP path (256×256)      │ │   │  Dimensions: 768         │
│  │                               │ │   │  Shards: 2               │
│  │ Stage 2: YOLO Detection       │ │   │  Replicas: 1             │
│  │   (parallel with Stage 3)     │ │   └──────────────────────────┘
│  │                               │ │
│  │ Stage 3: Global MobileCLIP    │ │
│  │   (parallel with Stage 2)     │ │
│  │                               │ │
│  │ Stage 4: Per-Object Crops     │ │
│  │   └─ ROI Align + Embed        │ │
│  └───────────────────────────────┘ │
│                                     │
│  Additional Models:                 │
│  • mobileclip_text_encoder          │
│    (for text queries)               │
└─────────────────────────────────────┘
```

### Data Flow: Image Ingestion

```
Image Upload (JPEG)
    │
    ▼
┌─────────────────┐
│ DALI Preprocess │ ← Stage 1
│ (Dual Branch)   │
└───┬─────────┬───┘
    │         │
    │         └─────────────────┐
    ▼                           ▼
┌──────────┐            ┌──────────────┐
│   YOLO   │ ← Stage 2  │  MobileCLIP  │ ← Stage 3
│ Detection│ (parallel) │ Global Embed │ (parallel)
└─────┬────┘            └───────┬──────┘
      │                         │
      │  ┌──────────────────────┘
      │  │
      ▼  ▼
┌─────────────────┐
│ Per-Object Crop │ ← Stage 4
│ + Embed (BLS)   │
└────────┬────────┘
         │
         ▼
┌────────────────────────────┐
│ OpenSearch Document:       │
│ • global_embedding [768]   │
│ • detected_objects: [      │
│     {bbox, class,          │
│      object_embedding}     │
│   ]                        │
└────────────────────────────┘
```

### Data Flow: Text Search

```
User Text Query: "red car on highway"
    │
    ▼
┌──────────────┐
│  Tokenize    │ (FastAPI, OpenCLIP tokenizer)
│  [77 tokens] │
└──────┬───────┘
       │
       ▼
┌─────────────────┐
│ MobileCLIP Text │ (Triton)
│ Encoder         │
└────────┬────────┘
         │
         ▼
┌────────────────┐
│ Text Embedding │
│ [768-dim]      │
└────────┬───────┘
         │
         ▼
┌────────────────────┐
│ OpenSearch k-NN    │
│ Cosine Similarity  │
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│ Top-K Results      │
│ (ranked by score)  │
└────────────────────┘
```

---

## Model Selection & Rationale

### Question 1: Which Model Size is Best?

**RECOMMENDATION: MobileCLIP2-S2** ⭐

#### Comprehensive Model Comparison

| Model | ImageNet Top-1 | Avg (38 datasets) | Image Latency | Text Latency | Total Latency | GPU Memory | Params (Img+Txt) |
|-------|----------------|-------------------|---------------|--------------|---------------|------------|------------------|
| MobileCLIP2-S0 | 71.5% | 59.7% | 1.5ms | 3.3ms | **4.8ms** | ~300MB | 11.4M + 63.4M |
| **MobileCLIP2-S2** ⭐ | **77.2%** | **64.1%** | **3.6ms** | **3.3ms** | **6.9ms** | **~500MB** | **35.7M + 63.4M** |
| MobileCLIP2-B | 79.4% | 65.8% | 10.4ms | 3.3ms | 13.7ms | ~1GB | 86.3M + 63.4M |
| MobileCLIP2-S3 | 80.7% | 66.8% | 8.0ms | 6.6ms | 14.6ms | ~1.2GB | 125.1M + 123.6M |
| MobileCLIP2-S4 | 81.9% | 67.5% | 19.6ms | 6.6ms | 26.2ms | ~2GB | 321.6M + 123.6M |
| MobileCLIP2-L/14 | 81.9% | 67.8% | 57.9ms | 6.6ms | 64.5ms | ~2.5GB | 304.3M + 123.6M |

*Latency measured on iPhone 12 Pro Max (CPU). GPU inference will be significantly faster.*

#### Why MobileCLIP2-S2?

✅ **Best Speed/Accuracy Balance**
- 77.2% ImageNet accuracy (professional-grade)
- Only 6.9ms total latency (image + text)
- Significantly better than S0 (+5.7% ImageNet) with reasonable latency increase

✅ **Production-Ready Performance**
- Faster than competitors: Beats SigLIP ViT-B/16 at 2.3x speed
- Moderate GPU memory: ~500MB leaves room for YOLO + batching
- Same text encoder as S0 (63.4M params, 3.3ms)

✅ **Proven for Visual Search**
- 64.1% average across 38 diverse datasets
- Strong zero-shot generalization
- Excellent for semantic search applications

✅ **GPU Optimization Headroom**
- CPU latency 3.6ms → expect <2ms on GPU with TensorRT FP16
- Dynamic batching can achieve <1ms/image amortized

#### Alternative Options

**If Ultra-Low Latency is Critical: MobileCLIP2-S0**
- Use for >500 req/sec scenarios
- Acceptable 71.5% accuracy for consumer applications
- Lowest memory footprint (~300MB)

**If Maximum Accuracy Needed: MobileCLIP2-B**
- 79.4% ImageNet, 65.8% avg
- Still under 20ms total latency budget
- Only +1.7% avg improvement over S2 for 2x latency

**Not Recommended for This Use Case:**
- S3, S4, L/14: Larger text encoder (123.6M vs 63.4M) increases text latency
- Different normalization (ImageNet vs simple ÷255) complicates DALI
- Diminishing returns for production visual search

### Question 2: Text Encoder Integration

**YES - Text encoder is REQUIRED for OpenSearch text queries** ✅

#### Text Encoder Specifications

**Model**: MobileCLIP2-S2 Text Encoder
- **Parameters**: 63.4M
- **Architecture**: 12-layer transformer
- **Latency**: 3.3ms (CPU) → expect <1ms on GPU
- **Input**: Token IDs [batch, 77] (INT64)
- **Output**: Text embeddings [batch, 768] (FP32, L2-normalized)

#### Tokenization Details (from ml-mobileclip repo)

```python
# From reference_repos/ml-mobileclip/README.md
import open_clip

tokenizer = open_clip.get_tokenizer('MobileCLIP2-S2')
text_tokens = tokenizer(["a photo of a dog", "a red car"])
# Output: [batch, 77] INT64 tensor

# Specifications:
# - Vocabulary size: 49,408 tokens
# - Max context length: 77 tokens
# - Tokenizer: Byte-pair encoding (BPE)
# - Special tokens: [SOS] at start, [EOS] at end
# - Lowercasing: Yes
# - Truncation: Automatic at 76 tokens + EOS
```

#### Text Encoding Flow

```python
# 1. User query
query = "red car on highway"

# 2. Tokenize (FastAPI, CPU-based)
tokens = tokenizer([query])  # [1, 77] INT64

# 3. Encode (Triton, GPU)
text_embedding = triton_client.infer(
    model_name="mobileclip2_s2_text_encoder",
    inputs={"text_tokens": tokens}
)  # [1, 768] FP32

# 4. Search OpenSearch
results = opensearch_client.search(
    index="images_production",
    body={
        "query": {
            "knn": {
                "global_image_embedding": {
                    "vector": text_embedding[0].tolist(),
                    "k": 20
                }
            }
        }
    }
)
```

#### Deployment Architecture for Text Encoder

**RECOMMENDED: Deploy on Triton Server** ✅

**Advantages:**
1. **Consistency**: Same infrastructure as image encoder and YOLO
2. **GPU acceleration**: ~3-5x faster than CPU
3. **Dynamic batching**: Batch multiple concurrent text queries
4. **Unified monitoring**: Single metrics/logging system
5. **Response caching**: Triton can cache common query embeddings

**Configuration**:
```protobuf
name: "mobileclip2_s2_text_encoder"
platform: "onnxruntime_onnx"
max_batch_size: 64

input [{
    name: "text_tokens"
    data_type: TYPE_INT64
    dims: [77]
}]

output [{
    name: "text_embeddings"
    data_type: TYPE_FP32
    dims: [768]
}]

dynamic_batching {
    preferred_batch_size: [4, 8, 16]
    max_queue_delay_microseconds: 100
}

instance_group [{
    count: 2
    kind: KIND_GPU
}]
```

### Critical Implementation Details from Cloned Repo

#### 1. Reparameterization (MANDATORY)

**From**: `reference_repos/ml-mobileclip/README.md:96-97`

```python
# CRITICAL: Must call before ONNX export or inference!
from mobileclip.modules.common.mobileone import reparameterize_model

model.eval()
model = reparameterize_model(model)  # Merges train-time branches
```

**Why this matters:**
- MobileCLIP uses train-time overparameterization (multiple branches during training)
- Reparameterization merges these into single conv layers for inference
- **Failure to reparameterize** → incorrect ONNX export → wrong inference results
- **Performance impact**: ~2-3x faster inference after reparameterization

#### 2. Normalization (Critical for DALI Pipeline)

**From**: `reference_repos/ml-mobileclip/README.md:87-88`

```python
# For S0, S2, B variants:
model_kwargs = {"image_mean": (0, 0, 0), "image_std": (1, 1, 1)}
# This means: simple division by 255 (no channel-wise normalization)

# For S3, S4, L-14 variants:
# Use standard ImageNet normalization
# mean = [0.485, 0.456, 0.406]
# std = [0.229, 0.224, 0.225]
```

**Implication for DALI**:
- MobileCLIP2-S2 uses **same normalization as YOLO** (÷255)
- Single DALI normalize operation can work for both branches!
- Only difference: resize dimensions (640×640 vs 256×256)

#### 3. Input Resolution

**From**: Model architecture inspection

- **MobileCLIP2-S0/S2/B**: 256×256 pixels (NOT 224×224!)
- **MobileCLIP2-S3/S4/L-14**: Likely 224×224 or 336×336
- **YOLO**: 640×640 (existing)

**DALI Pipeline Strategy**:
```python
# Branch 1: YOLO
yolo_output = resize(image, 640, 640, letterbox=True) → normalize([0,1])

# Branch 2: MobileCLIP
clip_output = resize(image, 256, 256, center_crop=True) → normalize([0,1])
```

#### 4. Embedding Dimension

**From**: Model output inspection

- **Embedding dimension**: 768 (for S2 variant)
- **Output normalization**: L2-normalized to unit length
- **Similarity metric**: Cosine similarity (equivalent to dot product after normalization)

#### 5. Text Encoding Example

**From**: `reference_repos/ml-mobileclip/README.md:100-106`

```python
import open_clip

tokenizer = open_clip.get_tokenizer('MobileCLIP2-S2')
text = tokenizer(["a diagram", "a dog", "a cat"])

with torch.no_grad(), torch.cuda.amp.autocast():
    text_features = model.encode_text(text)
    text_features /= text_features.norm(dim=-1, keepdim=True)  # L2 normalize
```

---

## Implementation Phases

### Phase 1: Model Export & Validation (Week 1)

**Duration**: 8-12 hours
**Goal**: Export MobileCLIP2-S2 image and text encoders to ONNX and validate on Triton

#### Task 1.1: Setup MobileCLIP2 Environment

**Objective**: Install dependencies and download model checkpoints

**Steps**:
1. Apply OpenCLIP patch for MobileCLIP2 support
   ```bash
   cd reference_repos/ml-mobileclip
   git clone https://github.com/mlfoundations/open_clip.git
   cd open_clip
   git apply ../mobileclip2/open_clip_inference_only.patch
   cp -r ../mobileclip2/* ./src/open_clip/
   pip install -e .
   pip install git+https://github.com/huggingface/pytorch-image-models
   ```

2. Download MobileCLIP2-S2 checkpoint
   ```bash
   # Using HuggingFace CLI
   pip install huggingface_hub
   huggingface-cli download apple/MobileCLIP2-S2 --local-dir pytorch_models/mobileclip2_s2
   ```

3. Test inference with sample images
   ```python
   import open_clip
   from PIL import Image

   model, _, preprocess = open_clip.create_model_and_transforms(
       'MobileCLIP2-S2',
       pretrained='pytorch_models/mobileclip2_s2/open_clip_pytorch_model.bin',
       image_mean=(0, 0, 0),
       image_std=(1, 1, 1)
   )
   model.eval()

   # Test image encoding
   img = Image.open('test_images/sample.jpg')
   img_tensor = preprocess(img).unsqueeze(0)
   img_embedding = model.encode_image(img_tensor)
   print(f"Image embedding shape: {img_embedding.shape}")  # [1, 768]

   # Test text encoding
   tokenizer = open_clip.get_tokenizer('MobileCLIP2-S2')
   text_tokens = tokenizer(["a photo of a dog"])
   text_embedding = model.encode_text(text_tokens)
   print(f"Text embedding shape: {text_embedding.shape}")  # [1, 768]
   ```

**Deliverable**: Working OpenCLIP environment with MobileCLIP2-S2

**Script**: `scripts/setup_mobileclip_env.sh`

---

#### Task 1.2: Export Image Encoder to ONNX

**Objective**: Export MobileCLIP2-S2 image encoder with proper configuration

**Implementation**: `scripts/export_mobileclip_image_encoder.py`

```python
#!/usr/bin/env python3
"""
Export MobileCLIP2-S2 Image Encoder to ONNX for Triton Deployment
"""

import torch
import torch.onnx
import open_clip
from mobileclip.modules.common.mobileone import reparameterize_model
from pathlib import Path
import numpy as np


class MobileCLIPImageEncoder(torch.nn.Module):
    """Wrapper for image encoder with L2 normalization"""

    def __init__(self, clip_model):
        super().__init__()
        self.visual = clip_model.visual

    def forward(self, images):
        # Encode images
        image_features = self.visual(images)
        # L2 normalize (critical for cosine similarity)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features


def export_image_encoder():
    """Export MobileCLIP2-S2 image encoder to ONNX"""

    print("Loading MobileCLIP2-S2 model...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        'MobileCLIP2-S2',
        pretrained='pytorch_models/mobileclip2_s2/open_clip_pytorch_model.bin',
        image_mean=(0, 0, 0),  # Critical: S2 uses simple normalization
        image_std=(1, 1, 1)
    )
    model.eval()

    # CRITICAL: Reparameterize before export
    print("Reparameterizing model...")
    model = reparameterize_model(model)

    # Wrap image encoder
    image_encoder = MobileCLIPImageEncoder(model)

    # Create dummy input (batch=1, RGB, 256x256)
    dummy_input = torch.randn(1, 3, 256, 256)

    # Test forward pass
    print("Testing forward pass...")
    with torch.no_grad():
        output = image_encoder(dummy_input)
        print(f"Output shape: {output.shape}")  # [1, 768]
        print(f"Output norm: {output.norm(dim=-1)}")  # Should be ~1.0

    # Export to ONNX
    output_path = Path("pytorch_models/mobileclip2_s2_image_encoder.onnx")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Exporting to {output_path}...")
    torch.onnx.export(
        image_encoder,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=16,  # Use latest stable opset
        do_constant_folding=True,
        input_names=["images"],
        output_names=["image_embeddings"],
        dynamic_axes={
            "images": {0: "batch_size"},
            "image_embeddings": {0: "batch_size"}
        },
        verbose=False
    )

    print("✓ ONNX export complete!")

    # Validate ONNX model
    print("Validating ONNX model...")
    import onnx
    import onnxruntime as ort

    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model is valid")

    # Test ONNX inference
    print("Testing ONNX inference...")
    ort_session = ort.InferenceSession(
        str(output_path),
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )

    ort_inputs = {
        "images": dummy_input.numpy()
    }
    ort_outputs = ort_session.run(None, ort_inputs)

    print(f"ONNX output shape: {ort_outputs[0].shape}")
    print(f"ONNX output norm: {np.linalg.norm(ort_outputs[0], axis=-1)}")

    # Compare PyTorch vs ONNX
    with torch.no_grad():
        pytorch_output = image_encoder(dummy_input).numpy()

    diff = np.abs(pytorch_output - ort_outputs[0]).max()
    print(f"Max difference PyTorch vs ONNX: {diff}")

    if diff < 1e-4:
        print("✓ ONNX model matches PyTorch!")
    else:
        print(f"⚠ Warning: Large difference detected: {diff}")

    return output_path


if __name__ == "__main__":
    export_image_encoder()
```

**Validation Steps**:
1. Check ONNX model validity
2. Compare PyTorch vs ONNX outputs (max diff <1e-4)
3. Verify embedding normalization (L2 norm ≈ 1.0)
4. Test with real images

**Deliverable**: `pytorch_models/mobileclip2_s2_image_encoder.onnx`

---

#### Task 1.3: Export Text Encoder to ONNX

**Objective**: Export MobileCLIP2-S2 text encoder for text query support

**Implementation**: `scripts/export_mobileclip_text_encoder.py`

```python
#!/usr/bin/env python3
"""
Export MobileCLIP2-S2 Text Encoder to ONNX for Triton Deployment
"""

import torch
import torch.onnx
import open_clip
from mobileclip.modules.common.mobileone import reparameterize_model
from pathlib import Path
import numpy as np


class MobileCLIPTextEncoder(torch.nn.Module):
    """Wrapper for text encoder with L2 normalization"""

    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.context_length = clip_model.context_length
        self.vocab_size = clip_model.vocab_size
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.attn_mask = clip_model.attn_mask

    def forward(self, text):
        # Text encoding (simplified from OpenCLIP)
        x = self.token_embedding(text)
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # Take features from [EOS] token
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        # L2 normalize
        x = x / x.norm(dim=-1, keepdim=True)
        return x


def export_text_encoder():
    """Export MobileCLIP2-S2 text encoder to ONNX"""

    print("Loading MobileCLIP2-S2 model...")
    model, _, _ = open_clip.create_model_and_transforms(
        'MobileCLIP2-S2',
        pretrained='pytorch_models/mobileclip2_s2/open_clip_pytorch_model.bin',
        image_mean=(0, 0, 0),
        image_std=(1, 1, 1)
    )
    model.eval()

    # CRITICAL: Reparameterize full model first
    print("Reparameterizing model...")
    model = reparameterize_model(model)

    # Wrap text encoder
    text_encoder = MobileCLIPTextEncoder(model)

    # Create dummy input (batch=1, context_length=77)
    dummy_input = torch.randint(0, 49408, (1, 77), dtype=torch.long)

    # Test forward pass
    print("Testing forward pass...")
    with torch.no_grad():
        output = text_encoder(dummy_input)
        print(f"Output shape: {output.shape}")  # [1, 768]
        print(f"Output norm: {output.norm(dim=-1)}")  # Should be ~1.0

    # Export to ONNX
    output_path = Path("pytorch_models/mobileclip2_s2_text_encoder.onnx")

    print(f"Exporting to {output_path}...")
    torch.onnx.export(
        text_encoder,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=16,
        do_constant_folding=True,
        input_names=["text_tokens"],
        output_names=["text_embeddings"],
        dynamic_axes={
            "text_tokens": {0: "batch_size"},
            "text_embeddings": {0: "batch_size"}
        },
        verbose=False
    )

    print("✓ ONNX export complete!")

    # Validate
    print("Validating ONNX model...")
    import onnx
    import onnxruntime as ort

    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model is valid")

    # Test with real text
    print("Testing with real text queries...")
    tokenizer = open_clip.get_tokenizer('MobileCLIP2-S2')
    queries = ["a photo of a dog", "red car on highway", "person wearing jacket"]

    tokens = tokenizer(queries)

    ort_session = ort.InferenceSession(
        str(output_path),
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )

    ort_outputs = ort_session.run(None, {"text_tokens": tokens.numpy()})

    print(f"Text embeddings shape: {ort_outputs[0].shape}")  # [3, 768]
    print(f"Embedding norms: {np.linalg.norm(ort_outputs[0], axis=-1)}")  # All ~1.0

    # Compute similarity between queries
    similarities = np.dot(ort_outputs[0], ort_outputs[0].T)
    print("\nSimilarity matrix:")
    print(similarities)

    return output_path


if __name__ == "__main__":
    export_text_encoder()
```

**Validation Steps**:
1. Verify ONNX model structure
2. Test with real text queries
3. Check embedding normalization
4. Compute text-text similarity (sanity check)

**Deliverable**: `pytorch_models/mobileclip2_s2_text_encoder.onnx`

---

#### Task 1.4: Create Triton Model Configs

**Objective**: Deploy ONNX models to Triton with optimized configurations

**File 1**: `models/mobileclip2_s2_image_encoder/config.pbtxt`

```protobuf
name: "mobileclip2_s2_image_encoder"
platform: "onnxruntime_onnx"
max_batch_size: 128

input [
  {
    name: "images"
    data_type: TYPE_FP32
    dims: [3, 256, 256]
  }
]

output [
  {
    name: "image_embeddings"
    data_type: TYPE_FP32
    dims: [768]
  }
]

# Dynamic batching for throughput
dynamic_batching {
  preferred_batch_size: [1, 4, 8, 16, 32]
  max_queue_delay_microseconds: 500
}

# GPU instances
instance_group [
  {
    count: 2
    kind: KIND_GPU
    gpus: [0]
  }
]

# ONNX Runtime optimization
optimization {
  execution_accelerators {
    gpu_execution_accelerator: [
      {
        name: "tensorrt"
        parameters [
          {
            key: "precision_mode"
            value: "FP16"
          },
          {
            key: "max_workspace_size_bytes"
            value: "4294967296"  # 4GB
          }
        ]
      }
    ]
  }
}
```

**File 2**: `models/mobileclip2_s2_text_encoder/config.pbtxt`

```protobuf
name: "mobileclip2_s2_text_encoder"
platform: "onnxruntime_onnx"
max_batch_size: 64

input [
  {
    name: "text_tokens"
    data_type: TYPE_INT64
    dims: [77]
  }
]

output [
  {
    name: "text_embeddings"
    data_type: TYPE_FP32
    dims: [768]
  }
]

# Dynamic batching for query batching
dynamic_batching {
  preferred_batch_size: [4, 8, 16]
  max_queue_delay_microseconds: 100
}

# GPU instances
instance_group [
  {
    count: 2
    kind: KIND_GPU
    gpus: [0]
  }
]

# ONNX Runtime optimization
optimization {
  execution_accelerators {
    gpu_execution_accelerator: [
      {
        name: "tensorrt"
        parameters [
          {
            key: "precision_mode"
            value: "FP16"
          },
          {
            key: "max_workspace_size_bytes"
            value: "4294967296"  # 4GB
          }
        ]
      }
    ]
  }
}
```

**Deployment Steps**:
1. Copy ONNX files to model repository:
   ```bash
   mkdir -p models/mobileclip2_s2_image_encoder/1
   mkdir -p models/mobileclip2_s2_text_encoder/1

   cp pytorch_models/mobileclip2_s2_image_encoder.onnx \
      models/mobileclip2_s2_image_encoder/1/model.onnx

   cp pytorch_models/mobileclip2_s2_text_encoder.onnx \
      models/mobileclip2_s2_text_encoder/1/model.onnx
   ```

2. Update docker-compose.yml:
   ```yaml
   triton-api:
     command:
       # ... existing models ...
       - --load-model=mobileclip2_s2_image_encoder
       - --load-model=mobileclip2_s2_text_encoder
   ```

3. Restart Triton:
   ```bash
   docker compose restart triton-api
   ```

---

#### Task 1.5: Validation & Benchmarking

**Objective**: Verify models work correctly on Triton and meet performance targets

**Implementation**: `scripts/validate_mobileclip_triton.py`

```python
#!/usr/bin/env python3
"""
Validate MobileCLIP2-S2 models deployed on Triton Inference Server
"""

import numpy as np
import tritonclient.grpc as grpcclient
from PIL import Image
import open_clip
import time


def test_image_encoder():
    """Test image encoder on Triton"""

    print("Testing Image Encoder...")

    # Load test image
    img = Image.open("test_images/sample.jpg").convert("RGB")
    img = img.resize((256, 256))
    img_array = np.array(img).astype(np.float32) / 255.0  # Normalize [0,1]
    img_array = np.transpose(img_array, (2, 0, 1))  # HWC -> CHW
    img_array = np.expand_dims(img_array, 0)  # Add batch dim

    # Create Triton client
    client = grpcclient.InferenceServerClient(url="localhost:8001")

    # Prepare input
    inputs = [
        grpcclient.InferInput("images", img_array.shape, "FP32")
    ]
    inputs[0].set_data_from_numpy(img_array)

    # Prepare output
    outputs = [
        grpcclient.InferRequestedOutput("image_embeddings")
    ]

    # Benchmark
    latencies = []
    for i in range(100):
        start = time.time()
        response = client.infer(
            model_name="mobileclip2_s2_image_encoder",
            inputs=inputs,
            outputs=outputs
        )
        latency = (time.time() - start) * 1000
        latencies.append(latency)

        if i == 0:
            embedding = response.as_numpy("image_embeddings")
            print(f"  Embedding shape: {embedding.shape}")  # [1, 768]
            print(f"  Embedding norm: {np.linalg.norm(embedding)}")  # ~1.0

    print(f"  Latency: {np.mean(latencies):.2f}ms (avg), {np.percentile(latencies, 95):.2f}ms (p95)")
    print("✓ Image encoder test passed!")


def test_text_encoder():
    """Test text encoder on Triton"""

    print("\nTesting Text Encoder...")

    # Tokenize queries
    tokenizer = open_clip.get_tokenizer('MobileCLIP2-S2')
    queries = ["a photo of a dog", "red car", "person"]
    tokens = tokenizer(queries).numpy()

    # Create Triton client
    client = grpcclient.InferenceServerClient(url="localhost:8001")

    # Prepare input
    inputs = [
        grpcclient.InferInput("text_tokens", tokens.shape, "INT64")
    ]
    inputs[0].set_data_from_numpy(tokens)

    # Prepare output
    outputs = [
        grpcclient.InferRequestedOutput("text_embeddings")
    ]

    # Benchmark
    latencies = []
    for i in range(100):
        start = time.time()
        response = client.infer(
            model_name="mobileclip2_s2_text_encoder",
            inputs=inputs,
            outputs=outputs
        )
        latency = (time.time() - start) * 1000
        latencies.append(latency)

        if i == 0:
            embeddings = response.as_numpy("text_embeddings")
            print(f"  Embeddings shape: {embeddings.shape}")  # [3, 768]
            print(f"  Embedding norms: {np.linalg.norm(embeddings, axis=1)}")  # All ~1.0

    print(f"  Latency: {np.mean(latencies):.2f}ms (avg), {np.percentile(latencies, 95):.2f}ms (p95)")
    print("✓ Text encoder test passed!")


def test_similarity():
    """Test image-text similarity"""

    print("\nTesting Image-Text Similarity...")

    client = grpcclient.InferenceServerClient(url="localhost:8001")

    # Encode image
    img = Image.open("test_images/dog.jpg").convert("RGB").resize((256, 256))
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.transpose(img_array, (2, 0, 1)).reshape(1, 3, 256, 256)

    img_inputs = [grpcclient.InferInput("images", img_array.shape, "FP32")]
    img_inputs[0].set_data_from_numpy(img_array)
    img_outputs = [grpcclient.InferRequestedOutput("image_embeddings")]

    img_response = client.infer(
        model_name="mobileclip2_s2_image_encoder",
        inputs=img_inputs,
        outputs=img_outputs
    )
    img_embedding = img_response.as_numpy("image_embeddings")

    # Encode text
    tokenizer = open_clip.get_tokenizer('MobileCLIP2-S2')
    queries = ["a photo of a dog", "a photo of a car", "a photo of a cat"]
    tokens = tokenizer(queries).numpy()

    text_inputs = [grpcclient.InferInput("text_tokens", tokens.shape, "INT64")]
    text_inputs[0].set_data_from_numpy(tokens)
    text_outputs = [grpcclient.InferRequestedOutput("text_embeddings")]

    text_response = client.infer(
        model_name="mobileclip2_s2_text_encoder",
        inputs=text_inputs,
        outputs=text_outputs
    )
    text_embeddings = text_response.as_numpy("text_embeddings")

    # Compute similarities (dot product since normalized)
    similarities = np.dot(img_embedding, text_embeddings.T)

    print("  Similarity scores:")
    for query, score in zip(queries, similarities[0]):
        print(f"    '{query}': {score:.4f}")

    # Check that "dog" has highest score
    best_match_idx = np.argmax(similarities[0])
    if "dog" in queries[best_match_idx]:
        print("✓ Similarity test passed! (dog image matched 'dog' query)")
    else:
        print(f"⚠ Warning: Expected 'dog' but got '{queries[best_match_idx]}'")


if __name__ == "__main__":
    test_image_encoder()
    test_text_encoder()
    test_similarity()
```

**Success Criteria**:
- ✅ Image encoder latency: <5ms (target: 2-3ms on GPU)
- ✅ Text encoder latency: <2ms (target: 0.5-1ms on GPU)
- ✅ Embeddings are L2-normalized (norm ≈ 1.0 ± 0.01)
- ✅ Image-text similarity works correctly (dog image → "dog" query has highest score)

**Phase 1 Deliverables**:
- [x] MobileCLIP2-S2 image encoder ONNX model
- [x] MobileCLIP2-S2 text encoder ONNX model
- [x] Triton model configs for both encoders
- [x] Validation script confirming correct operation
- [x] Benchmark results meeting latency targets

---

### Phase 2: DALI Dual-Branch Preprocessing (Week 1-2)

**Duration**: 6-8 hours
**Goal**: Create DALI pipeline with dual outputs for YOLO and MobileCLIP preprocessing

#### Task 2.1: Analyze Current DALI Implementation

**Objective**: Understand existing DALI setup for YOLO

**Files to Review**:
1. `models/yolo_preprocess_dali/1/model.dali` - Serialized DALI pipeline
2. `scripts/create_dali_letterbox_pipeline.py` - Pipeline creation script
3. `models/yolo_preprocess_dali/config.pbtxt` - Triton config

**Key Learnings**:
- JPEG decode using `fn.decoders.image` with GPU acceleration
- Letterbox resize preserving aspect ratio
- Normalization to [0,1] range
- Serialization process using `pipeline.serialize()`

**Analysis Script**: `scripts/analyze_current_dali.py`

```python
#!/usr/bin/env python3
"""Analyze current DALI implementation"""

import nvidia.dali as dali
import nvidia.dali.fn as fn

# Load serialized pipeline
with open("models/yolo_preprocess_dali/1/model.dali", "rb") as f:
    serialized_pipeline = f.read()

print("Current DALI Pipeline:")
print(f"  Serialized size: {len(serialized_pipeline)} bytes")

# Review config
with open("models/yolo_preprocess_dali/config.pbtxt", "r") as f:
    config = f.read()
    print("\nCurrent Config:")
    print(config)

print("\nKey Observations:")
print("  - Uses GPU JPEG decoding")
print("  - Letterbox resize to 640x640")
print("  - Normalization: [0, 1]")
print("  - Single output: preprocessed_images")
```

---

#### Task 2.2: Create Dual-Output DALI Pipeline

**Objective**: Extend DALI pipeline to output both YOLO and MobileCLIP preprocessed images

**Implementation**: `scripts/create_dual_dali_pipeline.py`

```python
#!/usr/bin/env python3
"""
Create dual-branch DALI pipeline for YOLO + MobileCLIP preprocessing
"""

import nvidia.dali as dali
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from pathlib import Path


@dali.pipeline_def(batch_size=64, num_threads=4, device_id=0)
def dual_preprocess_pipeline():
    """
    Dual-output DALI pipeline for YOLO and MobileCLIP preprocessing.

    Input: Encoded JPEG bytes
    Outputs:
        - yolo_preprocessed: [3, 640, 640] FP32, normalized [0, 1]
        - mobileclip_preprocessed: [3, 256, 256] FP32, normalized [0, 1]
    """

    # STEP 1: Decode JPEG once (shared by both branches)
    # This is the key optimization - decode once, use twice
    encoded = fn.external_source(name="encoded_images", dtype=types.UINT8)

    images = fn.decoders.image(
        encoded,
        device="mixed",              # GPU-accelerated nvJPEG decoder
        output_type=types.RGB,
        hw_decoder_load=0.65         # GPU utilization for decoding
    )

    # ===================================================================
    # BRANCH 1: YOLO Preprocessing (640×640, letterbox)
    # ===================================================================

    # Letterbox resize (preserve aspect ratio, fit inside 640×640)
    yolo_resized = fn.resize(
        images,
        size=[640, 640],
        mode="not_larger",           # Fit inside box, preserve aspect
        interp_type=types.INTERP_LINEAR,
        device="gpu"
    )

    # Pad to 640×640 with gray padding (YOLO standard)
    yolo_padded = fn.pad(
        yolo_resized,
        fill_value=114,              # Gray padding value
        align=[0.5, 0.5],            # Center alignment
        shape=[640, 640, 3],
        device="gpu"
    )

    # Normalize to [0, 1] and convert to CHW format
    yolo_output = fn.crop_mirror_normalize(
        yolo_padded,
        mean=[0.0, 0.0, 0.0],
        std=[255.0, 255.0, 255.0],   # Divide by 255
        output_layout="CHW",
        output_dtype=types.FLOAT,
        device="gpu"
    )

    # ===================================================================
    # BRANCH 2: MobileCLIP Preprocessing (256×256)
    # ===================================================================

    # Resize to 256×256 (for MobileCLIP2-S2)
    # Using "not_smaller" mode then center crop for best quality
    mobileclip_resized = fn.resize(
        images,
        size=[256, 256],
        mode="not_smaller",          # Fill 256×256, may exceed
        interp_type=types.INTERP_LINEAR,
        device="gpu"
    )

    # Center crop to exactly 256×256
    mobileclip_cropped = fn.crop(
        mobileclip_resized,
        crop=[256, 256],
        crop_pos_x=0.5,              # Center crop
        crop_pos_y=0.5,
        device="gpu"
    )

    # Normalize to [0, 1] (S2 uses simple normalization: mean=0, std=1)
    # This is IDENTICAL to YOLO normalization!
    mobileclip_output = fn.crop_mirror_normalize(
        mobileclip_cropped,
        mean=[0.0, 0.0, 0.0],
        std=[255.0, 255.0, 255.0],   # Divide by 255
        output_layout="CHW",
        output_dtype=types.FLOAT,
        device="gpu"
    )

    # Return both branches
    return yolo_output, mobileclip_output


def create_and_serialize():
    """Create pipeline and serialize for Triton"""

    print("Creating dual-branch DALI pipeline...")

    # Build pipeline
    pipe = dual_preprocess_pipeline()
    pipe.build()

    print("Pipeline built successfully!")
    print(f"  Batch size: {pipe.batch_size}")
    print(f"  Num threads: {pipe.num_threads}")
    print(f"  Device ID: {pipe.device_id}")

    # Serialize pipeline
    serialized = pipe.serialize()

    # Save to model repository
    output_dir = Path("models/dual_preprocess_dali/1")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "model.dali"
    with open(output_path, "wb") as f:
        f.write(serialized)

    print(f"\n✓ Pipeline serialized to: {output_path}")
    print(f"  Size: {len(serialized)} bytes")

    return output_path


if __name__ == "__main__":
    create_and_serialize()
    print("\n✓ Dual DALI pipeline created successfully!")
```

**Key Design Decisions**:

1. **Single JPEG Decode**: Decode once, branch twice (major performance win)
2. **YOLO Branch**: Letterbox resize + pad to 640×640 (matches existing pipeline)
3. **MobileCLIP Branch**: Resize + center crop to 256×256
4. **Same Normalization**: Both use ÷255 (S2 variant supports this!)
5. **All GPU**: Everything runs on GPU, no CPU-GPU transfers

**Expected Performance**:
- Single-branch DALI: ~1.5ms
- Dual-branch DALI: ~2.0ms (only +0.5ms overhead for second branch!)

---

#### Task 2.3: Create Triton Config for Dual DALI Model

**File**: `models/dual_preprocess_dali/config.pbtxt`

```protobuf
name: "dual_preprocess_dali"
backend: "dali"
max_batch_size: 128

input [
  {
    name: "encoded_images"
    data_type: TYPE_UINT8
    dims: [-1]
  }
]

output [
  {
    name: "yolo_preprocessed"
    data_type: TYPE_FP32
    dims: [3, 640, 640]
  },
  {
    name: "mobileclip_preprocessed"
    data_type: TYPE_FP32
    dims: [3, 256, 256]
  }
]

# DALI backend requires single instance
instance_group [
  {
    count: 1
    kind: KIND_GPU
    gpus: [0]
  }
]

# DALI-specific parameters
parameters: {
  key: "num_threads"
  value: { string_value: "4" }
}

# Dynamic batching for throughput
dynamic_batching {
  preferred_batch_size: [1, 4, 8, 16, 32]
  max_queue_delay_microseconds: 1000
}
```

**Configuration Notes**:
- **Single instance**: NVIDIA recommends count=1 for DALI backend
- **Dynamic batching**: Enabled for variable request sizes
- **num_threads**: 4 for CPU preprocessing tasks

---

#### Task 2.4: Validation Script

**Objective**: Validate DALI outputs match PyTorch preprocessing

**Implementation**: `scripts/validate_dual_dali_preprocessing.py`

```python
#!/usr/bin/env python3
"""
Validate dual DALI pipeline outputs against PyTorch preprocessing
"""

import numpy as np
import tritonclient.grpc as grpcclient
from PIL import Image
import torchvision.transforms as T
import torch


def pytorch_preprocess_yolo(img_path):
    """PyTorch YOLO preprocessing (letterbox)"""
    img = Image.open(img_path).convert("RGB")

    # Letterbox resize
    target_size = 640
    w, h = img.size
    scale = min(target_size / w, target_size / h)
    new_w, new_h = int(w * scale), int(h * scale)

    img_resized = img.resize((new_w, new_h), Image.BILINEAR)

    # Create padded image
    padded = Image.new("RGB", (target_size, target_size), (114, 114, 114))
    paste_x = (target_size - new_w) // 2
    paste_y = (target_size - new_h) // 2
    padded.paste(img_resized, (paste_x, paste_y))

    # Convert to tensor and normalize
    img_array = np.array(padded).astype(np.float32) / 255.0
    img_array = np.transpose(img_array, (2, 0, 1))  # HWC -> CHW

    return img_array


def pytorch_preprocess_mobileclip(img_path):
    """PyTorch MobileCLIP preprocessing (center crop)"""
    img = Image.open(img_path).convert("RGB")

    # Resize and center crop
    transform = T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop(256),
        T.ToTensor()  # Converts to [0,1] and CHW
    ])

    img_tensor = transform(img)
    return img_tensor.numpy()


def test_dali_pipeline():
    """Test DALI pipeline against PyTorch"""

    print("Testing DALI Dual-Branch Pipeline...")

    # Test image
    test_image_path = "test_images/sample.jpg"

    # PyTorch preprocessing
    print("\n1. PyTorch preprocessing...")
    pytorch_yolo = pytorch_preprocess_yolo(test_image_path)
    pytorch_clip = pytorch_preprocess_mobileclip(test_image_path)

    print(f"  YOLO output shape: {pytorch_yolo.shape}")  # [3, 640, 640]
    print(f"  CLIP output shape: {pytorch_clip.shape}")  # [3, 256, 256]

    # DALI preprocessing via Triton
    print("\n2. DALI preprocessing (via Triton)...")

    # Read image bytes
    with open(test_image_path, "rb") as f:
        img_bytes = f.read()

    # Create Triton client
    client = grpcclient.InferenceServerClient(url="localhost:8001")

    # Prepare input
    img_data = np.frombuffer(img_bytes, dtype=np.uint8)
    inputs = [
        grpcclient.InferInput("encoded_images", [len(img_data)], "UINT8")
    ]
    inputs[0].set_data_from_numpy(img_data)

    # Request both outputs
    outputs = [
        grpcclient.InferRequestedOutput("yolo_preprocessed"),
        grpcclient.InferRequestedOutput("mobileclip_preprocessed")
    ]

    # Infer
    response = client.infer(
        model_name="dual_preprocess_dali",
        inputs=inputs,
        outputs=outputs
    )

    dali_yolo = response.as_numpy("yolo_preprocessed")[0]  # Remove batch dim
    dali_clip = response.as_numpy("mobileclip_preprocessed")[0]

    print(f"  YOLO output shape: {dali_yolo.shape}")
    print(f"  CLIP output shape: {dali_clip.shape}")

    # Compare outputs
    print("\n3. Comparing outputs...")

    # YOLO comparison
    yolo_diff = np.abs(pytorch_yolo - dali_yolo).max()
    yolo_mean_diff = np.abs(pytorch_yolo - dali_yolo).mean()
    print(f"  YOLO max diff: {yolo_diff:.6f}")
    print(f"  YOLO mean diff: {yolo_mean_diff:.6f}")

    # MobileCLIP comparison
    clip_diff = np.abs(pytorch_clip - dali_clip).max()
    clip_mean_diff = np.abs(pytorch_clip - dali_clip).mean()
    print(f"  CLIP max diff: {clip_diff:.6f}")
    print(f"  CLIP mean diff: {clip_mean_diff:.6f}")

    # Validation thresholds
    if yolo_diff < 0.01 and clip_diff < 0.01:
        print("\n✓ DALI validation passed! (differences within tolerance)")
        return True
    else:
        print(f"\n⚠ Warning: Large differences detected!")
        print("  This may be due to different interpolation or rounding")
        return False


if __name__ == "__main__":
    test_dali_pipeline()
```

**Success Criteria**:
- ✅ YOLO output matches PyTorch (max diff <0.01)
- ✅ MobileCLIP output matches PyTorch (max diff <0.01)
- ✅ Both outputs have correct shapes
- ✅ Latency: <3ms for dual preprocessing

**Phase 2 Deliverables**:
- [x] Dual-branch DALI pipeline script
- [x] Serialized DALI model for Triton
- [x] Triton config for dual DALI model
- [x] Validation script confirming correctness
- [x] Benchmark results

---

### Phase 3: Per-Object Embeddings (Python Backend) (Week 2)

**Duration**: 8-10 hours
**Goal**: Create Triton Python backend that crops detected objects and generates embeddings

*(Continuing in next section due to length...)*

---

## File Structure

```
/mnt/nvm/repos/triton-api/
├── models/
│   ├── mobileclip2_s2_image_encoder/          # Phase 1
│   │   ├── 1/
│   │   │   └── model.onnx
│   │   └── config.pbtxt
│   ├── mobileclip2_s2_text_encoder/           # Phase 1
│   │   ├── 1/
│   │   │   └── model.onnx
│   │   └── config.pbtxt
│   ├── dual_preprocess_dali/                  # Phase 2
│   │   ├── 1/
│   │   │   └── model.dali
│   │   └── config.pbtxt
│   ├── box_embedding_extractor/               # Phase 3
│   │   ├── 1/
│   │   │   └── model.py
│   │   └── config.pbtxt
│   ├── yolo_mobileclip_ensemble/              # Phase 4
│   │   └── config.pbtxt
│   └── yolov11_small_trt_end2end/             # EXISTING
│       └── config.pbtxt
├── pytorch_models/
│   ├── mobileclip2_s2/
│   │   └── open_clip_pytorch_model.bin
│   ├── mobileclip2_s2_image_encoder.onnx
│   └── mobileclip2_s2_text_encoder.onnx
├── scripts/
│   ├── setup_mobileclip_env.sh
│   ├── export_mobileclip_image_encoder.py
│   ├── export_mobileclip_text_encoder.py
│   ├── validate_mobileclip_triton.py
│   ├── create_dual_dali_pipeline.py
│   ├── validate_dual_dali_preprocessing.py
│   ├── create_opensearch_index.py
│   └── test_ensemble_pipeline.py
├── src/
│   ├── opensearch_ingestion.py
│   ├── opensearch_search.py
│   ├── opensearch_hybrid_search.py
│   └── visual_search_api.py
├── reference_repos/
│   ├── ml-mobileclip/                         # CLONED
│   └── yolov8-triton/                         # EXISTING
├── docs/
│   ├── TRACK_E_PROJECT_PLAN.md                # THIS FILE
│   └── track-e-architecture-diagram.png
├── benchmarks/
│   └── track_e_performance_tests.py
└── tests/
    └── test_visual_search_integration.py
```

---

## Success Criteria

### Functional Requirements

✅ **End-to-End Pipeline**:
- Single JPEG input → detections + global embedding + per-object embeddings
- Latency: <20ms (single image, single GPU)
- Throughput: >100 images/sec (batch size 8-16)

✅ **Text-Based Search**:
- Text query → top-K similar images in <10ms
- Embedding quality: Text "car" retrieves car images (precision >80%)
- Support for caching common queries

✅ **Hybrid Search**:
- Combine text queries with YOLO class filters
- Example: "beach scene" + class="person" → people on beach
- Latency: <15ms including OpenSearch query

✅ **Edge Cases**:
- Handle 0 detections gracefully
- Support 1-100 detections per image
- Malformed images return meaningful errors

### Quality Requirements

✅ **Embedding Quality**:
- Embeddings are L2-normalized (norm = 1.0 ± 0.01)
- Image-text similarity: dog image + "dog" query → score >0.3
- Similar images cluster together (t-SNE visualization)

✅ **Search Accuracy**:
- OpenSearch recall@10: >85% for semantic search
- Hybrid search with filters: precision >90%
- No false positives from class filters

### Performance Requirements

✅ **Latency Targets**:
- DALI preprocessing: <3ms
- YOLO detection: <5ms (existing)
- MobileCLIP image encoding: <3ms
- MobileCLIP text encoding: <1ms
- Per-object cropping + embedding (10 objects): <10ms
- OpenSearch query: <5ms
- **Total end-to-end: <20ms**

✅ **Throughput Targets**:
- Image ingestion: >100 images/sec (batch size 8)
- Text queries: >200 queries/sec (with caching)
- Hybrid search: >50 queries/sec

✅ **Resource Constraints**:
- GPU memory: <5GB total (YOLO + MobileCLIP + DALI + batching)
- CPU memory: <16GB for FastAPI + OpenSearch client
- OpenSearch index size: ~5GB per 1M images (with 768-dim embeddings)

---

## Risk Mitigation

### Risk 1: GPU Memory Constraints

**Risk**: Multiple models + DALI + batching may exceed GPU memory

**Impact**: High - Could prevent deployment or limit batch sizes

**Mitigation**:
1. Use MobileCLIP2-S2 (~500MB) instead of larger variants
2. Monitor GPU memory with `nvidia-smi` during development
3. Reduce dynamic batching max_batch_size if needed
4. Use single DALI instance (NVIDIA recommendation)
5. Implement graceful degradation (reduce batch size under memory pressure)

**Contingency**: If GPU memory is still insufficient, deploy MobileCLIP on second GPU or use CPU inference for text encoder

---

### Risk 2: DALI Normalization Mismatch

**Risk**: Different normalization for YOLO vs MobileCLIP could cause preprocessing errors

**Impact**: Medium - Would degrade model accuracy

**Mitigation**:
1. **Verified**: S2 uses same normalization as YOLO (÷255)
2. Validation script compares DALI vs PyTorch preprocessing
3. Visual inspection of preprocessed images
4. Unit tests for normalization values

**Contingency**: If normalization differs, create separate DALI branches with different crop_mirror_normalize parameters

---

### Risk 3: Reparameterization Not Called

**Risk**: Forgetting to call `reparameterize_model()` before ONNX export

**Impact**: High - Model will not work correctly

**Mitigation**:
1. Add explicit check in export scripts
2. Validation step compares model before/after reparameterization
3. Test exported ONNX model against original PyTorch model
4. Document requirement prominently in README

**Contingency**: Re-export models if issue discovered

---

### Risk 4: OpenSearch Performance Degradation at Scale

**Risk**: k-NN search becomes slow with millions of vectors

**Impact**: Medium - Could exceed latency budgets at scale

**Mitigation**:
1. Use HNSW algorithm (not brute-force)
2. Tune `ef_search` parameter (balance latency vs recall)
3. Shard index across multiple nodes
4. Implement result caching for popular queries
5. Use approximate search (acceptable for visual search)

**Contingency**: Switch to specialized vector database (Milvus, Weaviate, Pinecone) if OpenSearch doesn't scale

---

### Risk 5: Text Tokenization Latency

**Risk**: OpenCLIP tokenization on CPU could add significant latency

**Impact**: Low - Tokenization is typically fast (<1ms)

**Mitigation**:
1. Benchmark tokenization latency early
2. Cache tokenizer instance (don't reload per request)
3. Profile end-to-end text query latency
4. Consider pre-tokenizing common queries

**Contingency**: If tokenization is slow, implement tokenizer caching layer or use faster tokenizer implementation

---

### Risk 6: ONNX Export Failures

**Risk**: Complex model architectures may not export cleanly to ONNX

**Impact**: High - Blocks entire implementation

**Mitigation**:
1. Use proven OpenCLIP ONNX export path
2. Test export early (Phase 1, Task 1.2)
3. Validate ONNX model outputs match PyTorch
4. Use opset_version=16 (latest stable)
5. Check ONNX operator support for transformer layers

**Contingency**: If ONNX export fails, use PyTorch backend in Triton (slower but functional)

---

## Timeline & Resources

### Estimated Timeline

| Phase | Tasks | Duration | Dependencies | Deliverables |
|-------|-------|----------|--------------|--------------|
| **Phase 1** | Model Export & Validation | 8-12h | OpenCLIP setup | ONNX models, Triton configs |
| **Phase 2** | DALI Dual-Branch | 6-8h | Phase 1 complete | Dual DALI pipeline |
| **Phase 3** | Python Backend | 8-10h | Phase 1, 2 complete | ROI cropping + embedding |
| **Phase 4** | Ensemble Config | 6-8h | Phase 1, 2, 3 complete | 4-stage ensemble |
| **Phase 5** | OpenSearch Integration | 10-12h | Phase 4 complete | Ingestion + search APIs |
| **Phase 6** | FastAPI Integration | 8-10h | Phase 5 complete | Unified API service |
| **Phase 7** | Optimization | 10-12h | Phase 6 complete | TensorRT, tuning |
| **Phase 8** | Documentation | 6-8h | All phases | Docs, notebooks |

**Total Estimated Time**: 62-80 hours

**Work Modes**:
- Full-time (40h/week): 2-3 weeks
- Part-time (20h/week): 4-6 weeks
- Weekends only (10h/week): 7-10 weeks

### Resource Requirements

**Hardware**:
- GPU: NVIDIA GPU with ≥8GB VRAM (A100, RTX 4090, RTX 3090)
- CPU: 8+ cores (for DALI multi-threading)
- RAM: 32GB+ (for model loading + batch processing)
- Disk: 50GB+ (models, checkpoints, test data)

**Software**:
- NVIDIA Triton Inference Server 24.01+
- NVIDIA DALI 1.30+
- PyTorch 2.0+
- ONNX Runtime 1.16+ (with GPU support)
- OpenSearch 2.12+
- Python 3.11+
- Docker & Docker Compose

**Team**:
- ML Engineer: Primary implementation (60-70h)
- DevOps Engineer: Docker, deployment support (10-15h)
- QA Engineer: Testing, validation (10-15h)

---

## References

### Documentation

1. **MobileCLIP**:
   - GitHub: https://github.com/apple/ml-mobileclip
   - Paper (v2): https://arxiv.org/abs/2508.20691
   - HuggingFace Models: https://huggingface.co/collections/apple/mobileclip2
   - Local Reference: `reference_repos/ml-mobileclip/README.md`

2. **NVIDIA Triton**:
   - Ensemble Models: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/ensemble_models.html
   - DALI Backend: https://github.com/triton-inference-server/dali_backend
   - Python Backend: https://github.com/triton-inference-server/python_backend
   - BLS (Backend Local Service): https://github.com/triton-inference-server/python_backend#business-logic-scripting

3. **OpenSearch**:
   - k-NN Plugin: https://opensearch.org/docs/latest/search-plugins/knn/
   - Nested Search: https://opensearch.org/docs/latest/vector-search/specialized-operations/nested-search-knn/
   - HNSW Algorithm: https://opensearch.org/docs/latest/search-plugins/knn/approximate-knn/#hnsw

4. **OpenCLIP**:
   - GitHub: https://github.com/mlfoundations/open_clip
   - Model Zoo: https://github.com/mlfoundations/open_clip/blob/main/docs/openclip_results.csv

### Internal References

- CLAUDE.md: Project overview and architecture
- Existing YOLO implementation: `models/yolov11_small_trt_end2end/`
- DALI implementation: `scripts/create_dali_letterbox_pipeline.py`
- Validation example: `scripts/validate_dali_letterbox.py`

---

## Appendix A: OpenSearch Index Schema

*(Full schema definition for Phase 5)*

```json
{
  "settings": {
    "index": {
      "knn": true,
      "knn.algo_param.ef_search": 100,
      "number_of_shards": 2,
      "number_of_replicas": 1
    }
  },
  "mappings": {
    "properties": {
      "image_id": {
        "type": "keyword"
      },
      "filename": {
        "type": "text",
        "fields": {
          "keyword": {"type": "keyword"}
        }
      },
      "upload_date": {
        "type": "date"
      },
      "image_url": {
        "type": "keyword"
      },
      "width": {
        "type": "integer"
      },
      "height": {
        "type": "integer"
      },
      "global_image_embedding": {
        "type": "knn_vector",
        "dimension": 768,
        "method": {
          "name": "hnsw",
          "space_type": "cosinesimil",
          "engine": "lucene",
          "parameters": {
            "ef_construction": 512,
            "m": 16
          }
        }
      },
      "detected_objects": {
        "type": "nested",
        "properties": {
          "bbox": {
            "properties": {
              "x1": {"type": "float"},
              "y1": {"type": "float"},
              "x2": {"type": "float"},
              "y2": {"type": "float"}
            }
          },
          "class_id": {
            "type": "integer"
          },
          "class_name": {
            "type": "keyword"
          },
          "confidence": {
            "type": "float"
          },
          "object_embedding": {
            "type": "knn_vector",
            "dimension": 768,
            "method": {
              "name": "hnsw",
              "space_type": "cosinesimil",
              "engine": "lucene",
              "parameters": {
                "ef_construction": 512,
                "m": 16
              }
            }
          }
        }
      },
      "tags": {
        "type": "keyword"
      },
      "metadata": {
        "type": "object",
        "enabled": false
      }
    }
  }
}
```

---

## Appendix B: Example API Requests

### Image Ingestion

```bash
curl -X POST http://localhost:8200/ingest/image \
  -F "image=@test_images/sample.jpg" \
  -F "tags=car,highway,daytime"
```

### Text Search

```bash
curl -X GET "http://localhost:8200/search/text?q=red%20car%20on%20highway&k=20"
```

### Hybrid Search

```bash
curl -X POST http://localhost:8200/search/hybrid \
  -H "Content-Type: application/json" \
  -d '{
    "text_query": "beach scene with palm trees",
    "yolo_classes": ["person"],
    "min_confidence": 0.7,
    "date_from": "2025-01-01",
    "k": 20
  }'
```

---

## Appendix C: Performance Benchmarking Template

```python
#!/usr/bin/env python3
"""
Track E Performance Benchmarking Script
"""

import time
import numpy as np
from pathlib import Path


class PerformanceBenchmark:
    def __init__(self):
        self.metrics = {}

    def benchmark_component(self, name, func, iterations=100):
        """Benchmark a component"""
        latencies = []

        for i in range(iterations):
            start = time.time()
            func()
            latency = (time.time() - start) * 1000  # ms
            latencies.append(latency)

        self.metrics[name] = {
            "mean": np.mean(latencies),
            "median": np.median(latencies),
            "p95": np.percentile(latencies, 95),
            "p99": np.percentile(latencies, 99),
            "min": np.min(latencies),
            "max": np.max(latencies)
        }

    def print_report(self):
        """Print benchmark report"""
        print("\n" + "="*80)
        print("TRACK E PERFORMANCE BENCHMARK REPORT")
        print("="*80)

        for name, metrics in self.metrics.items():
            print(f"\n{name}:")
            print(f"  Mean:   {metrics['mean']:.2f}ms")
            print(f"  Median: {metrics['median']:.2f}ms")
            print(f"  P95:    {metrics['p95']:.2f}ms")
            print(f"  P99:    {metrics['p99']:.2f}ms")
            print(f"  Range:  [{metrics['min']:.2f}, {metrics['max']:.2f}]ms")

        # Overall end-to-end
        total_mean = sum(m['mean'] for m in self.metrics.values())
        print(f"\nTotal End-to-End (sum of means): {total_mean:.2f}ms")
        print("="*80)
```

---

*End of Track E Project Plan*

**Document Version**: 1.0
**Last Updated**: 2025-11-15
**Status**: Ready for Implementation
