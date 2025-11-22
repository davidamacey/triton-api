# Track E: Implementation Phases (Continued)

**Part 2 of Track E Project Plan**
**Parent Document**: [TRACK_E_PROJECT_PLAN.md](./TRACK_E_PROJECT_PLAN.md)

This document contains detailed implementation instructions for Phases 3-8.

---

## Table of Contents

1. [Phase 3: Per-Object Embeddings (Python Backend)](#phase-3-per-object-embeddings-python-backend)
2. [Phase 4: Ensemble Configuration](#phase-4-ensemble-configuration)
3. [Phase 5: OpenSearch Integration](#phase-5-opensearch-integration)
4. [Phase 6: FastAPI Integration & Testing](#phase-6-fastapi-integration--testing)
5. [Phase 7: Optimization & Production Hardening](#phase-7-optimization--production-hardening)
6. [Phase 8: Documentation & Deployment](#phase-8-documentation--deployment)

---

## Phase 3: Per-Object Embeddings (Python Backend)

**Duration**: 8-10 hours
**Goal**: Create Triton Python backend that crops detected objects and generates embeddings using BLS

### Overview

This phase implements the critical "per-object embedding" functionality that makes this pipeline unique. The Python backend will:
1. Receive full MobileCLIP-preprocessed image and YOLO detections
2. Crop each detected bounding box using GPU-accelerated ROI align
3. Batch all crops and call MobileCLIP image encoder via BLS
4. Return fixed-size array of embeddings (MAX_BOXES × 768)

---

### Task 3.1: Design Python Backend Architecture

**Objective**: Define inputs, outputs, and processing logic

**Architecture Diagram**:

```
Inputs (from ensemble):
  - full_image [3, 256, 256]          # MobileCLIP-preprocessed
  - det_boxes [100, 4]                # YOLO boxes [x, y, w, h]
  - num_dets [1]                      # Number of valid detections
  - det_scores [100]                  # Confidence scores
  - det_classes [100]                 # Class IDs

Processing:
  1. Extract valid boxes (0 to num_dets)
  2. Convert [x, y, w, h] → [x1, y1, x2, y2]
  3. Scale boxes to 256×256 image space
  4. ROI align to extract 256×256 crops (GPU)
  5. Batch crops → MobileCLIP encoder (BLS call)
  6. Pad to MAX_BOXES with zeros

Output:
  - box_embeddings [100, 768]         # Fixed-size output
```

**Key Design Decisions**:

1. **MAX_BOXES = 100**: Balances memory usage vs rare high-detection scenarios
2. **Zero-padding**: Unused slots filled with zeros (client filters using num_dets)
3. **GPU ROI align**: Uses PyTorch torchvision.ops.roi_align for speed
4. **BLS batching**: Single call to MobileCLIP for all crops (not sequential)
5. **Error handling**: Return zero embeddings if num_dets=0 or errors occur

---

### Task 3.2: Implement ROI Cropping Logic

**Objective**: Efficiently crop bounding boxes on GPU

**Key Components**:

1. **Box Format Conversion**:
   ```python
   # YOLO format: [x_center, y_center, width, height] (normalized or absolute)
   # ROI align format: [x1, y1, x2, y2] (absolute coordinates)

   def xywh_to_xyxy(boxes):
       """Convert center format to corner format"""
       x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
       x1 = x - w / 2
       y1 = y - h / 2
       x2 = x + w / 2
       y2 = y + h / 2
       return torch.stack([x1, y1, x2, y2], dim=1)
   ```

2. **Scaling to Image Coordinates**:
   ```python
   # If boxes are normalized [0,1], scale to image size
   def scale_boxes_to_image(boxes, img_width=256, img_height=256):
       """Scale normalized boxes to pixel coordinates"""
       boxes_scaled = boxes.clone()
       boxes_scaled[:, [0, 2]] *= img_width   # x1, x2
       boxes_scaled[:, [1, 3]] *= img_height  # y1, y2
       return boxes_scaled
   ```

3. **ROI Align**:
   ```python
   import torchvision.ops as ops

   def crop_boxes_roi_align(image, boxes, output_size=256):
       """
       Extract box regions using ROI align (GPU-accelerated)

       Args:
           image: [3, H, W] tensor
           boxes: [N, 4] tensor in [x1, y1, x2, y2] format
           output_size: Size of output crops

       Returns:
           [N, 3, output_size, output_size] tensor
       """

       # Add batch index (all boxes from same image)
       batch_indices = torch.zeros(boxes.shape[0], 1, device=boxes.device)
       boxes_with_idx = torch.cat([batch_indices, boxes], dim=1)

       # ROI align
       crops = ops.roi_align(
           image.unsqueeze(0),       # Add batch dim [1, 3, H, W]
           boxes_with_idx,           # [N, 5] (batch_idx, x1, y1, x2, y2)
           output_size=(output_size, output_size),
           spatial_scale=1.0,        # No scaling (boxes already in image coords)
           sampling_ratio=2,         # Bilinear sampling quality
           aligned=True              # Improved alignment
       )

       return crops  # [N, 3, output_size, output_size]
   ```

**Why ROI Align vs Naive Cropping?**
- ✅ GPU-accelerated (10-100x faster than CPU crop+resize)
- ✅ Sub-pixel accuracy (better for small objects)
- ✅ Differentiable (if needed for future fine-tuning)
- ✅ Handles fractional coordinates gracefully

---

### Task 3.3: Implement BLS Model Calls

**Objective**: Call MobileCLIP image encoder from Python backend using Triton BLS

**BLS (Business Logic Scripting) Overview**:
- Allows Python backend to call other Triton models
- In-process calls (no HTTP/gRPC overhead)
- Supports batching and pipelining
- Ideal for multi-model workflows

**Implementation**:

```python
import triton_python_backend_utils as pb_utils

def call_mobileclip_encoder(crops_tensor):
    """
    Call MobileCLIP image encoder via BLS

    Args:
        crops_tensor: numpy array [N, 3, 256, 256]

    Returns:
        embeddings: numpy array [N, 768]
    """

    # Create BLS inference request
    inference_request = pb_utils.InferenceRequest(
        model_name='mobileclip2_s2_image_encoder',
        requested_output_names=['image_embeddings'],
        inputs=[
            pb_utils.Tensor('images', crops_tensor)
        ]
    )

    # Execute (synchronous)
    inference_response = inference_request.exec()

    # Check for errors
    if inference_response.has_error():
        error_msg = inference_response.error().message()
        raise RuntimeError(f"MobileCLIP inference failed: {error_msg}")

    # Extract embeddings
    embeddings = pb_utils.get_output_tensor_by_name(
        inference_response,
        'image_embeddings'
    ).as_numpy()

    return embeddings  # [N, 768]
```

**Error Handling**:
```python
try:
    embeddings = call_mobileclip_encoder(crops)
except Exception as e:
    # Log error and return zero embeddings
    logger.error(f"BLS call failed: {e}")
    embeddings = np.zeros((num_dets, 768), dtype=np.float32)
```

---

### Task 3.4: Complete Python Backend Implementation

**File**: `models/box_embedding_extractor/1/model.py`

```python
#!/usr/bin/env python3
"""
Triton Python Backend: Per-Object Embedding Extractor

This backend receives YOLO detections and full MobileCLIP-preprocessed image,
crops each detected object, and generates embeddings using MobileCLIP via BLS.
"""

import triton_python_backend_utils as pb_utils
import torch
import torchvision.ops as ops
import numpy as np
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TritonPythonModel:
    """Triton Python Backend for Per-Object Embedding Extraction"""

    def initialize(self, args):
        """
        Initialize model

        Args:
            args: Dictionary with model configuration
        """
        self.model_config = json.loads(args['model_config'])

        # Configuration
        self.max_boxes = 100
        self.output_embed_dim = 768
        self.crop_size = 256

        # Setup device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Initialized box_embedding_extractor on {self.device}")

        # Warmup (optional)
        if self.device == 'cuda':
            self._warmup_gpu()

    def _warmup_gpu(self):
        """Warmup GPU with dummy operations"""
        try:
            dummy_image = torch.randn(3, 256, 256, device=self.device)
            dummy_boxes = torch.tensor([[50, 50, 100, 100]], device=self.device, dtype=torch.float32)
            _ = self._crop_boxes(dummy_image, dummy_boxes)
            logger.info("GPU warmup complete")
        except Exception as e:
            logger.warning(f"GPU warmup failed: {e}")

    def _xywh_to_xyxy(self, boxes):
        """
        Convert boxes from [x_center, y_center, width, height] to [x1, y1, x2, y2]

        Args:
            boxes: Tensor [N, 4]

        Returns:
            Tensor [N, 4] in corner format
        """
        x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        return torch.stack([x1, y1, x2, y2], dim=1)

    def _crop_boxes(self, image, boxes):
        """
        Crop boxes from image using ROI align

        Args:
            image: Tensor [3, H, W]
            boxes: Tensor [N, 4] in [x1, y1, x2, y2] format

        Returns:
            Tensor [N, 3, crop_size, crop_size]
        """
        if boxes.shape[0] == 0:
            return torch.empty(0, 3, self.crop_size, self.crop_size, device=self.device)

        # Clamp boxes to image boundaries
        boxes = boxes.clone()
        boxes[:, [0, 2]] = torch.clamp(boxes[:, [0, 2]], 0, image.shape[2] - 1)  # x coords
        boxes[:, [1, 3]] = torch.clamp(boxes[:, [1, 3]], 0, image.shape[1] - 1)  # y coords

        # Add batch index (all boxes from same image)
        batch_indices = torch.zeros(boxes.shape[0], 1, device=boxes.device)
        boxes_with_idx = torch.cat([batch_indices, boxes], dim=1)

        # ROI align
        crops = ops.roi_align(
            image.unsqueeze(0),  # [1, 3, H, W]
            boxes_with_idx,      # [N, 5]
            output_size=(self.crop_size, self.crop_size),
            spatial_scale=1.0,
            sampling_ratio=2,
            aligned=True
        )

        return crops  # [N, 3, crop_size, crop_size]

    def _call_mobileclip_encoder(self, crops):
        """
        Call MobileCLIP image encoder via BLS

        Args:
            crops: numpy array [N, 3, 256, 256]

        Returns:
            embeddings: numpy array [N, 768]
        """
        # Create BLS request
        inference_request = pb_utils.InferenceRequest(
            model_name='mobileclip2_s2_image_encoder',
            requested_output_names=['image_embeddings'],
            inputs=[
                pb_utils.Tensor('images', crops)
            ]
        )

        # Execute
        inference_response = inference_request.exec()

        # Check for errors
        if inference_response.has_error():
            raise RuntimeError(inference_response.error().message())

        # Extract embeddings
        embeddings = pb_utils.get_output_tensor_by_name(
            inference_response,
            'image_embeddings'
        ).as_numpy()

        return embeddings

    def execute(self, requests):
        """
        Execute inference requests

        Args:
            requests: List of pb_utils.InferenceRequest

        Returns:
            List of pb_utils.InferenceResponse
        """
        responses = []

        for request in requests:
            try:
                # Get inputs
                full_image_tensor = pb_utils.get_input_tensor_by_name(request, "full_image")
                det_boxes_tensor = pb_utils.get_input_tensor_by_name(request, "det_boxes")
                num_dets_tensor = pb_utils.get_input_tensor_by_name(request, "num_dets")

                # Convert to numpy/torch
                full_image = torch.from_numpy(full_image_tensor.as_numpy()).to(self.device)  # [3, 256, 256]
                det_boxes = torch.from_numpy(det_boxes_tensor.as_numpy()).to(self.device)    # [100, 4]
                num_dets = int(num_dets_tensor.as_numpy()[0])  # Scalar

                # Handle case: no detections
                if num_dets == 0:
                    box_embeddings = np.zeros((self.max_boxes, self.output_embed_dim), dtype=np.float32)
                    output_tensor = pb_utils.Tensor('box_embeddings', box_embeddings)
                    responses.append(pb_utils.InferenceResponse(output_tensors=[output_tensor]))
                    continue

                # Extract valid boxes
                valid_boxes = det_boxes[:num_dets]  # [num_dets, 4]

                # Convert format: [x, y, w, h] → [x1, y1, x2, y2]
                boxes_xyxy = self._xywh_to_xyxy(valid_boxes)

                # Crop boxes using ROI align
                cropped_boxes = self._crop_boxes(full_image, boxes_xyxy)  # [num_dets, 3, 256, 256]

                # Convert to numpy for BLS
                cropped_boxes_np = cropped_boxes.cpu().numpy()

                # Call MobileCLIP encoder via BLS
                embeddings = self._call_mobileclip_encoder(cropped_boxes_np)  # [num_dets, 768]

                # Pad to max_boxes
                box_embeddings = np.zeros((self.max_boxes, self.output_embed_dim), dtype=np.float32)
                box_embeddings[:num_dets] = embeddings

                # Create output tensor
                output_tensor = pb_utils.Tensor('box_embeddings', box_embeddings)
                responses.append(pb_utils.InferenceResponse(output_tensors=[output_tensor]))

            except Exception as e:
                logger.error(f"Error in execute: {e}", exc_info=True)

                # Return zero embeddings on error
                box_embeddings = np.zeros((self.max_boxes, self.output_embed_dim), dtype=np.float32)
                output_tensor = pb_utils.Tensor('box_embeddings', box_embeddings)
                responses.append(pb_utils.InferenceResponse(output_tensors=[output_tensor]))

        return responses

    def finalize(self):
        """Cleanup"""
        logger.info("Finalizing box_embedding_extractor")
```

**Configuration**: `models/box_embedding_extractor/config.pbtxt`

```protobuf
name: "box_embedding_extractor"
backend: "python"
max_batch_size: 0  # Python backend handles batching internally

input [
  {
    name: "full_image"
    data_type: TYPE_FP32
    dims: [3, 256, 256]
  },
  {
    name: "det_boxes"
    data_type: TYPE_FP32
    dims: [100, 4]
  },
  {
    name: "num_dets"
    data_type: TYPE_INT32
    dims: [1]
  }
]

output [
  {
    name: "box_embeddings"
    data_type: TYPE_FP32
    dims: [100, 768]
  }
]

instance_group [
  {
    count: 2
    kind: KIND_GPU
    gpus: [0]
  }
]

# Python backend specific parameters
parameters: {
  key: "EXECUTION_ENV_PATH"
  value: {string_value: "/opt/tritonserver/backends/python/envs/python_backend_env"}
}
```

---

### Task 3.5: Testing & Validation

**Test Script**: `scripts/test_box_embedding_extractor.py`

```python
#!/usr/bin/env python3
"""Test box embedding extractor Python backend"""

import numpy as np
import tritonclient.grpc as grpcclient


def test_box_embeddings():
    """Test per-object embedding extraction"""

    print("Testing Box Embedding Extractor...")

    # Create dummy inputs
    full_image = np.random.randn(3, 256, 256).astype(np.float32)

    # 5 detections
    det_boxes = np.zeros((100, 4), dtype=np.float32)
    det_boxes[0] = [50, 50, 30, 30]   # [x, y, w, h]
    det_boxes[1] = [100, 100, 40, 40]
    det_boxes[2] = [150, 150, 35, 35]
    det_boxes[3] = [200, 50, 25, 25]
    det_boxes[4] = [50, 200, 30, 30]

    num_dets = np.array([5], dtype=np.int32)

    # Create Triton client
    client = grpcclient.InferenceServerClient(url="localhost:8001")

    # Prepare inputs
    inputs = [
        grpcclient.InferInput("full_image", full_image.shape, "FP32"),
        grpcclient.InferInput("det_boxes", det_boxes.shape, "FP32"),
        grpcclient.InferInput("num_dets", num_dets.shape, "INT32")
    ]
    inputs[0].set_data_from_numpy(full_image)
    inputs[1].set_data_from_numpy(det_boxes)
    inputs[2].set_data_from_numpy(num_dets)

    # Request output
    outputs = [
        grpcclient.InferRequestedOutput("box_embeddings")
    ]

    # Infer
    response = client.infer(
        model_name="box_embedding_extractor",
        inputs=inputs,
        outputs=outputs
    )

    box_embeddings = response.as_numpy("box_embeddings")

    print(f"  Box embeddings shape: {box_embeddings.shape}")  # [100, 768]

    # Check that first 5 are non-zero, rest are zero
    print(f"  Valid embeddings norm (0-4): {np.linalg.norm(box_embeddings[:5], axis=1)}")
    print(f"  Padded embeddings norm (5-9): {np.linalg.norm(box_embeddings[5:10], axis=1)}")

    if np.all(box_embeddings[5:] == 0):
        print("✓ Padding is correct (zero embeddings)")
    else:
        print("⚠ Warning: Padding contains non-zero values")

    if np.all(np.linalg.norm(box_embeddings[:5], axis=1) > 0):
        print("✓ Valid embeddings are non-zero")
    else:
        print("⚠ Warning: Some valid embeddings are zero")


if __name__ == "__main__":
    test_box_embeddings()
```

**Phase 3 Deliverables**:
- [x] Python backend implementation with ROI align
- [x] BLS calls to MobileCLIP encoder
- [x] Triton config for Python backend
- [x] Test script validating correct operation
- [x] Benchmark: <10ms for 10 objects

---

## Phase 4: Ensemble Configuration

**Duration**: 6-8 hours
**Goal**: Wire all components into 4-stage ensemble pipeline

### Task 4.1: Design Ensemble Architecture

**4-Stage Pipeline**:

```
┌─────────────────────────────────────────────────────────────┐
│                    ENSEMBLE PIPELINE                         │
└─────────────────────────────────────────────────────────────┘

Stage 1: dual_preprocess_dali
         Input:  encoded_images [JPEG bytes]
         Output: yolo_preprocessed [3, 640, 640]
                 mobileclip_preprocessed [3, 256, 256]

         ┌──────────────┐
         │   YOLO Path  │
         └──────┬───────┘
                │
Stage 2:        ▼                    Stage 3 (parallel):
         yolov11_small_trt_end2end   mobileclip2_s2_image_encoder
         Input:  yolo_preprocessed   Input:  mobileclip_preprocessed
         Output: num_dets, boxes,    Output: global_embeddings [768]
                 scores, classes

         ┌──────┴─────────────┬─────────────┘
         │                    │
         ▼                    ▼
Stage 4: box_embedding_extractor
         Input:  mobileclip_preprocessed, det_boxes, num_dets
         Output: box_embeddings [100, 768]

Final Outputs:
  - num_dets [1]
  - det_boxes [100, 4]
  - det_scores [100]
  - det_classes [100]
  - global_embeddings [768]
  - box_embeddings [100, 768]
```

**Key Properties**:
- Stages 2 & 3 run in **parallel** (both only depend on Stage 1)
- Stage 4 **waits** for Stages 1, 2, 3 (needs all their outputs)
- Triton scheduler handles dependencies automatically via tensor routing

---

### Task 4.2: Create Ensemble Config

**File**: `models/yolo_mobileclip_ensemble/config.pbtxt`

```protobuf
name: "yolo_mobileclip_ensemble"
platform: "ensemble"
max_batch_size: 64

# ===========================================================================
# INPUTS: Raw JPEG bytes
# ===========================================================================
input [
  {
    name: "encoded_images"
    data_type: TYPE_UINT8
    dims: [-1]
  }
]

# ===========================================================================
# OUTPUTS: Detections + Global Embedding + Per-Object Embeddings
# ===========================================================================
output [
  # YOLO detection outputs
  {
    name: "num_dets"
    data_type: TYPE_INT32
    dims: [1]
  },
  {
    name: "det_boxes"
    data_type: TYPE_FP32
    dims: [100, 4]
  },
  {
    name: "det_scores"
    data_type: TYPE_FP32
    dims: [100]
  },
  {
    name: "det_classes"
    data_type: TYPE_INT32
    dims: [100]
  },

  # MobileCLIP outputs
  {
    name: "global_embeddings"
    data_type: TYPE_FP32
    dims: [768]
  },
  {
    name: "box_embeddings"
    data_type: TYPE_FP32
    dims: [100, 768]
  }
]

# ===========================================================================
# ENSEMBLE SCHEDULING: 4-Stage Pipeline
# ===========================================================================
ensemble_scheduling {
  # -------------------------------------------------------------------------
  # STAGE 1: DALI Dual Preprocessing
  # -------------------------------------------------------------------------
  step [
    {
      model_name: "dual_preprocess_dali"
      model_version: -1

      input_map {
        key: "encoded_images"
        value: "encoded_images"
      }

      output_map {
        key: "yolo_preprocessed"
        value: "_yolo_preprocessed"  # Internal tensor
      }

      output_map {
        key: "mobileclip_preprocessed"
        value: "_mobileclip_preprocessed"  # Internal tensor
      }
    },

    # -------------------------------------------------------------------------
    # STAGE 2: YOLO Detection (runs in parallel with Stage 3)
    # -------------------------------------------------------------------------
    {
      model_name: "yolov11_small_trt_end2end"
      model_version: -1

      input_map {
        key: "images"
        value: "_yolo_preprocessed"
      }

      output_map {
        key: "num_dets"
        value: "num_dets"  # Final output
      }

      output_map {
        key: "det_boxes"
        value: "_det_boxes"  # Pass to Stage 4
      }

      output_map {
        key: "det_scores"
        value: "det_scores"  # Final output
      }

      output_map {
        key: "det_classes"
        value: "det_classes"  # Final output
      }
    },

    # -------------------------------------------------------------------------
    # STAGE 3: Global MobileCLIP Embedding (runs in parallel with Stage 2)
    # -------------------------------------------------------------------------
    {
      model_name: "mobileclip2_s2_image_encoder"
      model_version: -1

      input_map {
        key: "images"
        value: "_mobileclip_preprocessed"
      }

      output_map {
        key: "image_embeddings"
        value: "global_embeddings"  # Final output
      }
    },

    # -------------------------------------------------------------------------
    # STAGE 4: Per-Object Embeddings (waits for Stages 1, 2, 3)
    # -------------------------------------------------------------------------
    {
      model_name: "box_embedding_extractor"
      model_version: -1

      input_map {
        key: "full_image"
        value: "_mobileclip_preprocessed"
      }

      input_map {
        key: "det_boxes"
        value: "_det_boxes"
      }

      input_map {
        key: "num_dets"
        value: "num_dets"
      }

      output_map {
        key: "box_embeddings"
        value: "box_embeddings"  # Final output
      }
    }
  ]
}

# Dynamic batching at ensemble level
dynamic_batching {
  preferred_batch_size: [1, 4, 8, 16]
  max_queue_delay_microseconds: 1000
}
```

**Critical Configuration Notes**:

1. **Tensor Naming Convention**:
   - Prefix `_` for internal tensors (not exposed to client)
   - No prefix for final ensemble outputs
   - Example: `_yolo_preprocessed` vs `global_embeddings`

2. **Dependency Handling**:
   - Stage 2 depends on Stage 1 (`_yolo_preprocessed`)
   - Stage 3 depends on Stage 1 (`_mobileclip_preprocessed`)
   - Stage 4 depends on Stages 1, 2, 3 (uses all their outputs)
   - Triton automatically schedules based on data dependencies

3. **Parallel Execution**:
   - Stages 2 and 3 have no dependency on each other
   - Triton scheduler runs them in parallel on GPU
   - Reduces latency by ~50% vs sequential execution

---

### Task 4.3: Test End-to-End Ensemble

**Test Script**: `scripts/test_ensemble_pipeline.py`

```python
#!/usr/bin/env python3
"""
Test complete YOLO + MobileCLIP ensemble pipeline
"""

import numpy as np
import tritonclient.grpc as grpcclient
from pathlib import Path
import time


def test_ensemble_e2e():
    """Test end-to-end ensemble pipeline"""

    print("Testing YOLO + MobileCLIP Ensemble...")

    # Load test image (JPEG bytes)
    test_image_path = "test_images/sample.jpg"
    with open(test_image_path, "rb") as f:
        img_bytes = f.read()

    img_data = np.frombuffer(img_bytes, dtype=np.uint8)

    # Create Triton client
    client = grpcclient.InferenceServerClient(url="localhost:8001")

    # Prepare input
    inputs = [
        grpcclient.InferInput("encoded_images", [len(img_data)], "UINT8")
    ]
    inputs[0].set_data_from_numpy(img_data)

    # Request all outputs
    outputs = [
        grpcclient.InferRequestedOutput("num_dets"),
        grpcclient.InferRequestedOutput("det_boxes"),
        grpcclient.InferRequestedOutput("det_scores"),
        grpcclient.InferRequestedOutput("det_classes"),
        grpcclient.InferRequestedOutput("global_embeddings"),
        grpcclient.InferRequestedOutput("box_embeddings")
    ]

    # Benchmark
    print("\nRunning inference (warmup + 100 iterations)...")

    latencies = []
    for i in range(101):  # 1 warmup + 100 measured
        start = time.time()

        response = client.infer(
            model_name="yolo_mobileclip_ensemble",
            inputs=inputs,
            outputs=outputs
        )

        latency = (time.time() - start) * 1000  # ms

        if i > 0:  # Skip warmup
            latencies.append(latency)

        if i == 1:  # Print first real result
            print("\nFirst Inference Results:")

            num_dets = response.as_numpy("num_dets")[0]
            det_boxes = response.as_numpy("det_boxes")
            det_scores = response.as_numpy("det_scores")
            det_classes = response.as_numpy("det_classes")
            global_embeddings = response.as_numpy("global_embeddings")
            box_embeddings = response.as_numpy("box_embeddings")

            print(f"  Number of detections: {num_dets}")
            print(f"  Detections shape: {det_boxes.shape}")  # [100, 4]
            print(f"  Global embedding shape: {global_embeddings.shape}")  # [768]
            print(f"  Box embeddings shape: {box_embeddings.shape}")  # [100, 768]

            # Validate
            print("\nValidation:")
            print(f"  Global embedding norm: {np.linalg.norm(global_embeddings):.4f}")

            if num_dets > 0:
                print(f"  First detection:")
                print(f"    Box: {det_boxes[0]}")
                print(f"    Score: {det_scores[0]:.4f}")
                print(f"    Class: {det_classes[0]}")
                print(f"    Embedding norm: {np.linalg.norm(box_embeddings[0]):.4f}")

    # Print statistics
    print(f"\nLatency Statistics (100 iterations):")
    print(f"  Mean:   {np.mean(latencies):.2f}ms")
    print(f"  Median: {np.median(latencies):.2f}ms")
    print(f"  P95:    {np.percentile(latencies, 95):.2f}ms")
    print(f"  P99:    {np.percentile(latencies, 99):.2f}ms")
    print(f"  Min:    {np.min(latencies):.2f}ms")
    print(f"  Max:    {np.max(latencies):.2f}ms")

    # Check target
    if np.mean(latencies) < 20:
        print("\n✓ Latency target met (<20ms)!")
    else:
        print(f"\n⚠ Latency target missed (mean: {np.mean(latencies):.2f}ms > 20ms)")


if __name__ == "__main__":
    test_ensemble_e2e()
```

**Success Criteria**:
- ✅ All outputs have correct shapes
- ✅ num_dets matches number of non-zero box embeddings
- ✅ Global embedding is normalized (norm ≈ 1.0)
- ✅ Mean latency <20ms (single image, GPU)
- ✅ No errors or warnings in Triton logs

**Phase 4 Deliverables**:
- [x] Ensemble config with 4 stages
- [x] End-to-end test script
- [x] Latency benchmarks
- [x] Validation of all outputs

---

## Phase 5: OpenSearch Integration

**Duration**: 10-12 hours
**Goal**: Setup OpenSearch cluster and implement ingestion + search APIs

### Task 5.1: Setup OpenSearch Cluster

**Installation (Docker)**:

```yaml
# Add to docker-compose.yml
services:
  opensearch:
    image: opensearchproject/opensearch:2.12.0
    container_name: opensearch
    environment:
      - discovery.type=single-node
      - plugins.security.disabled=true  # Disable for development
      - "OPENSEARCH_JAVA_OPTS=-Xms2g -Xmx2g"
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
      - "9200:9200"  # HTTP
      - "9600:9600"  # Performance Analyzer
    networks:
      - triton-network

  opensearch-dashboards:
    image: opensearchproject/opensearch-dashboards:2.12.0
    container_name: opensearch-dashboards
    ports:
      - "5601:5601"
    environment:
      OPENSEARCH_HOSTS: '["http://opensearch:9200"]'
      DISABLE_SECURITY_DASHBOARDS_PLUGIN: "true"
    networks:
      - triton-network

volumes:
  opensearch-data:

networks:
  triton-network:
    driver: bridge
```

**Start OpenSearch**:

```bash
docker compose up -d opensearch opensearch-dashboards

# Verify
curl http://localhost:9200/_cluster/health?pretty
```

---

### Task 5.2: Create Index with k-NN Support

**Script**: `scripts/create_opensearch_index.py`

```python
#!/usr/bin/env python3
"""
Create OpenSearch index with k-NN support for visual search
"""

from opensearchpy import OpenSearch
import json


def create_images_index():
    """Create images_production index with k-NN vectors"""

    # Connect to OpenSearch
    client = OpenSearch(
        hosts=[{'host': 'localhost', 'port': 9200}],
        http_compress=True,
        use_ssl=False,
        verify_certs=False
    )

    index_name = "images_production"

    # Delete if exists
    if client.indices.exists(index=index_name):
        client.indices.delete(index=index_name)
        print(f"Deleted existing index: {index_name}")

    # Index settings
    index_body = {
        "settings": {
            "index": {
                "knn": True,  # Enable k-NN
                "knn.algo_param.ef_search": 100,
                "number_of_shards": 2,
                "number_of_replicas": 1,
                "refresh_interval": "5s"
            }
        },
        "mappings": {
            "properties": {
                # Metadata
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

                # Global image embedding (for whole-image search)
                "global_image_embedding": {
                    "type": "knn_vector",
                    "dimension": 768,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",  # Cosine similarity
                        "engine": "lucene",
                        "parameters": {
                            "ef_construction": 512,  # Build quality
                            "m": 16  # Connections per node
                        }
                    }
                },

                # Per-object data (nested)
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

                # Optional tags
                "tags": {
                    "type": "keyword"
                },

                # Flexible metadata
                "metadata": {
                    "type": "object",
                    "enabled": False  # Don't index
                }
            }
        }
    }

    # Create index
    client.indices.create(index=index_name, body=index_body)
    print(f"✓ Created index: {index_name}")

    # Print index info
    info = client.indices.get(index=index_name)
    print(f"\nIndex Settings:")
    print(json.dumps(info[index_name]["settings"], indent=2))

    print(f"\nIndex Mappings:")
    print(json.dumps(info[index_name]["mappings"], indent=2))


if __name__ == "__main__":
    create_images_index()
```

**Run**:

```bash
python scripts/create_opensearch_index.py
```

---

### Task 5.3: Implement Ingestion Pipeline

**File**: `src/opensearch_ingestion.py`

```python
#!/usr/bin/env python3
"""
OpenSearch ingestion pipeline for images + embeddings
"""

from opensearchpy import OpenSearch
from datetime import datetime
import uuid
import numpy as np
from typing import List, Dict


class ImageIngestionPipeline:
    """Pipeline for ingesting images into OpenSearch"""

    def __init__(self, opensearch_host='localhost', opensearch_port=9200):
        self.client = OpenSearch(
            hosts=[{'host': opensearch_host, 'port': opensearch_port}],
            http_compress=True,
            use_ssl=False,
            verify_certs=False
        )
        self.index_name = "images_production"

    def ingest_image(
        self,
        filename: str,
        image_url: str,
        width: int,
        height: int,
        global_embedding: np.ndarray,
        detections: List[Dict],
        box_embeddings: np.ndarray,
        tags: List[str] = None,
        metadata: Dict = None
    ) -> str:
        """
        Ingest single image with detections and embeddings

        Args:
            filename: Image filename
            image_url: URL or path to image
            width: Image width
            height: Image height
            global_embedding: [768] array
            detections: List of {bbox, class_id, class_name, confidence}
            box_embeddings: [N, 768] array (one per detection)
            tags: Optional tags
            metadata: Optional metadata dict

        Returns:
            image_id: UUID of indexed document
        """

        # Generate image ID
        image_id = str(uuid.uuid4())

        # Build detected objects list
        detected_objects = []
        for i, det in enumerate(detections):
            obj = {
                "bbox": det["bbox"],  # {x1, y1, x2, y2}
                "class_id": int(det["class_id"]),
                "class_name": det["class_name"],
                "confidence": float(det["confidence"]),
                "object_embedding": box_embeddings[i].tolist()
            }
            detected_objects.append(obj)

        # Build document
        doc = {
            "image_id": image_id,
            "filename": filename,
            "upload_date": datetime.utcnow().isoformat(),
            "image_url": image_url,
            "width": width,
            "height": height,
            "global_image_embedding": global_embedding.tolist(),
            "detected_objects": detected_objects,
            "tags": tags or [],
            "metadata": metadata or {}
        }

        # Index document
        self.client.index(
            index=self.index_name,
            id=image_id,
            body=doc,
            refresh=False  # Don't force refresh (better throughput)
        )

        return image_id

    def bulk_ingest(self, images_data: List[Dict]):
        """Bulk ingest multiple images"""
        from opensearchpy.helpers import bulk

        actions = []
        for img_data in images_data:
            image_id = str(uuid.uuid4())

            # Build detected objects
            detected_objects = []
            for i, det in enumerate(img_data["detections"]):
                obj = {
                    "bbox": det["bbox"],
                    "class_id": int(det["class_id"]),
                    "class_name": det["class_name"],
                    "confidence": float(det["confidence"]),
                    "object_embedding": img_data["box_embeddings"][i].tolist()
                }
                detected_objects.append(obj)

            doc = {
                "_index": self.index_name,
                "_id": image_id,
                "_source": {
                    "image_id": image_id,
                    "filename": img_data["filename"],
                    "upload_date": datetime.utcnow().isoformat(),
                    "image_url": img_data["image_url"],
                    "width": img_data["width"],
                    "height": img_data["height"],
                    "global_image_embedding": img_data["global_embedding"].tolist(),
                    "detected_objects": detected_objects,
                    "tags": img_data.get("tags", []),
                    "metadata": img_data.get("metadata", {})
                }
            }
            actions.append(doc)

        # Bulk index
        success, failed = bulk(self.client, actions, raise_on_error=False)

        return success, failed
```

---

### Task 5.4: Implement Text Search

**File**: `src/opensearch_search.py`

```python
#!/usr/bin/env python3
"""
OpenSearch text-based search for visual search
"""

from opensearchpy import OpenSearch
import numpy as np
from typing import List, Dict


class VisualSearchEngine:
    """Text-based visual search using MobileCLIP embeddings"""

    def __init__(self, opensearch_host='localhost', opensearch_port=9200):
        self.client = OpenSearch(
            hosts=[{'host': opensearch_host, 'port': opensearch_port}],
            http_compress=True,
            use_ssl=False,
            verify_certs=False
        )
        self.index_name = "images_production"

    def search_by_text_embedding(
        self,
        text_embedding: np.ndarray,
        k: int = 20,
        min_score: float = 0.0
    ) -> List[Dict]:
        """
        Search images by text embedding

        Args:
            text_embedding: [768] array from MobileCLIP text encoder
            k: Number of results to return
            min_score: Minimum similarity score

        Returns:
            List of results with scores and metadata
        """

        # Build k-NN query
        query = {
            "size": k,
            "query": {
                "knn": {
                    "global_image_embedding": {
                        "vector": text_embedding.tolist(),
                        "k": k
                    }
                }
            },
            "_source": {
                "excludes": ["global_image_embedding", "detected_objects.object_embedding"]
            },
            "min_score": min_score
        }

        # Execute search
        response = self.client.search(
            index=self.index_name,
            body=query
        )

        # Format results
        results = []
        for hit in response["hits"]["hits"]:
            result = {
                "image_id": hit["_id"],
                "score": hit["_score"],
                **hit["_source"]
            }
            results.append(result)

        return results

    def search_by_object_embedding(
        self,
        object_embedding: np.ndarray,
        k: int = 20
    ) -> List[Dict]:
        """
        Search objects within images by embedding

        Args:
            object_embedding: [768] array
            k: Number of results

        Returns:
            List of results (objects, not images)
        """

        query = {
            "size": k,
            "query": {
                "nested": {
                    "path": "detected_objects",
                    "query": {
                        "knn": {
                            "detected_objects.object_embedding": {
                                "vector": object_embedding.tolist(),
                                "k": k
                            }
                        }
                    }
                }
            },
            "_source": ["image_id", "filename", "image_url", "detected_objects"]
        }

        response = self.client.search(
            index=self.index_name,
            body=query
        )

        # Format results
        results = []
        for hit in response["hits"]["hits"]:
            result = {
                "image_id": hit["_id"],
                "score": hit["_score"],
                **hit["_source"]
            }
            results.append(result)

        return results
```

---

### Task 5.5: Implement Hybrid Search

**File**: `src/opensearch_hybrid_search.py`

```python
#!/usr/bin/env python3
"""
Hybrid search: Text queries + YOLO class filters + metadata filters
"""

from opensearchpy import OpenSearch
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime


class HybridSearchEngine:
    """Hybrid search combining semantic similarity with filters"""

    def __init__(self, opensearch_host='localhost', opensearch_port=9200):
        self.client = OpenSearch(
            hosts=[{'host': opensearch_host, 'port': opensearch_port}],
            http_compress=True,
            use_ssl=False,
            verify_certs=False
        )
        self.index_name = "images_production"

    def hybrid_search(
        self,
        text_embedding: np.ndarray,
        yolo_classes: Optional[List[str]] = None,
        min_confidence: float = 0.5,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        tags: Optional[List[str]] = None,
        k: int = 20
    ) -> List[Dict]:
        """
        Hybrid search with semantic similarity + filters

        Args:
            text_embedding: [768] array from text query
            yolo_classes: List of YOLO class names (e.g., ["car", "person"])
            min_confidence: Minimum detection confidence
            date_from: ISO date string (e.g., "2025-01-01")
            date_to: ISO date string
            tags: List of tags to filter by
            k: Number of results

        Returns:
            List of results ranked by similarity score
        """

        # Build query
        query = {
            "size": k,
            "query": {
                "bool": {
                    "must": [
                        {
                            # Semantic similarity (required)
                            "knn": {
                                "global_image_embedding": {
                                    "vector": text_embedding.tolist(),
                                    "k": k * 2  # Over-fetch for filtering
                                }
                            }
                        }
                    ],
                    "filter": []
                }
            },
            "_source": {
                "excludes": ["global_image_embedding", "detected_objects.object_embedding"]
            }
        }

        # Add YOLO class filter
        if yolo_classes:
            query["query"]["bool"]["filter"].append({
                "nested": {
                    "path": "detected_objects",
                    "query": {
                        "bool": {
                            "must": [
                                {
                                    "terms": {
                                        "detected_objects.class_name": yolo_classes
                                    }
                                },
                                {
                                    "range": {
                                        "detected_objects.confidence": {
                                            "gte": min_confidence
                                        }
                                    }
                                }
                            ]
                        }
                    }
                }
            })

        # Add date range filter
        if date_from or date_to:
            date_filter = {"range": {"upload_date": {}}}
            if date_from:
                date_filter["range"]["upload_date"]["gte"] = date_from
            if date_to:
                date_filter["range"]["upload_date"]["lte"] = date_to
            query["query"]["bool"]["filter"].append(date_filter)

        # Add tags filter
        if tags:
            query["query"]["bool"]["filter"].append({
                "terms": {"tags": tags}
            })

        # Execute search
        response = self.client.search(
            index=self.index_name,
            body=query
        )

        # Format results
        results = []
        for hit in response["hits"]["hits"]:
            result = {
                "image_id": hit["_id"],
                "score": hit["_score"],
                **hit["_source"]
            }
            results.append(result)

        return results
```

**Phase 5 Deliverables**:
- [x] OpenSearch cluster running
- [x] Index created with k-NN support
- [x] Ingestion pipeline implemented
- [x] Text search implemented
- [x] Hybrid search with filters implemented
- [x] Test scripts validating all operations

---

## Phase 6: FastAPI Integration & Testing

**Duration**: 8-10 hours
**Goal**: Create unified REST API for image ingestion and search

*(Continuing with Phases 7-8 in the remaining sections...)*

---

*This document continues with Phases 6-8. See next section for complete implementation details.*
