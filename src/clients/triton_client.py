"""
Unified Triton Client for All Tracks (C, D, E).

Single client class that handles all Triton inference with proper connection pooling
and shared utilities. Sync-only for backpressure safety (async causes server deadlock
at 256+ clients).

Architecture:
- Shared gRPC connection pool via TritonClientManager
- Shared affine matrix calculation and JPEG parsing from utils/affine.py
- Shared detection formatting from utils/affine.py
- Clear method names per track: infer_track_c(), infer_track_d(), infer_track_e()

Performance:
- Sync client with thread pool: Proper backpressure, 200+ RPS
- Shared connection enables Triton dynamic batching (5-10x throughput)
"""

import io
import logging
from typing import Any

import cv2
import numpy as np
from PIL import Image
from tritonclient.grpc import InferInput, InferRequestedOutput
from ultralytics.data.augment import LetterBox

from src.clients.triton_pool import TritonClientManager
from src.utils.affine import (
    calculate_affine_matrix,
    format_detections_from_triton,
    get_jpeg_dimensions_fast,
)


logger = logging.getLogger(__name__)


class TritonClient:
    """
    Unified Triton client for all inference tracks (C, D, E).

    CRITICAL: Uses sync gRPC only (not async) to prevent server deadlock
    at high concurrency (256+ clients). FastAPI handles async via thread pool.

    Tracks:
    - Track C: End2End TRT + GPU NMS (CPU preprocessing)
    - Track D: DALI + TRT (100% GPU pipeline)
    - Track E: YOLO + MobileCLIP ensemble (visual search)
    """

    def __init__(self, triton_url: str = 'triton-api:8001'):
        """
        Initialize unified Triton client.

        Args:
            triton_url: Triton gRPC endpoint
        """
        self.triton_url = triton_url
        self.client = TritonClientManager.get_sync_client(triton_url)
        self.input_size = 640  # YOLO input size
        logger.info('Unified Triton client initialized (sync, backpressure-safe)')

    # =========================================================================
    # Track C: End2End TRT (CPU Preprocessing + GPU NMS)
    # =========================================================================
    def infer_track_c(self, image_array: np.ndarray, model_name: str) -> dict[str, Any]:
        """
        Track C inference: CPU preprocessing + TensorRT End2End (GPU NMS).

        Args:
            image_array: Preprocessed image (HWC, BGR, 0-255) from cv2.imdecode
            model_name: Triton model name (e.g., "yolov11_small_trt_end2end")

        Returns:
            Dict with num_dets, boxes, scores, classes, orig_shape, scale, padding
        """
        # Store original dimensions
        orig_h, orig_w = image_array.shape[:2]

        # Convert BGR (from cv2.imdecode) to RGB (YOLO trained on RGB)
        image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

        # Use Ultralytics LetterBox for exact preprocessing match
        letterbox = LetterBox(
            new_shape=(self.input_size, self.input_size), auto=False, scaleup=False
        )
        img_letterbox = letterbox(image=image_rgb)

        # Calculate transformation parameters (for inverse transform)
        scale = min(self.input_size / orig_h, self.input_size / orig_w)
        scale = min(scale, 1.0)  # scaleup=False

        # Calculate padding
        new_unpad_w = round(orig_w * scale)
        new_unpad_h = round(orig_h * scale)
        pad_w = (self.input_size - new_unpad_w) / 2.0
        pad_h = (self.input_size - new_unpad_h) / 2.0
        padding = (pad_w, pad_h)

        # Normalize to 0-1
        img_norm = img_letterbox.astype(np.float32) / 255.0

        # HWC to CHW
        img_chw = np.transpose(img_norm, (2, 0, 1))

        # Add batch dimension
        input_data = np.expand_dims(img_chw, axis=0)

        # Create Triton inputs
        inputs = [InferInput('images', input_data.shape, 'FP32')]
        inputs[0].set_data_from_numpy(input_data)

        # Create Triton outputs
        outputs = [
            InferRequestedOutput('num_dets'),
            InferRequestedOutput('det_boxes'),
            InferRequestedOutput('det_scores'),
            InferRequestedOutput('det_classes'),
        ]

        # Run inference
        response = self.client.infer(model_name=model_name, inputs=inputs, outputs=outputs)

        # Parse outputs
        num_dets = int(response.as_numpy('num_dets')[0][0])
        boxes = response.as_numpy('det_boxes')[0][:num_dets]
        scores = response.as_numpy('det_scores')[0][:num_dets]
        classes = response.as_numpy('det_classes')[0][:num_dets]

        return {
            'num_dets': num_dets,
            'boxes': boxes,
            'scores': scores,
            'classes': classes,
            'orig_shape': (orig_h, orig_w),
            'scale': scale,
            'padding': padding,
        }

    def infer_track_c_batch(self, images: list, model_name: str) -> list:
        """
        Track C batch inference: CPU preprocessing + TensorRT End2End (GPU NMS).

        Args:
            images: List of images (HWC, BGR, 0-255) from cv2.imdecode
            model_name: Triton model name (e.g., "yolov11_small_trt_end2end")

        Returns:
            List of dicts with num_dets, boxes, scores, classes, orig_shape, scale, padding
        """
        letterbox = LetterBox(
            new_shape=(self.input_size, self.input_size), auto=False, scaleup=False
        )

        # Store original dimensions and preprocess
        orig_shapes = []
        scales = []
        paddings = []
        preprocessed = []

        for img in images:
            orig_h, orig_w = img.shape[:2]
            orig_shapes.append((orig_h, orig_w))

            # Convert BGR to RGB (YOLO trained on RGB)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Letterbox transform
            img_letterbox = letterbox(image=img_rgb)

            # Calculate scale and padding
            scale = min(self.input_size / orig_h, self.input_size / orig_w)
            scale = min(scale, 1.0)
            scales.append(scale)

            new_unpad_w = round(orig_w * scale)
            new_unpad_h = round(orig_h * scale)
            pad_w = (self.input_size - new_unpad_w) / 2.0
            pad_h = (self.input_size - new_unpad_h) / 2.0
            paddings.append((pad_w, pad_h))

            # Normalize and transpose
            img_norm = img_letterbox.astype(np.float32) / 255.0
            img_chw = np.transpose(img_norm, (2, 0, 1))
            preprocessed.append(img_chw)

        # Stack into batch
        input_data = np.stack(preprocessed, axis=0)
        batch_size = input_data.shape[0]

        # Create Triton inputs/outputs
        inputs = [InferInput('images', input_data.shape, 'FP32')]
        inputs[0].set_data_from_numpy(input_data)

        outputs = [
            InferRequestedOutput('num_dets'),
            InferRequestedOutput('det_boxes'),
            InferRequestedOutput('det_scores'),
            InferRequestedOutput('det_classes'),
        ]

        # Run inference
        response = self.client.infer(model_name=model_name, inputs=inputs, outputs=outputs)

        # Parse batch outputs
        num_dets_batch = response.as_numpy('num_dets')
        boxes_batch = response.as_numpy('det_boxes')
        scores_batch = response.as_numpy('det_scores')
        classes_batch = response.as_numpy('det_classes')

        results = []
        for i in range(batch_size):
            num_dets = int(num_dets_batch[i][0])
            results.append(
                {
                    'num_dets': num_dets,
                    'boxes': boxes_batch[i][:num_dets],
                    'scores': scores_batch[i][:num_dets],
                    'classes': classes_batch[i][:num_dets],
                    'orig_shape': orig_shapes[i],
                    'scale': scales[i],
                    'padding': paddings[i],
                }
            )

        return results

    # =========================================================================
    # Track D: DALI + TRT (100% GPU Pipeline)
    # =========================================================================
    def infer_track_d(
        self, image_bytes: bytes, model_name: str, auto_affine: bool = False
    ) -> dict[str, Any]:
        """
        Track D inference: Full GPU pipeline with DALI preprocessing.

        GPU pipeline:
        1. nvJPEG decode (GPU)
        2. warp_affine letterbox (GPU)
        3. normalize + CHW transpose (GPU)
        4. TensorRT inference + NMS (GPU)

        CPU overhead: Only JPEG header parse (~0.1ms) + affine calc (~0.001ms)

        Args:
            image_bytes: Raw JPEG/PNG bytes
            model_name: Triton ensemble name (e.g., "yolov11_small_gpu_e2e")
            auto_affine: If True, use auto-affine (100% GPU, no CPU affine calc)

        Returns:
            Dict with num_dets, boxes, scores, classes, orig_shape, scale, padding
        """
        # Fast JPEG header parse for dimensions (~0.1ms)
        orig_w, orig_h = get_jpeg_dimensions_fast(image_bytes)

        # Prepare encoded image input
        input_data = np.frombuffer(image_bytes, dtype=np.uint8)
        input_data = np.expand_dims(input_data, axis=0)

        inputs = [InferInput('encoded_images', input_data.shape, 'UINT8')]
        inputs[0].set_data_from_numpy(input_data)

        if not auto_affine:
            # CPU affine calculation (cached, ~0.001ms for cache hit)
            affine_matrix, scale, padding = calculate_affine_matrix(orig_w, orig_h, self.input_size)
            input_affine = InferInput('affine_matrices', [1, 2, 3], 'FP32')
            input_affine.set_data_from_numpy(np.expand_dims(affine_matrix, axis=0))
            inputs.append(input_affine)
        else:
            # Auto-affine: GPU calculates affine matrix
            scale = min(self.input_size / orig_w, self.input_size / orig_h)
            padding = (0.0, 0.0)

        # Create Triton outputs
        outputs = [
            InferRequestedOutput('num_dets'),
            InferRequestedOutput('det_boxes'),
            InferRequestedOutput('det_scores'),
            InferRequestedOutput('det_classes'),
        ]

        # Run inference
        response = self.client.infer(model_name=model_name, inputs=inputs, outputs=outputs)

        # Parse outputs
        num_dets = int(response.as_numpy('num_dets')[0][0])
        boxes = response.as_numpy('det_boxes')[0][:num_dets]
        scores = response.as_numpy('det_scores')[0][:num_dets]
        classes = response.as_numpy('det_classes')[0][:num_dets]

        return {
            'num_dets': num_dets,
            'boxes': boxes,
            'scores': scores,
            'classes': classes,
            'orig_shape': (orig_h, orig_w),
            'scale': scale,
            'padding': padding,
        }

    # =========================================================================
    # Track E: Visual Search Ensemble (YOLO + MobileCLIP)
    # =========================================================================
    def infer_track_e(self, image_bytes: bytes, full_pipeline: bool = False) -> dict[str, Any]:
        """
        Track E ensemble inference: YOLO + MobileCLIP.

        Ensembles:
        - yolo_clip_ensemble: YOLO detections + global image embedding
        - yolo_mobileclip_ensemble: + per-box embeddings (full_pipeline=True)

        Pipeline (all in single Triton call):
        1. GPU JPEG decode (nvJPEG)
        2. GPU letterbox for YOLO (warp_affine)
        3. GPU resize/crop for CLIP
        4. GPU inference + NMS (TensorRT End2End)
        5. GPU image encoding (MobileCLIP)
        6. [Full only] Per-box embeddings (ROI align + MobileCLIP)

        Args:
            image_bytes: Raw JPEG/PNG bytes
            full_pipeline: If True, include per-box embeddings

        Returns:
            Dict with detections, embeddings, and transformation params
        """
        # Fast JPEG header parse for dimensions (~0.1ms)
        orig_w, orig_h = get_jpeg_dimensions_fast(image_bytes)

        # Get cached affine matrix (~0.001ms for cache hit)
        affine_matrix, scale, padding = calculate_affine_matrix(orig_w, orig_h, self.input_size)

        # Prepare inputs
        input_data = np.frombuffer(image_bytes, dtype=np.uint8)
        input_data = np.expand_dims(input_data, axis=0)

        inputs = [
            InferInput('encoded_images', input_data.shape, 'UINT8'),
            InferInput('affine_matrices', [1, 2, 3], 'FP32'),
        ]
        inputs[0].set_data_from_numpy(input_data)
        inputs[1].set_data_from_numpy(np.expand_dims(affine_matrix, axis=0))

        # Choose ensemble and outputs
        if full_pipeline:
            ensemble_name = 'yolo_mobileclip_ensemble'
            outputs = [
                InferRequestedOutput('num_dets'),
                InferRequestedOutput('det_boxes'),
                InferRequestedOutput('det_scores'),
                InferRequestedOutput('det_classes'),
                InferRequestedOutput('global_embeddings'),
                InferRequestedOutput('box_embeddings'),
                InferRequestedOutput('normalized_boxes'),
            ]
        else:
            ensemble_name = 'yolo_clip_ensemble'
            outputs = [
                InferRequestedOutput('num_dets'),
                InferRequestedOutput('det_boxes'),
                InferRequestedOutput('det_scores'),
                InferRequestedOutput('det_classes'),
                InferRequestedOutput('global_embeddings'),
            ]

        # Sync inference - proper backpressure handling
        response = self.client.infer(model_name=ensemble_name, inputs=inputs, outputs=outputs)

        # Parse outputs
        num_dets = int(response.as_numpy('num_dets')[0][0])
        boxes = response.as_numpy('det_boxes')[0][:num_dets]
        scores = response.as_numpy('det_scores')[0][:num_dets]
        classes = response.as_numpy('det_classes')[0][:num_dets]
        image_embedding = response.as_numpy('global_embeddings')[0]

        result = {
            'num_dets': num_dets,
            'boxes': boxes,
            'scores': scores,
            'classes': classes,
            'image_embedding': image_embedding,
            'orig_shape': (orig_h, orig_w),
            'scale': scale,
            'padding': padding,
        }

        if full_pipeline and num_dets > 0:
            # Reshape to ensure 2D arrays (Triton may return flattened data)
            box_emb = response.as_numpy('box_embeddings')[0]
            box_emb = box_emb.reshape(-1, 512)[:num_dets]  # [num_dets, 512]
            result['box_embeddings'] = box_emb

            norm_boxes = response.as_numpy('normalized_boxes')[0]
            norm_boxes = norm_boxes.reshape(-1, 4)[:num_dets]  # [num_dets, 4]
            result['normalized_boxes'] = norm_boxes
        elif full_pipeline:
            result['box_embeddings'] = np.array([])
            result['normalized_boxes'] = np.array([])

        return result

    # =========================================================================
    # Individual Model Inference (Track E Components)
    # =========================================================================
    def encode_image(self, image_bytes: bytes) -> np.ndarray:
        """
        Encode image to 512-dim embedding via MobileCLIP.

        Args:
            image_bytes: JPEG/PNG bytes

        Returns:
            512-dim L2-normalized embedding
        """
        # Preprocess for MobileCLIP (256x256, normalized)
        img_array = self._preprocess_for_mobileclip(image_bytes)

        # Prepare input
        input_tensor = InferInput('images', [1, 3, 256, 256], 'FP32')
        input_tensor.set_data_from_numpy(img_array)

        # Output
        output = InferRequestedOutput('image_embeddings')

        # Sync inference
        response = self.client.infer(
            model_name='mobileclip2_s2_image_encoder', inputs=[input_tensor], outputs=[output]
        )

        return response.as_numpy('image_embeddings')[0]

    def encode_text(self, tokens: np.ndarray) -> np.ndarray:
        """
        Encode tokenized text to 512-dim embedding via MobileCLIP.

        Args:
            tokens: Tokenized text [1, 77] INT64

        Returns:
            512-dim L2-normalized embedding
        """
        # Prepare input
        input_tensor = InferInput('text_tokens', [1, 77], 'INT64')
        input_tensor.set_data_from_numpy(tokens.astype(np.int64))

        # Output
        output = InferRequestedOutput('text_embeddings')

        # Sync inference
        response = self.client.infer(
            model_name='mobileclip2_s2_text_encoder', inputs=[input_tensor], outputs=[output]
        )

        return response.as_numpy('text_embeddings')[0]

    # =========================================================================
    # Preprocessing Utilities
    # =========================================================================
    def _preprocess_for_mobileclip(self, image_bytes: bytes) -> np.ndarray:
        """
        Preprocess image for MobileCLIP inference.

        Args:
            image_bytes: JPEG/PNG bytes

        Returns:
            Preprocessed image array [1, 3, 256, 256] FP32
        """
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # Resize shortest edge to 256, then center crop (matches OpenCLIP)
        width, height = img.size
        scale = 256 / min(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Center crop to 256x256
        left = (new_width - 256) // 2
        top = (new_height - 256) // 2
        img = img.crop((left, top, left + 256, top + 256))

        # Convert to numpy and normalize
        img_array = np.array(img).astype(np.float32)
        img_array = img_array / 255.0  # [0, 1]

        # Transpose HWC -> CHW
        img_array = np.transpose(img_array, (2, 0, 1))

        return img_array[np.newaxis, ...]  # Add batch dimension

    # =========================================================================
    # Formatting Utilities (Shared across all tracks)
    # =========================================================================
    @staticmethod
    def format_detections(result: dict[str, Any]) -> list:
        """
        Format detections with coordinates normalized to original image dimensions.

        Uses shared utility from affine.py to apply inverse letterbox transformation.
        This matches Track A (PyTorch boxes.xyxyn) coordinate output.

        Args:
            result: Inference result with boxes, scores, classes,
                   and optionally orig_shape, scale, padding for inverse transform

        Returns:
            List of detection dicts with x1, y1, x2, y2 normalized to original image
        """
        return format_detections_from_triton(result, input_size=640)


# =============================================================================
# Singleton Instance (for shared client across requests)
# =============================================================================
# Global singleton for shared Triton client instance
# Justification: Required for connection pooling and efficient resource usage
# - Single gRPC connection shared across all requests enables Triton's dynamic batching
# - Thread-safe sync client prevents deadlock at high concurrency (256+ clients)
# - Alternative (per-request clients) would exhaust connections and disable batching
_client_instance: TritonClient | None = None


def get_triton_client(triton_url: str = 'triton-api:8001') -> TritonClient:
    """
    Get singleton Triton client instance.

    Uses shared sync gRPC connection for proper backpressure handling.
    Thread-safe because gRPC sync client handles concurrent requests internally.

    Returns:
        Shared TritonClient instance
    """
    global _client_instance  # noqa: PLW0603 - Singleton pattern documented above
    if _client_instance is None:
        _client_instance = TritonClient(triton_url)
    return _client_instance
