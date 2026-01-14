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
from src.utils.retry import retry_sync


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

    def __init__(
        self,
        triton_url: str = 'triton-api:8001',
        max_retries: int = 3,
        retry_base_delay: float = 0.1,
        retry_max_delay: float = 5.0,
    ):
        """
        Initialize unified Triton client with retry support.

        Args:
            triton_url: Triton gRPC endpoint
            max_retries: Maximum retry attempts for failed requests
            retry_base_delay: Initial retry delay in seconds
            retry_max_delay: Maximum retry delay in seconds
        """
        self.triton_url = triton_url
        self.client = TritonClientManager.get_sync_client(triton_url)
        self.input_size = 640  # YOLO input size
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.retry_max_delay = retry_max_delay
        logger.info(f'Unified Triton client initialized (sync, retries={max_retries})')

    def _infer_with_retry(self, model_name: str, inputs: list, outputs: list):
        """
        Execute Triton inference with automatic retry on transient failures.

        Retries on: queue full, resource exhausted, timeout, unavailable.
        Does NOT retry on: invalid input, model not found, etc.

        Args:
            model_name: Triton model name
            inputs: List of InferInput objects
            outputs: List of InferRequestedOutput objects

        Returns:
            InferResult from Triton

        Raises:
            RetryExhaustedError: If all retries exhausted
            Exception: For non-retryable errors
        """
        return retry_sync(
            self.client.infer,
            model_name=model_name,
            inputs=inputs,
            outputs=outputs,
            max_retries=self.max_retries,
            base_delay=self.retry_base_delay,
            max_delay=self.retry_max_delay,
        )

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

        # Run inference with retry
        response = self._infer_with_retry(model_name, inputs, outputs)

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

        # Run inference with retry
        response = self._infer_with_retry(model_name, inputs, outputs)

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

        # Run inference with retry
        response = self._infer_with_retry(model_name, inputs, outputs)

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

        # Sync inference with retry - proper backpressure handling
        response = self._infer_with_retry(ensemble_name, inputs, outputs)

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

    def infer_track_e_batch(
        self, images_bytes: list[bytes], max_workers: int = 32
    ) -> list[dict[str, Any]]:
        """
        Track E batch inference: Process multiple images in parallel.

        Uses ThreadPoolExecutor to send parallel requests to Triton,
        which batches them via dynamic batching for optimal GPU utilization.

        For large photo libraries (50K+ images), batch sizes of 16-64
        significantly improve throughput by:
        - Reducing HTTP overhead
        - Ensuring full DALI/TRT batch utilization
        - Maximizing GPU parallelism

        Args:
            images_bytes: List of raw JPEG/PNG bytes (up to 64 images)
            max_workers: Max parallel threads (default 32)

        Returns:
            List of result dicts with detections and embeddings
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        batch_size = len(images_bytes)
        if batch_size == 0:
            return []

        # Limit batch size to max_batch_size
        if batch_size > 64:
            logger.warning(f'Batch size {batch_size} exceeds max 64, truncating')
            images_bytes = images_bytes[:64]
            batch_size = 64

        results = [None] * batch_size

        def process_single(idx: int, img_bytes: bytes) -> tuple[int, dict]:
            """Process a single image and return (index, result)."""
            result = self.infer_track_e(img_bytes, full_pipeline=False)
            return idx, result

        # Process in parallel
        workers = min(max_workers, batch_size)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process_single, i, img): i for i, img in enumerate(images_bytes)
            }

            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result

        return results

    def infer_track_e_batch_full(
        self,
        images_bytes: list[bytes],
        max_workers: int = 64,
    ) -> list[dict[str, Any]]:
        """
        Track E batch inference with full pipeline (YOLO + CLIP + per-box embeddings).

        Optimized for large photo library ingestion:
        - Submits ALL images to Triton simultaneously
        - Triton's dynamic batcher groups them (16-48 avg batch size)
        - Returns full pipeline results including box embeddings

        Performance: 3-5x faster than sequential /ingest calls.
        Target: 300+ RPS with batch sizes of 32-64.

        Args:
            images_bytes: List of raw JPEG/PNG bytes (max 64 images)
            max_workers: Max parallel threads for Triton submission

        Returns:
            List of result dicts with detections, embeddings, and box data
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        batch_size = len(images_bytes)
        if batch_size == 0:
            return []

        # Limit batch size
        if batch_size > 64:
            logger.warning(f'Batch size {batch_size} exceeds max 64, truncating')
            images_bytes = images_bytes[:64]
            batch_size = 64

        results = [None] * batch_size

        def process_single(idx: int, img_bytes: bytes) -> tuple[int, dict]:
            """Process a single image with full pipeline."""
            try:
                result = self.infer_track_e(img_bytes, full_pipeline=True)
                return idx, result
            except Exception as e:
                logger.error(f'Batch inference failed for image {idx}: {e}')
                return idx, {'error': str(e), 'num_dets': 0}

        # Submit ALL images in parallel - Triton will batch them
        workers = min(max_workers, batch_size)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process_single, i, img): i for i, img in enumerate(images_bytes)
            }

            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result

        return results

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

        # Sync inference with retry
        response = self._infer_with_retry('mobileclip2_s2_image_encoder', [input_tensor], [output])

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

        # Sync inference with retry
        response = self._infer_with_retry('mobileclip2_s2_text_encoder', [input_tensor], [output])

        return response.as_numpy('text_embeddings')[0]

    # =========================================================================
    # Track F: CPU Preprocessing + Direct Model Calls (No DALI)
    # =========================================================================
    def infer_track_f(self, image_bytes: bytes) -> dict[str, Any]:
        """
        Track F: CPU preprocessing + direct YOLO TRT + MobileCLIP TRT calls.

        Unlike Track E which uses DALI ensemble, Track F does:
        1. CPU decode (PIL/cv2)
        2. CPU letterbox for YOLO (640x640)
        3. CPU resize/crop for CLIP (256x256)
        4. Direct TRT inference for YOLO and CLIP

        This allows comparison of CPU vs GPU preprocessing overhead,
        and enables more TRT instances since DALI doesn't reserve VRAM.

        Args:
            image_bytes: Raw JPEG/PNG bytes

        Returns:
            Dict with detections and image embedding
        """
        # CPU decode image
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        orig_w, orig_h = img.size
        img_array = np.array(img)  # HWC, RGB, uint8

        # ---- YOLO Preprocessing (CPU letterbox) ----
        yolo_input, scale, padding = self._preprocess_yolo_cpu(img_array)

        # ---- CLIP Preprocessing (CPU resize/crop) ----
        clip_input = self._preprocess_clip_cpu(img_array)

        # ---- Run YOLO TRT Inference ----
        yolo_inputs = [InferInput('images', yolo_input.shape, 'FP32')]
        yolo_inputs[0].set_data_from_numpy(yolo_input)
        yolo_outputs = [
            InferRequestedOutput('num_dets'),
            InferRequestedOutput('det_boxes'),
            InferRequestedOutput('det_scores'),
            InferRequestedOutput('det_classes'),
        ]
        yolo_response = self._infer_with_retry(
            'yolov11_small_trt_end2end', yolo_inputs, yolo_outputs
        )

        # ---- Run CLIP TRT Inference ----
        clip_inputs = [InferInput('images', clip_input.shape, 'FP32')]
        clip_inputs[0].set_data_from_numpy(clip_input)
        clip_outputs = [InferRequestedOutput('image_embeddings')]
        clip_response = self._infer_with_retry(
            'mobileclip2_s2_image_encoder', clip_inputs, clip_outputs
        )

        # Parse YOLO outputs
        num_dets = int(yolo_response.as_numpy('num_dets')[0][0])
        boxes = yolo_response.as_numpy('det_boxes')[0][:num_dets]
        scores = yolo_response.as_numpy('det_scores')[0][:num_dets]
        classes = yolo_response.as_numpy('det_classes')[0][:num_dets]

        # Parse CLIP output
        image_embedding = clip_response.as_numpy('image_embeddings')[0]

        return {
            'num_dets': num_dets,
            'boxes': boxes,
            'scores': scores,
            'classes': classes,
            'image_embedding': image_embedding,
            'orig_shape': (orig_h, orig_w),
            'scale': scale,
            'padding': padding,
        }

    def _preprocess_yolo_cpu(self, img_array: np.ndarray) -> tuple[np.ndarray, float, tuple]:
        """
        CPU letterbox preprocessing for YOLO (matches DALI output exactly).

        Args:
            img_array: HWC, RGB, uint8 numpy array

        Returns:
            Tuple of:
            - preprocessed: [1, 3, 640, 640] FP32 normalized
            - scale: float for inverse transform
            - padding: (pad_x, pad_y) tuple
        """
        orig_h, orig_w = img_array.shape[:2]
        target_size = self.input_size  # 640

        # Calculate scale (don't upscale)
        scale = min(target_size / orig_h, target_size / orig_w)
        scale = min(scale, 1.0)

        # New dimensions after scaling
        new_w = round(orig_w * scale)
        new_h = round(orig_h * scale)

        # Resize using cv2 (faster than PIL for numpy arrays)
        if scale < 1.0:
            resized = cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            resized = img_array

        # Calculate padding
        pad_w = (target_size - new_w) / 2.0
        pad_h = (target_size - new_h) / 2.0
        top, bottom = round(pad_h - 0.1), round(pad_h + 0.1)
        left, right = round(pad_w - 0.1), round(pad_w + 0.1)

        # Add padding (gray = 114)
        letterboxed = cv2.copyMakeBorder(
            resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )

        # Ensure exact size (rounding may cause 1px difference)
        if letterboxed.shape[:2] != (target_size, target_size):
            letterboxed = cv2.resize(letterboxed, (target_size, target_size))

        # Normalize to [0, 1]
        normalized = letterboxed.astype(np.float32) / 255.0

        # HWC -> CHW
        transposed = np.transpose(normalized, (2, 0, 1))

        # Add batch dimension
        batched = np.expand_dims(transposed, axis=0)

        return batched, scale, (pad_w, pad_h)

    def _preprocess_clip_cpu(self, img_array: np.ndarray) -> np.ndarray:
        """
        CPU preprocessing for MobileCLIP (256x256, center crop).

        Args:
            img_array: HWC, RGB, uint8 numpy array

        Returns:
            Preprocessed array [1, 3, 256, 256] FP32
        """
        orig_h, orig_w = img_array.shape[:2]
        target_size = 256

        # Resize shortest edge to 256
        scale = target_size / min(orig_h, orig_w)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        resized = cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Center crop to 256x256
        start_x = (new_w - target_size) // 2
        start_y = (new_h - target_size) // 2
        cropped = resized[start_y : start_y + target_size, start_x : start_x + target_size]

        # Normalize to [0, 1]
        normalized = cropped.astype(np.float32) / 255.0

        # HWC -> CHW
        transposed = np.transpose(normalized, (2, 0, 1))

        # Add batch dimension
        return np.expand_dims(transposed, axis=0)

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
    # Track E Faces: YOLO + Face + CLIP Unified Ensemble
    # =========================================================================
    def infer_faces_full(self, image_bytes: bytes) -> dict[str, Any]:
        """
        Full face recognition pipeline: YOLO + SCRFD + MobileCLIP + ArcFace.

        All processing happens in Triton - single request, no round-trips:
        1. GPU JPEG decode (nvJPEG via DALI)
        2. Quad-branch preprocessing (YOLO 640, CLIP 256, SCRFD 640, HD original)
        3. YOLO object detection (parallel with 4, 5)
        4. MobileCLIP global embedding (parallel with 3, 5)
        5. SCRFD face detection + NMS (parallel with 3, 4)
        6. Per-box MobileCLIP embeddings (depends on 3)
        7. Per-face ArcFace embeddings (depends on 5)

        Args:
            image_bytes: Raw JPEG/PNG bytes

        Returns:
            Dict with:
                - YOLO detections (num_dets, boxes, scores, classes)
                - Global MobileCLIP embedding (512-dim)
                - Per-box MobileCLIP embeddings (300 x 512-dim)
                - Face detections (num_faces, face_boxes, face_landmarks, face_scores)
                - Face ArcFace embeddings (128 x 512-dim)
                - Face quality scores
        """
        # Fast JPEG header parse for dimensions (~0.1ms)
        orig_w, orig_h = get_jpeg_dimensions_fast(image_bytes)

        # Get cached affine matrix (~0.001ms for cache hit)
        affine_matrix, scale, padding = calculate_affine_matrix(orig_w, orig_h, self.input_size)

        # Prepare inputs
        input_data = np.frombuffer(image_bytes, dtype=np.uint8)
        input_data = np.expand_dims(input_data, axis=0)

        orig_shape = np.array([orig_h, orig_w], dtype=np.int32)
        orig_shape = np.expand_dims(orig_shape, axis=0)

        inputs = [
            InferInput('encoded_images', input_data.shape, 'UINT8'),
            InferInput('affine_matrices', [1, 2, 3], 'FP32'),
            InferInput('orig_shape', [1, 2], 'INT32'),
        ]
        inputs[0].set_data_from_numpy(input_data)
        inputs[1].set_data_from_numpy(np.expand_dims(affine_matrix, axis=0))
        inputs[2].set_data_from_numpy(orig_shape)

        # Request all outputs from unified ensemble
        outputs = [
            # YOLO detections
            InferRequestedOutput('num_dets'),
            InferRequestedOutput('det_boxes'),
            InferRequestedOutput('det_scores'),
            InferRequestedOutput('det_classes'),
            # MobileCLIP global embedding
            InferRequestedOutput('global_embeddings'),
            # Face detections and embeddings
            InferRequestedOutput('num_faces'),
            InferRequestedOutput('face_boxes'),
            InferRequestedOutput('face_landmarks'),
            InferRequestedOutput('face_scores'),
            InferRequestedOutput('face_embeddings'),
            InferRequestedOutput('face_quality'),
        ]

        # Sync inference with retry
        response = self._infer_with_retry('yolo_face_clip_ensemble', inputs, outputs)

        # Parse YOLO outputs
        num_dets = int(response.as_numpy('num_dets')[0][0])
        boxes = response.as_numpy('det_boxes')[0][:num_dets]
        scores = response.as_numpy('det_scores')[0][:num_dets]
        classes = response.as_numpy('det_classes')[0][:num_dets]
        image_embedding = response.as_numpy('global_embeddings')[0]

        # Parse face outputs
        num_faces_arr = response.as_numpy('num_faces')
        num_faces = int(num_faces_arr.flatten()[0])
        face_boxes_raw = response.as_numpy('face_boxes')
        face_landmarks_raw = response.as_numpy('face_landmarks')
        face_scores_raw = response.as_numpy('face_scores')

        # Handle batch dimension
        if face_boxes_raw.ndim == 3:
            face_boxes = face_boxes_raw[0][:num_faces]
            face_landmarks = face_landmarks_raw[0][:num_faces]
            face_scores_arr = face_scores_raw[0][:num_faces]
        else:
            face_boxes = face_boxes_raw[:num_faces]
            face_landmarks = face_landmarks_raw[:num_faces]
            face_scores_arr = face_scores_raw[:num_faces]

        # Parse face embeddings
        if num_faces > 0:
            face_emb_raw = response.as_numpy('face_embeddings')
            face_quality_raw = response.as_numpy('face_quality')

            # Handle batch dimension
            if face_emb_raw.ndim == 3:
                face_emb = face_emb_raw[0].reshape(-1, 512)[:num_faces]
                face_quality_arr = face_quality_raw[0][:num_faces]
            else:
                face_emb = face_emb_raw.reshape(-1, 512)[:num_faces]
                face_quality_arr = face_quality_raw[:num_faces]
        else:
            face_emb = np.array([])
            face_quality_arr = np.array([])

        return {
            # YOLO detections
            'num_dets': num_dets,
            'boxes': boxes,
            'scores': scores,
            'classes': classes,
            # MobileCLIP global embedding
            'image_embedding': image_embedding,
            # Face detections and embeddings
            'num_faces': num_faces,
            'face_boxes': face_boxes,
            'face_landmarks': face_landmarks,
            'face_scores': face_scores_arr,
            'face_embeddings': face_emb,
            'face_quality': face_quality_arr,
            # Transform params
            'orig_shape': (orig_h, orig_w),
            'scale': scale,
            'padding': padding,
        }

    def infer_faces_only(self, image_bytes: bytes) -> dict[str, Any]:
        """
        Face detection and recognition only (no YOLO).

        Simpler pipeline for face-focused applications:
        1. GPU JPEG decode + preprocess for SCRFD (640x640)
        2. SCRFD face detection + NMS
        3. Face alignment + ArcFace embedding extraction

        Args:
            image_bytes: Raw JPEG/PNG bytes

        Returns:
            Dict with face detections and ArcFace embeddings
        """
        # Use the full pipeline and extract face parts only
        # This ensures we use the optimized DALI quad-branch
        result = self.infer_faces_full(image_bytes)

        return {
            'num_faces': result['num_faces'],
            'face_boxes': result['face_boxes'],
            'face_landmarks': result['face_landmarks'],
            'face_scores': result['face_scores'],
            'face_embeddings': result['face_embeddings'],
            'face_quality': result['face_quality'],
            'orig_shape': result['orig_shape'],
        }

    # =========================================================================
    # Formatting Utilities (Shared across all tracks)
    # =========================================================================
    # Track E Unified: YOLO + MobileCLIP + Person-only Face Detection
    # =========================================================================
    def infer_unified(self, image_bytes: bytes) -> dict[str, Any]:
        """
        Unified pipeline: YOLO + MobileCLIP + person-only face detection.

        More efficient than full-image face detection - runs SCRFD only on
        person bounding box crops. Combines all embeddings in one request.

        Pipeline:
        1. GPU JPEG decode (nvJPEG via DALI)
        2. Triple preprocessing (YOLO 640, CLIP 256, HD original)
        3. YOLO object detection (parallel with 4)
        4. MobileCLIP global embedding (parallel with 3)
        5. Unified extraction:
           - MobileCLIP per-box embeddings (all boxes)
           - SCRFD face detection (person boxes only)
           - ArcFace face embeddings

        Args:
            image_bytes: Raw JPEG/PNG bytes

        Returns:
            Dict with:
                - YOLO detections (num_dets, boxes, scores, classes)
                - Global MobileCLIP embedding (512-dim)
                - Per-box MobileCLIP embeddings (300 x 512-dim)
                - Face detections (num_faces, face_boxes, face_landmarks, face_scores)
                - Face ArcFace embeddings (64 x 512-dim)
                - Face person index (which person box each face belongs to)
        """
        # Fast JPEG header parse for dimensions
        orig_w, orig_h = get_jpeg_dimensions_fast(image_bytes)

        # Get cached affine matrix
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

        # Request all outputs from unified ensemble
        outputs = [
            # YOLO detections
            InferRequestedOutput('num_dets'),
            InferRequestedOutput('det_boxes'),
            InferRequestedOutput('det_scores'),
            InferRequestedOutput('det_classes'),
            # MobileCLIP embeddings
            InferRequestedOutput('global_embeddings'),
            InferRequestedOutput('box_embeddings'),
            InferRequestedOutput('normalized_boxes'),
            # Face detections and embeddings
            InferRequestedOutput('num_faces'),
            InferRequestedOutput('face_embeddings'),
            InferRequestedOutput('face_boxes'),
            InferRequestedOutput('face_landmarks'),
            InferRequestedOutput('face_scores'),
            InferRequestedOutput('face_person_idx'),
        ]

        # Sync inference with retry
        response = self._infer_with_retry('yolo_unified_ensemble', inputs, outputs)

        # Parse YOLO outputs
        num_dets = int(response.as_numpy('num_dets')[0][0])
        boxes = response.as_numpy('det_boxes')[0][:num_dets]
        scores = response.as_numpy('det_scores')[0][:num_dets]
        classes = response.as_numpy('det_classes')[0][:num_dets]

        # Parse MobileCLIP outputs
        global_embedding = response.as_numpy('global_embeddings')[0]
        box_embeddings_raw = response.as_numpy('box_embeddings')
        normalized_boxes_raw = response.as_numpy('normalized_boxes')

        # Handle batch dimension
        if box_embeddings_raw.ndim == 3:
            box_embeddings = box_embeddings_raw[0][:num_dets]
            normalized_boxes = normalized_boxes_raw[0][:num_dets]
        else:
            box_embeddings = box_embeddings_raw[:num_dets]
            normalized_boxes = normalized_boxes_raw[:num_dets]

        # Parse face outputs
        num_faces_arr = response.as_numpy('num_faces')
        num_faces = int(num_faces_arr.flatten()[0])

        face_boxes_raw = response.as_numpy('face_boxes')
        face_landmarks_raw = response.as_numpy('face_landmarks')
        face_scores_raw = response.as_numpy('face_scores')
        face_person_idx_raw = response.as_numpy('face_person_idx')
        face_emb_raw = response.as_numpy('face_embeddings')

        # Handle batch dimension for faces
        if face_boxes_raw.ndim == 3:
            face_boxes = face_boxes_raw[0][:num_faces]
            face_landmarks = face_landmarks_raw[0][:num_faces]
            face_scores_arr = face_scores_raw[0][:num_faces]
            face_person_idx = face_person_idx_raw[0][:num_faces]
            face_embeddings = face_emb_raw[0][:num_faces] if num_faces > 0 else np.array([])
        else:
            face_boxes = face_boxes_raw[:num_faces]
            face_landmarks = face_landmarks_raw[:num_faces]
            face_scores_arr = face_scores_raw[:num_faces]
            face_person_idx = face_person_idx_raw[:num_faces]
            face_embeddings = face_emb_raw[:num_faces] if num_faces > 0 else np.array([])

        return {
            # YOLO detections
            'num_dets': num_dets,
            'boxes': boxes,
            'scores': scores,
            'classes': classes,
            # MobileCLIP embeddings
            'global_embedding': global_embedding,
            'box_embeddings': box_embeddings,
            'normalized_boxes': normalized_boxes,
            # Face detections and embeddings (from person crops only)
            'num_faces': num_faces,
            'face_boxes': face_boxes,
            'face_landmarks': face_landmarks,
            'face_scores': face_scores_arr,
            'face_embeddings': face_embeddings,
            'face_person_idx': face_person_idx,
            # Transform params
            'orig_shape': (orig_h, orig_w),
            'scale': scale,
            'padding': padding,
        }

    # =========================================================================
    # OCR: PP-OCRv5 Text Detection and Recognition
    # =========================================================================

    def infer_ocr(self, image_bytes: bytes) -> dict[str, Any]:
        """
        OCR inference: PP-OCRv5 text detection and recognition.

        Pipeline:
        1. Preprocess image (resize, normalize)
        2. Call OCR pipeline BLS (detection + recognition)
        3. Return text, boxes, and confidence scores

        Args:
            image_bytes: Raw JPEG/PNG bytes

        Returns:
            Dict with OCR results:
            - num_texts: Number of text regions detected
            - texts: List of detected text strings
            - text_boxes: [N, 8] Quadrilateral boxes
            - text_boxes_normalized: [N, 4] Axis-aligned boxes normalized
            - text_scores: Detection confidence scores
            - rec_scores: Recognition confidence scores
        """
        import cv2

        # Decode image (handle RGBA by reading with alpha then converting)
        img_array = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
        if img_array is None:
            return {
                'num_texts': 0,
                'texts': [],
                'text_boxes': np.array([]),
                'text_boxes_normalized': np.array([]),
                'text_scores': np.array([]),
                'rec_scores': np.array([]),
            }

        # Handle different channel formats
        if len(img_array.shape) == 2:
            # Grayscale -> BGR
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        elif img_array.shape[2] == 4:
            # RGBA/BGRA -> BGR (composite alpha onto white background)
            alpha = img_array[:, :, 3:4] / 255.0
            rgb = img_array[:, :, :3]
            white_bg = np.ones_like(rgb) * 255
            img_array = (rgb * alpha + white_bg * (1 - alpha)).astype(np.uint8)

        orig_h, orig_w = img_array.shape[:2]

        # Preprocess for OCR detection
        # PP-OCR approach: scale to fit within limit, pad to 32-boundary
        # For small images (< 640px), upscale to ensure text is detectable
        ocr_limit_side = 960
        ocr_min_side = 640  # Minimum dimension for reliable text detection

        # Calculate scaling ratio
        max_side = max(orig_h, orig_w)
        if max_side > ocr_limit_side:
            # Downscale large images
            ratio = ocr_limit_side / max_side
        elif max_side < ocr_min_side:
            # Upscale small images for better detection
            # Use conservative upscaling (up to min_side, not limit_side)
            ratio = ocr_min_side / max_side
        else:
            # Keep original size
            ratio = 1.0

        resize_h = int(orig_h * ratio)
        resize_w = int(orig_w * ratio)

        # Ensure minimum size
        resize_h = max(32, resize_h)
        resize_w = max(32, resize_w)

        # Resize
        resized = cv2.resize(img_array, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)

        # Pad to 32-boundary
        pad_h = (32 - resize_h % 32) % 32
        pad_w = (32 - resize_w % 32) % 32
        if pad_h > 0 or pad_w > 0:
            padded = np.zeros((resize_h + pad_h, resize_w + pad_w, 3), dtype=np.uint8)
            padded[:resize_h, :resize_w, :] = resized
            resized = padded

        # Normalize: (x / 127.5) - 1 for PP-OCR
        ocr_input = resized.astype(np.float32) / 127.5 - 1.0

        # HWC -> CHW (OpenCV reads as BGR, PP-OCR expects BGR)
        ocr_input = ocr_input.transpose(2, 0, 1)

        # Original image for cropping (normalize to [0, 1])
        orig_normalized = img_array.astype(np.float32) / 255.0
        orig_normalized = orig_normalized.transpose(2, 0, 1)

        # Prepare inputs for ocr_pipeline (max_batch_size: 0, no batch dim)
        inputs = [
            InferInput('ocr_images', list(ocr_input.shape), 'FP32'),
            InferInput('original_image', list(orig_normalized.shape), 'FP32'),
            InferInput('orig_shape', [2], 'INT32'),
        ]

        inputs[0].set_data_from_numpy(ocr_input)
        inputs[1].set_data_from_numpy(orig_normalized)
        inputs[2].set_data_from_numpy(np.array([orig_h, orig_w], dtype=np.int32))

        outputs = [
            InferRequestedOutput('num_texts'),
            InferRequestedOutput('text_boxes'),
            InferRequestedOutput('text_boxes_normalized'),
            InferRequestedOutput('texts'),
            InferRequestedOutput('text_scores'),
            InferRequestedOutput('rec_scores'),
        ]

        try:
            response = self._infer_with_retry('ocr_pipeline', inputs, outputs)
        except Exception as e:
            logger.error(f'OCR inference failed: {e}')
            return {
                'num_texts': 0,
                'texts': [],
                'text_boxes': np.array([]),
                'text_boxes_normalized': np.array([]),
                'text_scores': np.array([]),
                'rec_scores': np.array([]),
            }

        # Parse outputs
        num_texts_raw = response.as_numpy('num_texts')
        logger.info(
            f'OCR response: num_texts_raw shape={num_texts_raw.shape}, value={num_texts_raw}'
        )
        num_texts = int(num_texts_raw[0])
        text_boxes = response.as_numpy('text_boxes')[:num_texts]
        text_boxes_norm = response.as_numpy('text_boxes_normalized')[:num_texts]
        text_scores = response.as_numpy('text_scores')[:num_texts]
        rec_scores = response.as_numpy('rec_scores')[:num_texts]

        # Decode text strings (Triton returns bytes)
        texts_raw = response.as_numpy('texts')[:num_texts]
        texts = []
        for t in texts_raw:
            if isinstance(t, bytes):
                texts.append(t.decode('utf-8', errors='ignore'))
            elif isinstance(t, np.bytes_):
                texts.append(str(t, 'utf-8', errors='ignore'))
            else:
                texts.append(str(t))

        return {
            'num_texts': num_texts,
            'texts': texts,
            'text_boxes': text_boxes,
            'text_boxes_normalized': text_boxes_norm,
            'text_scores': text_scores,
            'rec_scores': rec_scores,
            'image_width': orig_w,
            'image_height': orig_h,
        }

    # =========================================================================
    # Unified Complete Pipeline: Detection + Embeddings + Faces + OCR
    # =========================================================================

    def infer_unified_complete(
        self, image_bytes: bytes, face_model: str = 'scrfd'
    ) -> dict[str, Any]:
        """
        Unified complete analysis pipeline: YOLO + CLIP + Faces + OCR.

        Single request that returns all analysis results from an image:
        1. YOLO object detection
        2. MobileCLIP global and per-box embeddings
        3. Face detection + ArcFace embeddings (selectable model)
        4. PP-OCRv5 text detection and recognition

        Face Model Selection:
        - "scrfd" (default): SCRFD on person crops only (more efficient)
        - "yolo11": YOLO11-face on full image (may detect more faces)

        Args:
            image_bytes: Raw JPEG/PNG bytes
            face_model: "scrfd" or "yolo11"

        Returns:
            Dict with all analysis results:
            - Detection: num_dets, boxes, scores, classes
            - Embeddings: global_embedding, box_embeddings, normalized_boxes
            - Faces: num_faces, face_boxes, face_landmarks, face_scores, face_embeddings, face_person_idx
            - OCR: num_texts, texts, text_boxes, text_det_scores, text_rec_scores
            - Metadata: face_model_used, orig_shape
        """
        # Fast JPEG header parse for dimensions
        orig_w, orig_h = get_jpeg_dimensions_fast(image_bytes)

        # Get cached affine matrix
        affine_matrix, scale, padding = calculate_affine_matrix(orig_w, orig_h, self.input_size)

        # Prepare inputs
        input_data = np.frombuffer(image_bytes, dtype=np.uint8)
        input_data = np.expand_dims(input_data, axis=0)

        inputs = [
            InferInput('encoded_images', input_data.shape, 'UINT8'),
            InferInput('affine_matrices', [1, 2, 3], 'FP32'),
            InferInput('face_model', [1, 1], 'BYTES'),
        ]
        inputs[0].set_data_from_numpy(input_data)
        inputs[1].set_data_from_numpy(np.expand_dims(affine_matrix, axis=0))
        inputs[2].set_data_from_numpy(np.array([[face_model.encode('utf-8')]], dtype=object))

        # Request all outputs
        outputs = [
            # Detection
            InferRequestedOutput('num_dets'),
            InferRequestedOutput('det_boxes'),
            InferRequestedOutput('det_scores'),
            InferRequestedOutput('det_classes'),
            # Embeddings
            InferRequestedOutput('global_embeddings'),
            InferRequestedOutput('box_embeddings'),
            InferRequestedOutput('normalized_boxes'),
            # Faces
            InferRequestedOutput('num_faces'),
            InferRequestedOutput('face_embeddings'),
            InferRequestedOutput('face_boxes'),
            InferRequestedOutput('face_landmarks'),
            InferRequestedOutput('face_scores'),
            InferRequestedOutput('face_person_idx'),
            # OCR
            InferRequestedOutput('num_texts'),
            InferRequestedOutput('text_boxes'),
            InferRequestedOutput('text_boxes_normalized'),
            InferRequestedOutput('texts'),
            InferRequestedOutput('text_det_scores'),
            InferRequestedOutput('text_rec_scores'),
            # Metadata
            InferRequestedOutput('face_model_used'),
        ]

        # Sync inference with retry
        response = self._infer_with_retry('unified_complete_pipeline', inputs, outputs)

        # Parse detection outputs (handle batch dimension variations)
        num_dets_raw = response.as_numpy('num_dets')
        num_dets = int(num_dets_raw.flatten()[0])

        det_boxes_raw = response.as_numpy('det_boxes')
        det_scores_raw = response.as_numpy('det_scores')
        det_classes_raw = response.as_numpy('det_classes')

        if det_boxes_raw.ndim == 3:
            boxes = det_boxes_raw[0][:num_dets]
            scores = det_scores_raw[0][:num_dets]
            classes = det_classes_raw[0][:num_dets]
        else:
            boxes = det_boxes_raw[:num_dets]
            scores = det_scores_raw[:num_dets]
            classes = det_classes_raw[:num_dets]

        # Parse embedding outputs
        global_embedding_raw = response.as_numpy('global_embeddings')
        if global_embedding_raw.ndim == 2:
            global_embedding = global_embedding_raw[0]
        else:
            global_embedding = global_embedding_raw

        box_embeddings_raw = response.as_numpy('box_embeddings')
        normalized_boxes_raw = response.as_numpy('normalized_boxes')

        if box_embeddings_raw.ndim == 3:
            box_embeddings = box_embeddings_raw[0][:num_dets]
            normalized_boxes = normalized_boxes_raw[0][:num_dets]
        else:
            box_embeddings = box_embeddings_raw[:num_dets]
            normalized_boxes = normalized_boxes_raw[:num_dets]

        # Parse face outputs
        num_faces_arr = response.as_numpy('num_faces')
        num_faces = int(num_faces_arr.flatten()[0])

        face_boxes_raw = response.as_numpy('face_boxes')
        face_landmarks_raw = response.as_numpy('face_landmarks')
        face_scores_raw = response.as_numpy('face_scores')
        face_person_idx_raw = response.as_numpy('face_person_idx')
        face_emb_raw = response.as_numpy('face_embeddings')

        if face_boxes_raw.ndim == 3:
            face_boxes = face_boxes_raw[0][:num_faces]
            face_landmarks = face_landmarks_raw[0][:num_faces]
            face_scores_arr = face_scores_raw[0][:num_faces]
            face_person_idx = face_person_idx_raw[0][:num_faces]
            face_embeddings = face_emb_raw[0][:num_faces] if num_faces > 0 else np.array([])
        else:
            face_boxes = face_boxes_raw[:num_faces]
            face_landmarks = face_landmarks_raw[:num_faces]
            face_scores_arr = face_scores_raw[:num_faces]
            face_person_idx = face_person_idx_raw[:num_faces]
            face_embeddings = face_emb_raw[:num_faces] if num_faces > 0 else np.array([])

        # Parse OCR outputs
        num_texts_raw = response.as_numpy('num_texts')
        num_texts = int(num_texts_raw.flatten()[0])
        text_boxes_raw = response.as_numpy('text_boxes')
        text_boxes_norm_raw = response.as_numpy('text_boxes_normalized')
        texts_raw = response.as_numpy('texts')
        text_det_scores_raw = response.as_numpy('text_det_scores')
        text_rec_scores_raw = response.as_numpy('text_rec_scores')

        if text_boxes_raw.ndim == 3:
            text_boxes = text_boxes_raw[0][:num_texts]
            text_boxes_norm = text_boxes_norm_raw[0][:num_texts]
            text_det_scores = text_det_scores_raw[0][:num_texts]
            text_rec_scores = text_rec_scores_raw[0][:num_texts]
        else:
            text_boxes = text_boxes_raw[:num_texts]
            text_boxes_norm = text_boxes_norm_raw[:num_texts]
            text_det_scores = text_det_scores_raw[:num_texts]
            text_rec_scores = text_rec_scores_raw[:num_texts]

        # Decode text strings
        texts = []
        if texts_raw.size > 0:
            texts_flat = texts_raw.flatten()[:num_texts]
            for t in texts_flat:
                if isinstance(t, bytes):
                    texts.append(t.decode('utf-8', errors='replace'))
                else:
                    texts.append(str(t))

        # Parse metadata
        face_model_used_raw = response.as_numpy('face_model_used')
        if face_model_used_raw.size > 0:
            fm = face_model_used_raw.flatten()[0]
            if isinstance(fm, bytes):
                face_model_used = fm.decode('utf-8')
            else:
                face_model_used = str(fm)
        else:
            face_model_used = face_model

        return {
            # Detection
            'num_dets': num_dets,
            'boxes': boxes,
            'scores': scores,
            'classes': classes,
            # Embeddings
            'global_embedding': global_embedding,
            'box_embeddings': box_embeddings,
            'normalized_boxes': normalized_boxes,
            # Faces
            'num_faces': num_faces,
            'face_boxes': face_boxes,
            'face_landmarks': face_landmarks,
            'face_scores': face_scores_arr,
            'face_embeddings': face_embeddings,
            'face_person_idx': face_person_idx,
            # OCR
            'num_texts': num_texts,
            'texts': texts,
            'text_boxes': text_boxes,
            'text_boxes_normalized': text_boxes_norm,
            'text_det_scores': text_det_scores,
            'text_rec_scores': text_rec_scores,
            # Metadata
            'face_model_used': face_model_used,
            'orig_shape': (orig_h, orig_w),
            'scale': scale,
            'padding': padding,
        }

    # =========================================================================
    # YOLO11-Face: Alternative Face Detection Pipeline
    # =========================================================================

    def infer_faces_yolo11(self, image_bytes: bytes, confidence: float = 0.5) -> dict[str, Any]:
        """
        Face detection and recognition using YOLO11-face + ArcFace.

        Alternative to SCRFD-based face detection. Uses YOLO11-face which is a
        single-stage pose-based face detector trained on face datasets.

        Pipeline:
        1. Decode and preprocess image for YOLO11-face (640x640)
        2. Preserve HD original for face alignment cropping
        3. Call yolo11_face_pipeline BLS model
        4. Return face boxes, landmarks, and ArcFace embeddings

        CRITICAL: Face alignment crops from HD ORIGINAL image, not detection input.
        This is the industry standard for maximum recognition accuracy.

        Args:
            image_bytes: Raw JPEG/PNG bytes
            confidence: Minimum detection confidence (filtered post-inference)

        Returns:
            Dict with:
                - num_faces: Number of faces detected
                - face_boxes: [N, 4] normalized boxes [x1, y1, x2, y2]
                - face_landmarks: [N, 10] 5-point landmarks (pixel coords)
                - face_scores: [N] detection confidence
                - face_embeddings: [N, 512] ArcFace embeddings
                - face_quality: [N] quality scores
                - orig_shape: (height, width)
        """
        # Decode image to get original dimensions
        img_array = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img_array is None:
            # Try PIL as fallback for unusual formats
            pil_img = Image.open(io.BytesIO(image_bytes))
            pil_img = pil_img.convert('RGB')
            img_array = np.array(pil_img)[:, :, ::-1]  # RGB to BGR

        orig_h, orig_w = img_array.shape[:2]

        # Preprocess for YOLO11-face (640x640 letterbox)
        letterbox = LetterBox(new_shape=(640, 640), auto=False, stride=32)
        preprocessed = letterbox(image=img_array)

        # Calculate letterbox affine matrix for inverse transformation
        scale = min(640 / orig_h, 640 / orig_w)
        new_w = orig_w * scale
        new_h = orig_h * scale
        pad_x = (640 - new_w) / 2
        pad_y = (640 - new_h) / 2
        affine_matrix = np.array([[scale, 0, pad_x], [0, scale, pad_y]], dtype=np.float32)

        # Convert to CHW, normalize to [0, 1], float32
        face_input = preprocessed.transpose(2, 0, 1).astype(np.float32) / 255.0

        # Prepare HD original for cropping (CHW, float32, [0, 1])
        # Convert BGR to RGB for consistency
        orig_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        original_image = orig_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0

        # Prepare inputs for yolo11_face_pipeline
        inputs = [
            InferInput('face_images', [1, 3, 640, 640], 'FP32'),
            InferInput('original_image', [1, 3, orig_h, orig_w], 'FP32'),
            InferInput('orig_shape', [1, 2], 'INT32'),
            InferInput('affine_matrix', [1, 2, 3], 'FP32'),
        ]
        inputs[0].set_data_from_numpy(np.expand_dims(face_input, axis=0))
        inputs[1].set_data_from_numpy(np.expand_dims(original_image, axis=0))
        inputs[2].set_data_from_numpy(np.array([[orig_h, orig_w]], dtype=np.int32))
        inputs[3].set_data_from_numpy(np.expand_dims(affine_matrix, axis=0))

        outputs = [
            InferRequestedOutput('num_faces'),
            InferRequestedOutput('face_boxes'),
            InferRequestedOutput('face_landmarks'),
            InferRequestedOutput('face_scores'),
            InferRequestedOutput('face_embeddings'),
            InferRequestedOutput('face_quality'),
        ]

        # Sync inference with retry
        response = self._infer_with_retry('yolo11_face_pipeline', inputs, outputs)

        # Parse outputs
        num_faces_arr = response.as_numpy('num_faces')
        num_faces = int(num_faces_arr.flatten()[0])

        face_boxes_raw = response.as_numpy('face_boxes')
        face_landmarks_raw = response.as_numpy('face_landmarks')
        face_scores_raw = response.as_numpy('face_scores')
        face_embeddings_raw = response.as_numpy('face_embeddings')
        face_quality_raw = response.as_numpy('face_quality')

        # Handle batch dimension
        if face_boxes_raw.ndim == 3:
            face_boxes = face_boxes_raw[0][:num_faces]
            face_landmarks = face_landmarks_raw[0][:num_faces]
            face_scores = face_scores_raw[0][:num_faces]
            face_embeddings = face_embeddings_raw[0][:num_faces] if num_faces > 0 else np.array([])
            face_quality = face_quality_raw[0][:num_faces]
        else:
            face_boxes = face_boxes_raw[:num_faces]
            face_landmarks = face_landmarks_raw[:num_faces]
            face_scores = face_scores_raw[:num_faces]
            face_embeddings = face_embeddings_raw[:num_faces] if num_faces > 0 else np.array([])
            face_quality = face_quality_raw[:num_faces]

        # Filter by confidence threshold (post-inference since pipeline doesn't accept it)
        if confidence > 0 and num_faces > 0:
            mask = face_scores >= confidence
            face_boxes = face_boxes[mask]
            face_landmarks = face_landmarks[mask]
            face_scores = face_scores[mask]
            face_embeddings = face_embeddings[mask] if len(face_embeddings) > 0 else face_embeddings
            face_quality = face_quality[mask]
            num_faces = len(face_scores)

        return {
            'num_faces': num_faces,
            'face_boxes': face_boxes,
            'face_landmarks': face_landmarks,
            'face_scores': face_scores,
            'face_embeddings': face_embeddings,
            'face_quality': face_quality,
            'orig_shape': (orig_h, orig_w),
        }

    def infer_faces_full_yolo11(
        self, image_bytes: bytes, confidence: float = 0.5
    ) -> dict[str, Any]:
        """
        Full pipeline with YOLO11-face: YOLO detection + MobileCLIP + YOLO11-face + ArcFace.

        Combines Track E visual search capabilities with YOLO11-face detection:
        1. YOLO object detection (parallel with 2, 3)
        2. MobileCLIP global embedding (parallel with 1, 3)
        3. YOLO11-face detection + ArcFace embeddings (parallel with 1, 2)

        This is an alternative to infer_faces_full() which uses SCRFD for face detection.
        YOLO11-face may detect more faces in challenging conditions (occlusion, pose).

        Args:
            image_bytes: Raw JPEG/PNG bytes
            confidence: Minimum face detection confidence (default 0.5)

        Returns:
            Dict with:
                - YOLO detections (num_dets, boxes, scores, classes)
                - MobileCLIP global embedding (512-dim)
                - Face detections (num_faces, face_boxes, face_landmarks, face_scores)
                - Face ArcFace embeddings (128 x 512-dim)
                - Face quality scores
                - Transform params (orig_shape, scale, padding)
        """
        from concurrent.futures import ThreadPoolExecutor

        # Fast JPEG header parse for dimensions
        orig_w, orig_h = get_jpeg_dimensions_fast(image_bytes)

        # Get cached affine matrix
        _affine_matrix, scale, padding = calculate_affine_matrix(orig_w, orig_h, self.input_size)

        # Run Track E (YOLO + MobileCLIP) and YOLO11-face in parallel
        def run_track_e():
            return self.infer_track_e(image_bytes, full_pipeline=False)

        def run_yolo11_face():
            return self.infer_faces_yolo11(image_bytes, confidence=confidence)

        with ThreadPoolExecutor(max_workers=2) as executor:
            track_e_future = executor.submit(run_track_e)
            face_future = executor.submit(run_yolo11_face)

            track_e_result = track_e_future.result()
            face_result = face_future.result()

        # Combine results
        return {
            # YOLO detections from Track E
            'num_dets': track_e_result['num_dets'],
            'boxes': track_e_result['boxes'],
            'scores': track_e_result['scores'],
            'classes': track_e_result['classes'],
            # MobileCLIP global embedding from Track E
            'image_embedding': track_e_result['image_embedding'],
            # Face detections from YOLO11-face pipeline
            'num_faces': face_result['num_faces'],
            'face_boxes': face_result['face_boxes'],
            'face_landmarks': face_result['face_landmarks'],
            'face_scores': face_result['face_scores'],
            'face_embeddings': face_result['face_embeddings'],
            'face_quality': face_result['face_quality'],
            # Transform params
            'orig_shape': (orig_h, orig_w),
            'scale': scale,
            'padding': padding,
        }

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
