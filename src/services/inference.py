"""
Inference service handling all YOLO detection tracks.

Orchestrates inference across PyTorch and Triton backends.
Returns industry-standard response format with metadata.
Timing is handled by FastAPI middleware (total_time_ms injected).
"""

import hashlib
import logging
from typing import Any

import numpy as np
from ultralytics import YOLO

from src.clients.triton_client import get_triton_client
from src.config import get_settings
from src.utils.affine import get_jpeg_dimensions_fast
from src.utils.cache import get_clip_tokenizer, get_image_cache, get_text_cache
from src.utils.image_processing import decode_image, validate_image
from src.utils.pytorch_utils import (
    format_detections,
    thread_safe_predict,
    thread_safe_predict_batch,
)


logger = logging.getLogger(__name__)


def build_response(
    detections: list,
    image_shape: tuple[int, int],
    model_name: str,
    track: str,
    backend: str = 'triton',
    preprocessing: str | None = None,
    nms_location: str | None = None,
    embedding: np.ndarray | None = None,
    box_embeddings: list | None = None,
    normalized_boxes: list | None = None,
) -> dict[str, Any]:
    """
    Build standardized inference response with all metadata.

    Args:
        detections: List of detection dicts
        image_shape: Original image (height, width)
        model_name: Model name used
        track: Performance track (A, B, C, D, E)
        backend: Inference backend (pytorch, triton)
        preprocessing: Preprocessing method (cpu, gpu_dali, gpu_dali_auto)
        nms_location: NMS location (cpu, gpu)
        embedding: Optional image embedding (Track E)
        box_embeddings: Optional per-box embeddings (Track E full)
        normalized_boxes: Optional normalized boxes (Track E full)

    Returns:
        Standardized response dict
    """
    response = {
        'detections': detections,
        'num_detections': len(detections),
        'status': 'success',
        'image': {
            'height': image_shape[0],
            'width': image_shape[1],
        },
        'model': {
            'name': model_name,
            'backend': backend,
            'device': 'gpu',
        },
        'track': track,
        'preprocessing': preprocessing,
        'nms_location': nms_location,
        # total_time_ms injected by middleware
    }

    # Add Track E specific fields if provided
    if embedding is not None:
        response['embedding_norm'] = float(np.linalg.norm(embedding))

    if box_embeddings is not None:
        response['box_embeddings'] = box_embeddings

    if normalized_boxes is not None:
        response['normalized_boxes'] = normalized_boxes

    return response


class InferenceService:
    """
    Unified inference service for all YOLO tracks.

    Handles:
    - Track A: PyTorch direct inference
    - Track B: Standard TRT + CPU NMS
    - Track C: End2End TRT + GPU NMS
    - Track D: DALI + TRT (Full GPU)
    - Track E: Visual search ensemble

    All responses include:
    - detections: List of normalized [0,1] bounding boxes
    - num_detections: Count of detections
    - image: Original image dimensions
    - model: Model name and backend info
    - track: Performance track identifier
    - total_time_ms: Injected by middleware
    """

    def __init__(self):
        """Initialize inference service."""
        self.settings = get_settings()

    # =========================================================================
    # Track A: PyTorch
    # =========================================================================
    def infer_pytorch(
        self,
        image_bytes: bytes,
        filename: str,
        model: Any,
        model_name: str = 'yolov11_small',
    ) -> dict[str, Any]:
        """
        Track A: PyTorch inference.

        Args:
            image_bytes: Raw image bytes
            filename: Original filename for validation
            model: YOLO model instance
            model_name: Model name for metadata

        Returns:
            Standardized response dict
        """

        img = decode_image(image_bytes, filename)
        validate_image(img, filename)
        image_shape = img.shape[:2]  # (height, width)

        detections = thread_safe_predict(model, img)
        results = format_detections(detections)

        return build_response(
            detections=results,
            image_shape=image_shape,
            model_name=model_name,
            track='A',
            backend='pytorch',
            preprocessing='cpu',
            nms_location='cpu',
        )

    def infer_pytorch_batch(
        self,
        images_data: list[tuple[bytes, str]],
        model: Any,
    ) -> dict[str, Any]:
        """Track A: Batch PyTorch inference."""

        decoded_images = []
        decoded_filenames = []
        image_shapes = []
        failed_images = []

        for idx, (image_bytes, filename) in enumerate(images_data):
            try:
                img = decode_image(image_bytes, filename)
                validate_image(img, filename)
                decoded_images.append(img)
                decoded_filenames.append(filename)
                image_shapes.append(img.shape[:2])
            except ValueError as e:
                failed_images.append({'filename': filename, 'index': idx, 'error': str(e)})

        all_results = []
        if decoded_images:
            detections_batch = thread_safe_predict_batch(model, decoded_images)

            for idx, (detections, filename, shape) in enumerate(
                zip(detections_batch, decoded_filenames, image_shapes, strict=False)
            ):
                results = format_detections(detections)
                all_results.append(
                    {
                        'filename': filename,
                        'image_index': idx,
                        'detections': results,
                        'num_detections': len(results),
                        'status': 'success',
                        'track': 'A',
                        'image': {'height': shape[0], 'width': shape[1]},
                    }
                )

        return {
            'total_images': len(images_data),
            'processed_images': len(all_results),
            'failed_images': len(failed_images),
            'results': all_results,
            'failures': failed_images if failed_images else None,
            'status': 'success',
        }

    # =========================================================================
    # Track B/C: Standard and End2End TRT
    # =========================================================================
    def infer_track_b(
        self,
        image_bytes: bytes,
        filename: str,
        model_url: str,
        model_name: str = 'yolov11_small',
    ) -> dict[str, Any]:
        """Track B: Standard TRT + CPU NMS using Ultralytics client."""

        img = decode_image(image_bytes, filename)
        validate_image(img, filename)
        image_shape = img.shape[:2]

        model = YOLO(model_url, task='detect')
        detections = model(img, verbose=False)
        results = format_detections(detections)

        return build_response(
            detections=results,
            image_shape=image_shape,
            model_name=model_name,
            track='B',
            backend='triton',
            preprocessing='cpu',
            nms_location='cpu',
        )

    def infer_track_c(
        self,
        image_bytes: bytes,
        filename: str,
        model_name: str,
    ) -> dict[str, Any]:
        """Track C: End2End TRT + GPU NMS."""
        img = decode_image(image_bytes, filename)
        validate_image(img, filename)
        image_shape = img.shape[:2]

        client = get_triton_client(self.settings.triton_url)
        detections = client.infer_track_c(img, model_name)
        results = client.format_detections(detections)

        return build_response(
            detections=results,
            image_shape=image_shape,
            model_name=model_name,
            track='C',
            backend='triton',
            preprocessing='cpu',
            nms_location='gpu',
        )

    # =========================================================================
    # Track D: DALI + TRT
    # =========================================================================
    def infer_track_d(
        self,
        image_bytes: bytes,
        model_name: str,
        auto_affine: bool = False,
    ) -> dict[str, Any]:
        """Track D: Full GPU pipeline with DALI preprocessing."""
        # Get original dimensions from JPEG header (no decode needed)
        orig_w, orig_h = get_jpeg_dimensions_fast(image_bytes)
        image_shape = (orig_h, orig_w)

        client = get_triton_client(self.settings.triton_url)
        detections = client.infer_track_d(image_bytes, model_name, auto_affine)
        results = client.format_detections(detections)

        return build_response(
            detections=results,
            image_shape=image_shape,
            model_name=model_name,
            track='D',
            backend='triton',
            preprocessing='gpu_dali_auto' if auto_affine else 'gpu_dali',
            nms_location='gpu',
        )

    async def infer_track_d_async(
        self,
        image_bytes: bytes,
        model_name: str,
        auto_affine: bool = False,
    ) -> dict[str, Any]:
        """Async Track D inference for FastAPI endpoints."""
        # Use sync client with thread pool (FastAPI handles async)
        return self.infer_track_d(image_bytes, model_name, auto_affine)

    # =========================================================================
    # Track E: Visual Search
    # =========================================================================
    def infer_track_e(
        self,
        image_bytes: bytes,
        full_pipeline: bool = False,
    ) -> dict[str, Any]:
        """
        Track E: Visual search ensemble inference (sync).

        Args:
            image_bytes: Raw JPEG/PNG bytes
            full_pipeline: If True, include per-box embeddings

        Returns:
            Standardized response dict with embeddings
        """
        client = get_triton_client(self.settings.triton_url)
        result = client.infer_track_e(image_bytes, full_pipeline)
        detections = client.format_detections(result)

        orig_h, orig_w = result['orig_shape']

        # Prepare Track E specific fields
        embedding = result['image_embedding']
        box_embeddings = None
        normalized_boxes = None

        if full_pipeline and result['num_dets'] > 0:
            box_embeddings = result.get('box_embeddings', np.array([])).tolist()
            normalized_boxes = result.get('normalized_boxes', np.array([])).tolist()

        return build_response(
            detections=detections,
            image_shape=(orig_h, orig_w),
            model_name='yolo_clip_ensemble' if not full_pipeline else 'yolo_mobileclip_ensemble',
            track='E_full' if full_pipeline else 'E',
            backend='triton',
            preprocessing='gpu_dali',
            nms_location='gpu',
            embedding=embedding,
            box_embeddings=box_embeddings,
            normalized_boxes=normalized_boxes,
        )

    async def infer_track_e_async(
        self,
        image_bytes: bytes,
        full_pipeline: bool = False,
    ) -> dict[str, Any]:
        """Async wrapper for Track E inference (for FastAPI endpoints)."""
        return self.infer_track_e(image_bytes, full_pipeline)

    def infer_track_e_batch(
        self,
        images_bytes: list[bytes],
        max_workers: int = 32,
    ) -> list[dict[str, Any]]:
        """
        Track E batch inference: Process multiple images in parallel.

        For large photo libraries (50K+ images), sending batches of 16-64
        images per request significantly improves throughput by:
        - Reducing HTTP round-trip overhead
        - Ensuring full DALI/TRT batch utilization
        - Maximizing GPU parallelism

        Args:
            images_bytes: List of raw JPEG/PNG bytes (up to 64 images)
            max_workers: Max parallel inference threads

        Returns:
            List of standardized response dicts with embeddings
        """
        client = get_triton_client(self.settings.triton_url)
        raw_results = client.infer_track_e_batch(images_bytes, max_workers)

        # Format each result
        formatted_results = []
        for result in raw_results:
            detections = client.format_detections(result)
            orig_h, orig_w = result['orig_shape']
            embedding = result['image_embedding']

            formatted_results.append(
                build_response(
                    detections=detections,
                    image_shape=(orig_h, orig_w),
                    model_name='yolo_clip_ensemble',
                    track='E_batch',
                    backend='triton',
                    preprocessing='gpu_dali',
                    nms_location='gpu',
                    embedding=embedding,
                )
            )

        return formatted_results

    # =========================================================================
    # Track F: CPU Preprocessing + Direct TRT Models (No DALI)
    # =========================================================================
    def infer_track_f(self, image_bytes: bytes) -> dict[str, Any]:
        """
        Track F: CPU preprocessing + direct YOLO TRT + MobileCLIP TRT.

        Unlike Track E (DALI ensemble), Track F uses:
        - CPU decode (PIL)
        - CPU letterbox for YOLO (custom, not Ultralytics)
        - CPU resize/crop for CLIP (custom)
        - Direct TRT inference (no ensemble scheduler overhead)

        Benefits:
        - Lower VRAM usage (no DALI instances)
        - More TRT instances possible
        - Baseline comparison for CPU vs GPU preprocessing

        Args:
            image_bytes: Raw JPEG/PNG bytes

        Returns:
            Standardized response dict with embeddings
        """
        client = get_triton_client(self.settings.triton_url)
        result = client.infer_track_f(image_bytes)
        detections = client.format_detections(result)

        orig_h, orig_w = result['orig_shape']
        embedding = result['image_embedding']

        return build_response(
            detections=detections,
            image_shape=(orig_h, orig_w),
            model_name='yolov11_small_trt_end2end + mobileclip2_s2',
            track='F',
            backend='triton',
            preprocessing='cpu',
            nms_location='gpu',
            embedding=embedding,
        )

    def encode_text_sync(self, text: str, use_cache: bool = True) -> np.ndarray:
        """
        Encode text to embedding using MobileCLIP text encoder.

        Args:
            text: Query text
            use_cache: Use text embedding cache

        Returns:
            512-dim L2-normalized embedding
        """
        # Check cache first
        if use_cache:
            cache = get_text_cache()
            cache_key = hashlib.sha256(text.encode()).hexdigest()
            cached_embedding = cache.get(cache_key)
            if cached_embedding is not None:
                return cached_embedding

        # Tokenize (using cached singleton)
        tokenizer = get_clip_tokenizer()
        tokens = tokenizer(
            text, padding='max_length', max_length=77, truncation=True, return_tensors='np'
        )

        # Get Triton client and encode
        client = get_triton_client(self.settings.triton_url)
        embedding = client.encode_text(tokens['input_ids'])

        # Cache the result
        if use_cache:
            cache.set(cache_key, embedding)

        return embedding

    def encode_image_sync(self, image_bytes: bytes, use_cache: bool = True) -> np.ndarray:
        """
        Encode image to embedding using MobileCLIP image encoder.

        Args:
            image_bytes: Raw JPEG/PNG bytes
            use_cache: Use image embedding cache

        Returns:
            512-dim L2-normalized embedding
        """
        # Check cache first
        if use_cache:
            cache = get_image_cache()
            cache_key = hashlib.sha256(image_bytes).hexdigest()
            cached_embedding = cache.get(cache_key)
            if cached_embedding is not None:
                return cached_embedding

        # Get Triton client and encode
        client = get_triton_client(self.settings.triton_url)
        embedding = client.encode_image(image_bytes)

        # Cache the result
        if use_cache:
            cache.set(cache_key, embedding)

        return embedding

    # =========================================================================
    # Track E Faces: SCRFD + ArcFace Pipeline
    # =========================================================================
    def infer_faces(self, image_bytes: bytes) -> dict[str, Any]:
        """
        Face detection and recognition pipeline (SCRFD + ArcFace).

        100% GPU pipeline via DALI preprocessing:
        1. GPU JPEG decode (nvJPEG via DALI)
        2. Quad-branch preprocessing (SCRFD 640, HD original)
        3. SCRFD face detection + NMS
        4. Face alignment + ArcFace embedding extraction

        Args:
            image_bytes: Raw JPEG/PNG bytes

        Returns:
            Dict with face detections, landmarks, and ArcFace embeddings
        """
        client = get_triton_client(self.settings.triton_url)
        result = client.infer_faces_only(image_bytes)

        orig_h, orig_w = result['orig_shape']

        # Format face detections
        # Note: face_pipeline already returns boxes normalized to [0,1]
        # and landmarks in pixel coordinates (need normalization)
        faces = []
        for i in range(result['num_faces']):
            box = result['face_boxes'][i]
            landmarks = result['face_landmarks'][i]
            score = float(result['face_scores'][i])
            quality = float(result['face_quality'][i]) if len(result['face_quality']) > i else None

            # Box is already normalized from face_pipeline
            norm_box = [float(x) for x in box]

            # Landmarks are in pixel coordinates - normalize to [0,1]
            norm_landmarks = []
            for j in range(0, 10, 2):
                norm_landmarks.append(float(landmarks[j]) / orig_w)
                norm_landmarks.append(float(landmarks[j + 1]) / orig_h)

            faces.append(
                {
                    'box': norm_box,
                    'landmarks': norm_landmarks,
                    'score': score,
                    'quality': quality,
                }
            )

        # Format embeddings
        embeddings = []
        if result['num_faces'] > 0 and len(result['face_embeddings']) > 0:
            embeddings = result['face_embeddings'].tolist()

        return {
            'num_faces': result['num_faces'],
            'faces': faces,
            'embeddings': embeddings,
            'orig_shape': (orig_h, orig_w),
        }

    def infer_faces_full(self, image_bytes: bytes) -> dict[str, Any]:
        """
        Full unified pipeline: YOLO + SCRFD + MobileCLIP + ArcFace.

        All processing happens in Triton via quad-branch ensemble:
        1. GPU JPEG decode (nvJPEG via DALI)
        2. Quad-branch preprocessing (YOLO 640, CLIP 256, SCRFD 640, HD original)
        3. YOLO object detection (parallel with 4, 5)
        4. MobileCLIP global embedding (parallel with 3, 5)
        5. SCRFD face detection + NMS (parallel with 3, 4)
        6. ArcFace face embeddings (depends on 5)

        Args:
            image_bytes: Raw JPEG/PNG bytes

        Returns:
            Dict with YOLO detections, faces, and all embeddings
        """
        client = get_triton_client(self.settings.triton_url)
        result = client.infer_faces_full(image_bytes)

        # Format YOLO detections
        detections = client.format_detections(result)

        orig_h, orig_w = result['orig_shape']

        # Format face detections
        # Note: face_pipeline already returns boxes normalized to [0,1]
        # and landmarks in pixel coordinates (need normalization)
        faces = []
        for i in range(result['num_faces']):
            box = result['face_boxes'][i]
            landmarks = result['face_landmarks'][i]
            score = float(result['face_scores'][i])
            quality = float(result['face_quality'][i]) if len(result['face_quality']) > i else None

            # Box is already normalized from face_pipeline
            norm_box = [float(x) for x in box]

            # Landmarks are in pixel coordinates - normalize to [0,1]
            norm_landmarks = []
            for j in range(0, 10, 2):
                norm_landmarks.append(float(landmarks[j]) / orig_w)
                norm_landmarks.append(float(landmarks[j + 1]) / orig_h)

            faces.append(
                {
                    'box': norm_box,
                    'landmarks': norm_landmarks,
                    'score': score,
                    'quality': quality,
                }
            )

        # Format face embeddings
        face_embeddings = []
        if result['num_faces'] > 0 and len(result['face_embeddings']) > 0:
            face_embeddings = result['face_embeddings'].tolist()

        return {
            # YOLO detections
            'detections': detections,
            'num_detections': len(detections),
            # Face detections and embeddings
            'num_faces': result['num_faces'],
            'faces': faces,
            'face_embeddings': face_embeddings,
            # Global embedding
            'image_embedding': result['image_embedding'],
            'embedding_norm': float(np.linalg.norm(result['image_embedding'])),
            # Image metadata
            'orig_shape': (orig_h, orig_w),
        }

    def infer_unified(self, image_bytes: bytes) -> dict[str, Any]:
        """
        Unified pipeline: YOLO + MobileCLIP + person-only face detection.

        More efficient than infer_faces_full - runs SCRFD only on person
        bounding box crops instead of full 640x640 image.

        Pipeline:
        1. GPU JPEG decode (nvJPEG via DALI)
        2. Triple preprocessing (YOLO 640, CLIP 256, HD original)
        3. YOLO object detection (parallel with 4)
        4. MobileCLIP global embedding (parallel with 3)
        5. Unified extraction (after 2, 3):
           - MobileCLIP per-box embeddings (all boxes)
           - SCRFD face detection (person boxes only)
           - ArcFace face embeddings

        Args:
            image_bytes: Raw JPEG/PNG bytes

        Returns:
            Dict with YOLO detections, embeddings, and face data
        """
        client = get_triton_client(self.settings.triton_url)
        return client.infer_unified(image_bytes)

    # =========================================================================
    # Model Resolution Helpers
    # =========================================================================
    def resolve_model_name(self, model_name: str) -> tuple[str, str, bool]:
        """
        Resolve model name to track and Triton model name.

        Supports both predefined models (in settings) and dynamically uploaded models.

        Returns:
            Tuple of (track, triton_model_name, is_auto_affine)
        """
        models = self.settings.models

        # Store original name for dynamic model fallback
        original_name = model_name

        # Normalize naming for predefined model lookup
        normalized_name = model_name
        if '_trt_end2end' in normalized_name:
            normalized_name = normalized_name.replace('_trt_end2end', '_end2end')
        elif normalized_name.endswith('_trt') and not normalized_name.endswith('_end2end'):
            normalized_name = normalized_name.replace('_trt', '')

        # Check Track D variants
        is_gpu_e2e = any(
            normalized_name.endswith(suffix)
            for suffix in [
                '_gpu_e2e_auto_streaming',
                '_gpu_e2e_auto_batch',
                '_gpu_e2e_auto',
                '_gpu_e2e_streaming',
                '_gpu_e2e_batch',
                '_gpu_e2e',
            ]
        )

        if is_gpu_e2e:
            is_auto = '_auto' in normalized_name
            base = (
                normalized_name.replace('_gpu_e2e_auto_streaming', '')
                .replace('_gpu_e2e_auto_batch', '')
                .replace('_gpu_e2e_auto', '')
                .replace('_gpu_e2e_streaming', '')
                .replace('_gpu_e2e_batch', '')
                .replace('_gpu_e2e', '')
            )

            if '_batch' in normalized_name:
                triton_name = models.GPU_E2E_BATCH_MODELS.get(base)
            elif '_streaming' in normalized_name:
                triton_name = models.GPU_E2E_STREAMING_MODELS.get(base)
            else:
                triton_name = models.GPU_E2E_MODELS.get(base)

            if is_auto and triton_name:
                triton_name = triton_name.replace('_gpu_e2e', '_gpu_e2e_auto')

            return 'D', triton_name, is_auto

        # Check Track C (End2End models)
        if normalized_name.endswith('_end2end') or '_trt_end2end' in original_name:
            base = normalized_name.replace('_end2end', '')
            triton_name = models.END2END_MODELS.get(base)

            # Fallback for dynamically uploaded models not in settings
            if triton_name is None:
                # Use original name directly - it's a dynamically uploaded model
                triton_name = original_name
                logger.info(f'Dynamic model detected: {original_name} -> Track C')

            return 'C', triton_name, False

        # Track B
        model_url = models.STANDARD_MODELS.get(normalized_name)

        # Fallback for dynamically uploaded standard models
        if model_url is None:
            # Use original name as Triton model name
            model_url = f'grpc://{self.settings.triton_url}/{original_name}'
            logger.info(f'Dynamic model detected: {original_name} -> Track B')

        return 'B', model_url, False
