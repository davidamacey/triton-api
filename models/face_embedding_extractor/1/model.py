#!/usr/bin/env python3
"""
Triton Python Backend: Face Embedding Extractor

Production-grade face embedding extraction using techniques from:
- InsightFace (Alibaba): Umeyama alignment, ArcFace embeddings
- DeepFace (Meta): Similarity transform, L2 normalization
- FaceNet (Google): Centered face alignment

Architecture:
    Inputs:
        - original_image: [3, H, W] Full-resolution decoded image
        - face_boxes: [N, 4] Face bounding boxes [x1, y1, x2, y2] in pixel coords
        - face_landmarks: [N, 10] Facial landmarks (5 points × 2 coords)
        - num_faces: [1] Number of valid faces

    Processing:
        1. Extract valid faces from original image
        2. Align each face using 5-point landmarks (Umeyama algorithm)
        3. Preprocess for ArcFace: (x - 127.5) / 128.0
        4. Batch call ArcFace via BLS
        5. L2-normalize embeddings (critical for cosine similarity)
        6. Pad output to MAX_FACES

    Outputs:
        - face_embeddings: [MAX_FACES, 512] L2-normalized ArcFace embeddings
        - aligned_boxes: [MAX_FACES, 4] Normalized face boxes [0, 1]
        - face_quality: [MAX_FACES] Quality scores for each face
"""

import json
import logging

import numpy as np
import torch
import torch.nn.functional as F
import triton_python_backend_utils as pb_utils

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# ArcFace Reference Landmarks (112x112)
# =============================================================================
# Standard landmarks used by InsightFace, DeepFace, and other production systems
ARCFACE_REF_LANDMARKS = np.array([
    [38.2946, 51.6963],   # Left eye center
    [73.5318, 51.5014],   # Right eye center
    [56.0252, 71.7366],   # Nose tip
    [41.5493, 92.3655],   # Left mouth corner
    [70.7299, 92.2041],   # Right mouth corner
], dtype=np.float32)

ARCFACE_SIZE = 112


# =============================================================================
# Umeyama Algorithm (Industry Standard)
# =============================================================================

def umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """
    Umeyama algorithm for similarity transform estimation.

    Industry standard used by InsightFace, DeepFace, and FaceNet.
    """
    num = src.shape[0]
    dim = src.shape[1]

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    A = dst_demean.T @ src_demean / num
    U, S, Vt = np.linalg.svd(A)

    d = np.ones(dim, dtype=np.float64)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1

    T = np.eye(dim + 1, dtype=np.float64)

    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.nan * T

    if rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(Vt) > 0:
            T[:dim, :dim] = U @ Vt
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = U @ np.diag(d) @ Vt
            d[dim - 1] = s
    else:
        T[:dim, :dim] = U @ np.diag(d) @ Vt

    src_var = src_demean.var(axis=0).sum()
    scale = 1.0 / src_var * (S @ d) if src_var > 0 else 1.0

    T[:dim, :dim] *= scale
    T[:dim, dim] = dst_mean - scale * (T[:dim, :dim] @ src_mean)

    return T


def get_alignment_matrix(landmarks: np.ndarray) -> np.ndarray:
    """Get 2x3 alignment matrix from 5 landmarks."""
    T = umeyama(landmarks.astype(np.float64), ARCFACE_REF_LANDMARKS.astype(np.float64))
    return T[:2, :].astype(np.float32)


# =============================================================================
# Triton Python Model
# =============================================================================

class TritonPythonModel:
    """Triton Python Backend for Face Embedding Extraction"""

    def initialize(self, args):
        """Initialize model configuration."""
        self.model_config = json.loads(args['model_config'])

        # Configuration
        self.max_faces = 128  # Maximum faces per image
        self.embed_dim = 512  # ArcFace embedding dimension
        self.arcface_size = ARCFACE_SIZE  # 112x112

        # Device setup
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        logger.info(f"Initialized face_embedding_extractor on {self.device}")
        logger.info(f"  Max faces: {self.max_faces}")
        logger.info(f"  Embedding dim: {self.embed_dim}")
        logger.info(f"  ArcFace size: {self.arcface_size}")

        # Warmup
        if self.device == 'cuda':
            self._warmup_gpu()

    def _warmup_gpu(self):
        """Warmup GPU with dummy operations."""
        try:
            dummy = torch.randn(1, 3, 112, 112, device=self.device)
            _ = F.interpolate(dummy, size=(112, 112), mode='bilinear')
            logger.info("GPU warmup complete")
        except Exception as e:
            logger.warning(f"GPU warmup failed: {e}")

    def _align_face_cv2(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """
        Align single face using OpenCV warpAffine.

        Args:
            image: [H, W, 3] RGB image (uint8 or float32)
            landmarks: [5, 2] facial landmarks

        Returns:
            aligned: [112, 112, 3] aligned face
        """
        import cv2

        M = get_alignment_matrix(landmarks)

        aligned = cv2.warpAffine(
            image,
            M,
            (self.arcface_size, self.arcface_size),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        return aligned

    def _preprocess_for_arcface(self, faces: np.ndarray) -> np.ndarray:
        """
        Preprocess aligned faces for ArcFace.

        ArcFace expects: [B, 3, 112, 112] FP32, normalized as (x - 127.5) / 128.0

        Args:
            faces: [N, 112, 112, 3] aligned faces (uint8 or float32)

        Returns:
            tensor: [N, 3, 112, 112] FP32 preprocessed
        """
        # Ensure float
        if faces.dtype == np.uint8:
            faces = faces.astype(np.float32)

        # Normalize
        faces = (faces - 127.5) / 128.0

        # NHWC -> NCHW
        faces = np.transpose(faces, (0, 3, 1, 2))

        return faces.astype(np.float32)

    def _compute_face_quality(
        self,
        landmarks: np.ndarray,
        box: np.ndarray,
        image_size: tuple,
    ) -> float:
        """Compute face quality score based on size, position, and symmetry."""
        h, w = image_size
        x1, y1, x2, y2 = box

        # Size score (larger faces are better)
        face_area = (x2 - x1) * (y2 - y1)
        image_area = h * w
        size_ratio = face_area / max(image_area, 1)
        size_score = min(1.0, size_ratio * 10)

        # Boundary score
        margin = 0.02
        boundary_score = 1.0
        if x1 < w * margin or x2 > w * (1 - margin):
            boundary_score *= 0.8
        if y1 < h * margin or y2 > h * (1 - margin):
            boundary_score *= 0.8

        # Symmetry score (eyes should be level)
        left_eye, right_eye = landmarks[0], landmarks[1]
        eye_dist = max(right_eye[0] - left_eye[0], 1)
        eye_tilt = abs(left_eye[1] - right_eye[1]) / eye_dist
        symmetry_score = max(0, 1 - eye_tilt * 2)

        quality = size_score * boundary_score * symmetry_score
        return float(np.clip(quality, 0, 1))

    def execute(self, requests):
        """Process batch of requests."""
        responses = []

        for request in requests:
            try:
                response = self._process_single_request(request)
                responses.append(response)
            except Exception as e:
                logger.error(f"Error processing request: {e}")
                error = pb_utils.TritonError(str(e))
                responses.append(pb_utils.InferenceResponse(error=error))

        return responses

    def _process_single_request(self, request):
        """Process a single inference request."""
        # Get inputs
        original_image = pb_utils.get_input_tensor_by_name(request, "original_image")
        face_boxes = pb_utils.get_input_tensor_by_name(request, "face_boxes")
        face_landmarks = pb_utils.get_input_tensor_by_name(request, "face_landmarks")
        num_faces = pb_utils.get_input_tensor_by_name(request, "num_faces")

        # Convert to numpy
        image_np = original_image.as_numpy()  # [3, H, W] or [H, W, 3]
        boxes_np = face_boxes.as_numpy()  # [N, 4]
        landmarks_np = face_landmarks.as_numpy()  # [N, 10]
        n_faces = int(num_faces.as_numpy().flatten()[0])

        # Handle image format
        if image_np.shape[0] == 3:
            # CHW -> HWC
            image_np = np.transpose(image_np, (1, 2, 0))

        # Convert to uint8 if normalized
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image_np.astype(np.uint8)

        h, w = image_np.shape[:2]

        # Initialize outputs
        face_embeddings = np.zeros((self.max_faces, self.embed_dim), dtype=np.float32)
        aligned_boxes = np.zeros((self.max_faces, 4), dtype=np.float32)
        face_quality = np.zeros(self.max_faces, dtype=np.float32)

        if n_faces == 0:
            return self._create_response(face_embeddings, aligned_boxes, face_quality)

        # Process each face
        n_valid = min(n_faces, self.max_faces, len(boxes_np))
        aligned_faces = []

        for i in range(n_valid):
            box = boxes_np[i]
            landmarks_flat = landmarks_np[i]  # [10]
            landmarks = landmarks_flat.reshape(5, 2)

            # Align face
            try:
                aligned = self._align_face_cv2(image_np, landmarks)
                aligned_faces.append(aligned)

                # Store normalized box
                aligned_boxes[i] = [
                    box[0] / w,
                    box[1] / h,
                    box[2] / w,
                    box[3] / h,
                ]

                # Compute quality
                face_quality[i] = self._compute_face_quality(landmarks, box, (h, w))

            except Exception as e:
                logger.warning(f"Failed to align face {i}: {e}")
                aligned_faces.append(np.zeros((112, 112, 3), dtype=np.uint8))

        if not aligned_faces:
            return self._create_response(face_embeddings, aligned_boxes, face_quality)

        # Stack and preprocess
        faces_batch = np.stack(aligned_faces, axis=0)
        faces_preprocessed = self._preprocess_for_arcface(faces_batch)

        # Call ArcFace via BLS
        embeddings = self._call_arcface(faces_preprocessed)

        if embeddings is not None:
            # L2 normalize (critical for cosine similarity)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)  # Avoid division by zero
            embeddings = embeddings / norms

            # Store embeddings
            face_embeddings[:n_valid] = embeddings

        return self._create_response(face_embeddings, aligned_boxes, face_quality)

    def _call_arcface(self, faces: np.ndarray) -> np.ndarray:
        """
        Call ArcFace model via BLS.

        Args:
            faces: [N, 3, 112, 112] preprocessed faces

        Returns:
            embeddings: [N, 512] face embeddings
        """
        try:
            # Create input tensor
            input_tensor = pb_utils.Tensor("input.1", faces)

            # Create inference request
            infer_request = pb_utils.InferenceRequest(
                model_name="arcface_w600k_r50",
                requested_output_names=["683"],
                inputs=[input_tensor],
            )

            # Execute
            infer_response = infer_request.exec()

            if infer_response.has_error():
                logger.error(f"ArcFace error: {infer_response.error().message()}")
                return None

            # Get output
            output = pb_utils.get_output_tensor_by_name(infer_response, "683")
            embeddings = output.as_numpy()

            return embeddings

        except Exception as e:
            logger.error(f"ArcFace BLS call failed: {e}")
            return None

    def _create_response(
        self,
        embeddings: np.ndarray,
        boxes: np.ndarray,
        quality: np.ndarray,
    ):
        """Create inference response."""
        output_tensors = [
            pb_utils.Tensor("face_embeddings", embeddings),
            pb_utils.Tensor("aligned_boxes", boxes),
            pb_utils.Tensor("face_quality", quality),
        ]
        return pb_utils.InferenceResponse(output_tensors=output_tensors)

    def finalize(self):
        """Cleanup."""
        logger.info("Finalizing face_embedding_extractor")
