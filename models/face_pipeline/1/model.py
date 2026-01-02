#!/usr/bin/env python3
"""
Triton Python Backend: Complete Face Detection & Recognition Pipeline (GPU-Accelerated)

Unified face pipeline that handles everything in one model via BLS with FULL GPU processing:
1. Calls SCRFD TensorRT for face detection (GPU)
2. Decodes raw outputs on GPU and runs GPU NMS via torchvision
3. Aligns faces on GPU using torch grid_sample
4. Calls ArcFace TensorRT for embedding extraction (GPU)
5. L2-normalizes embeddings on GPU

GPU Processing Pipeline:
- Anchor decoding: PyTorch CUDA operations
- NMS: torchvision.ops.nms (CUDA-accelerated)
- Face alignment: torch.nn.functional.grid_sample (CUDA-accelerated)
- Embedding normalization: PyTorch CUDA

Architecture:
    Inputs:
        - face_images: [3, 640, 640] Preprocessed image for SCRFD (normalized [0, 1])
        - original_image: [3, H, W] Full-resolution image for face cropping
        - orig_shape: [2] Original image shape [H, W]

    Outputs:
        - num_faces: [1] Number of detected faces
        - face_boxes: [128, 4] Face bounding boxes [x1, y1, x2, y2] normalized
        - face_landmarks: [128, 10] 5-point landmarks flattened
        - face_scores: [128] Detection confidence scores
        - face_embeddings: [128, 512] L2-normalized ArcFace embeddings
        - face_quality: [128] Quality scores
"""

import json
import logging

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import triton_python_backend_utils as pb_utils

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# ArcFace Reference Landmarks (112x112) - on GPU
# =============================================================================
ARCFACE_REF_LANDMARKS = torch.tensor([
    [38.2946, 51.6963],   # Left eye center
    [73.5318, 51.5014],   # Right eye center
    [56.0252, 71.7366],   # Nose tip
    [41.5493, 92.3655],   # Left mouth corner
    [70.7299, 92.2041],   # Right mouth corner
], dtype=torch.float32)

ARCFACE_SIZE = 112


def umeyama_torch(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Umeyama algorithm for similarity transform estimation (GPU version)."""
    num = src.shape[0]
    dim = src.shape[1]

    src_mean = src.mean(dim=0)
    dst_mean = dst.mean(dim=0)

    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    A = dst_demean.T @ src_demean / num
    U, S, Vt = torch.linalg.svd(A)

    d = torch.ones(dim, dtype=torch.float64, device=src.device)
    if torch.linalg.det(A) < 0:
        d[dim - 1] = -1

    T = torch.eye(dim + 1, dtype=torch.float64, device=src.device)

    T[:dim, :dim] = U @ torch.diag(d) @ Vt

    src_var = src_demean.var(dim=0).sum()
    scale = 1.0 / src_var * (S @ d) if src_var > 0 else 1.0

    T[:dim, :dim] *= scale
    T[:dim, dim] = dst_mean - scale * (T[:dim, :dim] @ src_mean)

    return T


def get_alignment_matrix_torch(landmarks: torch.Tensor, ref_landmarks: torch.Tensor) -> torch.Tensor:
    """Get 2x3 alignment matrix from 5 landmarks (GPU version)."""
    T = umeyama_torch(landmarks.double(), ref_landmarks.double())
    return T[:2, :].float()


class TritonPythonModel:
    """Complete Face Detection & Recognition Pipeline via BLS (GPU-Accelerated)"""

    def initialize(self, args):
        """Initialize model configuration."""
        self.model_config = json.loads(args['model_config'])

        # GPU device
        self.device = torch.device('cuda:0')

        # SCRFD configuration
        self.scrfd_size = 640
        self.max_faces = 128
        self.conf_threshold = 0.5
        self.nms_threshold = 0.4
        self.strides = [8, 16, 32]

        # ArcFace configuration
        self.arcface_size = ARCFACE_SIZE
        self.embed_dim = 512

        # Move reference landmarks to GPU
        self.ref_landmarks = ARCFACE_REF_LANDMARKS.to(self.device)

        # SCRFD output names
        self.output_names = {
            8: {'scores': '448', 'boxes': '451', 'landmarks': '454'},
            16: {'scores': '471', 'boxes': '474', 'landmarks': '477'},
            32: {'scores': '494', 'boxes': '497', 'landmarks': '500'},
        }

        # Pre-generate anchor centers on GPU
        self._init_anchors_gpu()

        logger.info("Initialized face_pipeline (GPU-accelerated)")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  SCRFD size: {self.scrfd_size}")
        logger.info(f"  ArcFace size: {self.arcface_size}")
        logger.info(f"  Max faces: {self.max_faces}")

    def _init_anchors_gpu(self):
        """Pre-compute anchor centers and strides on GPU."""
        all_centers = []
        all_strides = []
        self.anchors_per_stride = {}

        for stride in self.strides:
            feat_h = self.scrfd_size // stride
            feat_w = self.scrfd_size // stride

            y, x = torch.meshgrid(
                torch.arange(feat_h, dtype=torch.float32),
                torch.arange(feat_w, dtype=torch.float32),
                indexing='ij'
            )

            centers = torch.stack([x, y], dim=-1).reshape(-1, 2)
            centers = (centers + 0.5) * stride
            centers = centers.repeat_interleave(2, dim=0)

            all_centers.append(centers)
            all_strides.append(torch.full((centers.shape[0],), stride, dtype=torch.float32))
            self.anchors_per_stride[stride] = centers.shape[0]

        self.anchor_centers = torch.cat(all_centers, dim=0).to(self.device)
        self.anchor_strides = torch.cat(all_strides, dim=0).to(self.device)
        self.total_anchors = len(self.anchor_centers)

        logger.info(f"  Total anchors: {self.total_anchors}")

    def _call_scrfd(self, images):
        """Call SCRFD TensorRT via BLS with batch support."""
        try:
            if images.ndim == 3:
                images = images[np.newaxis, ...]
            batch_size = images.shape[0]

            input_tensor = pb_utils.Tensor("input.1", images)

            output_names = []
            for stride in self.strides:
                output_names.extend([
                    self.output_names[stride]['scores'],
                    self.output_names[stride]['boxes'],
                    self.output_names[stride]['landmarks'],
                ])

            infer_request = pb_utils.InferenceRequest(
                model_name="scrfd_10g_face_detect",
                requested_output_names=output_names,
                inputs=[input_tensor],
                preferred_memory=pb_utils.PreferredMemory(
                    pb_utils.TRITONSERVER_MEMORY_CPU
                ),
            )

            infer_response = infer_request.exec()

            if infer_response.has_error():
                logger.error(f"SCRFD error: {infer_response.error().message()}")
                return None

            # Extract outputs and convert to GPU tensors
            raw_outputs = {}
            for stride in self.strides:
                raw_outputs[stride] = {
                    'scores': torch.from_numpy(
                        pb_utils.get_output_tensor_by_name(
                            infer_response, self.output_names[stride]['scores']
                        ).as_numpy()
                    ).to(self.device),
                    'boxes': torch.from_numpy(
                        pb_utils.get_output_tensor_by_name(
                            infer_response, self.output_names[stride]['boxes']
                        ).as_numpy()
                    ).to(self.device),
                    'landmarks': torch.from_numpy(
                        pb_utils.get_output_tensor_by_name(
                            infer_response, self.output_names[stride]['landmarks']
                        ).as_numpy()
                    ).to(self.device),
                }

            # Split outputs by batch element
            batch_outputs = []
            for b in range(batch_size):
                outputs = {}
                for stride in self.strides:
                    anchors = self.anchors_per_stride[stride]
                    start_idx = b * anchors
                    end_idx = (b + 1) * anchors
                    outputs[stride] = {
                        'scores': raw_outputs[stride]['scores'][start_idx:end_idx],
                        'boxes': raw_outputs[stride]['boxes'][start_idx:end_idx],
                        'landmarks': raw_outputs[stride]['landmarks'][start_idx:end_idx],
                    }
                batch_outputs.append(outputs)

            return batch_outputs

        except Exception as e:
            logger.error(f"SCRFD BLS call failed: {e}")
            return None

    def _call_arcface(self, faces):
        """Call ArcFace via BLS."""
        try:
            input_tensor = pb_utils.Tensor("input.1", faces)

            infer_request = pb_utils.InferenceRequest(
                model_name="arcface_w600k_r50",
                requested_output_names=["683"],
                inputs=[input_tensor],
                preferred_memory=pb_utils.PreferredMemory(
                    pb_utils.TRITONSERVER_MEMORY_CPU
                ),
            )

            infer_response = infer_request.exec()

            if infer_response.has_error():
                logger.error(f"ArcFace error: {infer_response.error().message()}")
                return None

            output = pb_utils.get_output_tensor_by_name(infer_response, "683")
            return torch.from_numpy(output.as_numpy()).to(self.device)

        except Exception as e:
            logger.error(f"ArcFace BLS call failed: {e}")
            return None

    def _decode_and_nms_gpu(self, scrfd_outputs, orig_h, orig_w):
        """GPU-accelerated anchor decoding and NMS."""
        # Concatenate all strides on GPU
        scores = torch.cat([
            scrfd_outputs[8]['scores'].reshape(-1),
            scrfd_outputs[16]['scores'].reshape(-1),
            scrfd_outputs[32]['scores'].reshape(-1),
        ])

        boxes = torch.cat([
            scrfd_outputs[8]['boxes'].reshape(-1, 4),
            scrfd_outputs[16]['boxes'].reshape(-1, 4),
            scrfd_outputs[32]['boxes'].reshape(-1, 4),
        ])

        landmarks = torch.cat([
            scrfd_outputs[8]['landmarks'].reshape(-1, 10),
            scrfd_outputs[16]['landmarks'].reshape(-1, 10),
            scrfd_outputs[32]['landmarks'].reshape(-1, 10),
        ])

        # GPU: Confidence filter
        conf_mask = scores > self.conf_threshold
        if not conf_mask.any():
            return None, None, None

        scores = scores[conf_mask]
        boxes = boxes[conf_mask]
        landmarks = landmarks[conf_mask]
        anchor_centers = self.anchor_centers[conf_mask]
        anchor_strides = self.anchor_strides[conf_mask]

        # GPU: Decode boxes
        x1 = anchor_centers[:, 0] - boxes[:, 0] * anchor_strides
        y1 = anchor_centers[:, 1] - boxes[:, 1] * anchor_strides
        x2 = anchor_centers[:, 0] + boxes[:, 2] * anchor_strides
        y2 = anchor_centers[:, 1] + boxes[:, 3] * anchor_strides
        decoded_boxes = torch.stack([x1, y1, x2, y2], dim=-1)

        # GPU: Decode landmarks
        lmk = landmarks.reshape(-1, 5, 2)
        decoded_landmarks = (lmk * anchor_strides[:, None, None] + anchor_centers[:, None, :]).reshape(-1, 10)

        # GPU: NMS via torchvision
        keep = torchvision.ops.nms(decoded_boxes, scores, self.nms_threshold)
        if len(keep) > self.max_faces:
            keep = keep[:self.max_faces]

        if len(keep) == 0:
            return None, None, None

        # GPU: Scale to original coordinates
        scale_x = orig_w / self.scrfd_size
        scale_y = orig_h / self.scrfd_size

        final_boxes = decoded_boxes[keep].clone()
        final_boxes[:, 0] *= scale_x
        final_boxes[:, 2] *= scale_x
        final_boxes[:, 1] *= scale_y
        final_boxes[:, 3] *= scale_y

        final_landmarks = decoded_landmarks[keep].clone()
        final_landmarks[:, 0::2] *= scale_x
        final_landmarks[:, 1::2] *= scale_y

        final_scores = scores[keep]

        return final_boxes, final_landmarks, final_scores

    def _align_faces_gpu(self, image_tensor, landmarks_batch, img_h, img_w, orig_h, orig_w):
        """
        GPU-accelerated face alignment using grid_sample.

        Args:
            image_tensor: [3, H, W] GPU tensor (normalized 0-1)
            landmarks_batch: [N, 10] GPU tensor of landmarks in original coords
            img_h, img_w: Dimensions of image_tensor
            orig_h, orig_w: Original image dimensions (for landmark scaling)

        Returns:
            [N, 3, 112, 112] aligned faces on GPU
        """
        n_faces = landmarks_batch.shape[0]

        # Pre-compute base grid once (cached)
        if not hasattr(self, '_base_grid'):
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(0, self.arcface_size - 1, self.arcface_size, device=self.device),
                torch.linspace(0, self.arcface_size - 1, self.arcface_size, device=self.device),
                indexing='ij'
            )
            ones = torch.ones_like(grid_x)
            self._base_grid = torch.stack([grid_x, grid_y, ones], dim=-1).reshape(-1, 3)  # [H*W, 3]

        # Pre-compute normalization factors
        norm_x = 2.0 / (img_w - 1)
        norm_y = 2.0 / (img_h - 1)

        # Add batch dimension to image once
        image_unsqueezed = image_tensor.unsqueeze(0)

        aligned_faces = []
        for i in range(n_faces):
            # Scale landmarks from orig coords to image_tensor coords
            lmk = landmarks_batch[i].reshape(5, 2).clone()
            lmk[:, 0] *= img_w / orig_w
            lmk[:, 1] *= img_h / orig_h

            # Get alignment matrix (on GPU)
            M = get_alignment_matrix_torch(lmk, self.ref_landmarks)

            # Create inverse transform for grid_sample
            M_full = torch.eye(3, dtype=torch.float32, device=self.device)
            M_full[:2, :] = M
            M_inv = torch.linalg.inv(M_full)[:2, :]

            # Apply inverse transform using cached base grid
            src_coords = (M_inv @ self._base_grid.T).T  # [H*W, 2]

            # Normalize to [-1, 1] for grid_sample
            src_coords[:, 0] = src_coords[:, 0] * norm_x - 1
            src_coords[:, 1] = src_coords[:, 1] * norm_y - 1

            grid = src_coords.reshape(1, self.arcface_size, self.arcface_size, 2)

            # Apply grid_sample (CUDA-accelerated)
            aligned = F.grid_sample(
                image_unsqueezed,
                grid,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=True
            )
            aligned_faces.append(aligned.squeeze(0))

        return torch.stack(aligned_faces, dim=0)  # [N, 3, 112, 112]

    def _compute_quality_batch(self, landmarks, boxes, orig_h, orig_w):
        """Compute quality scores for batch of faces (CPU for now)."""
        n_faces = len(boxes)
        quality = np.zeros(n_faces, dtype=np.float32)

        for i in range(n_faces):
            x1, y1, x2, y2 = boxes[i]
            face_area = (x2 - x1) * (y2 - y1)
            image_area = orig_h * orig_w
            size_score = min(1.0, (face_area / max(image_area, 1)) * 10)

            margin = 0.02
            boundary_score = 1.0
            if x1 < orig_w * margin or x2 > orig_w * (1 - margin):
                boundary_score *= 0.8
            if y1 < orig_h * margin or y2 > orig_h * (1 - margin):
                boundary_score *= 0.8

            left_eye = landmarks[i][0:2]
            right_eye = landmarks[i][2:4]
            eye_dist = max(right_eye[0] - left_eye[0], 1)
            eye_tilt = abs(left_eye[1] - right_eye[1]) / eye_dist
            symmetry_score = max(0, 1 - eye_tilt * 2)

            quality[i] = float(np.clip(size_score * boundary_score * symmetry_score, 0, 1))

        return quality

    def execute(self, requests):
        """Process batch of requests with GPU-accelerated pipeline."""
        if len(requests) == 0:
            return []

        # Collect all face images and metadata
        request_data = []
        all_face_images = []

        for request in requests:
            try:
                face_images = pb_utils.get_input_tensor_by_name(request, "face_images").as_numpy()
                original_image = pb_utils.get_input_tensor_by_name(request, "original_image").as_numpy()
                orig_shape = pb_utils.get_input_tensor_by_name(request, "orig_shape").as_numpy().flatten()

                if face_images.ndim == 3:
                    face_images = face_images[np.newaxis, ...]

                request_data.append({
                    'face_images': face_images,
                    'original_image': original_image,
                    'orig_shape': orig_shape,
                })
                all_face_images.append(face_images)
            except Exception as e:
                logger.error(f"Error collecting request data: {e}")
                request_data.append(None)

        # Batched SCRFD call
        valid_indices = [i for i, d in enumerate(request_data) if d is not None]
        if len(valid_indices) > 0:
            stacked_images = np.concatenate([request_data[i]['face_images'] for i in valid_indices], axis=0)
            batch_scrfd_outputs = self._call_scrfd(stacked_images)
        else:
            batch_scrfd_outputs = None

        # Process each request
        responses = []
        scrfd_idx = 0
        for i, request in enumerate(requests):
            try:
                if request_data[i] is None:
                    error = pb_utils.TritonError("Failed to get request data")
                    responses.append(pb_utils.InferenceResponse(error=error))
                    continue

                if batch_scrfd_outputs is not None and scrfd_idx < len(batch_scrfd_outputs):
                    scrfd_outputs = batch_scrfd_outputs[scrfd_idx]
                    scrfd_idx += 1
                else:
                    scrfd_outputs = None

                response = self._process_single_request_gpu(
                    request_data[i]['original_image'],
                    request_data[i]['orig_shape'],
                    scrfd_outputs
                )
                responses.append(response)
            except Exception as e:
                logger.error(f"Error processing request {i}: {e}")
                error = pb_utils.TritonError(str(e))
                responses.append(pb_utils.InferenceResponse(error=error))

        return responses

    def _process_single_request_gpu(self, original_image, orig_shape, scrfd_outputs):
        """Process a single request with GPU-accelerated pipeline."""
        orig_h, orig_w = float(orig_shape[0]), float(orig_shape[1])

        # Initialize outputs on GPU
        face_boxes = torch.zeros((self.max_faces, 4), dtype=torch.float32, device=self.device)
        face_landmarks = torch.zeros((self.max_faces, 10), dtype=torch.float32, device=self.device)
        face_scores = torch.zeros(self.max_faces, dtype=torch.float32, device=self.device)
        face_embeddings = torch.zeros((self.max_faces, self.embed_dim), dtype=torch.float32, device=self.device)
        face_quality = np.zeros(self.max_faces, dtype=np.float32)
        num_faces = np.array([0], dtype=np.int32)

        if scrfd_outputs is None:
            return self._create_response(
                num_faces,
                face_boxes.cpu().numpy(),
                face_landmarks.cpu().numpy(),
                face_scores.cpu().numpy(),
                face_embeddings.cpu().numpy(),
                face_quality
            )

        # GPU: Decode and NMS
        final_boxes, final_landmarks, final_scores = self._decode_and_nms_gpu(
            scrfd_outputs, orig_h, orig_w
        )

        if final_boxes is None:
            return self._create_response(
                num_faces,
                face_boxes.cpu().numpy(),
                face_landmarks.cpu().numpy(),
                face_scores.cpu().numpy(),
                face_embeddings.cpu().numpy(),
                face_quality
            )

        n_faces = len(final_boxes)

        # Prepare original image on GPU
        if original_image.ndim == 4:
            original_image = original_image[0]
        if original_image.shape[0] == 3:
            # Already CHW
            image_tensor = torch.from_numpy(original_image).to(self.device)
        else:
            # HWC -> CHW
            image_tensor = torch.from_numpy(original_image.transpose(2, 0, 1)).to(self.device)

        # Ensure float and proper range
        if image_tensor.dtype != torch.float32:
            image_tensor = image_tensor.float()
        if image_tensor.max() > 1.0:
            image_tensor = image_tensor / 255.0

        img_h, img_w = image_tensor.shape[1], image_tensor.shape[2]

        # GPU: Align faces
        aligned_faces = self._align_faces_gpu(
            image_tensor, final_landmarks, img_h, img_w, orig_h, orig_w
        )

        # GPU: Preprocess for ArcFace (already CHW, just normalize)
        # ArcFace expects: (x - 127.5) / 128.0, input is [0, 1]
        aligned_faces = aligned_faces * 255.0  # Back to [0, 255]
        aligned_faces = (aligned_faces - 127.5) / 128.0

        # Call ArcFace
        embeddings = self._call_arcface(aligned_faces.cpu().numpy().astype(np.float32))

        if embeddings is not None:
            # GPU: L2 normalize
            norms = torch.linalg.norm(embeddings, dim=1, keepdim=True)
            norms = torch.clamp(norms, min=1e-10)
            embeddings = embeddings / norms
            face_embeddings[:n_faces] = embeddings

        # Compute quality (CPU for now - could be GPU)
        face_quality[:n_faces] = self._compute_quality_batch(
            final_landmarks.cpu().numpy(),
            final_boxes.cpu().numpy(),
            orig_h, orig_w
        )

        # Store normalized boxes
        num_faces[0] = n_faces
        face_boxes[:n_faces, 0] = final_boxes[:, 0] / orig_w
        face_boxes[:n_faces, 1] = final_boxes[:, 1] / orig_h
        face_boxes[:n_faces, 2] = final_boxes[:, 2] / orig_w
        face_boxes[:n_faces, 3] = final_boxes[:, 3] / orig_h
        face_landmarks[:n_faces] = final_landmarks
        face_scores[:n_faces] = final_scores

        return self._create_response(
            num_faces,
            face_boxes.cpu().numpy(),
            face_landmarks.cpu().numpy(),
            face_scores.cpu().numpy(),
            face_embeddings.cpu().numpy(),
            face_quality
        )

    def _create_response(self, num_faces, face_boxes, face_landmarks, face_scores, face_embeddings, face_quality):
        """Create inference response."""
        output_tensors = [
            pb_utils.Tensor("num_faces", num_faces),
            pb_utils.Tensor("face_boxes", face_boxes),
            pb_utils.Tensor("face_landmarks", face_landmarks),
            pb_utils.Tensor("face_scores", face_scores),
            pb_utils.Tensor("face_embeddings", face_embeddings),
            pb_utils.Tensor("face_quality", face_quality),
        ]
        return pb_utils.InferenceResponse(output_tensors=output_tensors)

    def finalize(self):
        """Cleanup."""
        logger.info("Finalizing face_pipeline (GPU)")
