#!/usr/bin/env python3
"""
Triton Python Backend: YOLO11-face Detection & Recognition Pipeline (GPU-Accelerated)

Complete face pipeline using YOLO11-face as alternative to SCRFD:
1. Calls YOLO11-face TensorRT for face detection
2. Runs GPU NMS via torchvision
3. Crops faces using MTCNN/FaceNet-style margin expansion (from HD original!)
4. Calls ArcFace TensorRT for embedding extraction
5. L2-normalizes embeddings on GPU

FACE CROPPING APPROACH (Industry Standard - MTCNN/FaceNet):
============================================================
Since YOLO11-face (YapaLab) is detection-only (no landmarks), we use
MTCNN/FaceNet-style face cropping which is the industry standard for
detection-only models:

1. Get bounding box from YOLO11-face detection
2. Expand box with margin (40% default, like MTCNN)
3. Make square (preserves aspect ratio for recognition)
4. Crop from HD ORIGINAL image (not 640x640 detection input!)
5. Resize to 112x112 for ArcFace

This ensures embeddings are extracted from high-definition face crops,
matching the quality of landmark-based alignment used by SCRFD pipeline.

Reference:
- MTCNN: Joint Face Detection and Alignment using Multi-task CNNs (Zhang et al.)
- FaceNet: A Unified Embedding for Face Recognition (Schroff et al.)

Architecture:
    Inputs:
        - face_images: [3, 640, 640] Preprocessed image for YOLO11-face (normalized [0, 1])
        - original_image: [3, H, W] Full-resolution image for face cropping
        - orig_shape: [2] Original image shape [H, W]

    Outputs:
        - num_faces: [1] Number of detected faces
        - face_boxes: [128, 4] Face bounding boxes [x1, y1, x2, y2] normalized
        - face_landmarks: [128, 10] 5-point landmarks flattened (zeros for YOLO11-face)
        - face_scores: [128] Detection confidence scores
        - face_embeddings: [128, 512] L2-normalized ArcFace embeddings
        - face_quality: [128] Quality scores
"""

import json
import logging
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import triton_python_backend_utils as pb_utils

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Face Cropping Constants (MTCNN/FaceNet Style)
# =============================================================================

# Output size for ArcFace recognition
ARCFACE_SIZE = 112

# MTCNN-style margin factor: expand bounding box by this factor on each side
# MTCNN uses ~0.4 (40%), FaceNet uses ~0.44 (44%)
# This ensures we capture the full face including some context
FACE_MARGIN_FACTOR = 0.4


class TritonPythonModel:
    """YOLO11-face Detection & Recognition Pipeline via BLS (GPU-Accelerated)"""

    def initialize(self, args):
        """Initialize model configuration."""
        self.model_config = json.loads(args['model_config'])

        # GPU device
        self.device = torch.device('cuda:0')

        # YOLO11-face configuration
        self.yolo_size = 640
        self.max_faces = 128
        self.conf_threshold = 0.5
        self.nms_threshold = 0.4

        # ArcFace configuration
        self.arcface_size = ARCFACE_SIZE
        self.embed_dim = 512

        # MTCNN-style face margin (40% expansion on each side)
        self.face_margin = FACE_MARGIN_FACTOR

        # End2End mode: Use TensorRT EfficientNMS (faster, no Python NMS needed)
        # Set to True to use yolo11_face_small_trt_end2end model
        # Configurable via YOLO11_FACE_END2END environment variable (default: true)
        self.use_end2end = os.environ.get('YOLO11_FACE_END2END', 'true').lower() in ('true', '1', 'yes')

        # YOLO11-face model name (auto-select based on End2End mode)
        if self.use_end2end:
            self.yolo_model_name = "yolo11_face_small_trt_end2end"
        else:
            self.yolo_model_name = "yolo11_face_small_trt"

        logger.info("Initialized yolo11_face_pipeline (GPU-accelerated)")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  YOLO11-face size: {self.yolo_size}")
        logger.info(f"  YOLO11-face model: {self.yolo_model_name}")
        logger.info(f"  End2End (GPU NMS): {self.use_end2end}")
        logger.info(f"  ArcFace size: {self.arcface_size}")
        logger.info(f"  Face margin: {self.face_margin:.0%}")
        logger.info(f"  Max faces: {self.max_faces}")

    def _call_yolo11_face(self, images):
        """Call YOLO11-face TensorRT via BLS (standard or End2End)."""
        if self.use_end2end:
            return self._call_yolo11_face_end2end(images)
        else:
            return self._call_yolo11_face_standard(images)

    def _call_yolo11_face_standard(self, images):
        """Call standard YOLO11-face TensorRT via BLS (raw output, needs Python NMS)."""
        try:
            if images.ndim == 3:
                images = images[np.newaxis, ...]

            input_tensor = pb_utils.Tensor("images", images)

            infer_request = pb_utils.InferenceRequest(
                model_name=self.yolo_model_name,
                requested_output_names=["output0"],
                inputs=[input_tensor],
                preferred_memory=pb_utils.PreferredMemory(
                    pb_utils.TRITONSERVER_MEMORY_CPU
                ),
            )

            infer_response = infer_request.exec()

            if infer_response.has_error():
                logger.error(f"YOLO11-face error: {infer_response.error().message()}")
                return None

            output = pb_utils.get_output_tensor_by_name(infer_response, "output0")
            raw_output = output.as_numpy()

            # Convert to torch tensor on GPU
            return torch.from_numpy(raw_output).to(self.device)

        except Exception as e:
            logger.error(f"YOLO11-face BLS call failed: {e}")
            return None

    def _call_yolo11_face_end2end(self, images):
        """
        Call YOLO11-face End2End TensorRT via BLS (post-NMS output).

        End2End output format:
        - num_dets: [batch, 1] - number of detections per image
        - det_boxes: [batch, max_det, 4] - boxes in [0,1] normalized xyxy
        - det_scores: [batch, max_det] - confidence scores
        - det_classes: [batch, max_det] - class IDs (all 0 for face)

        Returns:
            Dict with 'num_dets', 'boxes', 'scores' tensors on GPU
        """
        try:
            if images.ndim == 3:
                images = images[np.newaxis, ...]

            input_tensor = pb_utils.Tensor("images", images)

            infer_request = pb_utils.InferenceRequest(
                model_name=self.yolo_model_name,
                requested_output_names=["num_dets", "det_boxes", "det_scores", "det_classes"],
                inputs=[input_tensor],
                preferred_memory=pb_utils.PreferredMemory(
                    pb_utils.TRITONSERVER_MEMORY_CPU
                ),
            )

            infer_response = infer_request.exec()

            if infer_response.has_error():
                logger.error(f"YOLO11-face End2End error: {infer_response.error().message()}")
                return None

            # Parse End2End outputs
            num_dets = pb_utils.get_output_tensor_by_name(infer_response, "num_dets").as_numpy()
            det_boxes = pb_utils.get_output_tensor_by_name(infer_response, "det_boxes").as_numpy()
            det_scores = pb_utils.get_output_tensor_by_name(infer_response, "det_scores").as_numpy()

            # Return as dict of GPU tensors
            return {
                'num_dets': torch.from_numpy(num_dets).to(self.device),
                'boxes': torch.from_numpy(det_boxes).to(self.device),
                'scores': torch.from_numpy(det_scores).to(self.device),
            }

        except Exception as e:
            logger.error(f"YOLO11-face End2End BLS call failed: {e}")
            import traceback
            traceback.print_exc()
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

    def _decode_end2end_output(self, output, orig_h, orig_w, affine_matrix=None):
        """
        Decode End2End YOLO11-face output to boxes and scores.

        End2End output format (already post-NMS):
        - num_dets: [batch, 1] - number of valid detections
        - boxes: [batch, max_det, 4] - boxes in normalized [0,1] xyxy format
        - scores: [batch, max_det] - confidence scores

        IMPORTANT: Boxes are normalized to the 640x640 letterbox space, NOT the original
        image space. We need to apply inverse letterbox transformation.

        Args:
            output: Dict with 'num_dets', 'boxes', 'scores' tensors
            orig_h, orig_w: Original image dimensions
            affine_matrix: Optional [2, 3] DALI letterbox affine matrix
                          Format: [[scale, 0, pad_x], [0, scale, pad_y]]

        Returns:
            (boxes, scores) - boxes in original pixel coordinates, or (None, None) if no detections
        """
        num_dets = int(output['num_dets'][0, 0].item())  # First batch, first element

        if num_dets == 0:
            return None, None

        # Clamp to max faces
        num_dets = min(num_dets, self.max_faces)

        # Get valid detections (first num_dets)
        boxes_norm = output['boxes'][0, :num_dets]  # [num_dets, 4] in [0,1]
        scores = output['scores'][0, :num_dets]      # [num_dets]

        # Step 1: Convert from normalized [0,1] to 640x640 letterbox pixel coordinates
        boxes_640 = boxes_norm.clone()
        boxes_640[:, 0] *= self.yolo_size  # x1
        boxes_640[:, 1] *= self.yolo_size  # y1
        boxes_640[:, 2] *= self.yolo_size  # x2
        boxes_640[:, 3] *= self.yolo_size  # y2

        # Step 2: Apply inverse letterbox transformation to get original image coordinates
        if affine_matrix is not None:
            # DALI letterbox: affine_matrix = [[scale, 0, pad_x], [0, scale, pad_y]]
            if isinstance(affine_matrix, np.ndarray):
                affine_matrix = torch.from_numpy(affine_matrix).to(boxes_640.device)
            scale = float(affine_matrix[0, 0])
            pad_x = float(affine_matrix[0, 2])
            pad_y = float(affine_matrix[1, 2])
        else:
            # No affine matrix provided - compute letterbox parameters
            # Standard letterbox: scale to fit, center with padding
            scale = min(self.yolo_size / orig_h, self.yolo_size / orig_w)
            new_w = orig_w * scale
            new_h = orig_h * scale
            pad_x = (self.yolo_size - new_w) / 2
            pad_y = (self.yolo_size - new_h) / 2

        # Inverse letterbox: orig_coord = (letterbox_coord - pad) / scale
        boxes_pixel = boxes_640.clone()
        boxes_pixel[:, 0] = (boxes_640[:, 0] - pad_x) / scale  # x1
        boxes_pixel[:, 1] = (boxes_640[:, 1] - pad_y) / scale  # y1
        boxes_pixel[:, 2] = (boxes_640[:, 2] - pad_x) / scale  # x2
        boxes_pixel[:, 3] = (boxes_640[:, 3] - pad_y) / scale  # y2

        # Clamp to valid image bounds
        boxes_pixel[:, 0].clamp_(min=0, max=orig_w)
        boxes_pixel[:, 1].clamp_(min=0, max=orig_h)
        boxes_pixel[:, 2].clamp_(min=0, max=orig_w)
        boxes_pixel[:, 3].clamp_(min=0, max=orig_h)

        return boxes_pixel, scores

    def _decode_yolo11_face_output(self, output, orig_h, orig_w, affine_matrix=None):
        """
        Decode YOLO11-face output to boxes, scores, and landmarks.

        YOLO11-face (pose model) output format:
        - Shape: [batch, 21, num_predictions] or [batch, num_predictions, 21]
        - 21 = 4 (x,y,w,h) + 1 (obj_conf) + 1 (cls_conf for face) + 15 (5 keypoints * 3: x,y,conf)

        Args:
            output: YOLO11-face raw output tensor
            orig_h, orig_w: Original image dimensions
            affine_matrix: Optional [2, 3] affine transformation matrix from DALI letterbox
                          Format: [[scale, 0, pad_x], [0, scale, pad_y]]
                          If provided, uses proper inverse letterbox transformation.
                          If None, assumes image was scaled to fill 640x640 (CPU LetterBox behavior).

        Returns boxes in original image coordinates, landmarks in original coordinates.
        """
        # Handle shape variations
        if output.ndim == 2:
            output = output.unsqueeze(0)

        batch_size = output.shape[0]

        # Check shape and transpose if needed
        # Expected: [batch, num_predictions, 21] where 21 is the feature dim
        if output.shape[1] < output.shape[2]:
            # Shape is [batch, 21, num_predictions], transpose
            output = output.transpose(1, 2)

        # Now output is [batch, num_predictions, 21]
        # Extract components
        # Boxes: x_center, y_center, width, height (first 4)
        # Confidence: obj_conf (index 4)
        # Class conf: index 5 (for face class)
        # Keypoints: indices 6-20 (5 keypoints * 3 values each)

        boxes_xywh = output[..., :4]  # [batch, num_preds, 4]
        obj_conf = output[..., 4]     # [batch, num_preds]

        # For YOLO11, confidence might be combined or separate
        # Try to handle both cases
        if output.shape[-1] > 5:
            cls_conf = output[..., 5] if output.shape[-1] > 5 else torch.ones_like(obj_conf)
            scores = obj_conf * cls_conf
        else:
            scores = obj_conf

        # Keypoints: [x, y, visibility] * 5
        # Indices 6 onwards (or 5 if no separate class conf)
        kpt_start = 6 if output.shape[-1] > 6 else 5
        if output.shape[-1] >= kpt_start + 15:
            keypoints = output[..., kpt_start:kpt_start + 15]  # [batch, num_preds, 15]
        else:
            # No keypoints in output, create dummy
            keypoints = torch.zeros(batch_size, output.shape[1], 15, device=output.device)

        # Process each batch element
        all_boxes = []
        all_scores = []
        all_landmarks = []

        for b in range(batch_size):
            # Filter by confidence
            mask = scores[b] > self.conf_threshold
            if not mask.any():
                all_boxes.append(None)
                all_scores.append(None)
                all_landmarks.append(None)
                continue

            b_boxes_xywh = boxes_xywh[b, mask]  # [N, 4]
            b_scores = scores[b, mask]          # [N]
            b_keypoints = keypoints[b, mask]   # [N, 15]

            # Convert xywh to xyxy
            x_center = b_boxes_xywh[:, 0]
            y_center = b_boxes_xywh[:, 1]
            w = b_boxes_xywh[:, 2]
            h = b_boxes_xywh[:, 3]

            x1 = x_center - w / 2
            y1 = y_center - h / 2
            x2 = x_center + w / 2
            y2 = y_center + h / 2

            b_boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)

            # Scale to original image coordinates using inverse letterbox transformation
            if affine_matrix is not None:
                # DALI letterbox: affine_matrix = [[scale, 0, pad_x], [0, scale, pad_y]]
                # Inverse: orig_coord = (yolo_coord - pad) / scale
                if isinstance(affine_matrix, np.ndarray):
                    affine_matrix = torch.from_numpy(affine_matrix).to(b_boxes_xyxy.device)
                scale = float(affine_matrix[0, 0])
                pad_x = float(affine_matrix[0, 2])
                pad_y = float(affine_matrix[1, 2])

                # Apply inverse letterbox transformation
                b_boxes_xyxy[:, 0] = (b_boxes_xyxy[:, 0] - pad_x) / scale  # x1
                b_boxes_xyxy[:, 2] = (b_boxes_xyxy[:, 2] - pad_x) / scale  # x2
                b_boxes_xyxy[:, 1] = (b_boxes_xyxy[:, 1] - pad_y) / scale  # y1
                b_boxes_xyxy[:, 3] = (b_boxes_xyxy[:, 3] - pad_y) / scale  # y2

                # Scale for landmarks
                lmk_scale_x = 1.0 / scale
                lmk_scale_y = 1.0 / scale
                lmk_offset_x = pad_x
                lmk_offset_y = pad_y
            else:
                # CPU LetterBox: assumes image was scaled to fill 640x640
                # Simple scale: orig_coord = yolo_coord * (orig_size / 640)
                scale_x = orig_w / self.yolo_size
                scale_y = orig_h / self.yolo_size

                b_boxes_xyxy[:, 0] *= scale_x
                b_boxes_xyxy[:, 2] *= scale_x
                b_boxes_xyxy[:, 1] *= scale_y
                b_boxes_xyxy[:, 3] *= scale_y

                # Scale for landmarks
                lmk_scale_x = scale_x
                lmk_scale_y = scale_y
                lmk_offset_x = 0.0
                lmk_offset_y = 0.0

            # Extract landmarks [N, 10] (x,y pairs only, drop visibility)
            # From [N, 15] where format is [x1,y1,v1, x2,y2,v2, ...]
            b_landmarks = torch.zeros(b_keypoints.shape[0], 10, device=b_keypoints.device)
            for i in range(5):
                if affine_matrix is not None:
                    b_landmarks[:, i*2] = (b_keypoints[:, i*3] - lmk_offset_x) * lmk_scale_x      # x
                    b_landmarks[:, i*2 + 1] = (b_keypoints[:, i*3 + 1] - lmk_offset_y) * lmk_scale_y  # y
                else:
                    b_landmarks[:, i*2] = b_keypoints[:, i*3] * lmk_scale_x      # x
                    b_landmarks[:, i*2 + 1] = b_keypoints[:, i*3 + 1] * lmk_scale_y  # y

            # Apply NMS
            keep = torchvision.ops.nms(b_boxes_xyxy, b_scores, self.nms_threshold)
            if len(keep) > self.max_faces:
                keep = keep[:self.max_faces]

            if len(keep) == 0:
                all_boxes.append(None)
                all_scores.append(None)
                all_landmarks.append(None)
            else:
                all_boxes.append(b_boxes_xyxy[keep])
                all_scores.append(b_scores[keep])
                all_landmarks.append(b_landmarks[keep])

        return all_boxes, all_scores, all_landmarks

    def _crop_faces_mtcnn_style(self, image_tensor, boxes, img_h, img_w, orig_h, orig_w):
        """
        Face cropping with ArcFace-compatible alignment using GPU grid_sample.

        For detection-only models (no landmarks), we use a center-based alignment
        that maps the face center to the appropriate position in the 112x112 output.

        Algorithm:
        1. Calculate face center from bounding box
        2. Determine scale factor (face width -> ~70 pixels in 112x112 output)
        3. Map face center to reference center position in output
        4. Apply affine transform using GPU grid_sample

        The reference position is based on ArcFace training data:
        - Face center (between eyes and nose) should be around (56, 60) in 112x112
        - Face width should span approximately 70% of the output (78 pixels)

        Args:
            image_tensor: [3, H, W] GPU tensor (normalized 0-1) - HD original
            boxes: [N, 4] GPU tensor of boxes [x1, y1, x2, y2] in original coords
            img_h, img_w: Dimensions of image_tensor
            orig_h, orig_w: Original image dimensions (for box scaling)

        Returns:
            [N, 3, 112, 112] cropped and aligned faces on GPU
        """
        n_faces = boxes.shape[0]

        # ArcFace alignment parameters (based on reference landmarks)
        # The face should be centered around (56, 64) in 112x112 output
        # with face width spanning about 70% of output (35 pixels on each side)
        REF_CENTER_X = 56.0  # Horizontal center of face in output
        REF_CENTER_Y = 64.0  # Vertical center (slightly below eye level)
        REF_FACE_WIDTH = 78.0  # Face should span ~70% of 112 pixels

        # Add batch dimension to image once
        image_unsqueezed = image_tensor.unsqueeze(0)

        cropped_faces = []
        for i in range(n_faces):
            # Get box in original coordinates
            x1, y1, x2, y2 = boxes[i].tolist()

            # Calculate face dimensions and center
            face_w = x2 - x1
            face_h = y2 - y1
            face_cx = (x1 + x2) / 2.0
            # Face center is slightly above box center (nose/mouth area is lower)
            # Use ~45% down from top of box as face center
            face_cy = y1 + face_h * 0.45

            # Calculate scale: map detected face width to reference width
            scale = REF_FACE_WIDTH / max(face_w, 1.0)

            # Scale from original coords to image_tensor coords
            scale_img_x = img_w / orig_w
            scale_img_y = img_h / orig_h
            face_cx_scaled = face_cx * scale_img_x
            face_cy_scaled = face_cy * scale_img_y

            # Combined scale (for mapping output pixels to input pixels)
            # We use the geometric mean to handle non-square images
            scale_combined = scale / math.sqrt(scale_img_x * scale_img_y)

            # Inverse scale for grid_sample (output -> input)
            scale_inv = 1.0 / scale_combined

            # Generate output grid [112, 112]
            grid_y, grid_x = torch.meshgrid(
                torch.arange(self.arcface_size, dtype=torch.float32, device=self.device),
                torch.arange(self.arcface_size, dtype=torch.float32, device=self.device),
                indexing='ij'
            )

            # Apply inverse affine transform
            # Center the grid on reference face center
            centered_x = grid_x - REF_CENTER_X
            centered_y = grid_y - REF_CENTER_Y

            # Scale and translate to input coordinates
            src_x = scale_inv * centered_x + face_cx_scaled
            src_y = scale_inv * centered_y + face_cy_scaled

            # Normalize to [-1, 1] for grid_sample
            norm_x = 2.0 * src_x / (img_w - 1) - 1
            norm_y = 2.0 * src_y / (img_h - 1) - 1

            # Stack and reshape for grid_sample: [1, H, W, 2]
            grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(0)

            # Apply grid_sample (CUDA-accelerated bilinear interpolation)
            cropped = F.grid_sample(
                image_unsqueezed,
                grid,
                mode='bilinear',
                padding_mode='zeros',  # Black border for out-of-bounds
                align_corners=True
            )
            cropped_faces.append(cropped.squeeze(0))

        return torch.stack(cropped_faces, dim=0)  # [N, 3, 112, 112]

    def _compute_quality_batch(self, boxes, orig_h, orig_w):
        """
        Compute quality scores for batch of faces (box-based, no landmarks).

        Quality factors:
        - Face size relative to image (larger = better)
        - Face position (centered = better)
        - Box aspect ratio (closer to 1:1 = better)
        """
        n_faces = len(boxes)
        quality = np.zeros(n_faces, dtype=np.float32)

        for i in range(n_faces):
            x1, y1, x2, y2 = boxes[i]
            face_w = x2 - x1
            face_h = y2 - y1

            # Size score: larger faces are better (up to 10% of image)
            face_area = face_w * face_h
            image_area = orig_h * orig_w
            size_score = min(1.0, (face_area / max(image_area, 1)) * 10)

            # Boundary score: penalize faces at image edge
            margin = 0.02
            boundary_score = 1.0
            if x1 < orig_w * margin or x2 > orig_w * (1 - margin):
                boundary_score *= 0.8
            if y1 < orig_h * margin or y2 > orig_h * (1 - margin):
                boundary_score *= 0.8

            # Aspect ratio score: square faces are better for recognition
            aspect_ratio = min(face_w, face_h) / max(face_w, face_h, 1)
            aspect_score = aspect_ratio  # 1.0 for square, less for elongated

            quality[i] = float(np.clip(size_score * boundary_score * aspect_score, 0, 1))

        return quality

    def execute(self, requests):
        """Process batch of requests with GPU-accelerated pipeline."""
        if len(requests) == 0:
            return []

        responses = []

        for request in requests:
            try:
                # Get inputs
                face_images = pb_utils.get_input_tensor_by_name(request, "face_images").as_numpy()
                original_image = pb_utils.get_input_tensor_by_name(request, "original_image").as_numpy()
                orig_shape = pb_utils.get_input_tensor_by_name(request, "orig_shape").as_numpy().flatten()

                # Get optional affine_matrix (for DALI letterbox transformation)
                affine_tensor = pb_utils.get_input_tensor_by_name(request, "affine_matrix")
                affine_matrix = affine_tensor.as_numpy() if affine_tensor is not None else None
                if affine_matrix is not None and affine_matrix.ndim == 3:
                    affine_matrix = affine_matrix[0]  # Remove batch dimension

                if face_images.ndim == 3:
                    face_images = face_images[np.newaxis, ...]

                response = self._process_single_request(
                    face_images[0],  # Take first batch element
                    original_image,
                    orig_shape,
                    affine_matrix
                )
                responses.append(response)

            except Exception as e:
                logger.error(f"Error processing request: {e}")
                import traceback
                traceback.print_exc()
                error = pb_utils.TritonError(str(e))
                responses.append(pb_utils.InferenceResponse(error=error))

        return responses

    def _process_single_request(self, face_image, original_image, orig_shape, affine_matrix=None):
        """
        Process a single request with GPU-accelerated pipeline.

        Full GPU pipeline:
        1. YOLO11-face detection (TensorRT on GPU)
        2. NMS (torchvision on GPU) OR End2End NMS (built into TensorRT)
        3. MTCNN-style face cropping (grid_sample on GPU)
        4. ArcFace embedding (TensorRT on GPU)
        5. L2 normalization (PyTorch on GPU)

        Args:
            face_image: [3, 640, 640] preprocessed image for YOLO11-face
            original_image: [3, H, W] full-resolution image for face cropping
            orig_shape: [2] original image dimensions [H, W]
            affine_matrix: Optional [2, 3] DALI letterbox affine matrix
        """
        orig_h, orig_w = float(orig_shape[0]), float(orig_shape[1])

        # Initialize outputs on GPU
        face_boxes = torch.zeros((self.max_faces, 4), dtype=torch.float32, device=self.device)
        face_landmarks = torch.zeros((self.max_faces, 10), dtype=torch.float32, device=self.device)
        face_scores = torch.zeros(self.max_faces, dtype=torch.float32, device=self.device)
        face_embeddings = torch.zeros((self.max_faces, self.embed_dim), dtype=torch.float32, device=self.device)
        face_quality = np.zeros(self.max_faces, dtype=np.float32)
        num_faces = np.array([0], dtype=np.int32)

        # GPU: Call YOLO11-face TensorRT
        yolo_output = self._call_yolo11_face(face_image[np.newaxis, ...])

        if yolo_output is None:
            return self._create_response(
                num_faces,
                face_boxes.cpu().numpy(),
                face_landmarks.cpu().numpy(),
                face_scores.cpu().numpy(),
                face_embeddings.cpu().numpy(),
                face_quality
            )

        # Parse output based on mode
        if self.use_end2end:
            # End2End output: already post-NMS, boxes in normalized [0,1] coords
            # Pass affine_matrix for proper inverse letterbox transformation
            final_boxes, final_scores = self._decode_end2end_output(
                yolo_output, orig_h, orig_w, affine_matrix
            )
            final_landmarks = torch.zeros(len(final_boxes), 10, device=self.device) if final_boxes is not None else None
        else:
            # Standard output: needs decoding and Python NMS
            all_boxes, all_scores, all_landmarks = self._decode_yolo11_face_output(
                yolo_output, orig_h, orig_w, affine_matrix
            )
            # Take first batch element
            final_boxes = all_boxes[0]
            final_scores = all_scores[0]
            final_landmarks = all_landmarks[0]  # Will be zeros for detection-only models

        if final_boxes is None or len(final_boxes) == 0:
            return self._create_response(
                num_faces,
                face_boxes.cpu().numpy(),
                face_landmarks.cpu().numpy(),
                face_scores.cpu().numpy(),
                face_embeddings.cpu().numpy(),
                face_quality
            )

        n_faces = len(final_boxes)

        # GPU: Prepare original image tensor
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

        # GPU: MTCNN-style face cropping from HD original
        # This uses grid_sample which is fully GPU-accelerated
        cropped_faces = self._crop_faces_mtcnn_style(
            image_tensor, final_boxes, img_h, img_w, orig_h, orig_w
        )

        # GPU: Preprocess for ArcFace
        # ArcFace expects: (x - 127.5) / 128.0, input is [0, 1]
        cropped_faces = cropped_faces * 255.0  # Back to [0, 255]
        cropped_faces = (cropped_faces - 127.5) / 128.0

        # GPU: Call ArcFace TensorRT
        embeddings = self._call_arcface(cropped_faces.cpu().numpy().astype(np.float32))

        if embeddings is not None:
            # GPU: L2 normalize
            norms = torch.linalg.norm(embeddings, dim=1, keepdim=True)
            norms = torch.clamp(norms, min=1e-10)
            embeddings = embeddings / norms
            face_embeddings[:n_faces] = embeddings

        # Compute quality (CPU - simple calculation)
        face_quality[:n_faces] = self._compute_quality_batch(
            final_boxes.cpu().numpy(),
            orig_h, orig_w
        )

        # Store normalized boxes (GPU)
        num_faces[0] = n_faces
        face_boxes[:n_faces, 0] = final_boxes[:, 0] / orig_w
        face_boxes[:n_faces, 1] = final_boxes[:, 1] / orig_h
        face_boxes[:n_faces, 2] = final_boxes[:, 2] / orig_w
        face_boxes[:n_faces, 3] = final_boxes[:, 3] / orig_h
        face_landmarks[:n_faces] = final_landmarks  # Zeros for detection-only
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
        logger.info("Finalizing yolo11_face_pipeline (GPU)")
