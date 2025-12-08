#!/usr/bin/env python3
"""
Triton Python Backend: Per-Object Embedding Extractor with Full-Resolution Cropping

This backend receives YOLO detections and the ORIGINAL NATIVE-RESOLUTION decoded image,
crops each detected object at full quality, resizes to 256×256, and generates embeddings
using MobileCLIP via BLS.

Key Features:
1. **Native Resolution:** No upscaling/downscaling - preserves original image quality
2. **Normalized Outputs:** Bounding boxes output in [0, 1] range for any image size
3. **MobileCLIP2 Compliant:** Follows official preprocessing guidelines

Architecture:
    Inputs:
        - original_image: [3, H, W] Native-resolution decoded image (NO resize/upscale)
        - det_boxes: [100, 4] YOLO boxes in [x1, y1, x2, y2] XYXY format
          Supports TWO input modes (auto-detected):
          - Normalized [0, 1]: If max(boxes) <= 1.0 (from YOLO with normalize_boxes=True)
          - Pixel coords: If max(boxes) > 1.0 (legacy 640×640 pixel space)
        - num_dets: [1] Number of valid detections
        - affine_matrix: [2, 3] YOLO letterbox transformation (to extract scale)

    Processing:
        1. Extract original image dimensions (H, W) - can be any size
        2. Extract scale factor from affine_matrix (YOLO 640→original mapping)
        3. Extract valid boxes (0 to num_dets)
        4. If normalized: denormalize to 640×640 pixel space first
        5. Scale boxes from YOLO 640×640 space to original image space
        6. Normalize boxes to [0, 1] range based on original dimensions
        7. ROI align to extract crops at original resolution (GPU)
        8. Resize crops to 256×256 for MobileCLIP (following official guidelines)
        9. Batch crops → MobileCLIP encoder (BLS call)
        10. Pad to MAX_BOXES with zeros

    Outputs:
        - box_embeddings: [300, 512] Fixed-size padded embeddings (MobileCLIP2-S2)
        - normalized_boxes: [300, 4] Normalized boxes in [0, 1] range (relative to original image)
"""

import triton_python_backend_utils as pb_utils
import torch
import torchvision.ops as ops
import torch.nn.functional as F
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
        self.max_boxes = 300
        self.output_embed_dim = 512  # MobileCLIP2-S2 outputs 512-dim embeddings
        self.mobileclip_size = 256  # MobileCLIP requires 256×256 input
        self.yolo_size = 640  # YOLO uses 640×640

        # Setup device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Initialized box_embedding_extractor on {self.device}")
        logger.info(f"  Max boxes: {self.max_boxes}")
        logger.info(f"  Embedding dim: {self.output_embed_dim}")
        logger.info(f"  MobileCLIP size: {self.mobileclip_size}")
        logger.info(f"  YOLO size: {self.yolo_size}")
        logger.info(f"  Strategy: Full-resolution cropping from original image")

        # Warmup (optional)
        if self.device == 'cuda':
            self._warmup_gpu()

    def _warmup_gpu(self):
        """Warmup GPU with dummy operations"""
        try:
            dummy_image = torch.randn(3, 1080, 1920, device=self.device)
            dummy_boxes = torch.tensor([[50, 50, 100, 100]], device=self.device, dtype=torch.float32)
            crops = self._crop_boxes(dummy_image, dummy_boxes)
            _ = self._resize_crops_to_mobileclip_size(crops)
            logger.info("GPU warmup complete")
        except Exception as e:
            logger.warning(f"GPU warmup failed: {e}")

    def _extract_scale_from_affine(self, affine_matrix):
        """
        Extract scale factor from YOLO letterbox affine matrix

        Args:
            affine_matrix: numpy array [2, 3] or torch tensor

        Returns:
            scale: float - scale factor from original→YOLO (to reverse: 1/scale)
        """
        if isinstance(affine_matrix, np.ndarray):
            affine_matrix = torch.from_numpy(affine_matrix)

        # Affine matrix structure:
        # [scale,   0,    pad_x]
        # [  0,  scale,  pad_y]
        scale = float(affine_matrix[0, 0])
        return scale

    def _denormalize_boxes_to_yolo_space(self, boxes):
        """
        Denormalize boxes from [0, 1] range to YOLO 640×640 pixel space.

        This is used when YOLO model was exported with normalize_boxes=True.

        Args:
            boxes: Tensor [N, 4] in [0, 1] range (XYXY format)

        Returns:
            Tensor [N, 4] in 640×640 pixel coordinates
        """
        if boxes.shape[0] == 0:
            return boxes

        return boxes * self.yolo_size

    def _is_normalized_boxes(self, boxes):
        """
        Auto-detect if boxes are in normalized [0, 1] range or pixel coordinates.

        Args:
            boxes: Tensor [N, 4]

        Returns:
            bool: True if boxes appear to be normalized (max <= 1.0)
        """
        if boxes.shape[0] == 0:
            return False

        # If all values are <= 1.0, assume normalized
        # (pixel coords in 640x640 would have values > 1.0)
        return float(boxes.max()) <= 1.0

    def _scale_boxes_to_dali_hd_space(self, boxes, yolo_scale, pad_x, pad_y, dali_h, dali_w):
        """
        Scale boxes from YOLO 640×640 space to DALI HD image space.

        The DALI HD image has max dimension 1920 while preserving aspect ratio.
        The boxes need to be transformed from letterbox space to HD image space.

        Pipeline:
        1. Remove letterbox padding from YOLO coords
        2. Divide by yolo_scale to get original image coords
        3. Scale to DALI HD dims (which is original * dali_scale)

        Args:
            boxes: Tensor [N, 4] in [x1, y1, x2, y2] format (YOLO 640×640 space)
            yolo_scale: float - YOLO letterbox scale factor
            pad_x: float - horizontal padding in YOLO space
            pad_y: float - vertical padding in YOLO space
            dali_h: int - DALI HD image height
            dali_w: int - DALI HD image width

        Returns:
            Tensor [N, 4] in DALI HD image pixel coordinates
        """
        if boxes.shape[0] == 0:
            return boxes

        # Calculate original image's longest dimension from YOLO scale
        original_max_dim = self.yolo_size / yolo_scale

        # Calculate DALI scale factor (how much DALI scaled from original)
        dali_max_dim = max(dali_h, dali_w)
        dali_scale = dali_max_dim / original_max_dim

        # Combined transformation: yolo → original → dali
        # dali_px = (yolo_px - pad) / yolo_scale * dali_scale
        # Simplify: dali_px = (yolo_px - pad) * (dali_scale / yolo_scale)
        combined_scale = dali_scale / yolo_scale

        boxes_out = boxes.clone()
        boxes_out[:, [0, 2]] = (boxes_out[:, [0, 2]] - pad_x) * combined_scale
        boxes_out[:, [1, 3]] = (boxes_out[:, [1, 3]] - pad_y) * combined_scale

        return boxes_out

    def _normalize_boxes(self, boxes, img_width, img_height):
        """
        Normalize boxes to [0, 1] range based on image dimensions

        Args:
            boxes: Tensor [N, 4] in [x1, y1, x2, y2] pixel coordinates
            img_width: Image width in pixels
            img_height: Image height in pixels

        Returns:
            Tensor [N, 4] with values in [0, 1] range
        """
        if boxes.shape[0] == 0:
            return boxes

        boxes_normalized = boxes.clone()
        boxes_normalized[:, [0, 2]] /= img_width   # x coordinates
        boxes_normalized[:, [1, 3]] /= img_height  # y coordinates

        # Clamp to [0, 1]
        boxes_normalized = torch.clamp(boxes_normalized, 0.0, 1.0)

        return boxes_normalized

    def _crop_boxes(self, image, boxes):
        """
        Crop boxes from image using ROI align

        Args:
            image: Tensor [3, H, W]
            boxes: Tensor [N, 4] in [x1, y1, x2, y2] format

        Returns:
            Tensor [N, 3, crop_h, crop_w] - variable-size crops
        """
        if boxes.shape[0] == 0:
            return torch.empty(0, 3, 0, 0, device=self.device)

        # Clamp boxes to image boundaries
        img_h, img_w = image.shape[1], image.shape[2]
        boxes = boxes.clone()
        boxes[:, [0, 2]] = torch.clamp(boxes[:, [0, 2]], 0, img_w - 1)  # x coords
        boxes[:, [1, 3]] = torch.clamp(boxes[:, [1, 3]], 0, img_h - 1)  # y coords

        # Validate boxes (avoid zero-width/height boxes)
        valid_mask = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        if not valid_mask.all():
            logger.warning(f"Found {(~valid_mask).sum()} invalid boxes (zero width/height)")
            boxes = boxes[valid_mask]

        if boxes.shape[0] == 0:
            return torch.empty(0, 3, 0, 0, device=self.device)

        # Calculate target size for each crop (maintain aspect ratio, then resize)
        # For now, we'll use ROI align with a reasonable size, then resize to 256×256
        # Using 256×256 for ROI align is fine since we'll resize anyway
        target_size = self.mobileclip_size

        # Add batch index (all boxes from same image)
        batch_indices = torch.zeros(boxes.shape[0], 1, device=boxes.device)
        boxes_with_idx = torch.cat([batch_indices, boxes], dim=1)

        # ROI align
        crops = ops.roi_align(
            image.unsqueeze(0),  # [1, 3, H, W]
            boxes_with_idx,      # [N, 5] (batch_idx, x1, y1, x2, y2)
            output_size=(target_size, target_size),
            spatial_scale=1.0,
            sampling_ratio=2,
            aligned=True
        )

        return crops  # [N, 3, 256, 256]

    def _resize_crops_to_mobileclip_size(self, crops):
        """
        Resize crops to MobileCLIP input size (256×256) if needed

        Args:
            crops: Tensor [N, 3, H, W]

        Returns:
            Tensor [N, 3, 256, 256]
        """
        if crops.shape[0] == 0:
            return torch.empty(0, 3, self.mobileclip_size, self.mobileclip_size, device=self.device)

        # Already at target size from ROI align
        if crops.shape[2] == self.mobileclip_size and crops.shape[3] == self.mobileclip_size:
            return crops

        # Resize using bilinear interpolation
        crops_resized = F.interpolate(
            crops,
            size=(self.mobileclip_size, self.mobileclip_size),
            mode='bilinear',
            align_corners=False
        )

        return crops_resized

    def _call_mobileclip_encoder(self, crops):
        """
        Call MobileCLIP image encoder via BLS

        Args:
            crops: numpy array [N, 3, 256, 256]

        Returns:
            embeddings: torch.Tensor [N, 512] on GPU
        """
        # Convert torch GPU tensor to Triton tensor via DLPack (zero-copy)
        if isinstance(crops, torch.Tensor):
            if crops.is_cuda:
                input_tensor = pb_utils.Tensor.from_dlpack('images', torch.to_dlpack(crops))
            else:
                input_tensor = pb_utils.Tensor('images', crops.numpy())
        else:
            input_tensor = pb_utils.Tensor('images', crops)

        # Create BLS request
        inference_request = pb_utils.InferenceRequest(
            model_name='mobileclip2_s2_image_encoder',
            requested_output_names=['image_embeddings'],
            inputs=[input_tensor]
        )

        # Execute
        inference_response = inference_request.exec()

        # Check for errors
        if inference_response.has_error():
            error_msg = inference_response.error().message()
            raise RuntimeError(f"MobileCLIP BLS call failed: {error_msg}")

        # Extract embeddings via DLPack (zero-copy on GPU)
        output_tensor = pb_utils.get_output_tensor_by_name(
            inference_response,
            'image_embeddings'
        )

        if output_tensor.is_cpu():
            embeddings = torch.from_numpy(output_tensor.as_numpy()).to(self.device)
        else:
            embeddings = torch.from_dlpack(output_tensor.to_dlpack())

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
                original_image_tensor = pb_utils.get_input_tensor_by_name(request, "original_image")
                det_boxes_tensor = pb_utils.get_input_tensor_by_name(request, "det_boxes")
                num_dets_tensor = pb_utils.get_input_tensor_by_name(request, "num_dets")
                affine_matrix_tensor = pb_utils.get_input_tensor_by_name(request, "affine_matrix")

                # Zero-copy GPU tensor conversion via DLPack (keeps data on GPU)
                def triton_to_torch(tensor):
                    """Convert Triton tensor to PyTorch tensor via DLPack (zero-copy on GPU)"""
                    if tensor.is_cpu():
                        return torch.from_numpy(tensor.as_numpy()).to(self.device)
                    else:
                        # GPU tensor - use DLPack for zero-copy GPU-to-GPU
                        dlpack = tensor.to_dlpack()
                        return torch.from_dlpack(dlpack)

                # Convert all tensors to torch on GPU (zero-copy)
                original_image = triton_to_torch(original_image_tensor)  # [1, 3, H, W] or [3, H, W]
                det_boxes = triton_to_torch(det_boxes_tensor)            # [1, 300, 4] or [300, 4]
                num_dets_t = triton_to_torch(num_dets_tensor)            # [1, 1] or [1]
                affine_matrix = triton_to_torch(affine_matrix_tensor)    # [1, 2, 3] or [2, 3]

                # Handle batching (squeeze batch dimension)
                if original_image.ndim == 4:
                    original_image = original_image[0]  # [3, H, W]
                if det_boxes.ndim == 3:
                    det_boxes = det_boxes[0]            # [300, 4]
                if num_dets_t.ndim == 2:
                    num_dets_t = num_dets_t[0]          # [1]
                if affine_matrix.ndim == 3:
                    affine_matrix = affine_matrix[0]    # [2, 3]

                # Extract scalar values (minimal CPU sync)
                num_dets = int(num_dets_t[0].item())
                scale = float(affine_matrix[0, 0].item())
                pad_x = float(affine_matrix[0, 2].item())
                pad_y = float(affine_matrix[1, 2].item())

                logger.debug(f"Processing: original_size={original_image.shape[1:3]}, num_dets={num_dets}, scale={scale:.4f}")

                # Handle case: no detections
                if num_dets == 0:
                    logger.debug("No detections, returning zero embeddings")
                    box_embeddings = np.zeros((self.max_boxes, self.output_embed_dim), dtype=np.float32)
                    normalized_boxes_padded = np.zeros((self.max_boxes, 4), dtype=np.float32)
                    output_tensors = [
                        pb_utils.Tensor('box_embeddings', box_embeddings),
                        pb_utils.Tensor('normalized_boxes', normalized_boxes_padded)
                    ]
                    responses.append(pb_utils.InferenceResponse(output_tensors=output_tensors))
                    continue

                # Clamp num_dets to max_boxes
                if num_dets > self.max_boxes:
                    logger.warning(f"num_dets={num_dets} exceeds max_boxes={self.max_boxes}, clamping")
                    num_dets = self.max_boxes

                # Extract valid boxes (already in XYXY format from EfficientNMS)
                valid_boxes = det_boxes[:num_dets]  # [num_dets, 4]

                # Auto-detect if boxes are normalized [0,1] or pixel coords
                # YOLO with normalize_boxes=True outputs [0,1], legacy outputs 640x640 pixels
                if self._is_normalized_boxes(valid_boxes):
                    logger.debug("Detected normalized boxes [0,1], denormalizing to YOLO space")
                    boxes_yolo_space = self._denormalize_boxes_to_yolo_space(valid_boxes)
                else:
                    logger.debug("Detected pixel coordinate boxes (640x640 space)")
                    boxes_yolo_space = valid_boxes

                # Get DALI HD image dimensions
                img_h, img_w = original_image.shape[1], original_image.shape[2]

                # Scale boxes from YOLO 640×640 space to DALI HD image space
                boxes_dali_space = self._scale_boxes_to_dali_hd_space(
                    boxes_yolo_space, scale, pad_x, pad_y, img_h, img_w
                )

                # Normalize boxes to [0, 1] range based on DALI HD dimensions
                boxes_normalized = self._normalize_boxes(boxes_dali_space, img_w, img_h)

                # Crop boxes from DALI HD image
                cropped_boxes = self._crop_boxes(original_image, boxes_dali_space)  # [num_dets, 3, 256, 256]

                if cropped_boxes.shape[0] == 0:
                    logger.warning("All boxes invalid after cropping, returning zero embeddings")
                    box_embeddings = np.zeros((self.max_boxes, self.output_embed_dim), dtype=np.float32)
                    normalized_boxes_padded = np.zeros((self.max_boxes, 4), dtype=np.float32)
                    output_tensors = [
                        pb_utils.Tensor('box_embeddings', box_embeddings),
                        pb_utils.Tensor('normalized_boxes', normalized_boxes_padded)
                    ]
                    responses.append(pb_utils.InferenceResponse(output_tensors=output_tensors))
                    continue

                # Ensure crops are 256×256 (should already be from ROI align)
                cropped_boxes_resized = self._resize_crops_to_mobileclip_size(cropped_boxes)

                # Call MobileCLIP encoder via BLS (keeps data on GPU via DLPack)
                embeddings = self._call_mobileclip_encoder(cropped_boxes_resized)  # [actual_num_dets, 512]

                # Pad embeddings to max_boxes (on GPU)
                actual_num = embeddings.shape[0]
                box_embeddings = torch.zeros(
                    (self.max_boxes, self.output_embed_dim),
                    dtype=torch.float32,
                    device=self.device
                )
                box_embeddings[:actual_num] = embeddings

                # Pad normalized boxes to max_boxes (on GPU)
                normalized_boxes_padded = torch.zeros(
                    (self.max_boxes, 4),
                    dtype=torch.float32,
                    device=self.device
                )
                normalized_boxes_padded[:actual_num] = boxes_normalized

                logger.debug(f"Extracted {actual_num} box embeddings (padded to {self.max_boxes})")
                logger.debug(f"Image size: {img_w}×{img_h}, Scale: {scale:.4f}")

                # Add batch dimension for Triton ensemble compatibility
                # Triton expects outputs with shape [batch, ...] when max_batch_size > 0
                box_embeddings = box_embeddings.unsqueeze(0)  # [1, 300, 512]
                normalized_boxes_padded = normalized_boxes_padded.unsqueeze(0)  # [1, 300, 4]

                # Create output tensors via DLPack (zero-copy on GPU)
                output_tensors = [
                    pb_utils.Tensor.from_dlpack('box_embeddings', torch.to_dlpack(box_embeddings)),
                    pb_utils.Tensor.from_dlpack('normalized_boxes', torch.to_dlpack(normalized_boxes_padded))
                ]
                responses.append(pb_utils.InferenceResponse(output_tensors=output_tensors))

            except Exception as e:
                logger.error(f"Error in execute: {e}", exc_info=True)

                # Return zero embeddings on error
                box_embeddings = np.zeros((self.max_boxes, self.output_embed_dim), dtype=np.float32)
                normalized_boxes_padded = np.zeros((self.max_boxes, 4), dtype=np.float32)
                output_tensors = [
                    pb_utils.Tensor('box_embeddings', box_embeddings),
                    pb_utils.Tensor('normalized_boxes', normalized_boxes_padded)
                ]
                error_response = pb_utils.InferenceResponse(
                    output_tensors=output_tensors,
                    error=pb_utils.TritonError(f"Box extraction failed: {str(e)}")
                )
                responses.append(error_response)

        return responses

    def finalize(self):
        """Cleanup"""
        logger.info("Finalizing box_embedding_extractor")
        # Free GPU memory if needed
        if self.device == 'cuda':
            torch.cuda.empty_cache()
