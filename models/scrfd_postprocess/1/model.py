#!/usr/bin/env python3
"""
Triton Python Backend: SCRFD Face Detection with GPU Post-Processing

Calls SCRFD TensorRT model via BLS and decodes raw outputs into usable face detections.
ALL post-processing (anchor decoding, filtering, NMS) runs on GPU using PyTorch CUDA.

SCRFD outputs raw anchor-based predictions at 3 scales (stride 8, 16, 32):
- Score outputs: 448, 471, 494 (class probabilities)
- Box outputs: 451, 474, 497 (bounding box offsets)
- Landmark outputs: 454, 477, 500 (5-point facial landmarks)

GPU Post-Processing Pipeline:
1. Receive SCRFD TensorRT outputs (already on GPU)
2. Convert to PyTorch CUDA tensors (zero-copy when possible)
3. GPU anchor decoding
4. GPU confidence filtering
5. GPU NMS via torchvision.ops.nms (CUDA-accelerated)
6. GPU landmark gathering using NMS indices
7. Output formatted detections

Architecture:
    Inputs:
        - images: [3, 640, 640] Preprocessed image (normalized [0, 1])
        - orig_shape: [2] Original image shape [H, W] for scaling detections

    Outputs:
        - num_faces: [1] Number of detected faces
        - face_boxes: [128, 4] Face bounding boxes [x1, y1, x2, y2] in pixel coords
        - face_landmarks: [128, 10] 5-point landmarks flattened
        - face_scores: [128] Detection confidence scores
"""

import json
import logging

import numpy as np
import torch
import torchvision
import triton_python_backend_utils as pb_utils

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TritonPythonModel:
    """SCRFD Face Detection with GPU Post-Processing via torchvision NMS"""

    def initialize(self, args):
        """Initialize model configuration."""
        self.model_config = json.loads(args['model_config'])

        # SCRFD configuration
        self.input_size = 640
        self.max_faces = 128
        self.conf_threshold = 0.5
        self.nms_threshold = 0.4
        self.strides = [8, 16, 32]

        # GPU device
        self.device = torch.device('cuda:0')

        # Output names from SCRFD TensorRT (mapped to scales)
        self.output_names = {
            8: {'scores': '448', 'boxes': '451', 'landmarks': '454'},
            16: {'scores': '471', 'boxes': '474', 'landmarks': '477'},
            32: {'scores': '494', 'boxes': '497', 'landmarks': '500'},
        }

        # Pre-generate anchor centers on GPU
        self._init_anchors_gpu()

        logger.info("Initialized SCRFD GPU postprocess")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Input size: {self.input_size}")
        logger.info(f"  Max faces: {self.max_faces}")
        logger.info(f"  Confidence threshold: {self.conf_threshold}")
        logger.info(f"  NMS threshold: {self.nms_threshold}")

    def _init_anchors_gpu(self):
        """Pre-compute anchor centers and strides on GPU."""
        all_centers = []
        all_strides = []

        for stride in self.strides:
            feat_h = self.input_size // stride
            feat_w = self.input_size // stride

            # Create grid of anchor centers using torch
            y, x = torch.meshgrid(
                torch.arange(feat_h, dtype=torch.float32),
                torch.arange(feat_w, dtype=torch.float32),
                indexing='ij'
            )

            # Stack x,y and reshape to [H*W, 2]
            centers = torch.stack([x, y], dim=-1).reshape(-1, 2)
            # Convert to pixel coordinates
            centers = (centers + 0.5) * stride
            # SCRFD uses 2 anchors per location
            centers = centers.repeat_interleave(2, dim=0)

            all_centers.append(centers)
            all_strides.append(torch.full((centers.shape[0],), stride, dtype=torch.float32))

        # Concatenate all and move to GPU
        self.anchor_centers = torch.cat(all_centers, dim=0).to(self.device)
        self.anchor_strides = torch.cat(all_strides, dim=0).to(self.device)

        # Store anchor counts per stride for batch splitting
        self.anchors_per_stride = {
            8: (self.input_size // 8) ** 2 * 2,
            16: (self.input_size // 16) ** 2 * 2,
            32: (self.input_size // 32) ** 2 * 2,
        }

        logger.info(f"  Total anchors: {len(self.anchor_centers)}")

    def _call_scrfd(self, images):
        """Call SCRFD TensorRT via BLS."""
        try:
            # Create input tensor for SCRFD
            input_tensor = pb_utils.Tensor("input.1", images)

            # Get all output names
            output_names = []
            for stride in self.strides:
                output_names.extend([
                    self.output_names[stride]['scores'],
                    self.output_names[stride]['boxes'],
                    self.output_names[stride]['landmarks'],
                ])

            # Create inference request
            infer_request = pb_utils.InferenceRequest(
                model_name="scrfd_10g_face_detect",
                requested_output_names=output_names,
                inputs=[input_tensor],
            )

            # Execute
            infer_response = infer_request.exec()

            if infer_response.has_error():
                logger.error(f"SCRFD error: {infer_response.error().message()}")
                return None

            # Extract outputs by stride and convert to GPU tensors
            outputs = {}
            for stride in self.strides:
                outputs[stride] = {
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

            return outputs

        except Exception as e:
            logger.error(f"SCRFD BLS call failed: {e}")
            return None

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
        """Process a single inference request with full GPU pipeline."""
        # Get inputs
        images = pb_utils.get_input_tensor_by_name(request, "images").as_numpy()
        orig_shape = pb_utils.get_input_tensor_by_name(request, "orig_shape").as_numpy()
        orig_shape = orig_shape.flatten()
        orig_h, orig_w = float(orig_shape[0]), float(orig_shape[1])

        # Call SCRFD via BLS
        scrfd_outputs = self._call_scrfd(images)

        if scrfd_outputs is None:
            return self._create_empty_response()

        # =========================================
        # GPU POST-PROCESSING
        # =========================================

        # Concatenate all strides (all on GPU)
        scores = torch.cat([
            scrfd_outputs[8]['scores'].reshape(-1),
            scrfd_outputs[16]['scores'].reshape(-1),
            scrfd_outputs[32]['scores'].reshape(-1),
        ])  # [N]

        boxes = torch.cat([
            scrfd_outputs[8]['boxes'].reshape(-1, 4),
            scrfd_outputs[16]['boxes'].reshape(-1, 4),
            scrfd_outputs[32]['boxes'].reshape(-1, 4),
        ])  # [N, 4]

        landmarks = torch.cat([
            scrfd_outputs[8]['landmarks'].reshape(-1, 10),
            scrfd_outputs[16]['landmarks'].reshape(-1, 10),
            scrfd_outputs[32]['landmarks'].reshape(-1, 10),
        ])  # [N, 10]

        # GPU: Confidence filter
        conf_mask = scores > self.conf_threshold

        if not conf_mask.any():
            return self._create_empty_response()

        # Filter by confidence (GPU)
        scores = scores[conf_mask]
        boxes = boxes[conf_mask]
        landmarks = landmarks[conf_mask]
        anchor_centers = self.anchor_centers[conf_mask]
        anchor_strides = self.anchor_strides[conf_mask]

        # GPU: Decode boxes from anchor distances
        # SCRFD format: (left, top, right, bottom) distances from anchor center
        x1 = anchor_centers[:, 0] - boxes[:, 0] * anchor_strides
        y1 = anchor_centers[:, 1] - boxes[:, 1] * anchor_strides
        x2 = anchor_centers[:, 0] + boxes[:, 2] * anchor_strides
        y2 = anchor_centers[:, 1] + boxes[:, 3] * anchor_strides

        decoded_boxes = torch.stack([x1, y1, x2, y2], dim=-1)  # [M, 4]

        # GPU: Decode landmarks
        lmk = landmarks.reshape(-1, 5, 2)
        lmk = lmk * anchor_strides[:, None, None] + anchor_centers[:, None, :]
        decoded_landmarks = lmk.reshape(-1, 10)  # [M, 10]

        # GPU: NMS via torchvision (CUDA-accelerated)
        keep = torchvision.ops.nms(decoded_boxes, scores, self.nms_threshold)

        # Limit to max faces
        if len(keep) > self.max_faces:
            keep = keep[:self.max_faces]

        n_faces = len(keep)

        if n_faces == 0:
            return self._create_empty_response()

        # GPU: Gather final detections using NMS indices
        final_boxes = decoded_boxes[keep]
        final_landmarks = decoded_landmarks[keep]
        final_scores = scores[keep]

        # GPU: Scale to original image coordinates
        scale_x = orig_w / self.input_size
        scale_y = orig_h / self.input_size

        final_boxes[:, 0] *= scale_x  # x1
        final_boxes[:, 2] *= scale_x  # x2
        final_boxes[:, 1] *= scale_y  # y1
        final_boxes[:, 3] *= scale_y  # y2

        final_landmarks[:, 0::2] *= scale_x  # x coords
        final_landmarks[:, 1::2] *= scale_y  # y coords

        # Create padded output arrays on GPU
        out_boxes = torch.zeros((self.max_faces, 4), dtype=torch.float32, device=self.device)
        out_landmarks = torch.zeros((self.max_faces, 10), dtype=torch.float32, device=self.device)
        out_scores = torch.zeros((self.max_faces,), dtype=torch.float32, device=self.device)

        out_boxes[:n_faces] = final_boxes
        out_landmarks[:n_faces] = final_landmarks
        out_scores[:n_faces] = final_scores

        # Transfer to CPU for Triton output
        num_faces = np.array([n_faces], dtype=np.int32)
        face_boxes = out_boxes.cpu().numpy()
        face_landmarks = out_landmarks.cpu().numpy()
        face_scores = out_scores.cpu().numpy()

        return self._create_response(num_faces, face_boxes, face_landmarks, face_scores)

    def _create_empty_response(self):
        """Create response with no detections."""
        num_faces = np.array([0], dtype=np.int32)
        face_boxes = np.zeros((self.max_faces, 4), dtype=np.float32)
        face_landmarks = np.zeros((self.max_faces, 10), dtype=np.float32)
        face_scores = np.zeros((self.max_faces,), dtype=np.float32)
        return self._create_response(num_faces, face_boxes, face_landmarks, face_scores)

    def _create_response(self, num_faces, face_boxes, face_landmarks, face_scores):
        """Create inference response."""
        output_tensors = [
            pb_utils.Tensor("num_faces", num_faces),
            pb_utils.Tensor("face_boxes", face_boxes),
            pb_utils.Tensor("face_landmarks", face_landmarks),
            pb_utils.Tensor("face_scores", face_scores),
        ]
        return pb_utils.InferenceResponse(output_tensors=output_tensors)

    def finalize(self):
        """Cleanup."""
        logger.info("Finalizing SCRFD GPU postprocess")
