#!/usr/bin/env python3
"""
Triton Python Backend: Unified Embedding Extractor with Face Detection

This backend combines per-box MobileCLIP embeddings with face detection/recognition.
For person detections, it runs SCRFD face detection on the crop and extracts ArcFace
embeddings for any detected faces.

Key Features:
1. MobileCLIP embeddings for ALL detected boxes
2. SCRFD face detection ONLY on person crops (class_id=0)
3. ArcFace embeddings for detected faces
4. Much more efficient than full-image face detection

Architecture:
    Inputs:
        - original_image: [3, H, W] HD image for cropping
        - det_boxes: [300, 4] YOLO boxes (normalized [0,1])
        - det_classes: [300] YOLO class IDs
        - det_scores: [300] YOLO confidence scores
        - num_dets: [1] Number of valid detections
        - affine_matrix: [2, 3] YOLO letterbox transformation

    Processing:
        1. Crop all boxes → MobileCLIP embeddings
        2. Filter person boxes (class_id=0)
        3. Resize person crops to 640x640 → SCRFD face detection
        4. For each detected face: Umeyama align → ArcFace embedding

    Outputs:
        - box_embeddings: [300, 512] MobileCLIP embeddings per box
        - normalized_boxes: [300, 4] Normalized boxes
        - num_faces: [1] Total faces detected across all persons
        - face_embeddings: [64, 512] ArcFace embeddings per face
        - face_boxes: [64, 4] Face boxes (normalized to original image)
        - face_landmarks: [64, 10] 5-point landmarks
        - face_scores: [64] Detection confidence
        - face_person_idx: [64] Which person box each face belongs to
"""

import json
import logging

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.ops as ops
import triton_python_backend_utils as pb_utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ArcFace reference landmarks for alignment
ARCFACE_REF_LANDMARKS = np.array([
    [38.2946, 51.6963],  # Left eye
    [73.5318, 51.5014],  # Right eye
    [56.0252, 71.7366],  # Nose
    [41.5493, 92.3655],  # Left mouth
    [70.7299, 92.2041],  # Right mouth
], dtype=np.float32)


def umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Umeyama similarity transform estimation."""
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


class TritonPythonModel:
    """Unified Embedding Extractor with Face Detection"""

    def initialize(self, args):
        self.model_config = json.loads(args['model_config'])

        # Box embedding config
        self.max_boxes = 300
        self.embed_dim = 512
        self.mobileclip_size = 256
        self.yolo_size = 640

        # Face detection config
        self.max_faces = 64
        self.scrfd_size = 640
        self.arcface_size = 112
        self.face_conf_threshold = 0.5
        self.face_nms_threshold = 0.4
        self.person_class_id = 0  # COCO person class

        # SCRFD anchor config
        self.strides = [8, 16, 32]
        self.output_names = {
            8: {'scores': '448', 'boxes': '451', 'landmarks': '454'},
            16: {'scores': '471', 'boxes': '474', 'landmarks': '477'},
            32: {'scores': '494', 'boxes': '497', 'landmarks': '500'},
        }

        # Pre-generate anchor centers
        self.anchor_centers = {}
        for stride in self.strides:
            feat_h = self.scrfd_size // stride
            feat_w = self.scrfd_size // stride
            y, x = np.meshgrid(np.arange(feat_h), np.arange(feat_w), indexing='ij')
            anchor_centers = np.stack([x, y], axis=-1).reshape(-1, 2)
            anchor_centers = (anchor_centers + 0.5) * stride
            self.anchor_centers[stride] = np.repeat(anchor_centers, 2, axis=0)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Initialized unified_embedding_extractor on {self.device}")
        logger.info(f"  Max boxes: {self.max_boxes}, Max faces: {self.max_faces}")
        logger.info(f"  Person-only face detection enabled")

    def _triton_to_torch(self, tensor):
        """Convert Triton tensor to PyTorch tensor (zero-copy on GPU)."""
        if tensor.is_cpu():
            return torch.from_numpy(tensor.as_numpy()).to(self.device)
        return torch.from_dlpack(tensor.to_dlpack())

    def _crop_boxes(self, image, boxes, target_size):
        """Crop boxes using ROI align."""
        if boxes.shape[0] == 0:
            return torch.empty(0, 3, target_size, target_size, device=self.device)

        img_h, img_w = image.shape[1], image.shape[2]
        boxes = boxes.clone()
        boxes[:, [0, 2]] = torch.clamp(boxes[:, [0, 2]], 0, img_w - 1)
        boxes[:, [1, 3]] = torch.clamp(boxes[:, [1, 3]], 0, img_h - 1)

        valid_mask = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        if not valid_mask.all():
            boxes = boxes[valid_mask]

        if boxes.shape[0] == 0:
            return torch.empty(0, 3, target_size, target_size, device=self.device)

        batch_indices = torch.zeros(boxes.shape[0], 1, device=boxes.device)
        boxes_with_idx = torch.cat([batch_indices, boxes], dim=1)

        crops = ops.roi_align(
            image.unsqueeze(0),
            boxes_with_idx,
            output_size=(target_size, target_size),
            spatial_scale=1.0,
            sampling_ratio=2,
            aligned=True
        )
        return crops

    def _call_mobileclip(self, crops):
        """Call MobileCLIP encoder via BLS."""
        if isinstance(crops, torch.Tensor) and crops.is_cuda:
            input_tensor = pb_utils.Tensor.from_dlpack('images', torch.to_dlpack(crops))
        else:
            input_tensor = pb_utils.Tensor('images', crops.cpu().numpy())

        request = pb_utils.InferenceRequest(
            model_name='mobileclip2_s2_image_encoder',
            requested_output_names=['image_embeddings'],
            inputs=[input_tensor]
        )
        response = request.exec()
        if response.has_error():
            raise RuntimeError(f"MobileCLIP error: {response.error().message()}")

        output = pb_utils.get_output_tensor_by_name(response, 'image_embeddings')
        if output.is_cpu():
            return torch.from_numpy(output.as_numpy()).to(self.device)
        return torch.from_dlpack(output.to_dlpack())

    def _call_scrfd(self, images):
        """Call SCRFD face detector via BLS (batch of person crops)."""
        results = []
        # SCRFD is static batch=1, so we process one at a time
        for i in range(images.shape[0]):
            img = images[i:i+1]  # [1, 3, 640, 640]
            if isinstance(img, torch.Tensor):
                img_np = img.cpu().numpy()
            else:
                img_np = img

            input_tensor = pb_utils.Tensor("input.1", img_np)
            output_names = []
            for stride in self.strides:
                output_names.extend([
                    self.output_names[stride]['scores'],
                    self.output_names[stride]['boxes'],
                    self.output_names[stride]['landmarks'],
                ])

            request = pb_utils.InferenceRequest(
                model_name="scrfd_10g_face_detect",
                requested_output_names=output_names,
                inputs=[input_tensor],
                preferred_memory=pb_utils.PreferredMemory(pb_utils.TRITONSERVER_MEMORY_CPU),
            )
            response = request.exec()
            if response.has_error():
                logger.warning(f"SCRFD error: {response.error().message()}")
                results.append(None)
                continue

            outputs = {}
            for stride in self.strides:
                score_tensor = pb_utils.get_output_tensor_by_name(
                    response, self.output_names[stride]['scores']
                )
                if score_tensor is None:
                    continue
                box_tensor = pb_utils.get_output_tensor_by_name(
                    response, self.output_names[stride]['boxes']
                )
                lmk_tensor = pb_utils.get_output_tensor_by_name(
                    response, self.output_names[stride]['landmarks']
                )
                outputs[stride] = {
                    'scores': score_tensor.as_numpy(),
                    'boxes': box_tensor.as_numpy(),
                    'landmarks': lmk_tensor.as_numpy(),
                }
            results.append(outputs)
        return results

    def _decode_scrfd_outputs(self, outputs, crop_size):
        """Decode SCRFD outputs to face boxes and landmarks."""
        if outputs is None:
            return np.array([]), np.array([]), np.array([])

        all_boxes, all_scores, all_landmarks = [], [], []

        for stride in self.strides:
            if stride not in outputs:
                continue

            scores = outputs[stride]['scores'].flatten()
            boxes = outputs[stride]['boxes'].reshape(-1, 4)
            landmarks = outputs[stride]['landmarks'].reshape(-1, 10)
            anchor_centers = self.anchor_centers[stride]

            mask = scores > self.face_conf_threshold
            if not np.any(mask):
                continue

            anchor_centers = anchor_centers[mask]
            scores = scores[mask]
            boxes = boxes[mask]
            landmarks = landmarks[mask]

            # Decode boxes (distance format)
            x1 = anchor_centers[:, 0] - boxes[:, 0] * stride
            y1 = anchor_centers[:, 1] - boxes[:, 1] * stride
            x2 = anchor_centers[:, 0] + boxes[:, 2] * stride
            y2 = anchor_centers[:, 1] + boxes[:, 3] * stride
            decoded_boxes = np.stack([x1, y1, x2, y2], axis=-1)

            # Decode landmarks
            lmks = landmarks.reshape(-1, 5, 2)
            lmks = lmks * stride + anchor_centers[:, np.newaxis, :]
            decoded_landmarks = lmks.reshape(-1, 10)

            all_boxes.append(decoded_boxes)
            all_scores.append(scores)
            all_landmarks.append(decoded_landmarks)

        if not all_boxes:
            return np.array([]), np.array([]), np.array([])

        all_boxes = np.concatenate(all_boxes, axis=0)
        all_scores = np.concatenate(all_scores, axis=0)
        all_landmarks = np.concatenate(all_landmarks, axis=0)

        # NMS
        keep = self._nms(all_boxes, all_scores, self.face_nms_threshold)
        if len(keep) == 0:
            return np.array([]), np.array([]), np.array([])

        return all_boxes[keep], all_scores[keep], all_landmarks[keep]

    def _nms(self, boxes, scores, threshold):
        """Non-Maximum Suppression."""
        if len(boxes) == 0:
            return np.array([], dtype=np.int32)

        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-10)
            inds = np.where(iou <= threshold)[0]
            order = order[inds + 1]

        return np.array(keep, dtype=np.int32)

    def _align_face(self, image, landmarks):
        """Align face using Umeyama transform."""
        M = umeyama(landmarks.astype(np.float64), ARCFACE_REF_LANDMARKS.astype(np.float64))
        M = M[:2, :].astype(np.float32)

        out_h, out_w = self.arcface_size, self.arcface_size
        aligned = np.zeros((out_h, out_w, 3), dtype=image.dtype)

        y_coords, x_coords = np.meshgrid(np.arange(out_h), np.arange(out_w), indexing='ij')
        coords = np.stack([x_coords.flatten(), y_coords.flatten(), np.ones_like(x_coords.flatten())], axis=0)

        M_full = np.vstack([M, [0, 0, 1]])
        M_inv = np.linalg.inv(M_full)[:2, :]
        src_coords = M_inv @ coords
        src_x = src_coords[0].reshape(out_h, out_w)
        src_y = src_coords[1].reshape(out_h, out_w)

        h, w = image.shape[:2]
        x0 = np.floor(src_x).astype(np.int32)
        y0 = np.floor(src_y).astype(np.int32)
        x1, y1 = x0 + 1, y0 + 1
        wx, wy = src_x - x0, src_y - y0

        valid = (x0 >= 0) & (x1 < w) & (y0 >= 0) & (y1 < h)
        x0c = np.clip(x0, 0, w - 1)
        x1c = np.clip(x1, 0, w - 1)
        y0c = np.clip(y0, 0, h - 1)
        y1c = np.clip(y1, 0, h - 1)

        Ia = image[y0c.flatten(), x0c.flatten()].reshape(out_h, out_w, -1)
        Ib = image[y1c.flatten(), x0c.flatten()].reshape(out_h, out_w, -1)
        Ic = image[y0c.flatten(), x1c.flatten()].reshape(out_h, out_w, -1)
        Id = image[y1c.flatten(), x1c.flatten()].reshape(out_h, out_w, -1)

        wx = wx[:, :, np.newaxis]
        wy = wy[:, :, np.newaxis]
        result = (1 - wx) * (1 - wy) * Ia + (1 - wx) * wy * Ib + wx * (1 - wy) * Ic + wx * wy * Id

        valid_expanded = np.repeat(valid[:, :, np.newaxis], 3, axis=2)
        aligned[valid_expanded] = result[valid_expanded]

        return aligned.astype(np.uint8)

    def _call_arcface(self, faces):
        """Call ArcFace via BLS."""
        if faces.shape[0] == 0:
            return np.zeros((0, 512), dtype=np.float32)

        # Preprocess: (x - 127.5) / 128.0, transpose to NCHW
        faces_float = faces.astype(np.float32)
        faces_float = (faces_float - 127.5) / 128.0
        faces_float = np.transpose(faces_float, (0, 3, 1, 2))

        input_tensor = pb_utils.Tensor("input.1", faces_float)
        request = pb_utils.InferenceRequest(
            model_name="arcface_w600k_r50",
            requested_output_names=["683"],
            inputs=[input_tensor],
            preferred_memory=pb_utils.PreferredMemory(pb_utils.TRITONSERVER_MEMORY_CPU),
        )
        response = request.exec()
        if response.has_error():
            logger.warning(f"ArcFace error: {response.error().message()}")
            return np.zeros((faces.shape[0], 512), dtype=np.float32)

        output = pb_utils.get_output_tensor_by_name(response, "683")
        embeddings = output.as_numpy()

        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        return embeddings / norms

    def execute(self, requests):
        responses = []

        for request in requests:
            try:
                response = self._process_request(request)
                responses.append(response)
            except Exception as e:
                logger.error(f"Error: {e}", exc_info=True)
                responses.append(self._create_error_response(str(e)))

        return responses

    def _process_request(self, request):
        # Get inputs
        original_image = self._triton_to_torch(
            pb_utils.get_input_tensor_by_name(request, "original_image")
        )
        det_boxes = self._triton_to_torch(
            pb_utils.get_input_tensor_by_name(request, "det_boxes")
        )
        det_classes = self._triton_to_torch(
            pb_utils.get_input_tensor_by_name(request, "det_classes")
        )
        num_dets_t = self._triton_to_torch(
            pb_utils.get_input_tensor_by_name(request, "num_dets")
        )
        affine_matrix = self._triton_to_torch(
            pb_utils.get_input_tensor_by_name(request, "affine_matrix")
        )

        # Handle batch dims
        if original_image.ndim == 4:
            original_image = original_image[0]
        if det_boxes.ndim == 3:
            det_boxes = det_boxes[0]
        if det_classes.ndim == 2:
            det_classes = det_classes[0]
        if num_dets_t.ndim == 2:
            num_dets_t = num_dets_t[0]
        if affine_matrix.ndim == 3:
            affine_matrix = affine_matrix[0]

        num_dets = int(num_dets_t[0].item())
        scale = float(affine_matrix[0, 0].item())
        pad_x = float(affine_matrix[0, 2].item())
        pad_y = float(affine_matrix[1, 2].item())

        img_h, img_w = original_image.shape[1], original_image.shape[2]

        # Initialize outputs
        box_embeddings = torch.zeros((self.max_boxes, self.embed_dim), device=self.device)
        normalized_boxes = torch.zeros((self.max_boxes, 4), device=self.device)
        face_embeddings = np.zeros((self.max_faces, 512), dtype=np.float32)
        face_boxes = np.zeros((self.max_faces, 4), dtype=np.float32)
        face_landmarks = np.zeros((self.max_faces, 10), dtype=np.float32)
        face_scores = np.zeros(self.max_faces, dtype=np.float32)
        face_person_idx = np.zeros(self.max_faces, dtype=np.int32) - 1
        num_faces = np.array([0], dtype=np.int32)

        if num_dets == 0:
            return self._create_response(
                box_embeddings, normalized_boxes,
                num_faces, face_embeddings, face_boxes, face_landmarks, face_scores, face_person_idx
            )

        # Get valid detections
        num_dets = min(num_dets, self.max_boxes)
        valid_boxes = det_boxes[:num_dets]
        valid_classes = det_classes[:num_dets]

        # Scale boxes from normalized [0,1] to HD pixel space
        boxes_pixel = valid_boxes.clone()
        boxes_pixel[:, [0, 2]] *= img_w
        boxes_pixel[:, [1, 3]] *= img_h

        # === Part 1: MobileCLIP embeddings for ALL boxes ===
        crops = self._crop_boxes(original_image, boxes_pixel, self.mobileclip_size)
        if crops.shape[0] > 0:
            embeddings = self._call_mobileclip(crops)
            actual_num = min(embeddings.shape[0], self.max_boxes)
            box_embeddings[:actual_num] = embeddings[:actual_num]
            normalized_boxes[:num_dets] = valid_boxes

        # === Part 2: Face detection on PERSON boxes only ===
        person_mask = valid_classes == self.person_class_id
        person_indices = torch.where(person_mask)[0]

        if len(person_indices) > 0:
            # Crop person boxes at SCRFD size
            person_boxes_pixel = boxes_pixel[person_indices]
            person_crops = self._crop_boxes(original_image, person_boxes_pixel, self.scrfd_size)

            # original_image is already normalized [0, 1] from DALI, so crops are too
            # No need to divide by 255 again!

            # Run SCRFD on each person crop (already normalized [0, 1])
            scrfd_results = self._call_scrfd(person_crops)

            # Convert original image to numpy for face alignment
            image_np = original_image.permute(1, 2, 0).cpu().numpy()
            if image_np.max() <= 1.0:
                image_np = (image_np * 255).astype(np.uint8)
            else:
                image_np = image_np.astype(np.uint8)

            # Process each person's faces
            all_aligned_faces = []
            all_face_info = []
            total_faces = 0

            for i, (person_idx, scrfd_out) in enumerate(zip(person_indices, scrfd_results)):
                person_idx = int(person_idx.item())
                person_box = person_boxes_pixel[i].cpu().numpy()
                p_x1, p_y1, p_x2, p_y2 = person_box

                # Decode SCRFD outputs (in 640x640 crop space)
                face_boxes_crop, scores, landmarks_crop = self._decode_scrfd_outputs(
                    scrfd_out, self.scrfd_size
                )

                if len(face_boxes_crop) == 0:
                    continue

                # Scale face coords from crop space to original image space
                crop_w = p_x2 - p_x1
                crop_h = p_y2 - p_y1
                scale_x = crop_w / self.scrfd_size
                scale_y = crop_h / self.scrfd_size

                for j in range(len(face_boxes_crop)):
                    if total_faces >= self.max_faces:
                        break

                    # Transform face box to original image coords
                    fx1 = face_boxes_crop[j, 0] * scale_x + p_x1
                    fy1 = face_boxes_crop[j, 1] * scale_y + p_y1
                    fx2 = face_boxes_crop[j, 2] * scale_x + p_x1
                    fy2 = face_boxes_crop[j, 3] * scale_y + p_y1

                    # Transform landmarks to original image coords
                    lmks = landmarks_crop[j].reshape(5, 2).copy()
                    lmks[:, 0] = lmks[:, 0] * scale_x + p_x1
                    lmks[:, 1] = lmks[:, 1] * scale_y + p_y1

                    # Align face
                    try:
                        aligned = self._align_face(image_np, lmks)
                        all_aligned_faces.append(aligned)
                        all_face_info.append({
                            'box': [fx1 / img_w, fy1 / img_h, fx2 / img_w, fy2 / img_h],
                            'landmarks': lmks.flatten() / np.array([img_w, img_h] * 5),
                            'score': scores[j],
                            'person_idx': person_idx,
                        })
                        total_faces += 1
                    except Exception as e:
                        logger.warning(f"Face alignment failed: {e}")

            # Get ArcFace embeddings for all faces
            if all_aligned_faces:
                faces_batch = np.stack(all_aligned_faces, axis=0)
                arcface_embeddings = self._call_arcface(faces_batch)

                num_faces[0] = len(all_face_info)
                for i, (info, emb) in enumerate(zip(all_face_info, arcface_embeddings)):
                    face_embeddings[i] = emb
                    face_boxes[i] = info['box']
                    face_landmarks[i] = info['landmarks']
                    face_scores[i] = info['score']
                    face_person_idx[i] = info['person_idx']

        return self._create_response(
            box_embeddings, normalized_boxes,
            num_faces, face_embeddings, face_boxes, face_landmarks, face_scores, face_person_idx
        )

    def _create_response(self, box_embeddings, normalized_boxes, num_faces,
                         face_embeddings, face_boxes, face_landmarks, face_scores, face_person_idx):
        # Add batch dim
        box_embeddings = box_embeddings.unsqueeze(0)
        normalized_boxes = normalized_boxes.unsqueeze(0)

        outputs = [
            pb_utils.Tensor.from_dlpack('box_embeddings', torch.to_dlpack(box_embeddings)),
            pb_utils.Tensor.from_dlpack('normalized_boxes', torch.to_dlpack(normalized_boxes)),
            pb_utils.Tensor('num_faces', num_faces),
            pb_utils.Tensor('face_embeddings', face_embeddings),
            pb_utils.Tensor('face_boxes', face_boxes),
            pb_utils.Tensor('face_landmarks', face_landmarks),
            pb_utils.Tensor('face_scores', face_scores),
            pb_utils.Tensor('face_person_idx', face_person_idx),
        ]
        return pb_utils.InferenceResponse(output_tensors=outputs)

    def _create_error_response(self, error_msg):
        box_embeddings = np.zeros((1, self.max_boxes, self.embed_dim), dtype=np.float32)
        normalized_boxes = np.zeros((1, self.max_boxes, 4), dtype=np.float32)
        return pb_utils.InferenceResponse(
            output_tensors=[
                pb_utils.Tensor('box_embeddings', box_embeddings),
                pb_utils.Tensor('normalized_boxes', normalized_boxes),
                pb_utils.Tensor('num_faces', np.array([0], dtype=np.int32)),
                pb_utils.Tensor('face_embeddings', np.zeros((self.max_faces, 512), dtype=np.float32)),
                pb_utils.Tensor('face_boxes', np.zeros((self.max_faces, 4), dtype=np.float32)),
                pb_utils.Tensor('face_landmarks', np.zeros((self.max_faces, 10), dtype=np.float32)),
                pb_utils.Tensor('face_scores', np.zeros(self.max_faces, dtype=np.float32)),
                pb_utils.Tensor('face_person_idx', np.zeros(self.max_faces, dtype=np.int32) - 1),
            ],
            error=pb_utils.TritonError(error_msg)
        )

    def finalize(self):
        logger.info("Finalizing unified_embedding_extractor")
