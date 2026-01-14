#!/usr/bin/env python3
"""
Triton Python Backend: Unified Embedding Extractor with Face Detection

This backend combines per-box MobileCLIP embeddings with face detection/recognition.
For person detections, it runs YOLO11-Face End2End on the crop and extracts ArcFace
embeddings for any detected faces.

Key Features:
1. MobileCLIP embeddings for ALL detected boxes
2. YOLO11-Face End2End detection ONLY on person crops (class_id=0)
3. BATCHED face detection with TensorRT GPU NMS for maximum throughput
4. ArcFace embeddings for detected faces
5. Much more efficient than full-image face detection

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
        3. Resize person crops to 640x640 → YOLO11-Face End2End (BATCHED)
        4. For each detected face: MTCNN-style crop → ArcFace embedding

    Outputs:
        - box_embeddings: [300, 512] MobileCLIP embeddings per box
        - normalized_boxes: [300, 4] Normalized boxes
        - num_faces: [1] Total faces detected across all persons
        - face_embeddings: [64, 512] ArcFace embeddings per face
        - face_boxes: [64, 4] Face boxes (normalized to original image)
        - face_landmarks: [64, 10] 5-point landmarks (zeros for YOLO11-Face)
        - face_scores: [64] Detection confidence
        - face_person_idx: [64] Which person box each face belongs to
"""

import json
import logging

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.ops as ops
import triton_python_backend_utils as pb_utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TritonPythonModel:
    """Unified Embedding Extractor with Face Detection"""

    def initialize(self, args):
        self.model_config = json.loads(args['model_config'])

        # Box embedding config
        self.max_boxes = 300
        self.embed_dim = 512
        self.mobileclip_size = 256
        self.yolo_size = 640

        # Face detection config (YOLO11-Face End2End with batching)
        self.max_faces = 64
        self.face_input_size = 640  # YOLO11-Face input size
        self.arcface_size = 112
        self.face_conf_threshold = 0.5
        self.person_class_id = 0  # COCO person class

        # YOLO11-Face End2End model name (supports batching up to 64)
        self.face_model_name = "yolo11_face_small_trt_end2end"
        self.face_max_det = 100  # Max detections per image from End2End model

        # MTCNN-style face margin for cropping (40% expansion)
        self.face_margin = 0.4

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Initialized unified_embedding_extractor on {self.device}")
        logger.info(f"  Max boxes: {self.max_boxes}, Max faces: {self.max_faces}")
        logger.info(f"  Face detector: {self.face_model_name} (batched End2End)")
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

    def _call_yolo11_face_batch(self, images):
        """
        Call YOLO11-Face End2End via BLS with batching support.

        The End2End model has GPU NMS built-in and supports batching up to 64 images.
        This is much faster than processing images one at a time.

        Args:
            images: [N, 3, 640, 640] tensor of person crops (normalized [0, 1])

        Returns:
            List of dicts, one per image, each containing:
                - num_dets: int, number of faces detected
                - boxes: [num_dets, 4] boxes in normalized [0,1] xyxy format
                - scores: [num_dets] confidence scores
        """
        if images.shape[0] == 0:
            return []

        if isinstance(images, torch.Tensor):
            images_np = images.cpu().numpy().astype(np.float32)
        else:
            images_np = images.astype(np.float32)

        # YOLO11-Face End2End supports batching - process all at once
        input_tensor = pb_utils.Tensor("images", images_np)

        request = pb_utils.InferenceRequest(
            model_name=self.face_model_name,
            requested_output_names=["num_dets", "det_boxes", "det_scores", "det_classes"],
            inputs=[input_tensor],
            preferred_memory=pb_utils.PreferredMemory(pb_utils.TRITONSERVER_MEMORY_CPU),
        )
        response = request.exec()

        if response.has_error():
            logger.warning(f"YOLO11-Face error: {response.error().message()}")
            return [None] * images.shape[0]

        # Parse batched outputs
        num_dets = pb_utils.get_output_tensor_by_name(response, "num_dets").as_numpy()
        det_boxes = pb_utils.get_output_tensor_by_name(response, "det_boxes").as_numpy()
        det_scores = pb_utils.get_output_tensor_by_name(response, "det_scores").as_numpy()

        # Split into per-image results
        results = []
        for i in range(images.shape[0]):
            n = int(num_dets[i, 0])
            results.append({
                'num_dets': n,
                'boxes': det_boxes[i, :n],  # [n, 4] normalized xyxy
                'scores': det_scores[i, :n],  # [n]
            })

        return results

    def _decode_yolo11_outputs(self, result, crop_size):
        """
        Decode YOLO11-Face End2End outputs to face boxes and scores.

        The End2End model outputs are already post-NMS in normalized [0,1] coordinates.

        Args:
            result: Dict with 'num_dets', 'boxes', 'scores' from _call_yolo11_face_batch
            crop_size: Size of the input crop (for scaling)

        Returns:
            (boxes, scores) - boxes in pixel coordinates for the crop, scores
        """
        if result is None or result['num_dets'] == 0:
            return np.array([]), np.array([])

        # Boxes are in normalized [0,1] format, convert to pixel coords
        boxes_norm = result['boxes']  # [n, 4]
        scores = result['scores']  # [n]

        # Filter by confidence threshold
        mask = scores > self.face_conf_threshold
        if not np.any(mask):
            return np.array([]), np.array([])

        boxes_norm = boxes_norm[mask]
        scores = scores[mask]

        # Convert to pixel coordinates
        boxes_pixel = boxes_norm.copy()
        boxes_pixel[:, 0] *= crop_size  # x1
        boxes_pixel[:, 1] *= crop_size  # y1
        boxes_pixel[:, 2] *= crop_size  # x2
        boxes_pixel[:, 3] *= crop_size  # y2

        return boxes_pixel, scores

    def _mtcnn_crop_face(self, image_np, box):
        """
        MTCNN-style face cropping with margin expansion.

        Since YOLO11-Face End2End doesn't provide landmarks, we use
        MTCNN/FaceNet-style bounding box expansion for face cropping.

        Args:
            image_np: [H, W, 3] numpy array (uint8)
            box: [4] face box in pixel coordinates [x1, y1, x2, y2]

        Returns:
            [112, 112, 3] cropped and resized face (uint8)
        """
        import cv2

        x1, y1, x2, y2 = box
        face_w = x2 - x1
        face_h = y2 - y1

        # Expand with margin (MTCNN style)
        margin_w = face_w * self.face_margin
        margin_h = face_h * self.face_margin

        x1_exp = max(0, x1 - margin_w)
        y1_exp = max(0, y1 - margin_h)
        x2_exp = min(image_np.shape[1], x2 + margin_w)
        y2_exp = min(image_np.shape[0], y2 + margin_h)

        # Make square (take larger dimension)
        new_w = x2_exp - x1_exp
        new_h = y2_exp - y1_exp
        if new_w > new_h:
            diff = (new_w - new_h) / 2
            y1_exp = max(0, y1_exp - diff)
            y2_exp = min(image_np.shape[0], y2_exp + diff)
        else:
            diff = (new_h - new_w) / 2
            x1_exp = max(0, x1_exp - diff)
            x2_exp = min(image_np.shape[1], x2_exp + diff)

        # Crop and resize
        x1_exp, y1_exp, x2_exp, y2_exp = int(x1_exp), int(y1_exp), int(x2_exp), int(y2_exp)
        if x2_exp <= x1_exp or y2_exp <= y1_exp:
            return np.zeros((self.arcface_size, self.arcface_size, 3), dtype=np.uint8)

        crop = image_np[y1_exp:y2_exp, x1_exp:x2_exp]
        resized = cv2.resize(crop, (self.arcface_size, self.arcface_size), interpolation=cv2.INTER_LINEAR)

        return resized

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

        # Transform boxes from YOLO 640x640 letterbox-normalized space to original pixel space
        # The det_boxes are normalized [0,1] in 640x640 letterbox space
        # We need to: (box_640 - pad) / scale to get original pixel coords
        boxes_pixel = valid_boxes.clone()

        # First convert to 640x640 pixel coords
        boxes_pixel *= self.face_input_size  # face_input_size = 640

        # Then apply inverse letterbox transformation using affine matrix
        # affine_matrix = [[scale, 0, pad_x], [0, scale, pad_y]]
        # Inverse: orig_coord = (letterbox_coord - pad) / scale
        boxes_pixel[:, 0] = (boxes_pixel[:, 0] - pad_x) / scale  # x1
        boxes_pixel[:, 2] = (boxes_pixel[:, 2] - pad_x) / scale  # x2
        boxes_pixel[:, 1] = (boxes_pixel[:, 1] - pad_y) / scale  # y1
        boxes_pixel[:, 3] = (boxes_pixel[:, 3] - pad_y) / scale  # y2

        # Clamp to image bounds
        boxes_pixel[:, [0, 2]] = torch.clamp(boxes_pixel[:, [0, 2]], 0, img_w)
        boxes_pixel[:, [1, 3]] = torch.clamp(boxes_pixel[:, [1, 3]], 0, img_h)

        # Also compute normalized boxes in original image space (for output)
        valid_boxes_normalized = boxes_pixel.clone()
        valid_boxes_normalized[:, [0, 2]] /= img_w
        valid_boxes_normalized[:, [1, 3]] /= img_h

        # === Part 1: MobileCLIP embeddings for ALL boxes ===
        crops = self._crop_boxes(original_image, boxes_pixel, self.mobileclip_size)
        if crops.shape[0] > 0:
            embeddings = self._call_mobileclip(crops)
            actual_num = min(embeddings.shape[0], self.max_boxes)
            box_embeddings[:actual_num] = embeddings[:actual_num]
            # Use properly transformed normalized boxes (not letterbox-normalized)
            normalized_boxes[:num_dets] = valid_boxes_normalized

        # === Part 2: Face detection on PERSON boxes only ===
        person_mask = valid_classes == self.person_class_id
        person_indices = torch.where(person_mask)[0]

        if len(person_indices) > 0:
            # Crop person boxes at YOLO11-Face input size
            person_boxes_pixel = boxes_pixel[person_indices]
            person_crops = self._crop_boxes(original_image, person_boxes_pixel, self.face_input_size)

            # original_image is already normalized [0, 1] from DALI, so crops are too
            # YOLO11-Face End2End expects [0, 1] normalized input

            # Run YOLO11-Face End2End on ALL person crops in ONE BATCHED CALL
            # This is much faster than processing one at a time
            yolo_face_results = self._call_yolo11_face_batch(person_crops)

            # Convert original image to numpy for face cropping
            image_np = original_image.permute(1, 2, 0).cpu().numpy()
            if image_np.max() <= 1.0:
                image_np = (image_np * 255).astype(np.uint8)
            else:
                image_np = image_np.astype(np.uint8)

            # Process each person's faces
            all_cropped_faces = []
            all_face_info = []
            total_faces = 0

            for i, (person_idx, yolo_out) in enumerate(zip(person_indices, yolo_face_results)):
                person_idx = int(person_idx.item())
                person_box = person_boxes_pixel[i].cpu().numpy()
                p_x1, p_y1, p_x2, p_y2 = person_box

                # Decode YOLO11-Face End2End outputs (in 640x640 crop space)
                face_boxes_crop, scores = self._decode_yolo11_outputs(
                    yolo_out, self.face_input_size
                )

                if len(face_boxes_crop) == 0:
                    continue

                # Scale face coords from crop space to original image space
                crop_w = p_x2 - p_x1
                crop_h = p_y2 - p_y1
                scale_x = crop_w / self.face_input_size
                scale_y = crop_h / self.face_input_size

                for j in range(len(face_boxes_crop)):
                    if total_faces >= self.max_faces:
                        break

                    # Transform face box to original image coords
                    fx1 = face_boxes_crop[j, 0] * scale_x + p_x1
                    fy1 = face_boxes_crop[j, 1] * scale_y + p_y1
                    fx2 = face_boxes_crop[j, 2] * scale_x + p_x1
                    fy2 = face_boxes_crop[j, 3] * scale_y + p_y1

                    # MTCNN-style face cropping (no landmarks needed)
                    try:
                        face_box_orig = np.array([fx1, fy1, fx2, fy2])
                        cropped = self._mtcnn_crop_face(image_np, face_box_orig)
                        all_cropped_faces.append(cropped)
                        all_face_info.append({
                            'box': [fx1 / img_w, fy1 / img_h, fx2 / img_w, fy2 / img_h],
                            'landmarks': np.zeros(10),  # No landmarks from YOLO11-Face End2End
                            'score': float(scores[j]),
                            'person_idx': person_idx,
                        })
                        total_faces += 1
                    except Exception as e:
                        logger.warning(f"Face cropping failed: {e}")

            # Get ArcFace embeddings for all faces
            if all_cropped_faces:
                faces_batch = np.stack(all_cropped_faces, axis=0)
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
