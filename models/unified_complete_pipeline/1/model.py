#!/usr/bin/env python3
"""
Triton Python Backend: Unified Complete Analysis Pipeline

Complete GPU-accelerated pipeline combining all analysis capabilities:
1. YOLO object detection
2. MobileCLIP global image embedding
3. MobileCLIP per-box embeddings
4. Face detection (selectable: SCRFD on person crops OR YOLO11-face full image)
5. ArcFace face embeddings
6. PP-OCRv5 text detection and recognition

Face Model Selection:
- "scrfd" (default): SCRFD face detection on person crops only (more efficient)
- "yolo11": YOLO11-face detection on full image (may detect more faces)

Architecture:
    Inputs:
        - encoded_images: Raw JPEG/PNG bytes
        - affine_matrices: YOLO letterbox transformation [2, 3]
        - face_model: STRING "scrfd" or "yolo11" (default: "scrfd")

    Processing:
        For SCRFD mode:
            1. Call yolo_unified_ensemble (detection + embeddings + faces in one pass)
            2. Run OCR

        For YOLO11 mode:
            1. Call yolo_mobileclip_ensemble (detection + embeddings only)
            2. Call dual_preprocess_dali for HD image + YOLO11-face preprocessing
            3. Call yolo11_face_pipeline for full-image face detection
            4. Run OCR

    Outputs:
        All detection, embedding, face, and OCR results
"""

import json
import logging
import math
from pathlib import Path

import cv2
import numpy as np
import triton_python_backend_utils as pb_utils

# GPU-accelerated libraries
try:
    import cupy as cp
    import torch
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    cp = None
    torch = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OCR constants
MAX_TEXTS = 128
DET_MODEL = 'paddleocr_det_trt'
REC_MODEL = 'paddleocr_rec_trt'
REC_HEIGHT = 48
DICT_PATH = Path('/models/paddleocr_rec_trt/en_ppocrv5_dict.txt')


class DBPostProcess:
    """Post-processing for DB text detection."""

    def __init__(
        self,
        thresh: float = 0.2,
        box_thresh: float = 0.4,
        max_candidates: int = 1000,
        unclip_ratio: float = 1.5,
        use_dilation: bool = True,
    ):
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        self.use_dilation = use_dilation

        if use_dilation:
            self.dilation_kernel = np.array([[1, 1], [1, 1]], dtype=np.uint8)
        else:
            self.dilation_kernel = None

    def __call__(self, pred: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        if pred.ndim == 4:
            pred = pred[0, 0]
        elif pred.ndim == 3:
            pred = pred[0]

        src_h, src_w = shape
        mask = (pred > self.thresh).astype(np.uint8) * 255

        if self.dilation_kernel is not None:
            mask = cv2.dilate(mask, self.dilation_kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        scores = []

        for contour in contours[: self.max_candidates]:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)

            x, y = int(rect[0][0]), int(rect[0][1])
            x = min(max(x, 0), pred.shape[1] - 1)
            y = min(max(y, 0), pred.shape[0] - 1)
            score = float(pred[y, x])

            if score < self.box_thresh:
                continue

            box = self._unclip(box)
            if box is None:
                continue

            rect = cv2.minAreaRect(box)
            if min(rect[1]) < 3:
                continue

            box = cv2.boxPoints(rect)
            scale_x = src_w / pred.shape[1]
            scale_y = src_h / pred.shape[0]
            box[:, 0] *= scale_x
            box[:, 1] *= scale_y
            box[:, 0] = np.clip(box[:, 0], 0, src_w - 1)
            box[:, 1] = np.clip(box[:, 1], 0, src_h - 1)
            box = self._order_points(box)

            boxes.append(box)
            scores.append(score)

        if len(boxes) == 0:
            return np.empty((0, 4, 2), dtype=np.float32), np.empty(0, dtype=np.float32)

        return np.array(boxes, dtype=np.float32), np.array(scores, dtype=np.float32)

    def _unclip(self, box: np.ndarray) -> np.ndarray | None:
        try:
            import pyclipper

            poly = pyclipper.Pyclipper()
            poly.AddPath(box.astype(np.int32).tolist(), pyclipper.PT_SUBJECT, True)

            area = cv2.contourArea(box)
            peri = cv2.arcLength(box, True)
            if peri == 0:
                return None
            distance = area * self.unclip_ratio / peri

            expanded = poly.Execute(
                pyclipper.CT_UNION, pyclipper.PFT_EVENODD, pyclipper.PFT_EVENODD
            )

            if len(expanded) == 0:
                return None

            offset = pyclipper.PyclipperOffset()
            offset.AddPath(expanded[0], pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            result = offset.Execute(distance)

            if len(result) == 0:
                return None

            return np.array(result[0], dtype=np.float32)
        except ImportError:
            center = box.mean(axis=0)
            expanded = center + (box - center) * self.unclip_ratio
            return expanded

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        d = np.diff(pts, axis=1).squeeze()
        rect[1] = pts[np.argmin(d)]
        rect[3] = pts[np.argmax(d)]
        return rect


class CTCLabelDecode:
    """CTC decoder for text recognition."""

    def __init__(self, character_dict_path: str | Path):
        self.character = ['blank']
        with open(character_dict_path, 'r', encoding='utf-8') as f:
            for line in f:
                char = line.strip('\n')
                self.character.append(char)
        self.character.append(' ')

    def __call__(self, preds: np.ndarray) -> list[tuple[str, float]]:
        results = []
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)

        for idx, prob in zip(preds_idx, preds_prob):
            text = ''
            conf_list = []
            prev_idx = 0

            for char_idx, char_prob in zip(idx, prob):
                if char_idx != 0 and char_idx != prev_idx:
                    if char_idx < len(self.character):
                        text += self.character[char_idx]
                        conf_list.append(char_prob)
                prev_idx = char_idx

            conf = float(np.mean(conf_list)) if conf_list else 0.0
            results.append((text, conf))

        return results


class TritonPythonModel:
    """Unified Complete Analysis Pipeline with Selectable Face Model"""

    def initialize(self, args):
        self.model_config = json.loads(args['model_config'])

        # Detection config
        self.max_dets = 300
        self.max_faces = 64
        self.max_texts = MAX_TEXTS
        self.embed_dim = 512

        # OCR config
        self.rec_height = REC_HEIGHT
        self.ocr_limit_side = 960
        self.ocr_min_side = 640

        self.device = 'cuda' if HAS_GPU and torch.cuda.is_available() else 'cpu'

        # DB post-processor
        self.db_postprocess = DBPostProcess(
            thresh=0.2, box_thresh=0.4, max_candidates=1000,
            unclip_ratio=1.5, use_dilation=True
        )

        # CTC decoder
        if DICT_PATH.exists():
            self.ctc_decode = CTCLabelDecode(DICT_PATH)
            logger.info(f'Loaded OCR dictionary: {len(self.ctc_decode.character)} characters')
        else:
            self.ctc_decode = None
            logger.warning(f'OCR dictionary not found: {DICT_PATH}')

        logger.info(f'Initialized unified_complete_pipeline (device={self.device})')
        logger.info('  Face models: SCRFD (person crops) / YOLO11 (full image)')
        logger.info('  Components: YOLO + CLIP + Face + ArcFace + OCR')

    def _triton_to_numpy(self, tensor):
        """Convert Triton tensor to NumPy."""
        if tensor.is_cpu():
            return tensor.as_numpy()
        if HAS_GPU:
            dlpack = tensor.to_dlpack()
            return cp.from_dlpack(dlpack).get()
        return tensor.as_numpy()

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                response = self._process_request(request)
                responses.append(response)
            except Exception as e:
                logger.error(f'Error: {e}', exc_info=True)
                responses.append(self._create_error_response(str(e)))
        return responses

    def _process_request(self, request):
        # Get inputs
        encoded_images = pb_utils.get_input_tensor_by_name(request, 'encoded_images')
        affine_matrices = pb_utils.get_input_tensor_by_name(request, 'affine_matrices')

        # Get optional face_model parameter (default: "scrfd")
        face_model_tensor = pb_utils.get_input_tensor_by_name(request, 'face_model')
        if face_model_tensor is not None:
            # Handle various array shapes from client ([1], [1,1], etc.)
            face_model_arr = face_model_tensor.as_numpy().flatten()
            if len(face_model_arr) > 0:
                face_model_bytes = face_model_arr[0]
                if isinstance(face_model_bytes, bytes):
                    face_model = face_model_bytes.decode('utf-8').lower().strip()
                else:
                    face_model = str(face_model_bytes).lower().strip()
            else:
                face_model = 'scrfd'
        else:
            face_model = 'scrfd'

        logger.info(f'Processing with face_model={face_model}')

        if face_model == 'yolo11':
            return self._process_with_yolo11_face(encoded_images, affine_matrices)
        else:
            return self._process_with_scrfd(encoded_images, affine_matrices)

    def _process_with_scrfd(self, encoded_images, affine_matrices):
        """Process using SCRFD face detection on person crops (default, more efficient)."""

        # =================================================================
        # Step 1: Call yolo_unified_ensemble for detection + embeddings + faces
        # =================================================================
        unified_request = pb_utils.InferenceRequest(
            model_name='yolo_unified_ensemble',
            requested_output_names=[
                'num_dets', 'det_boxes', 'det_scores', 'det_classes',
                'global_embeddings', 'box_embeddings', 'normalized_boxes',
                'num_faces', 'face_embeddings', 'face_boxes', 'face_landmarks',
                'face_scores', 'face_person_idx'
            ],
            inputs=[encoded_images, affine_matrices]
        )
        unified_response = unified_request.exec()

        if unified_response.has_error():
            raise RuntimeError(f'yolo_unified_ensemble failed: {unified_response.error().message()}')

        # Extract all outputs
        result = self._extract_unified_outputs(unified_response)

        # =================================================================
        # Step 2: Run OCR
        # =================================================================
        image_bytes = encoded_images.as_numpy().flatten().tobytes()
        ocr_result = self._run_ocr(image_bytes)

        return self._create_response(result, ocr_result, face_model='scrfd')

    def _process_with_yolo11_face(self, encoded_images, affine_matrices):
        """Process using YOLO11-face detection on full image."""

        # =================================================================
        # Step 1: Call yolo_mobileclip_ensemble for detection + embeddings (no faces)
        # =================================================================
        yolo_clip_request = pb_utils.InferenceRequest(
            model_name='yolo_mobileclip_ensemble',
            requested_output_names=[
                'num_dets', 'det_boxes', 'det_scores', 'det_classes',
                'global_embeddings', 'box_embeddings', 'normalized_boxes'
            ],
            inputs=[encoded_images, affine_matrices]
        )
        yolo_clip_response = yolo_clip_request.exec()

        if yolo_clip_response.has_error():
            raise RuntimeError(f'yolo_mobileclip_ensemble failed: {yolo_clip_response.error().message()}')

        # Extract detection + embedding outputs
        num_dets = self._triton_to_numpy(
            pb_utils.get_output_tensor_by_name(yolo_clip_response, 'num_dets')
        )
        det_boxes = self._triton_to_numpy(
            pb_utils.get_output_tensor_by_name(yolo_clip_response, 'det_boxes')
        )
        det_scores = self._triton_to_numpy(
            pb_utils.get_output_tensor_by_name(yolo_clip_response, 'det_scores')
        )
        det_classes = self._triton_to_numpy(
            pb_utils.get_output_tensor_by_name(yolo_clip_response, 'det_classes')
        )
        global_embeddings = self._triton_to_numpy(
            pb_utils.get_output_tensor_by_name(yolo_clip_response, 'global_embeddings')
        )
        box_embeddings = self._triton_to_numpy(
            pb_utils.get_output_tensor_by_name(yolo_clip_response, 'box_embeddings')
        )
        normalized_boxes = self._triton_to_numpy(
            pb_utils.get_output_tensor_by_name(yolo_clip_response, 'normalized_boxes')
        )

        # =================================================================
        # Step 2: Get HD image + YOLO preprocessed image from DALI
        # =================================================================
        dali_request = pb_utils.InferenceRequest(
            model_name='dual_preprocess_dali',
            requested_output_names=['yolo_images', 'clip_images', 'original_images'],
            inputs=[encoded_images, affine_matrices]
        )
        dali_response = dali_request.exec()

        if dali_response.has_error():
            raise RuntimeError(f'dual_preprocess_dali failed: {dali_response.error().message()}')

        yolo_images = pb_utils.get_output_tensor_by_name(dali_response, 'yolo_images')
        original_images = pb_utils.get_output_tensor_by_name(dali_response, 'original_images')

        # Get original shape from HD image
        orig_images_np = self._triton_to_numpy(original_images)
        if orig_images_np.ndim == 4:
            _, _, orig_h, orig_w = orig_images_np.shape
        else:
            _, orig_h, orig_w = orig_images_np.shape

        orig_shape = np.array([orig_h, orig_w], dtype=np.int32)

        # =================================================================
        # Step 3: Call yolo11_face_pipeline for face detection
        # =================================================================
        # Create properly named input tensors for yolo11_face_pipeline
        yolo_images_np = self._triton_to_numpy(yolo_images)
        original_images_np = self._triton_to_numpy(original_images)

        # Get affine matrix for coordinate transformation
        affine_matrix_np = self._triton_to_numpy(affine_matrices)
        if affine_matrix_np.ndim == 3:
            affine_matrix_np = affine_matrix_np[0]  # Remove batch dim: [1, 2, 3] -> [2, 3]

        face_request = pb_utils.InferenceRequest(
            model_name='yolo11_face_pipeline',
            requested_output_names=[
                'num_faces', 'face_boxes', 'face_landmarks',
                'face_scores', 'face_embeddings', 'face_quality'
            ],
            inputs=[
                pb_utils.Tensor('face_images', yolo_images_np),
                pb_utils.Tensor('original_image', original_images_np),
                pb_utils.Tensor('orig_shape', orig_shape[np.newaxis]),
                pb_utils.Tensor('affine_matrix', affine_matrix_np[np.newaxis])  # Pass affine matrix
            ]
        )
        face_response = face_request.exec()

        if face_response.has_error():
            logger.warning(f'yolo11_face_pipeline failed: {face_response.error().message()}')
            # Return with empty face results
            num_faces = np.array([[0]], dtype=np.int32)
            face_embeddings = np.zeros((1, self.max_faces, 512), dtype=np.float32)
            face_boxes = np.zeros((1, self.max_faces, 4), dtype=np.float32)
            face_landmarks = np.zeros((1, self.max_faces, 10), dtype=np.float32)
            face_scores = np.zeros((1, self.max_faces), dtype=np.float32)
            face_person_idx = np.zeros((1, self.max_faces), dtype=np.int32) - 1
        else:
            # Parse face outputs from yolo11_face_pipeline (max 128 faces)
            num_faces_raw = self._triton_to_numpy(
                pb_utils.get_output_tensor_by_name(face_response, 'num_faces')
            )
            n_faces = int(num_faces_raw.flatten()[0])
            # Cap at our max_faces limit (64)
            n_faces = min(n_faces, self.max_faces)
            num_faces = np.array([[n_faces]], dtype=np.int32)

            # Get raw outputs (shape [128, *] from yolo11_face_pipeline)
            face_emb_raw = self._triton_to_numpy(
                pb_utils.get_output_tensor_by_name(face_response, 'face_embeddings')
            )
            face_boxes_raw = self._triton_to_numpy(
                pb_utils.get_output_tensor_by_name(face_response, 'face_boxes')
            )
            face_lmk_raw = self._triton_to_numpy(
                pb_utils.get_output_tensor_by_name(face_response, 'face_landmarks')
            )
            face_scores_raw = self._triton_to_numpy(
                pb_utils.get_output_tensor_by_name(face_response, 'face_scores')
            )

            # Create output arrays with correct shape for unified_complete_pipeline (64 max)
            face_embeddings = np.zeros((1, self.max_faces, 512), dtype=np.float32)
            face_boxes = np.zeros((1, self.max_faces, 4), dtype=np.float32)
            face_landmarks = np.zeros((1, self.max_faces, 10), dtype=np.float32)
            face_scores = np.zeros((1, self.max_faces), dtype=np.float32)
            face_person_idx = np.zeros((1, self.max_faces), dtype=np.int32) - 1

            # Copy detected faces (up to max_faces)
            if n_faces > 0:
                # Handle different input shapes (with or without batch dim)
                if face_emb_raw.ndim == 3:
                    face_embeddings[0, :n_faces] = face_emb_raw[0, :n_faces]
                    face_boxes[0, :n_faces] = face_boxes_raw[0, :n_faces]
                    face_landmarks[0, :n_faces] = face_lmk_raw[0, :n_faces]
                    face_scores[0, :n_faces] = face_scores_raw[0, :n_faces] if face_scores_raw.ndim == 2 else face_scores_raw[:n_faces]
                else:
                    face_embeddings[0, :n_faces] = face_emb_raw[:n_faces]
                    face_boxes[0, :n_faces] = face_boxes_raw[:n_faces]
                    face_landmarks[0, :n_faces] = face_lmk_raw[:n_faces]
                    face_scores[0, :n_faces] = face_scores_raw[:n_faces]

            logger.info(f'YOLO11-face detected {n_faces} faces')

        result = {
            'num_dets': num_dets,
            'det_boxes': det_boxes,
            'det_scores': det_scores,
            'det_classes': det_classes,
            'global_embeddings': global_embeddings,
            'box_embeddings': box_embeddings,
            'normalized_boxes': normalized_boxes,
            'num_faces': num_faces,
            'face_embeddings': face_embeddings,
            'face_boxes': face_boxes,
            'face_landmarks': face_landmarks,
            'face_scores': face_scores,
            'face_person_idx': face_person_idx,
        }

        # =================================================================
        # Step 4: Run OCR
        # =================================================================
        image_bytes = encoded_images.as_numpy().flatten().tobytes()
        ocr_result = self._run_ocr(image_bytes)

        return self._create_response(result, ocr_result, face_model='yolo11')

    def _extract_unified_outputs(self, unified_response):
        """Extract all outputs from yolo_unified_ensemble."""
        return {
            'num_dets': self._triton_to_numpy(
                pb_utils.get_output_tensor_by_name(unified_response, 'num_dets')
            ),
            'det_boxes': self._triton_to_numpy(
                pb_utils.get_output_tensor_by_name(unified_response, 'det_boxes')
            ),
            'det_scores': self._triton_to_numpy(
                pb_utils.get_output_tensor_by_name(unified_response, 'det_scores')
            ),
            'det_classes': self._triton_to_numpy(
                pb_utils.get_output_tensor_by_name(unified_response, 'det_classes')
            ),
            'global_embeddings': self._triton_to_numpy(
                pb_utils.get_output_tensor_by_name(unified_response, 'global_embeddings')
            ),
            'box_embeddings': self._triton_to_numpy(
                pb_utils.get_output_tensor_by_name(unified_response, 'box_embeddings')
            ),
            'normalized_boxes': self._triton_to_numpy(
                pb_utils.get_output_tensor_by_name(unified_response, 'normalized_boxes')
            ),
            'num_faces': self._triton_to_numpy(
                pb_utils.get_output_tensor_by_name(unified_response, 'num_faces')
            ),
            'face_embeddings': self._triton_to_numpy(
                pb_utils.get_output_tensor_by_name(unified_response, 'face_embeddings')
            ),
            'face_boxes': self._triton_to_numpy(
                pb_utils.get_output_tensor_by_name(unified_response, 'face_boxes')
            ),
            'face_landmarks': self._triton_to_numpy(
                pb_utils.get_output_tensor_by_name(unified_response, 'face_landmarks')
            ),
            'face_scores': self._triton_to_numpy(
                pb_utils.get_output_tensor_by_name(unified_response, 'face_scores')
            ),
            'face_person_idx': self._triton_to_numpy(
                pb_utils.get_output_tensor_by_name(unified_response, 'face_person_idx')
            ),
        }

    def _run_ocr(self, image_bytes: bytes) -> dict:
        """Run OCR detection and recognition on the image."""
        if self.ctc_decode is None:
            return self._empty_ocr_result()

        try:
            img_array = np.frombuffer(image_bytes, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if img is None:
                logger.warning('OCR: Failed to decode image')
                return self._empty_ocr_result()

            orig_h, orig_w = img.shape[:2]
            ocr_input, resize_h, resize_w = self._preprocess_ocr(img)
            det_output = self._call_ocr_detection(ocr_input)
            boxes, det_scores = self.db_postprocess(det_output, (orig_h, orig_w))

            if len(boxes) == 0:
                return self._empty_ocr_result()

            boxes = self._sorted_boxes(boxes)
            if len(boxes) > self.max_texts:
                boxes = boxes[:self.max_texts]
                det_scores = det_scores[:self.max_texts]

            text_crops = self._get_text_crops(img, boxes)
            if len(text_crops) == 0:
                return self._empty_ocr_result()

            texts, rec_scores = self._call_recognition(text_crops)

            return {
                'num_texts': len(boxes),
                'boxes': boxes,
                'det_scores': det_scores,
                'texts': texts,
                'rec_scores': rec_scores,
                'orig_shape': (orig_h, orig_w)
            }

        except Exception as e:
            logger.error(f'OCR failed: {e}', exc_info=True)
            return self._empty_ocr_result()

    def _preprocess_ocr(self, img: np.ndarray) -> tuple[np.ndarray, int, int]:
        """Preprocess image for OCR detection."""
        orig_h, orig_w = img.shape[:2]
        max_side = max(orig_h, orig_w)

        if max_side > self.ocr_limit_side:
            ratio = self.ocr_limit_side / max_side
        elif max_side < self.ocr_min_side:
            ratio = self.ocr_min_side / max_side
        else:
            ratio = 1.0

        resize_h = max(32, int(orig_h * ratio))
        resize_w = max(32, int(orig_w * ratio))

        resized = cv2.resize(img, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)

        pad_h = (32 - resize_h % 32) % 32
        pad_w = (32 - resize_w % 32) % 32
        if pad_h > 0 or pad_w > 0:
            padded = np.zeros((resize_h + pad_h, resize_w + pad_w, 3), dtype=np.uint8)
            padded[:resize_h, :resize_w, :] = resized
            resized = padded

        ocr_input = resized[:, :, ::-1].astype(np.float32) / 127.5 - 1.0
        ocr_input = ocr_input.transpose(2, 0, 1)

        return ocr_input, resize_h, resize_w

    def _call_ocr_detection(self, ocr_input: np.ndarray) -> np.ndarray:
        """Call OCR detection model via BLS."""
        input_tensor = pb_utils.Tensor('x', ocr_input[np.newaxis].astype(np.float32))
        request = pb_utils.InferenceRequest(
            model_name=DET_MODEL,
            requested_output_names=['fetch_name_0'],
            inputs=[input_tensor]
        )
        response = request.exec()

        if response.has_error():
            raise RuntimeError(f'OCR detection failed: {response.error().message()}')

        output = pb_utils.get_output_tensor_by_name(response, 'fetch_name_0')
        return self._triton_to_numpy(output)

    def _call_recognition(self, text_crops: list[np.ndarray]) -> tuple[list[str], list[float]]:
        """Call OCR recognition model for each crop."""
        if not text_crops:
            return [], []

        MIN_WIDTH = 8
        MAX_WIDTH = 2048
        all_results = []

        for crop in text_crops:
            h, w = crop.shape[:2]
            ratio = w / float(h)
            target_w = int(math.ceil(self.rec_height * ratio))
            target_w = max(MIN_WIDTH, min(MAX_WIDTH, target_w))

            norm_img = self._resize_norm_img(crop, target_w)
            batch_input = norm_img[np.newaxis].astype(np.float32)

            input_tensor = pb_utils.Tensor('x', batch_input)
            request = pb_utils.InferenceRequest(
                model_name=REC_MODEL,
                requested_output_names=['fetch_name_0'],
                inputs=[input_tensor]
            )
            response = request.exec()

            if response.has_error():
                raise RuntimeError(f'OCR recognition failed: {response.error().message()}')

            output = pb_utils.get_output_tensor_by_name(response, 'fetch_name_0')
            output_np = self._triton_to_numpy(output)
            batch_results = self.ctc_decode(output_np)
            all_results.extend(batch_results)

        return [r[0] for r in all_results], [r[1] for r in all_results]

    def _resize_norm_img(self, img: np.ndarray, target_w: int) -> np.ndarray:
        """Resize and normalize image for recognition."""
        resized = cv2.resize(img, (target_w, self.rec_height))
        resized = resized.astype(np.float32)
        resized = resized.transpose(2, 0, 1) / 255.0
        resized -= 0.5
        resized /= 0.5
        return resized

    def _sorted_boxes(self, boxes: np.ndarray) -> np.ndarray:
        """Sort boxes in reading order."""
        if len(boxes) == 0:
            return boxes

        y_order = np.argsort(boxes[:, 0, 1], kind='stable')
        sorted_y = boxes[y_order, 0, 1]

        line_ids = np.zeros(len(boxes), dtype=np.int32)
        if len(sorted_y) > 1:
            line_ids[1:] = np.cumsum(np.abs(np.diff(sorted_y)) >= 10)

        sort_key = line_ids * 1e6 + boxes[y_order, 0, 0]
        final_order = np.argsort(sort_key, kind='stable')

        return boxes[y_order[final_order]]

    def _get_text_crops(self, img: np.ndarray, boxes: np.ndarray) -> list[np.ndarray]:
        """Crop text regions using perspective transform."""
        crops = []

        for box in boxes:
            width = max(
                np.linalg.norm(box[1] - box[0]),
                np.linalg.norm(box[2] - box[3]),
            )
            height = max(
                np.linalg.norm(box[0] - box[3]),
                np.linalg.norm(box[1] - box[2]),
            )

            width, height = int(width), int(height)
            if width < 3 or height < 3:
                continue

            dst_pts = np.array(
                [[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32
            )
            M = cv2.getPerspectiveTransform(box.astype(np.float32), dst_pts)
            crop = cv2.warpPerspective(img, M, (width, height), borderMode=cv2.BORDER_REPLICATE)

            if height / width >= 1.5:
                crop = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)

            crops.append(crop)

        return crops

    def _empty_ocr_result(self) -> dict:
        return {
            'num_texts': 0,
            'boxes': np.empty((0, 4, 2), dtype=np.float32),
            'det_scores': np.empty(0, dtype=np.float32),
            'texts': [],
            'rec_scores': [],
            'orig_shape': (0, 0)
        }

    def _create_response(self, result: dict, ocr_result: dict, face_model: str):
        """Create combined response with all outputs."""
        # Prepare OCR outputs
        num_texts = np.array([ocr_result['num_texts']], dtype=np.int32)

        text_boxes = np.zeros((self.max_texts, 8), dtype=np.float32)
        text_boxes_norm = np.zeros((self.max_texts, 4), dtype=np.float32)
        text_det_scores = np.zeros(self.max_texts, dtype=np.float32)
        text_rec_scores = np.zeros(self.max_texts, dtype=np.float32)
        texts_bytes = np.array([b'' for _ in range(self.max_texts)], dtype=object)

        if ocr_result['num_texts'] > 0:
            orig_h, orig_w = ocr_result['orig_shape']
            for i, box in enumerate(ocr_result['boxes'][:self.max_texts]):
                text_boxes[i] = box.flatten()
                x1 = box[:, 0].min() / orig_w
                y1 = box[:, 1].min() / orig_h
                x2 = box[:, 0].max() / orig_w
                y2 = box[:, 1].max() / orig_h
                text_boxes_norm[i] = [x1, y1, x2, y2]
                text_det_scores[i] = ocr_result['det_scores'][i] if i < len(ocr_result['det_scores']) else 0.0
                text_rec_scores[i] = ocr_result['rec_scores'][i] if i < len(ocr_result['rec_scores']) else 0.0
                texts_bytes[i] = ocr_result['texts'][i].encode('utf-8') if i < len(ocr_result['texts']) else b''

        # Face model indicator
        face_model_bytes = np.array([face_model.encode('utf-8')], dtype=object)

        output_tensors = [
            # Detection outputs
            pb_utils.Tensor('num_dets', result['num_dets']),
            pb_utils.Tensor('det_boxes', result['det_boxes']),
            pb_utils.Tensor('det_scores', result['det_scores']),
            pb_utils.Tensor('det_classes', result['det_classes']),
            # Embedding outputs
            pb_utils.Tensor('global_embeddings', result['global_embeddings']),
            pb_utils.Tensor('box_embeddings', result['box_embeddings']),
            pb_utils.Tensor('normalized_boxes', result['normalized_boxes']),
            # Face outputs
            pb_utils.Tensor('num_faces', result['num_faces']),
            pb_utils.Tensor('face_embeddings', result['face_embeddings']),
            pb_utils.Tensor('face_boxes', result['face_boxes']),
            pb_utils.Tensor('face_landmarks', result['face_landmarks']),
            pb_utils.Tensor('face_scores', result['face_scores']),
            pb_utils.Tensor('face_person_idx', result['face_person_idx']),
            # OCR outputs
            pb_utils.Tensor('num_texts', num_texts),
            pb_utils.Tensor('text_boxes', text_boxes),
            pb_utils.Tensor('text_boxes_normalized', text_boxes_norm),
            pb_utils.Tensor('texts', texts_bytes),
            pb_utils.Tensor('text_det_scores', text_det_scores),
            pb_utils.Tensor('text_rec_scores', text_rec_scores),
            # Metadata
            pb_utils.Tensor('face_model_used', face_model_bytes),
        ]

        return pb_utils.InferenceResponse(output_tensors=output_tensors)

    def _create_error_response(self, error_msg):
        """Create error response with empty outputs."""
        empty_dets = np.array([[0]], dtype=np.int32)
        empty_boxes = np.zeros((1, self.max_dets, 4), dtype=np.float32)
        empty_scores = np.zeros((1, self.max_dets), dtype=np.float32)
        empty_classes = np.zeros((1, self.max_dets), dtype=np.int32)
        empty_global = np.zeros((1, self.embed_dim), dtype=np.float32)
        empty_box_emb = np.zeros((1, self.max_dets, self.embed_dim), dtype=np.float32)
        empty_norm_boxes = np.zeros((1, self.max_dets, 4), dtype=np.float32)
        empty_faces = np.array([[0]], dtype=np.int32)
        empty_face_emb = np.zeros((1, self.max_faces, self.embed_dim), dtype=np.float32)
        empty_face_boxes = np.zeros((1, self.max_faces, 4), dtype=np.float32)
        empty_face_lmk = np.zeros((1, self.max_faces, 10), dtype=np.float32)
        empty_face_scores = np.zeros((1, self.max_faces), dtype=np.float32)
        empty_face_idx = np.zeros((1, self.max_faces), dtype=np.int32) - 1
        empty_texts = np.array([[0]], dtype=np.int32)
        empty_text_boxes = np.zeros((1, self.max_texts, 8), dtype=np.float32)
        empty_text_norm = np.zeros((1, self.max_texts, 4), dtype=np.float32)
        empty_text_str = np.array([[b'' for _ in range(self.max_texts)]], dtype=object)
        empty_text_det = np.zeros((1, self.max_texts), dtype=np.float32)
        empty_text_rec = np.zeros((1, self.max_texts), dtype=np.float32)
        empty_face_model = np.array([b'error'], dtype=object)

        return pb_utils.InferenceResponse(
            output_tensors=[
                pb_utils.Tensor('num_dets', empty_dets),
                pb_utils.Tensor('det_boxes', empty_boxes),
                pb_utils.Tensor('det_scores', empty_scores),
                pb_utils.Tensor('det_classes', empty_classes),
                pb_utils.Tensor('global_embeddings', empty_global),
                pb_utils.Tensor('box_embeddings', empty_box_emb),
                pb_utils.Tensor('normalized_boxes', empty_norm_boxes),
                pb_utils.Tensor('num_faces', empty_faces),
                pb_utils.Tensor('face_embeddings', empty_face_emb),
                pb_utils.Tensor('face_boxes', empty_face_boxes),
                pb_utils.Tensor('face_landmarks', empty_face_lmk),
                pb_utils.Tensor('face_scores', empty_face_scores),
                pb_utils.Tensor('face_person_idx', empty_face_idx),
                pb_utils.Tensor('num_texts', empty_texts),
                pb_utils.Tensor('text_boxes', empty_text_boxes),
                pb_utils.Tensor('text_boxes_normalized', empty_text_norm),
                pb_utils.Tensor('texts', empty_text_str),
                pb_utils.Tensor('text_det_scores', empty_text_det),
                pb_utils.Tensor('text_rec_scores', empty_text_rec),
                pb_utils.Tensor('face_model_used', empty_face_model),
            ],
            error=pb_utils.TritonError(error_msg)
        )

    def finalize(self):
        logger.info('Finalizing unified_complete_pipeline')
