#!/usr/bin/env python3
"""
Triton Python Backend: PP-OCRv5 Text Detection & Recognition Pipeline

Complete OCR pipeline that handles text detection and recognition via BLS:
1. Receives preprocessed OCR image from DALI
2. Calls PP-OCRv5 detection TensorRT for text region detection
3. Post-processes detection output (DBPostProcess)
4. Sorts text boxes in reading order
5. Crops text regions with perspective transform
6. Calls PP-OCRv5 recognition TensorRT for text reading
7. Decodes text using CTC and returns results

Architecture:
    Inputs:
        - ocr_images: [3, H, W] Preprocessed image for OCR (normalized [-1, 1])
        - original_image: [3, H, W] Original image for cropping
        - orig_shape: [2] Original image shape [H, W]

    Outputs:
        - num_texts: [1] Number of detected text regions
        - text_boxes: [128, 8] Quadrilateral boxes (8 coords per box)
        - text_boxes_normalized: [128, 4] Axis-aligned boxes [x1, y1, x2, y2] normalized
        - texts: [128] Detected text strings (as bytes)
        - text_scores: [128] Detection confidence scores
        - rec_scores: [128] Recognition confidence scores

Based on: Immich OCR implementation and RapidOCR
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
    import torch.nn.functional as F
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    cp = None
    torch = None

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MAX_TEXTS = 128
DET_MODEL = 'paddleocr_det_trt'
REC_MODEL = 'paddleocr_rec_trt'
REC_HEIGHT = 48
REC_WIDTH = 320
DICT_PATH = Path('/models/paddleocr_rec_trt/en_ppocrv5_dict.txt')


class DBPostProcess:
    """
    Post-processing for DB text detection (simplified from RapidOCR).

    DB (Differentiable Binarization) post-processing:
    1. Threshold probability map
    2. Find contours
    3. Expand boxes (unclip)
    4. Filter by area and score
    """

    def __init__(
        self,
        thresh: float = 0.3,
        box_thresh: float = 0.5,
        max_candidates: int = 1000,
        unclip_ratio: float = 1.6,
        use_dilation: bool = True,
        score_mode: str = 'fast',
    ):
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        self.use_dilation = use_dilation
        self.score_mode = score_mode

        # Dilation kernel
        if use_dilation:
            self.dilation_kernel = np.array([[1, 1], [1, 1]], dtype=np.uint8)
        else:
            self.dilation_kernel = None

    def __call__(self, pred: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        """
        Process detection output.

        Args:
            pred: [1, 1, H, W] probability map
            shape: Original image (H, W)

        Returns:
            boxes: [N, 4, 2] quadrilateral boxes
            scores: [N] detection scores
        """
        # Handle different output shapes from detection model
        if pred.ndim == 4:
            pred = pred[0, 0]  # [1, 1, H, W] -> [H, W]
        elif pred.ndim == 3:
            pred = pred[0]  # [1, H, W] -> [H, W]
        # else assume it's already [H, W]

        logger.debug(f'DBPost: pred shape={pred.shape}, range=[{pred.min():.3f}, {pred.max():.3f}]')
        src_h, src_w = shape

        # Threshold
        mask = (pred > self.thresh).astype(np.uint8) * 255
        logger.debug(f'DBPost: mask pixels > {self.thresh}: {np.sum(mask > 0)}')

        # Optional dilation
        if self.dilation_kernel is not None:
            mask = cv2.dilate(mask, self.dilation_kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        logger.debug(f'DBPost: found {len(contours)} contours')

        boxes = []
        scores = []
        filtered_reasons = {'small': 0, 'low_score': 0, 'unclip_fail': 0, 'too_small': 0}

        for contour in contours[: self.max_candidates]:
            # Get bounding box
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)

            # Calculate score
            if self.score_mode == 'fast':
                # Fast mode: use bounding box center
                x, y = int(rect[0][0]), int(rect[0][1])
                x = min(max(x, 0), pred.shape[1] - 1)
                y = min(max(y, 0), pred.shape[0] - 1)
                score = float(pred[y, x])
            else:
                # Slow mode: use polygon mean
                mask_box = np.zeros_like(pred, dtype=np.uint8)
                cv2.fillPoly(mask_box, [box.astype(np.int32)], 1)
                score = float(cv2.mean(pred, mask_box)[0])

            if score < self.box_thresh:
                filtered_reasons['low_score'] += 1
                continue

            # Unclip (expand box)
            box = self._unclip(box)
            if box is None:
                filtered_reasons['unclip_fail'] += 1
                continue

            # Verify expanded box is valid
            rect = cv2.minAreaRect(box)
            if min(rect[1]) < 3:  # Too small
                filtered_reasons['too_small'] += 1
                continue

            box = cv2.boxPoints(rect)

            # Scale to original image size
            scale_x = src_w / pred.shape[1]
            scale_y = src_h / pred.shape[0]
            box[:, 0] *= scale_x
            box[:, 1] *= scale_y

            # Clip to image bounds
            box[:, 0] = np.clip(box[:, 0], 0, src_w - 1)
            box[:, 1] = np.clip(box[:, 1], 0, src_h - 1)

            # Order points: top-left, top-right, bottom-right, bottom-left
            box = self._order_points(box)

            boxes.append(box)
            scores.append(score)

        logger.debug(f'DBPost: {len(boxes)} boxes passed, filtered: {filtered_reasons}')

        if len(boxes) == 0:
            return np.empty((0, 4, 2), dtype=np.float32), np.empty(0, dtype=np.float32)

        return np.array(boxes, dtype=np.float32), np.array(scores, dtype=np.float32)

    def _unclip(self, box: np.ndarray) -> np.ndarray | None:
        """Expand box by unclip_ratio."""
        try:
            import pyclipper

            poly = pyclipper.Pyclipper()
            poly.AddPath(box.astype(np.int32).tolist(), pyclipper.PT_SUBJECT, True)

            # Calculate expansion distance
            area = cv2.contourArea(box)
            peri = cv2.arcLength(box, True)
            if peri == 0:
                return None
            distance = area * self.unclip_ratio / peri

            expanded = poly.Execute(
                pyclipper.CT_UNION,
                pyclipper.PFT_EVENODD,
                pyclipper.PFT_EVENODD,
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
            # Fallback: simple box expansion
            center = box.mean(axis=0)
            expanded = center + (box - center) * self.unclip_ratio
            return expanded

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """Order points: top-left, top-right, bottom-right, bottom-left."""
        rect = np.zeros((4, 2), dtype=np.float32)

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # Top-left has smallest sum
        rect[2] = pts[np.argmax(s)]  # Bottom-right has largest sum

        d = np.diff(pts, axis=1).squeeze()
        rect[1] = pts[np.argmin(d)]  # Top-right has smallest diff
        rect[3] = pts[np.argmax(d)]  # Bottom-left has largest diff

        return rect


class CTCLabelDecode:
    """CTC decoder for text recognition."""

    def __init__(self, character_dict_path: str | Path):
        self.character = ['blank']  # CTC blank token

        with open(character_dict_path, 'r', encoding='utf-8') as f:
            for line in f:
                char = line.strip('\n')
                self.character.append(char)

        self.character.append(' ')  # Space at end

    def __call__(self, preds: np.ndarray) -> list[tuple[str, float]]:
        """
        Decode CTC output.

        Args:
            preds: [B, T, num_chars] softmax output

        Returns:
            List of (text, confidence) tuples
        """
        results = []
        preds_idx = preds.argmax(axis=2)  # [B, T]
        preds_prob = preds.max(axis=2)  # [B, T]

        for idx, prob in zip(preds_idx, preds_prob):
            text = ''
            conf_list = []
            prev_idx = 0

            for i, (char_idx, char_prob) in enumerate(zip(idx, prob)):
                # Skip blank and repeated
                if char_idx != 0 and char_idx != prev_idx:
                    if char_idx < len(self.character):
                        text += self.character[char_idx]
                        conf_list.append(char_prob)
                prev_idx = char_idx

            conf = float(np.mean(conf_list)) if conf_list else 0.0
            results.append((text, conf))

        return results


class TritonPythonModel:
    """PP-OCRv5 Text Detection & Recognition Pipeline via BLS"""

    def initialize(self, args):
        """Initialize model configuration."""
        self.model_config = json.loads(args['model_config'])

        # Configuration
        self.max_texts = MAX_TEXTS
        self.rec_height = REC_HEIGHT
        self.rec_width = REC_WIDTH

        # GPU device
        self.device = 'cuda' if HAS_GPU and torch.cuda.is_available() else 'cpu'

        # DB post-processor - lower thresholds for better detection
        self.db_postprocess = DBPostProcess(
            thresh=0.2,        # Lower for faint text
            box_thresh=0.4,    # Lower for more candidates
            max_candidates=1000,
            unclip_ratio=1.5,  # Standard expansion
            use_dilation=True,
            score_mode='fast',
        )

        # CTC decoder
        if DICT_PATH.exists():
            self.ctc_decode = CTCLabelDecode(DICT_PATH)
            logger.info(f'Loaded dictionary: {len(self.ctc_decode.character)} characters')
        else:
            self.ctc_decode = None
            logger.warning(f'Dictionary not found: {DICT_PATH}')

        logger.info(f'Initialized ocr_pipeline (device={self.device}, GPU={HAS_GPU})')
        logger.info(f'  Detection model: {DET_MODEL}')
        logger.info(f'  Recognition model: {REC_MODEL}')
        logger.info(f'  Max texts: {self.max_texts}')

    def _triton_to_torch(self, tensor):
        """Convert Triton tensor to PyTorch tensor (zero-copy on GPU)."""
        if not HAS_GPU:
            return torch.from_numpy(tensor.as_numpy())
        if tensor.is_cpu():
            return torch.from_numpy(tensor.as_numpy()).to(self.device)
        return torch.from_dlpack(tensor.to_dlpack())

    def _triton_to_numpy(self, tensor):
        """Convert Triton tensor to NumPy (handles GPU tensors via CuPy)."""
        if tensor.is_cpu():
            return tensor.as_numpy()
        # GPU tensor - convert via DLPack
        if HAS_GPU:
            dlpack = tensor.to_dlpack()
            return cp.from_dlpack(dlpack).get()  # Copy to CPU
        return tensor.as_numpy()

    def execute(self, requests):
        """Execute OCR pipeline for each request."""
        responses = []

        for request in requests:
            try:
                response = self._process_request(request)
                responses.append(response)
            except Exception as e:
                logger.error(f'Error processing request: {e}')
                import traceback

                traceback.print_exc()
                responses.append(
                    pb_utils.InferenceResponse(
                        error=pb_utils.TritonError(f'OCR pipeline error: {e}')
                    )
                )

        return responses

    def _process_request(self, request):
        """Process a single OCR request."""
        # Get inputs
        ocr_image = pb_utils.get_input_tensor_by_name(request, 'ocr_images').as_numpy()
        original_image = pb_utils.get_input_tensor_by_name(request, 'original_image').as_numpy()
        orig_shape = pb_utils.get_input_tensor_by_name(request, 'orig_shape').as_numpy()

        # ocr_image: [3, H, W] normalized [-1, 1]
        # original_image: [3, H, W] normalized [0, 1]
        # orig_shape: [2] = [H, W]

        orig_h, orig_w = int(orig_shape[0]), int(orig_shape[1])

        # Step 1: Call detection model
        det_output = self._call_detection(ocr_image)

        # Step 2: Post-process detection
        boxes, det_scores = self.db_postprocess(det_output, (orig_h, orig_w))

        if len(boxes) == 0:
            return self._create_empty_response()

        # Step 3: Sort boxes in reading order
        boxes = self._sorted_boxes(boxes)

        # Limit to max_texts
        if len(boxes) > self.max_texts:
            boxes = boxes[: self.max_texts]
            det_scores = det_scores[: self.max_texts]

        # Step 4: Crop text regions
        # Convert original image from CHW [0,1] to HWC [0,255] uint8
        orig_img_hwc = (original_image.transpose(1, 2, 0) * 255).astype(np.uint8)
        logger.debug(f'Processing: {len(boxes)} boxes, orig_img shape={orig_img_hwc.shape}')
        text_crops = self._get_text_crops(orig_img_hwc, boxes)
        logger.debug(f'Got {len(text_crops)} text crops')
        # Debug: log crop sizes
        for i, crop in enumerate(text_crops[:3]):
            logger.debug(f'  Crop {i}: shape={crop.shape}, dtype={crop.dtype}, range=[{crop.min()}, {crop.max()}]')

        if len(text_crops) == 0:
            return self._create_empty_response()

        # Step 5: Call recognition model
        texts, rec_scores = self._call_recognition(text_crops)
        logger.debug(f'Recognition: {len(texts)} texts recognized')
        if texts:
            logger.debug(f'First 3 texts: {texts[:3]}')

        # Step 6: Create output tensors
        return self._create_response(boxes, det_scores, texts, rec_scores, orig_h, orig_w)

    def _call_detection(self, ocr_image: np.ndarray) -> np.ndarray:
        """Call detection model via BLS with GPU tensor handling."""
        # ocr_image: [3, H, W] normalized [-1, 1], BGR (from OpenCV)
        # PP-OCR models were exported expecting RGB, so convert BGR -> RGB
        ocr_image_rgb = ocr_image[[2, 1, 0], :, :]  # BGR -> RGB

        # Use CPU tensor for now (GPU DLPack may have issues with TRT)
        input_tensor = pb_utils.Tensor('x', ocr_image_rgb[np.newaxis].astype(np.float32))

        infer_request = pb_utils.InferenceRequest(
            model_name=DET_MODEL,
            requested_output_names=['fetch_name_0'],
            inputs=[input_tensor],
        )

        infer_response = infer_request.exec()

        if infer_response.has_error():
            raise RuntimeError(f'Detection failed: {infer_response.error().message()}')

        output_tensor = pb_utils.get_output_tensor_by_name(infer_response, 'fetch_name_0')
        output = self._triton_to_numpy(output_tensor)
        return output  # [1, 1, H, W]

    def _call_recognition(self, text_crops: list[np.ndarray]) -> tuple[list[str], list[float]]:
        """
        Call recognition model via BLS using PaddleOCR preprocessing.

        Uses dynamic width TensorRT model (supports 8-2048 width).
        Each crop is processed individually to preserve aspect ratio.
        """
        if not text_crops or self.ctc_decode is None:
            return [], []

        # Dynamic width constraints from TensorRT engine
        MIN_WIDTH = 8
        MAX_WIDTH = 2048

        logger.debug(f'Recognition: {len(text_crops)} crops (dynamic width)')

        # Process each crop individually to preserve aspect ratio
        all_results = []

        for i, crop in enumerate(text_crops):
            h, w = crop.shape[:2]

            # Calculate target width maintaining aspect ratio
            ratio = w / float(h)
            target_w = int(math.ceil(self.rec_height * ratio))
            target_w = max(MIN_WIDTH, min(MAX_WIDTH, target_w))

            # Preprocess crop
            norm_img = self._resize_norm_img_dynamic(crop, target_w)

            # Call recognition model (batch size 1)
            batch_input = norm_img[np.newaxis].astype(np.float32)
            input_tensor = pb_utils.Tensor('x', batch_input)

            infer_request = pb_utils.InferenceRequest(
                model_name=REC_MODEL,
                requested_output_names=['fetch_name_0'],
                inputs=[input_tensor],
            )

            infer_response = infer_request.exec()

            if infer_response.has_error():
                raise RuntimeError(f'Recognition failed: {infer_response.error().message()}')

            output_tensor = pb_utils.get_output_tensor_by_name(infer_response, 'fetch_name_0')
            output = self._triton_to_numpy(output_tensor)

            # Decode CTC
            batch_results = self.ctc_decode(output)
            all_results.extend(batch_results)

            if i < 3:
                logger.debug(f'  Crop {i}: {w}x{h} -> width={target_w}, text="{batch_results[0][0][:30]}..."')

        # Extract texts and scores
        texts = [r[0] for r in all_results]
        scores = [r[1] for r in all_results]

        return texts, scores

    def _resize_norm_img(self, img: np.ndarray, imgW: int) -> np.ndarray:
        """
        Resize and normalize image for recognition (PaddleOCR style) with padding.

        Args:
            img: [H, W, 3] BGR uint8 image
            imgW: Target batch width (with padding)

        Returns:
            [3, 48, imgW] normalized float32 tensor
        """
        imgH = self.rec_height
        h, w = img.shape[:2]

        # Calculate resize width maintaining aspect ratio
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))

        # Resize
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype(np.float32)

        # Normalize: (x / 255 - 0.5) / 0.5 = x / 127.5 - 1
        resized_image = resized_image.transpose(2, 0, 1) / 255.0
        resized_image -= 0.5
        resized_image /= 0.5

        # Pad to batch width
        padding_im = np.zeros((3, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image

        return padding_im

    def _resize_norm_img_dynamic(self, img: np.ndarray, target_w: int) -> np.ndarray:
        """
        Resize and normalize image for recognition (no padding, exact width).

        For use with dynamic-width TensorRT model.

        Args:
            img: [H, W, 3] BGR uint8 image
            target_w: Target width (no padding)

        Returns:
            [3, 48, target_w] normalized float32 tensor
        """
        imgH = self.rec_height

        # Resize to target dimensions
        resized_image = cv2.resize(img, (target_w, imgH))
        resized_image = resized_image.astype(np.float32)

        # Normalize: (x / 255 - 0.5) / 0.5 = x / 127.5 - 1
        resized_image = resized_image.transpose(2, 0, 1) / 255.0
        resized_image -= 0.5
        resized_image /= 0.5

        return resized_image

    def _sorted_boxes(self, boxes: np.ndarray) -> np.ndarray:
        """Sort boxes in reading order (top-to-bottom, left-to-right)."""
        if len(boxes) == 0:
            return boxes

        # Sort by y, then identify lines, then sort by (line, x)
        y_order = np.argsort(boxes[:, 0, 1], kind='stable')
        sorted_y = boxes[y_order, 0, 1]

        # Group into lines (10px threshold)
        line_ids = np.zeros(len(boxes), dtype=np.int32)
        if len(sorted_y) > 1:
            line_ids[1:] = np.cumsum(np.abs(np.diff(sorted_y)) >= 10)

        # Sort by (line_id, x)
        sort_key = line_ids * 1e6 + boxes[y_order, 0, 0]
        final_order = np.argsort(sort_key, kind='stable')

        return boxes[y_order[final_order]]

    def _get_text_crops(self, img: np.ndarray, boxes: np.ndarray) -> list[np.ndarray]:
        """Crop text regions using perspective transform."""
        crops = []

        for box in boxes:
            # Calculate crop dimensions
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

            # Destination points
            dst_pts = np.array(
                [[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32
            )

            # Perspective transform
            M = cv2.getPerspectiveTransform(box.astype(np.float32), dst_pts)
            crop = cv2.warpPerspective(img, M, (width, height), borderMode=cv2.BORDER_REPLICATE)

            # Rotate if tall text
            if height / width >= 1.5:
                crop = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)

            crops.append(crop)

        return crops

    def _create_response(
        self,
        boxes: np.ndarray,
        det_scores: np.ndarray,
        texts: list[str],
        rec_scores: list[float],
        orig_h: int,
        orig_w: int,
    ):
        """Create inference response."""
        num_texts = len(boxes)
        logger.debug(f'Creating response: num_texts={num_texts}, texts_len={len(texts)}')

        # Pad to max_texts
        padded_boxes = np.zeros((self.max_texts, 8), dtype=np.float32)
        padded_boxes_norm = np.zeros((self.max_texts, 4), dtype=np.float32)
        padded_det_scores = np.zeros(self.max_texts, dtype=np.float32)
        padded_rec_scores = np.zeros(self.max_texts, dtype=np.float32)
        padded_texts = np.zeros(self.max_texts, dtype=object)

        for i, box in enumerate(boxes[:num_texts]):
            # Flatten quad box [4, 2] -> [8]
            padded_boxes[i] = box.flatten()

            # Axis-aligned normalized box [x1, y1, x2, y2]
            x1 = box[:, 0].min() / orig_w
            y1 = box[:, 1].min() / orig_h
            x2 = box[:, 0].max() / orig_w
            y2 = box[:, 1].max() / orig_h
            padded_boxes_norm[i] = [x1, y1, x2, y2]

            padded_det_scores[i] = det_scores[i] if i < len(det_scores) else 0.0
            padded_rec_scores[i] = rec_scores[i] if i < len(rec_scores) else 0.0
            padded_texts[i] = texts[i] if i < len(texts) else ''

        # Convert texts to bytes for Triton BYTES type
        texts_bytes = np.array([t.encode('utf-8') if t else b'' for t in padded_texts], dtype=object)

        output_tensors = [
            pb_utils.Tensor('num_texts', np.array([num_texts], dtype=np.int32)),
            pb_utils.Tensor('text_boxes', padded_boxes),
            pb_utils.Tensor('text_boxes_normalized', padded_boxes_norm),
            pb_utils.Tensor('texts', texts_bytes),
            pb_utils.Tensor('text_scores', padded_det_scores),
            pb_utils.Tensor('rec_scores', padded_rec_scores),
        ]

        return pb_utils.InferenceResponse(output_tensors=output_tensors)

    def _create_empty_response(self):
        """Create empty response when no text detected."""
        output_tensors = [
            pb_utils.Tensor('num_texts', np.array([0], dtype=np.int32)),
            pb_utils.Tensor('text_boxes', np.zeros((self.max_texts, 8), dtype=np.float32)),
            pb_utils.Tensor('text_boxes_normalized', np.zeros((self.max_texts, 4), dtype=np.float32)),
            pb_utils.Tensor('texts', np.array([b'' for _ in range(self.max_texts)], dtype=object)),
            pb_utils.Tensor('text_scores', np.zeros(self.max_texts, dtype=np.float32)),
            pb_utils.Tensor('rec_scores', np.zeros(self.max_texts, dtype=np.float32)),
        ]

        return pb_utils.InferenceResponse(output_tensors=output_tensors)

    def finalize(self):
        """Cleanup."""
        logger.info('Finalized ocr_pipeline')
