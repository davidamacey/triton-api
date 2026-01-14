"""
OCR Service for PP-OCRv5 Text Detection and Recognition.

This service provides OCR inference via Triton Server with PP-OCRv5 models.
It handles preprocessing, inference orchestration, and result formatting.

Pipeline:
1. Preprocess image with DALI (penta_preprocess_dali)
2. Call OCR pipeline BLS (detection + recognition)
3. Format results with text, boxes, and confidence scores

Usage:
    from src.services.ocr_service import OcrService

    service = OcrService()

    # Single image OCR
    result = service.extract_text(image_bytes)
    # Returns: {
    #     'texts': ['Hello', 'World'],
    #     'boxes': [[x1,y1,x2,y2,x3,y3,x4,y4], ...],
    #     'boxes_normalized': [[x1,y1,x2,y2], ...],
    #     'det_scores': [0.95, 0.88],
    #     'rec_scores': [0.99, 0.97],
    #     'num_texts': 2
    # }
"""

import io
import logging
from typing import Any

import numpy as np
from PIL import Image

from src.clients.triton_client import get_triton_client
from src.config import get_settings
from src.utils.image_processing import decode_image, validate_image


logger = logging.getLogger(__name__)


class OcrService:
    """
    OCR service for text detection and recognition.

    Uses PP-OCRv5 models via Triton Server:
    - paddleocr_det_trt: Text detection
    - paddleocr_rec_trt: Text recognition
    - ocr_pipeline: BLS orchestrator (optional, if using Triton BLS)

    For now, implements client-side orchestration for flexibility.
    """

    def __init__(self, min_det_score: float = 0.5, min_rec_score: float = 0.8):
        """
        Initialize OCR service.

        Args:
            min_det_score: Minimum detection confidence (default 0.5)
            min_rec_score: Minimum recognition confidence (default 0.8)
        """
        self.min_det_score = min_det_score
        self.min_rec_score = min_rec_score
        self.settings = get_settings()

    def extract_text(
        self, image_bytes: bytes, filter_by_score: bool = True
    ) -> dict[str, Any]:
        """
        Extract text from an image.

        Args:
            image_bytes: JPEG/PNG image bytes
            filter_by_score: If True, filter results by confidence thresholds

        Returns:
            Dict with OCR results:
            - texts: List of detected text strings
            - boxes: List of quad boxes [x1,y1,x2,y2,x3,y3,x4,y4]
            - boxes_normalized: List of axis-aligned boxes [x1,y1,x2,y2] normalized
            - det_scores: Detection confidence scores
            - rec_scores: Recognition confidence scores
            - num_texts: Number of text regions detected
            - image_size: [height, width] of original image
        """
        try:
            # Decode image first
            image = decode_image(image_bytes)
            if image is None:
                return self._empty_result(error='Failed to decode image')

            # Validate decoded image
            try:
                validate_image(image)
            except ValueError as e:
                return self._empty_result(error=str(e))

            img_h, img_w = image.shape[:2]

            # Call Triton OCR pipeline
            client = get_triton_client(self.settings.triton_url)
            result = client.infer_ocr(image_bytes)

            if result is None:
                return self._empty_result(error='OCR inference failed')

            # Parse results
            num_texts = int(result.get('num_texts', 0))
            if num_texts == 0:
                return self._empty_result(image_size=[img_h, img_w])

            # Extract arrays
            texts = result.get('texts', [])[:num_texts]
            boxes = result.get('text_boxes', [])[:num_texts]
            boxes_norm = result.get('text_boxes_normalized', [])[:num_texts]
            det_scores = result.get('text_scores', [])[:num_texts]
            rec_scores = result.get('rec_scores', [])[:num_texts]

            # Filter by confidence if requested
            if filter_by_score:
                filtered_indices = [
                    i
                    for i in range(len(texts))
                    if det_scores[i] >= self.min_det_score
                    and rec_scores[i] >= self.min_rec_score
                ]

                texts = [texts[i] for i in filtered_indices]
                boxes = [boxes[i].tolist() if hasattr(boxes[i], 'tolist') else list(boxes[i]) for i in filtered_indices]
                boxes_norm = [boxes_norm[i].tolist() if hasattr(boxes_norm[i], 'tolist') else list(boxes_norm[i]) for i in filtered_indices]
                det_scores = [float(det_scores[i]) for i in filtered_indices]
                rec_scores = [float(rec_scores[i]) for i in filtered_indices]
                num_texts = len(texts)
            else:
                # Convert to lists
                boxes = [b.tolist() if hasattr(b, 'tolist') else list(b) for b in boxes]
                boxes_norm = [b.tolist() if hasattr(b, 'tolist') else list(b) for b in boxes_norm]
                det_scores = [float(s) for s in det_scores]
                rec_scores = [float(s) for s in rec_scores]

            return {
                'status': 'success',
                'texts': texts,
                'boxes': boxes,
                'boxes_normalized': boxes_norm,
                'det_scores': det_scores,
                'rec_scores': rec_scores,
                'num_texts': num_texts,
                'image_size': [img_h, img_w],
            }

        except Exception as e:
            logger.error(f'OCR extraction failed: {e}')
            import traceback
            traceback.print_exc()
            return self._empty_result(error=str(e))

    def extract_text_batch(
        self, image_bytes_list: list[bytes], filter_by_score: bool = True
    ) -> list[dict[str, Any]]:
        """
        Extract text from multiple images.

        Args:
            image_bytes_list: List of JPEG/PNG image bytes
            filter_by_score: If True, filter results by confidence thresholds

        Returns:
            List of OCR results for each image
        """
        results = []
        for image_bytes in image_bytes_list:
            result = self.extract_text(image_bytes, filter_by_score)
            results.append(result)
        return results

    def get_full_text(self, ocr_result: dict[str, Any], separator: str = ' ') -> str:
        """
        Get full text from OCR result as a single string.

        Args:
            ocr_result: Result from extract_text()
            separator: Text separator (default: space)

        Returns:
            Full text as single string
        """
        texts = ocr_result.get('texts', [])
        return separator.join(texts)

    def _empty_result(
        self, image_size: list[int] | None = None, error: str | None = None
    ) -> dict[str, Any]:
        """Create empty OCR result."""
        result = {
            'status': 'error' if error else 'success',
            'texts': [],
            'boxes': [],
            'boxes_normalized': [],
            'det_scores': [],
            'rec_scores': [],
            'num_texts': 0,
            'image_size': image_size or [0, 0],
        }
        if error:
            result['error'] = error
        return result


# Singleton instance for reuse
_ocr_service: OcrService | None = None


def get_ocr_service() -> OcrService:
    """Get or create singleton OCR service instance."""
    global _ocr_service
    if _ocr_service is None:
        _ocr_service = OcrService()
    return _ocr_service
