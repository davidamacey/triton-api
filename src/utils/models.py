"""
Pydantic models for API responses.

Shared data models used across all API implementations.
"""

from pydantic import BaseModel


class InferenceResult(BaseModel):
    """
    Standard response schema for object detection inference.

    Used by all API endpoints (PyTorch, Triton Standard, Triton End2End)
    to ensure consistent response format.

    Attributes:
        detections (list): List of detections with bounding box coordinates,
                          confidence scores, and class IDs
        status (str): Status of the inference process ("success" or "error")
    """

    detections: list
    status: str
