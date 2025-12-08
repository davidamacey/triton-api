"""
Image processing service.

Provides image validation, resizing, and preprocessing utilities.
"""

import io
import logging

from PIL import Image

from src.config import get_settings
from src.utils.affine import get_jpeg_dimensions_fast


logger = logging.getLogger(__name__)


class ImageService:
    """
    Image preprocessing service.

    Handles:
    - Image resizing for large images
    - Image validation
    - Format conversion
    """

    def __init__(self):
        self.settings = get_settings()

    def resize_if_needed(
        self, image_bytes: bytes, max_size: int | None = None, min_model_size: int = 640
    ) -> bytes:
        """
        Resize image if larger than max_size while maintaining aspect ratio.

        Args:
            image_bytes: Original JPEG/PNG bytes
            max_size: Maximum dimension (default from settings)
            min_model_size: Minimum model input size (no upscaling below this)

        Returns:
            JPEG bytes (original or resized)
        """
        max_size = max_size or self.settings.default_max_resize

        try:
            img = Image.open(io.BytesIO(image_bytes))
            width, height = img.size
            max_dim = max(width, height)

            # No resizing needed
            if max_dim <= max_size:
                return image_bytes

            # Don't upscale small images
            if max_dim < min_model_size:
                return image_bytes

            # Calculate new size maintaining aspect ratio
            if width > height:
                new_width = max_size
                new_height = int(height * (max_size / width))
            else:
                new_height = max_size
                new_width = int(width * (max_size / height))

            # Resize using Lanczos
            img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Save to JPEG bytes
            buffer = io.BytesIO()
            img_resized.save(buffer, format='JPEG', quality=85, optimize=True)

            logger.debug(f'Resized image from {width}x{height} to {new_width}x{new_height}')
            return buffer.getvalue()

        except Exception as e:
            logger.warning(f'Failed to resize image: {e}')
            return image_bytes

    def get_dimensions(self, image_bytes: bytes) -> tuple[int, int]:
        """
        Get image dimensions without full decode.

        Uses JPEG header parsing for speed when possible.

        Args:
            image_bytes: Image bytes

        Returns:
            Tuple of (width, height)
        """
        try:
            return get_jpeg_dimensions_fast(image_bytes)
        except Exception:
            # Fallback to PIL
            img = Image.open(io.BytesIO(image_bytes))
            return img.size

    def validate_size(self, image_bytes: bytes, max_size_mb: int | None = None) -> bool:
        """
        Validate image file size.

        Args:
            image_bytes: Image bytes
            max_size_mb: Maximum size in MB (default from settings)

        Returns:
            True if valid, raises ValueError if too large
        """
        max_size_mb = max_size_mb or self.settings.max_file_size_mb
        size_mb = len(image_bytes) / (1024 * 1024)

        if size_mb > max_size_mb:
            raise ValueError(f'File size {size_mb:.2f}MB exceeds maximum {max_size_mb}MB')

        return True

    def convert_to_jpeg(self, image_bytes: bytes, quality: int = 85) -> bytes:
        """
        Convert image to JPEG format.

        Useful for normalizing input format.

        Args:
            image_bytes: Original image bytes
            quality: JPEG quality (0-100)

        Returns:
            JPEG bytes
        """
        try:
            img = Image.open(io.BytesIO(image_bytes))

            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')

            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=quality, optimize=True)
            return buffer.getvalue()

        except Exception as e:
            logger.warning(f'Failed to convert image to JPEG: {e}')
            return image_bytes
