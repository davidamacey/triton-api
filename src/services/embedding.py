"""
Embedding service for MobileCLIP encoding.

Handles image and text embedding generation with caching.
"""

import asyncio
import hashlib
import io
import logging

import numpy as np
from PIL import Image
from tritonclient.grpc import InferInput, InferRequestedOutput

from src.clients.triton_pool import TritonClientManager
from src.config import get_settings
from src.utils.cache import get_clip_tokenizer, get_image_cache, get_text_cache


logger = logging.getLogger(__name__)

# MobileCLIP2-S2 embedding dimension (from model config.pbtxt)
EMBEDDING_DIM = 512


class EmbeddingService:
    """
    MobileCLIP2-S2 embedding service with caching.

    Provides efficient encoding for:
    - Image embeddings (512-dim, L2-normalized)
    - Text embeddings (512-dim, L2-normalized)
    - Box embeddings (512-dim per detected object)

    Uses native async gRPC for Triton batching.
    """

    def __init__(self):
        self.settings = get_settings()
        self._triton_client = None
        self.embedding_dim = EMBEDDING_DIM

    async def _get_client(self):
        """Get async Triton client (lazy init)."""
        if self._triton_client is None:
            self._triton_client = await TritonClientManager.get_async_client(
                self.settings.triton_url
            )
        return self._triton_client

    # =========================================================================
    # Image Preprocessing
    # =========================================================================
    @staticmethod
    def preprocess_image(image_bytes: bytes) -> np.ndarray:
        """
        Preprocess image for MobileCLIP encoder.

        Applies OpenCLIP-style preprocessing:
        - Resize shortest edge to 256
        - Center crop to 256x256
        - Normalize to [0, 1]
        - Transpose to CHW format

        Args:
            image_bytes: JPEG/PNG bytes

        Returns:
            Preprocessed array [1, 3, 256, 256] FP32
        """
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # Resize shortest edge to 256
        width, height = img.size
        scale = 256 / min(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Center crop to 256x256
        left = (new_width - 256) // 2
        top = (new_height - 256) // 2
        img = img.crop((left, top, left + 256, top + 256))

        # Normalize and transpose
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = np.transpose(img_array, (2, 0, 1))

        return img_array[np.newaxis, ...]

    # =========================================================================
    # Embedding Generation
    # =========================================================================
    async def encode_image(self, image_bytes: bytes, use_cache: bool = True) -> np.ndarray:
        """
        Encode image to 512-dim MobileCLIP embedding.

        Uses MobileCLIP2-S2 image encoder via Triton.
        Enables async gRPC for concurrent request batching.

        Args:
            image_bytes: JPEG/PNG bytes
            use_cache: Use embedding cache (default True)

        Returns:
            512-dim L2-normalized embedding vector
        """
        # Check cache
        cache_key = None
        if use_cache:
            cache = get_image_cache()
            cache_key = hashlib.sha256(image_bytes).hexdigest()
            cached = cache.get(cache_key)
            if cached is not None:
                return cached

        # Compute embedding
        client = await self._get_client()
        img_array = self.preprocess_image(image_bytes)

        input_tensor = InferInput('images', [1, 3, 256, 256], 'FP32')
        input_tensor.set_data_from_numpy(img_array)

        output = InferRequestedOutput('image_embeddings')

        response = await client.infer(
            model_name=self.settings.models.CLIP_MODELS['image_encoder'],
            inputs=[input_tensor],
            outputs=[output],
        )

        embedding = response.as_numpy('image_embeddings')[0]

        # Cache result
        if use_cache and cache_key:
            cache.set(cache_key, embedding)

        return embedding

    async def encode_text(self, text: str, use_cache: bool = True) -> np.ndarray:
        """
        Encode text to 512-dim MobileCLIP embedding.

        Uses MobileCLIP2-S2 text encoder via Triton.
        Tokenizer is cached singleton (~500ms savings on first call).

        Args:
            text: Query text string
            use_cache: Use embedding cache (default True)

        Returns:
            512-dim L2-normalized embedding vector
        """
        # Check cache
        cache_key = None
        if use_cache:
            cache = get_text_cache()
            cache_key = hashlib.sha256(text.encode()).hexdigest()
            cached = cache.get(cache_key)
            if cached is not None:
                return cached

        # Tokenize (using cached singleton - 500ms savings)
        tokenizer = get_clip_tokenizer()
        tokens = tokenizer(
            text, padding='max_length', max_length=77, truncation=True, return_tensors='np'
        )

        # Compute embedding
        client = await self._get_client()

        input_tensor = InferInput('text_tokens', [1, 77], 'INT64')
        input_tensor.set_data_from_numpy(tokens['input_ids'].astype(np.int64))

        output = InferRequestedOutput('text_embeddings')

        response = await client.infer(
            model_name=self.settings.models.CLIP_MODELS['text_encoder'],
            inputs=[input_tensor],
            outputs=[output],
        )

        embedding = response.as_numpy('text_embeddings')[0]

        # Cache result
        if use_cache and cache_key:
            cache.set(cache_key, embedding)

        return embedding

    async def encode_image_batch(
        self, image_bytes_list: list[bytes], use_cache: bool = True
    ) -> list[np.ndarray]:
        """
        Encode multiple images concurrently.

        Uses asyncio.gather for concurrent Triton requests.
        Triton batches these requests automatically.

        Args:
            image_bytes_list: List of JPEG/PNG bytes
            use_cache: Use embedding cache (default True)

        Returns:
            List of 512-dim L2-normalized embeddings
        """
        tasks = [
            self.encode_image(img_bytes, use_cache=use_cache) for img_bytes in image_bytes_list
        ]

        return await asyncio.gather(*tasks)

    @staticmethod
    def get_embedding_norm(embedding: np.ndarray) -> float:
        """
        Calculate L2 norm of embedding.

        For L2-normalized embeddings, this should be ~1.0.

        Args:
            embedding: Embedding vector

        Returns:
            L2 norm (float)
        """
        return float(np.linalg.norm(embedding))
