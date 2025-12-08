"""
Track E: Caching Utilities for Performance Optimization

Provides in-memory caching for embeddings to reduce Triton inference calls.

Key Features:
- LRU cache for image embeddings
- TTL-based expiration
- Thread-safe operations
- Memory-efficient storage

Usage:
    from src.utils.cache import EmbeddingCache

    cache = EmbeddingCache(max_size=1000, ttl_seconds=3600)

    # Try cache first
    embedding = cache.get(image_hash)
    if embedding is None:
        embedding = await encode_image(image_bytes)
        cache.set(image_hash, embedding)
"""

import hashlib
import threading
import time
from collections import OrderedDict
from typing import Any

import numpy as np
from transformers import CLIPTokenizer


class EmbeddingCache:
    """
    Thread-safe LRU cache for image/text embeddings.

    Features:
    - LRU eviction policy
    - TTL-based expiration
    - Memory-efficient numpy storage
    - Thread-safe access
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        """
        Initialize embedding cache.

        Args:
            max_size: Maximum number of embeddings to cache
            ttl_seconds: Time-to-live for cached embeddings (seconds)
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.lock = threading.RLock()

        self.hits = 0
        self.misses = 0

    def _compute_hash(self, data: bytes) -> str:
        """
        Compute SHA256 hash of data.

        Args:
            data: Input bytes

        Returns:
            Hex digest string
        """
        return hashlib.sha256(data).hexdigest()

    def get(self, key: str) -> np.ndarray | None:
        """
        Get embedding from cache.

        Args:
            key: Cache key (hash of input data)

        Returns:
            Cached embedding or None if not found/expired
        """
        with self.lock:
            if key not in self.cache:
                self.misses += 1
                return None

            entry = self.cache[key]

            # Check TTL
            if time.time() - entry['timestamp'] > self.ttl_seconds:
                del self.cache[key]
                self.misses += 1
                return None

            # Move to end (LRU)
            self.cache.move_to_end(key)
            self.hits += 1

            return entry['embedding']

    def set(self, key: str, embedding: np.ndarray):
        """
        Store embedding in cache.

        Args:
            key: Cache key (hash of input data)
            embedding: Numpy array to cache
        """
        with self.lock:
            # Evict oldest if at capacity
            if len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)  # Remove oldest (first item)

            # Store embedding with timestamp
            self.cache[key] = {
                'embedding': embedding.copy(),  # Copy to avoid external mutations
                'timestamp': time.time(),
            }

    def get_or_compute(
        self, data: bytes, compute_fn: callable, use_hash: bool = True
    ) -> tuple[np.ndarray, bool]:
        """
        Get embedding from cache or compute if missing.

        Args:
            data: Input data bytes
            compute_fn: Function to compute embedding if not cached
            use_hash: Whether to hash the data (default: True)

        Returns:
            Tuple of (embedding, was_cached)
        """
        # Compute key
        key = self._compute_hash(data) if use_hash else data.decode()

        # Try cache
        embedding = self.get(key)
        if embedding is not None:
            return embedding, True

        # Compute and cache
        embedding = compute_fn(data)
        self.set(key, embedding)

        return embedding, False

    def clear(self):
        """Clear all cached embeddings."""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0

    def get_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache metrics
        """
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0.0

            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate,
                'ttl_seconds': self.ttl_seconds,
            }

    def __len__(self):
        """Return current cache size."""
        with self.lock:
            return len(self.cache)


# Global cache instances (singletons)
# Justification: Shared caches reduce redundant embedding computations across requests
# - Image embeddings are expensive (~10-20ms per image on GPU)
# - Text embeddings are expensive (~5-10ms per query)
# - Thread-safe LRU cache with TTL prevents unbounded memory growth
# - Alternative (per-request caches) would waste computation on duplicate queries
_image_embedding_cache: EmbeddingCache | None = None
_text_embedding_cache: EmbeddingCache | None = None


def get_image_cache(max_size: int = 1000, ttl_seconds: int = 3600) -> EmbeddingCache:
    """
    Get or create global image embedding cache.

    Thread-safe singleton with LRU eviction and TTL expiry.

    Args:
        max_size: Maximum cache size
        ttl_seconds: TTL for cached embeddings

    Returns:
        Singleton EmbeddingCache instance
    """
    global _image_embedding_cache  # noqa: PLW0603 - Singleton pattern documented above

    if _image_embedding_cache is None:
        _image_embedding_cache = EmbeddingCache(max_size=max_size, ttl_seconds=ttl_seconds)

    return _image_embedding_cache


def get_text_cache(max_size: int = 1000, ttl_seconds: int = 3600) -> EmbeddingCache:
    """
    Get or create global text embedding cache.

    Thread-safe singleton with LRU eviction and TTL expiry.

    Args:
        max_size: Maximum cache size
        ttl_seconds: TTL for cached embeddings

    Returns:
        Singleton EmbeddingCache instance
    """
    global _text_embedding_cache  # noqa: PLW0603 - Singleton pattern documented above

    if _text_embedding_cache is None:
        _text_embedding_cache = EmbeddingCache(max_size=max_size, ttl_seconds=ttl_seconds)

    return _text_embedding_cache


def clear_all_caches():
    """Clear all global caches."""
    if _image_embedding_cache:
        _image_embedding_cache.clear()
    if _text_embedding_cache:
        _text_embedding_cache.clear()


def get_all_cache_stats() -> dict[str, dict[str, Any]]:
    """
    Get statistics for all caches.

    Returns:
        Dictionary with stats for each cache
    """
    stats = {}

    if _image_embedding_cache:
        stats['image_cache'] = _image_embedding_cache.get_stats()

    if _text_embedding_cache:
        stats['text_cache'] = _text_embedding_cache.get_stats()

    return stats


# CLIPTokenizer singleton (loading is expensive - ~500ms)
# Justification: Tokenizer loading from HuggingFace is expensive (~500ms)
# - Shared instance across all text embedding requests prevents repeated loading
# - Thread-safe lock prevents concurrent initialization attempts
_clip_tokenizer = None
_tokenizer_lock = threading.Lock()


def get_clip_tokenizer():
    """
    Get cached CLIPTokenizer (singleton, thread-safe).

    Loading CLIPTokenizer.from_pretrained() takes ~500ms.
    This singleton ensures it's only loaded once per process.

    Returns:
        CLIPTokenizer instance
    """
    global _clip_tokenizer  # noqa: PLW0603 - Singleton pattern for CLIP tokenizer

    if _clip_tokenizer is None:
        with _tokenizer_lock:
            # Double-check locking pattern
            if _clip_tokenizer is None:
                _clip_tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')

    return _clip_tokenizer
