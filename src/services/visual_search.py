"""
Track E: Visual Search Service.

Orchestrates inference + OpenSearch operations for visual search.
Bridges InferenceService (Triton) and OpenSearchClient (vector search).

Key Features:
- Image ingestion with embedding extraction
- Image-to-image similarity search
- Text-to-image search
- Object-level search (per-box embeddings)
- Index lifecycle management

Usage:
    service = VisualSearchService(opensearch_client)

    # Ingest
    result = await service.ingest_image(image_bytes, image_id)

    # Search
    results = await service.search_by_image(query_bytes, top_k=10)
    results = await service.search_by_text("red car", top_k=10)
"""

import logging
from typing import Any

import numpy as np

from src.clients.opensearch import OpenSearchClient
from src.services.inference import InferenceService


logger = logging.getLogger(__name__)


class VisualSearchService:
    """
    Service for visual search operations combining inference and OpenSearch.

    Design:
    - Accepts OpenSearchClient via dependency injection (testability)
    - Creates InferenceService internally (follows existing patterns)
    - All OpenSearch methods are async (client is async)
    - Inference is sync (FastAPI thread pool handles blocking)
    """

    def __init__(self, opensearch_client: OpenSearchClient):
        """
        Initialize visual search service.

        Args:
            opensearch_client: Async OpenSearch client for vector operations
        """
        self.opensearch = opensearch_client
        self.inference = InferenceService()

    # =========================================================================
    # Ingestion Operations
    # =========================================================================

    async def ingest_image(
        self,
        image_bytes: bytes,
        image_id: str,
        image_path: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Ingest single image: run inference + index in OpenSearch.

        Pipeline:
        1. Run Track E full ensemble (YOLO + MobileCLIP)
        2. Extract global + box embeddings
        3. Index document in OpenSearch

        Args:
            image_bytes: Raw JPEG/PNG bytes
            image_id: Unique identifier for the image
            image_path: Optional file path (for retrieval)
            metadata: Optional metadata dictionary

        Returns:
            dict with status, image_id, num_detections, etc.
        """
        try:
            # Get raw inference result from Triton client (need embeddings, not formatted response)
            from src.clients.triton_client import get_triton_client
            from src.config import get_settings

            settings = get_settings()
            client = get_triton_client(settings.triton_url)
            result = client.infer_track_e(image_bytes, full_pipeline=True)

            global_embedding = np.array(result['image_embedding'])

            # Prepare box data
            box_embeddings = None
            normalized_boxes = None
            det_classes = None
            det_scores = None

            if result['num_dets'] > 0:
                box_embeddings = np.array(result.get('box_embeddings', []))
                normalized_boxes = np.array(result.get('normalized_boxes', []))
                det_classes = result.get('classes', [])
                det_scores = result.get('scores', [])

            # Index in OpenSearch
            success = await self.opensearch.ingest_image(
                image_id=image_id,
                image_path=image_path or image_id,
                global_embedding=global_embedding,
                box_embeddings=box_embeddings,
                normalized_boxes=normalized_boxes,
                det_classes=det_classes,
                det_scores=det_scores,
                metadata=metadata,
            )

            return {
                'status': 'success' if success else 'failed',
                'image_id': image_id,
                'num_detections': result['num_dets'],
                'embedding_norm': float(np.linalg.norm(global_embedding)),
                'indexed': success,
            }

        except Exception as e:
            logger.error(f'Failed to ingest image {image_id}: {e}')
            return {
                'status': 'error',
                'image_id': image_id,
                'error': str(e),
            }

    async def ingest_batch(
        self,
        images: list[tuple[bytes, str, str | None, dict | None]],
        refresh: bool = False,
    ) -> dict[str, Any]:
        """
        Batch ingest multiple images.

        Args:
            images: List of (image_bytes, image_id, image_path, metadata)
            refresh: Whether to refresh index after bulk operation

        Returns:
            dict with succeeded/failed counts
        """
        from src.clients.triton_client import get_triton_client
        from src.config import get_settings

        settings = get_settings()
        client = get_triton_client(settings.triton_url)

        documents = []
        failed = []

        for image_bytes, image_id, image_path, metadata in images:
            try:
                # Run inference
                result = client.infer_track_e(image_bytes, full_pipeline=True)

                doc = {
                    'image_id': image_id,
                    'image_path': image_path or image_id,
                    'global_embedding': np.array(result['image_embedding']),
                    'metadata': metadata,
                }

                if result['num_dets'] > 0:
                    doc['box_embeddings'] = np.array(result.get('box_embeddings', []))
                    doc['normalized_boxes'] = np.array(result.get('normalized_boxes', []))
                    doc['det_classes'] = result.get('classes', [])
                    doc['det_scores'] = result.get('scores', [])

                documents.append(doc)

            except Exception as e:
                logger.error(f'Failed to process {image_id}: {e}')
                failed.append({'image_id': image_id, 'error': str(e)})

        # Bulk index
        succeeded, bulk_failed = await self.opensearch.bulk_ingest_images(documents, refresh)

        return {
            'status': 'success',
            'total': len(images),
            'succeeded': succeeded,
            'failed': len(failed) + bulk_failed,
            'errors': failed if failed else None,
        }

    # =========================================================================
    # Search Operations
    # =========================================================================

    async def search_by_image(
        self,
        image_bytes: bytes,
        top_k: int = 10,
        min_score: float | None = None,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Image-to-image similarity search.

        Pipeline:
        1. Encode query image via MobileCLIP
        2. k-NN search on global_embedding field

        Args:
            image_bytes: Query image bytes
            top_k: Number of results to return
            min_score: Minimum similarity threshold
            filter_metadata: Metadata filters

        Returns:
            List of search results with scores
        """
        # Encode query image (sync, uses cache)
        query_embedding = self.inference.encode_image_sync(image_bytes, use_cache=True)

        # Search OpenSearch
        return await self.opensearch.search_by_image_embedding(
            query_embedding=query_embedding,
            top_k=top_k,
            min_score=min_score,
            filter_metadata=filter_metadata,
        )

    async def search_by_text(
        self,
        text: str,
        top_k: int = 10,
        min_score: float | None = None,
        filter_metadata: dict[str, Any] | None = None,
        use_cache: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Text-to-image search using MobileCLIP text encoder.

        Pipeline:
        1. Tokenize + encode text to embedding
        2. k-NN search on global_embedding field

        Args:
            text: Query text string
            top_k: Number of results
            min_score: Minimum similarity
            filter_metadata: Metadata filters
            use_cache: Use text embedding cache

        Returns:
            List of search results with scores
        """
        # Encode text (sync, uses cache)
        query_embedding = self.inference.encode_text_sync(text, use_cache)

        # Search OpenSearch
        return await self.opensearch.search_by_image_embedding(
            query_embedding=query_embedding,
            top_k=top_k,
            min_score=min_score,
            filter_metadata=filter_metadata,
        )

    async def search_by_object(
        self,
        image_bytes: bytes,
        box_index: int = 0,
        top_k: int = 10,
        min_score: float | None = None,
        class_filter: list[int] | None = None,
    ) -> dict[str, Any]:
        """
        Object-to-object search using per-box embeddings.

        Pipeline:
        1. Run Track E full ensemble to get box embeddings
        2. Use specified box's embedding for nested k-NN search

        Args:
            image_bytes: Query image bytes
            box_index: Index of detected box to use (default: 0)
            top_k: Number of results
            min_score: Minimum similarity
            class_filter: Filter by COCO class IDs

        Returns:
            dict with results and query box info
        """
        from src.clients.triton_client import get_triton_client
        from src.config import get_settings

        settings = get_settings()
        client = get_triton_client(settings.triton_url)

        # Run full inference to get box embeddings
        result = client.infer_track_e(image_bytes, full_pipeline=True)

        if result['num_dets'] == 0:
            return {
                'status': 'error',
                'error': 'No objects detected in query image',
                'results': [],
            }

        if box_index >= result['num_dets']:
            return {
                'status': 'error',
                'error': f'Box index {box_index} out of range (0-{result["num_dets"] - 1})',
                'results': [],
            }

        # Get the specified box embedding
        box_embeddings = np.array(result['box_embeddings'])
        query_embedding = box_embeddings[box_index]

        # Get query box info
        boxes = result.get('normalized_boxes', [])
        classes = result.get('classes', [])
        scores = result.get('scores', [])

        query_box_info = {
            'box': boxes[box_index].tolist() if len(boxes) > box_index else None,
            'class_id': int(classes[box_index]) if len(classes) > box_index else None,
            'score': float(scores[box_index]) if len(scores) > box_index else None,
        }

        # Search OpenSearch (nested k-NN on box embeddings)
        results = await self.opensearch.search_by_object_embedding(
            query_embedding=query_embedding,
            top_k=top_k,
            min_score=min_score,
            class_filter=class_filter,
        )

        return {
            'status': 'success',
            'query_box': query_box_info,
            'results': results,
        }

    # =========================================================================
    # Index Management
    # =========================================================================

    async def setup_index(self, force_recreate: bool = False) -> bool:
        """
        Create visual search index (idempotent).

        Args:
            force_recreate: Delete existing index if present

        Returns:
            True if index created/exists successfully
        """
        return await self.opensearch.create_visual_search_index(force_recreate=force_recreate)

    async def delete_index(self) -> bool:
        """Delete the visual search index."""
        return await self.opensearch.delete_index()

    async def get_index_stats(self) -> dict[str, Any]:
        """
        Get index statistics.

        Returns:
            dict with total_documents, index_size_mb, etc.
        """
        stats = await self.opensearch.get_index_stats()
        return {
            'status': 'success' if stats else 'error',
            **stats,
        }

    async def ping_opensearch(self) -> bool:
        """Check if OpenSearch is reachable."""
        return await self.opensearch.ping()
