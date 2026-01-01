"""
Track E: OpenSearch Client for Visual Search

This module provides a high-level interface for managing visual search indices
and performing k-NN similarity searches using MobileCLIP embeddings.

Key Features:
- Index creation with k-NN configuration (HNSW algorithm)
- Document ingestion with embeddings
- Cosine similarity search (L2-normalized embeddings)
- Hybrid search (text + visual)
- Bulk operations for performance

Index Schema:
- image_id: Unique identifier for the image
- image_path: File path or URL to the image
- global_embedding: 512-dim MobileCLIP embedding for entire image
- box_embeddings: List of 512-dim embeddings for detected objects
- normalized_boxes: List of [x1, y1, x2, y2] boxes in [0, 1] range
- det_classes: List of detected class IDs
- det_scores: List of detection confidence scores
- metadata: Optional dictionary for custom fields (tags, timestamps, etc.)

Usage:
    # Initialize client
    client = OpenSearchClient(hosts=["http://localhost:9200"])

    # Create index
    await client.create_visual_search_index()

    # Ingest documents
    await client.ingest_image(
        image_id="img_001",
        image_path="/path/to/image.jpg",
        global_embedding=np.array([...]),  # 512-dim
        box_embeddings=np.array([[...]]),  # [N, 512]
        normalized_boxes=np.array([[...]]),  # [N, 4]
        det_classes=[0, 15, 62],
        det_scores=[0.95, 0.87, 0.76],
        metadata={"source": "dataset_a", "timestamp": "2024-01-01"}
    )

    # Search by visual similarity
    results = await client.search_by_image_embedding(
        query_embedding=np.array([...]),  # 512-dim
        top_k=10
    )
"""

import logging
from typing import Any

import numpy as np
from opensearchpy import AsyncOpenSearch
from opensearchpy.helpers import async_bulk


logger = logging.getLogger(__name__)


class OpenSearchClient:
    """
    Async OpenSearch client for visual search with MobileCLIP embeddings.
    """

    def __init__(
        self,
        hosts: list[str] | None = None,
        http_auth: tuple | None = None,  # No auth needed when security plugin disabled
        verify_certs: bool = False,
        ssl_show_warn: bool = False,
        timeout: int = 30,
    ):
        """
        Initialize OpenSearch async client.

        Args:
            hosts: List of OpenSearch node URLs
            http_auth: Tuple of (username, password) for authentication (None when security disabled)
            verify_certs: Whether to verify SSL certificates
            ssl_show_warn: Whether to show SSL warnings
            timeout: Request timeout in seconds
        """
        hosts = hosts or ['http://localhost:9200']
        client_kwargs = {
            'hosts': hosts,
            'use_ssl': False,  # Set to True in production with proper certs
            'verify_certs': verify_certs,
            'ssl_show_warn': ssl_show_warn,
            'timeout': timeout,
        }

        # Only add auth if provided (security plugin may be disabled)
        if http_auth:
            client_kwargs['http_auth'] = http_auth

        self.client = AsyncOpenSearch(**client_kwargs)
        self.index_name = 'visual_search'
        self.embedding_dim = 512  # MobileCLIP2-S2

    async def close(self):
        """Close the OpenSearch client connection."""
        await self.client.close()

    async def ping(self) -> bool:
        """Check if OpenSearch is reachable."""
        try:
            return await self.client.ping()
        except Exception as e:
            logger.error(f'OpenSearch ping failed: {e}')
            return False

    async def create_visual_search_index(
        self, index_name: str | None = None, force_recreate: bool = False
    ) -> bool:
        """
        Create visual search index with k-NN configuration.

        Index Schema:
        - image_id (keyword): Unique identifier
        - image_path (keyword): File path or URL
        - global_embedding (knn_vector): 768-dim image embedding
        - box_embeddings (nested): Array of object embeddings
          - embedding (knn_vector): 768-dim object embedding
          - box (float[]): [x1, y1, x2, y2] normalized coordinates
          - class_id (integer): COCO class ID
          - score (float): Detection confidence
        - metadata (object): Flexible metadata storage
        - indexed_at (date): Timestamp

        Args:
            index_name: Custom index name (default: "visual_search")
            force_recreate: Delete existing index if present

        Returns:
            True if index created successfully
        """
        if index_name:
            self.index_name = index_name

        try:
            # Check if index exists
            exists = await self.client.indices.exists(index=self.index_name)

            if exists:
                if force_recreate:
                    logger.info(f'Deleting existing index: {self.index_name}')
                    await self.client.indices.delete(index=self.index_name)
                else:
                    logger.info(f'Index already exists: {self.index_name}')
                    return True

            # Index settings with k-NN configuration
            # OpenSearch 3.x: Use 'faiss' engine (nmslib is deprecated)
            index_body = {
                'settings': {
                    'index': {
                        'number_of_shards': 1,
                        'number_of_replicas': 0,  # Set to 1+ in production
                        'knn': True,  # Enable k-NN plugin
                    },
                    'analysis': {'analyzer': {'default': {'type': 'standard'}}},
                },
                'mappings': {
                    'properties': {
                        'image_id': {'type': 'keyword'},
                        'image_path': {'type': 'keyword'},
                        'global_embedding': {
                            'type': 'knn_vector',
                            'dimension': self.embedding_dim,
                            'method': {
                                'name': 'hnsw',
                                'space_type': 'cosinesimil',  # Cosine similarity (for L2-normalized)
                                'engine': 'faiss',  # OpenSearch 3.x: faiss replaces deprecated nmslib
                                'parameters': {
                                    'ef_construction': 512,  # Build-time quality
                                    'm': 16,  # Graph connectivity (higher = better quality, more memory)
                                    'ef_search': 512,  # Search-time quality
                                },
                            },
                        },
                        'box_embeddings': {
                            'type': 'nested',
                            'properties': {
                                'embedding': {
                                    'type': 'knn_vector',
                                    'dimension': self.embedding_dim,
                                    'method': {
                                        'name': 'hnsw',
                                        'space_type': 'cosinesimil',
                                        'engine': 'faiss',  # OpenSearch 3.x: faiss replaces deprecated nmslib
                                        'parameters': {
                                            'ef_construction': 512,
                                            'm': 16,
                                            'ef_search': 512,
                                        },
                                    },
                                },
                                'box': {
                                    'type': 'float',
                                    'index': False,  # No need to index coordinates
                                },
                                'class_id': {'type': 'integer'},
                                'score': {'type': 'float'},
                            },
                        },
                        'num_detections': {'type': 'integer'},
                        'metadata': {
                            'type': 'object',
                            'enabled': True,  # Allow flexible metadata
                        },
                        'indexed_at': {'type': 'date'},
                    }
                },
            }

            # Create index
            await self.client.indices.create(index=self.index_name, body=index_body)

            logger.info(f'Index created successfully: {self.index_name}')
            logger.info('k-NN enabled with HNSW algorithm (cosine similarity)')
            return True

        except Exception as e:
            logger.error(f'Failed to create index: {e}')
            return False

    async def ingest_image(
        self,
        image_id: str,
        image_path: str,
        global_embedding: np.ndarray,
        box_embeddings: np.ndarray | None = None,
        normalized_boxes: np.ndarray | None = None,
        det_classes: list[int] | None = None,
        det_scores: list[float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """
        Ingest single image document with embeddings.

        Args:
            image_id: Unique identifier for the image
            image_path: File path or URL to the image
            global_embedding: 768-dim L2-normalized embedding for entire image
            box_embeddings: [N, 768] embeddings for detected objects (optional)
            normalized_boxes: [N, 4] boxes in [0, 1] range (optional)
            det_classes: [N] class IDs for detected objects (optional)
            det_scores: [N] detection confidence scores (optional)
            metadata: Optional dictionary for custom fields

        Returns:
            True if ingestion successful
        """
        try:
            # Validate global embedding
            assert global_embedding.shape == (self.embedding_dim,), (
                f'Global embedding must be {self.embedding_dim}-dim'
            )

            # Build document
            from datetime import datetime, timezone

            doc = {
                'image_id': image_id,
                'image_path': image_path,
                'global_embedding': global_embedding.tolist(),
                'indexed_at': datetime.now(timezone.utc).isoformat(),
            }

            # Add box embeddings if provided
            if box_embeddings is not None:
                num_boxes = box_embeddings.shape[0]
                assert box_embeddings.shape == (num_boxes, self.embedding_dim), (
                    f'Box embeddings must be [N, {self.embedding_dim}]'
                )

                # Build nested box objects
                box_objects = []
                for i in range(num_boxes):
                    box_obj = {
                        'embedding': box_embeddings[i].tolist(),
                    }

                    if normalized_boxes is not None:
                        box_obj['box'] = normalized_boxes[i].tolist()

                    if det_classes is not None:
                        box_obj['class_id'] = int(det_classes[i])

                    if det_scores is not None:
                        box_obj['score'] = float(det_scores[i])

                    box_objects.append(box_obj)

                doc['box_embeddings'] = box_objects
                doc['num_detections'] = num_boxes
            else:
                doc['num_detections'] = 0

            # Add metadata
            if metadata:
                doc['metadata'] = metadata

            # Index document
            response = await self.client.index(
                index=self.index_name,
                id=image_id,
                body=doc,
                refresh=False,  # Set to True for immediate search visibility
            )

            return response['result'] in ['created', 'updated']

        except Exception as e:
            logger.error(f'Failed to ingest image {image_id}: {e}')
            return False

    async def bulk_ingest_images(
        self, documents: list[dict[str, Any]], refresh: bool = False
    ) -> tuple:
        """
        Bulk ingest multiple image documents.

        Args:
            documents: List of document dictionaries (same format as ingest_image)
            refresh: Whether to refresh index after bulk operation

        Returns:
            Tuple of (success_count, failed_count)
        """
        try:
            actions = []
            for doc in documents:
                # Convert numpy arrays to lists
                doc_copy = doc.copy()
                if 'global_embedding' in doc_copy:
                    doc_copy['global_embedding'] = doc_copy['global_embedding'].tolist()

                if 'box_embeddings' in doc_copy:
                    box_objects = []
                    num_boxes = len(doc_copy['box_embeddings'])
                    for i in range(num_boxes):
                        box_obj = {
                            'embedding': doc_copy['box_embeddings'][i].tolist(),
                        }
                        if 'normalized_boxes' in doc_copy:
                            box_obj['box'] = doc_copy['normalized_boxes'][i].tolist()
                        if 'det_classes' in doc_copy:
                            box_obj['class_id'] = int(doc_copy['det_classes'][i])
                        if 'det_scores' in doc_copy:
                            box_obj['score'] = float(doc_copy['det_scores'][i])
                        box_objects.append(box_obj)
                    doc_copy['box_embeddings'] = box_objects

                # Remove arrays used for construction
                for key in ['normalized_boxes', 'det_classes', 'det_scores']:
                    doc_copy.pop(key, None)

                from datetime import datetime, timezone

                doc_copy['indexed_at'] = datetime.now(timezone.utc).isoformat()

                actions.append(
                    {'_index': self.index_name, '_id': doc_copy['image_id'], '_source': doc_copy}
                )

            success, failed = await async_bulk(
                self.client, actions, refresh=refresh, raise_on_error=False
            )

            logger.info(f'Bulk ingestion: {success} succeeded, {len(failed)} failed')
            return success, len(failed)

        except Exception as e:
            logger.error(f'Bulk ingestion failed: {e}')
            return 0, len(documents)

    async def search_by_image_embedding(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        min_score: float | None = None,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search for similar images by global image embedding.

        Args:
            query_embedding: 768-dim L2-normalized query embedding
            top_k: Number of results to return
            min_score: Minimum similarity score threshold (optional)
            filter_metadata: Metadata filters (e.g., {"source": "dataset_a"})

        Returns:
            List of search results with scores and documents
        """
        try:
            assert query_embedding.shape == (self.embedding_dim,), (
                f'Query embedding must be {self.embedding_dim}-dim'
            )

            # Build k-NN query
            query = {
                'size': top_k,
                'query': {
                    'knn': {'global_embedding': {'vector': query_embedding.tolist(), 'k': top_k}}
                },
            }

            # Add metadata filters if provided
            if filter_metadata:
                filter_clauses = [
                    {'term': {f'metadata.{key}': value}} for key, value in filter_metadata.items()
                ]
                query['query'] = {'bool': {'must': [query['query']], 'filter': filter_clauses}}

            # Add minimum score filter
            if min_score is not None:
                query['min_score'] = min_score

            # Execute search
            response = await self.client.search(index=self.index_name, body=query)

            # Parse results
            return [
                {
                    'image_id': hit['_source']['image_id'],
                    'image_path': hit['_source']['image_path'],
                    'score': hit['_score'],
                    'num_detections': hit['_source'].get('num_detections', 0),
                    'metadata': hit['_source'].get('metadata', {}),
                }
                for hit in response['hits']['hits']
            ]

        except Exception as e:
            logger.error(f'Image search failed: {e}')
            return []

    async def search_by_object_embedding(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        min_score: float | None = None,
        class_filter: list[int] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search for similar objects by object embedding (nested k-NN).

        Args:
            query_embedding: 768-dim L2-normalized query embedding
            top_k: Number of results to return
            min_score: Minimum similarity score threshold (optional)
            class_filter: Filter by detected class IDs (e.g., [0, 15, 62])

        Returns:
            List of search results with matched objects
        """
        try:
            assert query_embedding.shape == (self.embedding_dim,), (
                f'Query embedding must be {self.embedding_dim}-dim'
            )

            # Build nested k-NN query
            query = {
                'size': top_k,
                'query': {
                    'nested': {
                        'path': 'box_embeddings',
                        'query': {
                            'knn': {
                                'box_embeddings.embedding': {
                                    'vector': query_embedding.tolist(),
                                    'k': top_k,
                                }
                            }
                        },
                        'inner_hits': {
                            'size': 5,  # Return top 5 matching boxes per image
                            '_source': [
                                'box_embeddings.box',
                                'box_embeddings.class_id',
                                'box_embeddings.score',
                            ],
                        },
                    }
                },
            }

            # Add class filter if provided
            if class_filter:
                query['query']['nested']['query'] = {
                    'bool': {
                        'must': [query['query']['nested']['query']],
                        'filter': [{'terms': {'box_embeddings.class_id': class_filter}}],
                    }
                }

            # Add minimum score filter
            if min_score is not None:
                query['min_score'] = min_score

            # Execute search
            response = await self.client.search(index=self.index_name, body=query)

            # Parse results
            results = []
            for hit in response['hits']['hits']:
                result = {
                    'image_id': hit['_source']['image_id'],
                    'image_path': hit['_source']['image_path'],
                    'score': hit['_score'],
                    'matched_objects': [],
                }

                # Extract matched objects from inner_hits
                if 'inner_hits' in hit:
                    for inner_hit in hit['inner_hits']['box_embeddings']['hits']['hits']:
                        result['matched_objects'].append(
                            {
                                'box': inner_hit['_source'].get('box'),
                                'class_id': inner_hit['_source'].get('class_id'),
                                'score': inner_hit['_source'].get('score'),
                                'similarity': inner_hit['_score'],
                            }
                        )

                results.append(result)

            return results

        except Exception as e:
            logger.error(f'Object search failed: {e}')
            return []

    async def delete_index(self, index_name: str | None = None) -> bool:
        """Delete the visual search index."""
        try:
            idx = index_name or self.index_name
            await self.client.indices.delete(index=idx)
            logger.info(f'Index deleted: {idx}')
            return True
        except Exception as e:
            logger.error(f'Failed to delete index: {e}')
            return False

    async def get_index_stats(self, index_name: str | None = None) -> dict[str, Any]:
        """Get statistics for the visual search index."""
        try:
            idx = index_name or self.index_name
            response = await self.client.indices.stats(index=idx)
            return {
                'total_documents': response['_all']['primaries']['docs']['count'],
                'index_size_bytes': response['_all']['primaries']['store']['size_in_bytes'],
                'index_size_mb': round(
                    response['_all']['primaries']['store']['size_in_bytes'] / 1024 / 1024, 2
                ),
            }
        except Exception as e:
            logger.error(f'Failed to get index stats: {e}')
            return {}


# Convenience function for standalone usage
async def create_client(**kwargs) -> OpenSearchClient:
    """Create and return an OpenSearch client instance."""
    return OpenSearchClient(**kwargs)
