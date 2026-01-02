"""
Track E: Visual Search Service.

Orchestrates inference + OpenSearch operations for visual search.
Bridges InferenceService (Triton) and OpenSearchClient (multi-index vector search).

Multi-Index Architecture:
- visual_search_global: Whole image similarity (scene matching)
- visual_search_vehicles: Vehicle detections (car, truck, motorcycle, bus, boat)
- visual_search_people: Person appearance (clothing, pose)
- visual_search_faces: Face identity matching (future: ArcFace)

Key Features:
- Auto-routing ingestion by detection class
- Category-specific search (vehicles, people, global)
- Index lifecycle management for all indexes

Usage:
    service = VisualSearchService(opensearch_client)

    # Ingest (auto-routes by class)
    result = await service.ingest_image(image_bytes, image_id)
    # Result: {global: True, vehicles: 2, people: 1, skipped: 0}

    # Category-specific search
    results = await service.search_global(query_bytes, top_k=10)
    results = await service.search_vehicles(query_embedding, top_k=10)
    results = await service.search_people(query_embedding, top_k=10)
"""

import logging
from typing import Any

import numpy as np

from src.clients.opensearch import DetectionCategory, OpenSearchClient, get_category
from src.services.inference import InferenceService


logger = logging.getLogger(__name__)


class VisualSearchService:
    """
    Service for visual search operations combining inference and OpenSearch.

    Google Photos-like capabilities:
    - Similar image search (whole scene matching)
    - Search for people by appearance
    - Search for vehicles
    - Search for faces (future - ArcFace identity matching)
    - Extensible for products, animals, etc.

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
    # Ingestion Operations (Auto-Routes to Category Indexes)
    # =========================================================================

    async def ingest_image(
        self,
        image_bytes: bytes,
        image_id: str,
        image_path: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Ingest single image with auto-routing to category indexes.

        Pipeline:
        1. Run Track E full ensemble (YOLO + MobileCLIP)
        2. Extract global + box embeddings
        3. Route to appropriate indexes:
           - Global embedding -> visual_search_global
           - Person detections -> visual_search_people
           - Vehicle detections -> visual_search_vehicles

        Args:
            image_bytes: Raw JPEG/PNG bytes
            image_id: Unique identifier for the image
            image_path: Optional file path (for retrieval)
            metadata: Optional metadata dictionary

        Returns:
            dict with status, routing info, and counts per category
        """
        try:
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

            # Index in OpenSearch (auto-routes by class)
            ingest_result = await self.opensearch.ingest_image(
                image_id=image_id,
                image_path=image_path or image_id,
                global_embedding=global_embedding,
                box_embeddings=box_embeddings,
                normalized_boxes=normalized_boxes,
                det_classes=det_classes,
                det_scores=det_scores,
                image_width=result.get('image_width'),
                image_height=result.get('image_height'),
                metadata=metadata,
            )

            return {
                'status': 'success' if ingest_result['global'] else 'failed',
                'image_id': image_id,
                'num_detections': result['num_dets'],
                'embedding_norm': float(np.linalg.norm(global_embedding)),
                'indexed': {
                    'global': ingest_result['global'],
                    'vehicles': ingest_result['vehicles'],
                    'people': ingest_result['people'],
                    'skipped': ingest_result['skipped'],
                },
                'errors': ingest_result.get('errors', []),
            }

        except Exception as e:
            logger.error(f'Failed to ingest image {image_id}: {e}')
            return {
                'status': 'error',
                'image_id': image_id,
                'error': str(e),
            }

    async def ingest_faces(
        self,
        image_bytes: bytes,
        image_id: str,
        image_path: str | None = None,
        person_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Detect faces in image and ingest with ArcFace embeddings.

        Pipeline:
        1. Run SCRFD face detection
        2. Extract ArcFace embeddings for each face
        3. Index faces to visual_search_faces

        Args:
            image_bytes: Raw JPEG/PNG bytes
            image_id: Unique identifier for the image
            image_path: Optional file path (for retrieval)
            person_name: Optional name/label for faces
            metadata: Optional metadata dictionary

        Returns:
            dict with status and face count
        """
        try:
            from src.clients.triton_client import get_triton_client
            from src.config import get_settings

            settings = get_settings()
            client = get_triton_client(settings.triton_url)

            # Run unified pipeline (face detection only on person crops - faster, fewer false positives)
            result = client.infer_unified(image_bytes)

            if result['num_faces'] == 0:
                return {
                    'status': 'success',
                    'image_id': image_id,
                    'num_faces': 0,
                    'indexed': 0,
                    'message': 'No faces detected',
                }

            # Prepare face data from unified pipeline output
            num_faces = result['num_faces']
            faces = [
                {
                    'box': result['face_boxes'][i].tolist()
                    if hasattr(result['face_boxes'][i], 'tolist')
                    else result['face_boxes'][i],
                    'landmarks': result['face_landmarks'][i].tolist()
                    if hasattr(result['face_landmarks'][i], 'tolist')
                    else result['face_landmarks'][i],
                    'score': float(result['face_scores'][i]),
                    'quality': 0.0,  # unified pipeline doesn't compute quality
                    'person_idx': int(
                        result['face_person_idx'][i]
                    ),  # which person box this face belongs to
                }
                for i in range(num_faces)
            ]
            embeddings = result['face_embeddings']

            # Ingest to OpenSearch
            ingest_result = await self.opensearch.ingest_faces(
                image_id=image_id,
                image_path=image_path or image_id,
                faces=faces,
                embeddings=embeddings,
                person_name=person_name,
                metadata=metadata,
            )

            return {
                'status': 'success' if ingest_result['faces'] > 0 else 'failed',
                'image_id': image_id,
                'num_faces': result['num_faces'],
                'indexed': ingest_result['faces'],
                'errors': ingest_result.get('errors', []),
            }

        except Exception as e:
            logger.error(f'Failed to ingest faces from {image_id}: {e}')
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
        Batch ingest multiple images with auto-routing.

        Args:
            images: List of (image_bytes, image_id, image_path, metadata)
            refresh: Whether to refresh indexes after bulk operation

        Returns:
            dict with counts per category
        """
        from src.clients.triton_client import get_triton_client
        from src.config import get_settings

        settings = get_settings()
        client = get_triton_client(settings.triton_url)

        documents = []
        failed = []

        for image_bytes, image_id, image_path, metadata in images:
            try:
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

        # Bulk index (auto-routes by class)
        bulk_result = await self.opensearch.bulk_ingest(documents, refresh)

        return {
            'status': 'success',
            'total': len(images),
            'indexed': {
                'global': bulk_result['global'],
                'vehicles': bulk_result['vehicles'],
                'people': bulk_result['people'],
            },
            'inference_failed': len(failed),
            'errors': failed + bulk_result.get('errors', []),
        }

    # =========================================================================
    # Search Operations (Google Photos-like)
    # =========================================================================

    async def search_by_image(
        self,
        image_bytes: bytes,
        top_k: int = 10,
        min_score: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Image-to-image similarity search (whole scene).

        Like Google Photos "Find similar images" - matches overall visual similarity.

        Pipeline:
        1. Encode query image via MobileCLIP
        2. k-NN search on visual_search_global index

        Args:
            image_bytes: Query image bytes
            top_k: Number of results to return
            min_score: Minimum similarity threshold

        Returns:
            List of similar images with scores
        """
        query_embedding = self.inference.encode_image_sync(image_bytes, use_cache=True)
        return await self.opensearch.search_global(
            query_embedding=query_embedding,
            top_k=top_k,
            min_score=min_score,
        )

    async def search_by_text(
        self,
        text: str,
        top_k: int = 10,
        min_score: float | None = None,
        use_cache: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Text-to-image search using MobileCLIP text encoder.

        Like Google Photos search - "beach sunset", "birthday party", etc.

        Pipeline:
        1. Tokenize + encode text to embedding
        2. k-NN search on visual_search_global index

        Args:
            text: Query text string
            top_k: Number of results
            min_score: Minimum similarity
            use_cache: Use text embedding cache

        Returns:
            List of matching images with scores
        """
        query_embedding = self.inference.encode_text_sync(text, use_cache)
        return await self.opensearch.search_global(
            query_embedding=query_embedding,
            top_k=top_k,
            min_score=min_score,
        )

    async def search_vehicles(
        self,
        image_bytes: bytes,
        box_index: int = 0,
        top_k: int = 10,
        min_score: float | None = None,
        class_filter: list[int] | None = None,
    ) -> dict[str, Any]:
        """
        Find similar vehicles across all images.

        Like "Find all red cars" or "Show me motorcycles like this one".

        Pipeline:
        1. Run Track E to get vehicle detection embedding
        2. k-NN search on visual_search_vehicles index

        Args:
            image_bytes: Query image with vehicle
            box_index: Which detected vehicle to use (0 = first)
            top_k: Number of results
            min_score: Minimum similarity
            class_filter: Filter by vehicle type [2=car, 3=motorcycle, 5=bus, 7=truck, 8=boat]

        Returns:
            dict with query vehicle info and matching vehicles
        """
        from src.clients.triton_client import get_triton_client
        from src.config import get_settings

        settings = get_settings()
        client = get_triton_client(settings.triton_url)
        result = client.infer_track_e(image_bytes, full_pipeline=True)

        if result['num_dets'] == 0:
            return {'status': 'error', 'error': 'No objects detected', 'results': []}

        # Find vehicle detections
        classes = result.get('classes', [])
        vehicle_indices = [
            i for i, c in enumerate(classes) if get_category(int(c)) == DetectionCategory.VEHICLE
        ]

        if not vehicle_indices:
            return {'status': 'error', 'error': 'No vehicles detected', 'results': []}

        if box_index >= len(vehicle_indices):
            return {
                'status': 'error',
                'error': f'Vehicle index {box_index} out of range (0-{len(vehicle_indices) - 1})',
                'results': [],
            }

        actual_idx = vehicle_indices[box_index]
        box_embeddings = np.array(result['box_embeddings'])
        query_embedding = box_embeddings[actual_idx]

        boxes = result.get('normalized_boxes', [])
        scores = result.get('scores', [])

        query_info = {
            'box': boxes[actual_idx].tolist() if len(boxes) > actual_idx else None,
            'class_id': int(classes[actual_idx]),
            'score': float(scores[actual_idx]) if len(scores) > actual_idx else None,
        }

        results = await self.opensearch.search_vehicles(
            query_embedding=query_embedding,
            top_k=top_k,
            min_score=min_score,
            class_filter=class_filter,
        )

        return {'status': 'success', 'query_vehicle': query_info, 'results': results}

    async def search_people(
        self,
        image_bytes: bytes,
        box_index: int = 0,
        top_k: int = 10,
        min_score: float | None = None,
    ) -> dict[str, Any]:
        """
        Find similar people by appearance (clothing, pose).

        Like "Find people wearing similar outfits" - NOT identity matching.
        For identity matching, use search_faces (future - requires ArcFace).

        Pipeline:
        1. Run Track E to get person detection embedding
        2. k-NN search on visual_search_people index

        Args:
            image_bytes: Query image with person
            box_index: Which detected person to use (0 = first)
            top_k: Number of results
            min_score: Minimum similarity

        Returns:
            dict with query person info and matching people
        """
        from src.clients.triton_client import get_triton_client
        from src.config import get_settings

        settings = get_settings()
        client = get_triton_client(settings.triton_url)
        result = client.infer_track_e(image_bytes, full_pipeline=True)

        if result['num_dets'] == 0:
            return {'status': 'error', 'error': 'No objects detected', 'results': []}

        # Find person detections
        classes = result.get('classes', [])
        person_indices = [
            i for i, c in enumerate(classes) if get_category(int(c)) == DetectionCategory.PERSON
        ]

        if not person_indices:
            return {'status': 'error', 'error': 'No people detected', 'results': []}

        if box_index >= len(person_indices):
            return {
                'status': 'error',
                'error': f'Person index {box_index} out of range (0-{len(person_indices) - 1})',
                'results': [],
            }

        actual_idx = person_indices[box_index]
        box_embeddings = np.array(result['box_embeddings'])
        query_embedding = box_embeddings[actual_idx]

        boxes = result.get('normalized_boxes', [])
        scores = result.get('scores', [])

        query_info = {
            'box': boxes[actual_idx].tolist() if len(boxes) > actual_idx else None,
            'class_id': int(classes[actual_idx]),
            'score': float(scores[actual_idx]) if len(scores) > actual_idx else None,
        }

        results = await self.opensearch.search_people(
            query_embedding=query_embedding,
            top_k=top_k,
            min_score=min_score,
        )

        return {'status': 'success', 'query_person': query_info, 'results': results}

    async def search_by_object(
        self,
        image_bytes: bytes,
        box_index: int = 0,
        top_k: int = 10,
        min_score: float | None = None,
        class_filter: list[int] | None = None,
    ) -> dict[str, Any]:
        """
        Object-to-object search with auto-routing to category index.

        Automatically routes to vehicles or people index based on detection class.

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
        result = client.infer_track_e(image_bytes, full_pipeline=True)

        if result['num_dets'] == 0:
            return {'status': 'error', 'error': 'No objects detected', 'results': []}

        if box_index >= result['num_dets']:
            return {
                'status': 'error',
                'error': f'Box index {box_index} out of range (0-{result["num_dets"] - 1})',
                'results': [],
            }

        box_embeddings = np.array(result['box_embeddings'])
        query_embedding = box_embeddings[box_index]
        classes = result.get('classes', [])
        boxes = result.get('normalized_boxes', [])
        scores = result.get('scores', [])

        class_id = int(classes[box_index])
        category = get_category(class_id)

        query_info = {
            'box': boxes[box_index].tolist() if len(boxes) > box_index else None,
            'class_id': class_id,
            'category': category.value,
            'score': float(scores[box_index]) if len(scores) > box_index else None,
        }

        # Route to appropriate index
        if category == DetectionCategory.VEHICLE:
            results = await self.opensearch.search_vehicles(
                query_embedding=query_embedding,
                top_k=top_k,
                min_score=min_score,
                class_filter=class_filter,
            )
        elif category == DetectionCategory.PERSON:
            results = await self.opensearch.search_people(
                query_embedding=query_embedding,
                top_k=top_k,
                min_score=min_score,
            )
        else:
            # For other classes, search global (fallback)
            results = await self.opensearch.search_global(
                query_embedding=query_embedding,
                top_k=top_k,
                min_score=min_score,
            )

        return {'status': 'success', 'query_object': query_info, 'results': results}

    # =========================================================================
    # Index Management (All Category Indexes)
    # =========================================================================

    async def setup_index(self, force_recreate: bool = False) -> dict[str, bool]:
        """
        Create all visual search indexes (global, vehicles, people, faces).

        Args:
            force_recreate: Delete existing indexes if present

        Returns:
            dict mapping index name to creation success
        """
        return await self.opensearch.create_all_indexes(force_recreate=force_recreate)

    async def delete_index(self) -> dict[str, bool]:
        """Delete all visual search indexes."""
        return await self.opensearch.delete_all_indexes()

    async def get_index_stats(self) -> dict[str, Any]:
        """
        Get statistics for all indexes.

        Returns:
            dict with stats per index (documents, size)
        """
        stats = await self.opensearch.get_all_index_stats()
        return {
            'status': 'success',
            'indexes': stats,
        }

    async def ping_opensearch(self) -> bool:
        """Check if OpenSearch is reachable."""
        return await self.opensearch.ping()

    # =========================================================================
    # Clustering Operations (FAISS IVF - Industry Standard)
    # =========================================================================

    def _get_cluster_index(self, index_name: str):
        """Convert string index name to ClusterIndex enum."""
        from src.services.clustering import ClusterIndex

        name_map = {
            'global': ClusterIndex.GLOBAL,
            'vehicles': ClusterIndex.VEHICLES,
            'people': ClusterIndex.PEOPLE,
            'faces': ClusterIndex.FACES,
        }
        if index_name.lower() not in name_map:
            raise ValueError(
                f'Invalid index name: {index_name}. Must be one of: {list(name_map.keys())}'
            )
        return name_map[index_name.lower()]

    def _get_opensearch_index(self, index_name: str):
        """Convert string index name to OpenSearch IndexName enum."""
        from src.clients.opensearch import IndexName

        name_map = {
            'global': IndexName.GLOBAL,
            'vehicles': IndexName.VEHICLES,
            'people': IndexName.PEOPLE,
            'faces': IndexName.FACES,
        }
        if index_name.lower() not in name_map:
            raise ValueError(
                f'Invalid index name: {index_name}. Must be one of: {list(name_map.keys())}'
            )
        return name_map[index_name.lower()]

    async def train_clusters(
        self,
        index_name: str,
        n_clusters: int | None = None,
        max_samples: int | None = None,
    ) -> dict:
        """
        Train FAISS IVF clustering for an index.

        Extracts embeddings from OpenSearch and trains FAISS IVF index.
        This is typically run once initially, then periodically for rebalancing.

        Args:
            index_name: Which index to train (global, vehicles, people, faces)
            n_clusters: Number of clusters (uses default if None)
            max_samples: Maximum samples for training (None = all)

        Returns:
            Training stats including n_clusters, n_vectors, timing
        """
        import time

        from src.services.clustering import get_clustering_service

        cluster_index = self._get_cluster_index(index_name)
        opensearch_index = self._get_opensearch_index(index_name)

        logger.info(f'Training clusters for {index_name}...')
        start_time = time.time()

        # Extract embeddings from OpenSearch
        embeddings, doc_ids = await self.opensearch.get_all_embeddings(
            index_name=opensearch_index,
            max_docs=max_samples,
        )

        if len(embeddings) == 0:
            return {
                'status': 'error',
                'error': f'No embeddings found in {index_name} index',
            }

        # Train FAISS index
        clustering_service = get_clustering_service()
        stats = await clustering_service.train_index(
            index_name=cluster_index,
            embeddings=embeddings,
            n_clusters=n_clusters,
        )

        # Assign clusters to all documents
        assignments = clustering_service.assign_clusters_batch(cluster_index, embeddings)
        cluster_ids = [a.cluster_id for a in assignments]
        cluster_distances = [a.distance for a in assignments]

        # Update OpenSearch with cluster assignments
        updated = await self.opensearch.update_cluster_assignments(
            index_name=opensearch_index,
            doc_ids=doc_ids,
            cluster_ids=cluster_ids,
            cluster_distances=cluster_distances,
        )

        training_time = time.time() - start_time

        return {
            'status': 'success',
            'index_name': index_name,
            'n_vectors': stats.n_vectors,
            'n_clusters': stats.n_clusters,
            'avg_cluster_size': stats.avg_cluster_size,
            'empty_clusters': stats.empty_clusters,
            'documents_updated': updated,
            'training_time_s': round(training_time, 2),
        }

    async def assign_unclustered(self, index_name: str) -> dict:
        """
        Assign clusters to unclustered documents.

        Finds documents without cluster_id and assigns them to nearest centroid.

        Args:
            index_name: Which index to process

        Returns:
            Assignment stats
        """
        from src.services.clustering import get_clustering_service

        cluster_index = self._get_cluster_index(index_name)
        opensearch_index = self._get_opensearch_index(index_name)

        clustering_service = get_clustering_service()

        if not clustering_service.is_trained(cluster_index):
            return {
                'status': 'error',
                'error': f'Clusters not trained for {index_name}. Run train_clusters first.',
            }

        # Get unclustered embeddings
        embeddings, doc_ids = await self.opensearch.get_unclustered_embeddings(
            index_name=opensearch_index,
        )

        if len(embeddings) == 0:
            return {
                'status': 'success',
                'index_name': index_name,
                'documents_updated': 0,
                'message': 'No unclustered documents found',
            }

        # Assign to clusters
        assignments = clustering_service.assign_clusters_batch(cluster_index, embeddings)
        cluster_ids = [a.cluster_id for a in assignments]
        cluster_distances = [a.distance for a in assignments]

        # Update OpenSearch
        updated = await self.opensearch.update_cluster_assignments(
            index_name=opensearch_index,
            doc_ids=doc_ids,
            cluster_ids=cluster_ids,
            cluster_distances=cluster_distances,
        )

        return {
            'status': 'success',
            'index_name': index_name,
            'documents_found': len(embeddings),
            'documents_updated': updated,
        }

    async def get_cluster_stats(self, index_name: str) -> dict:
        """
        Get detailed cluster statistics.

        Returns FAISS stats and OpenSearch aggregations.

        Args:
            index_name: Which index

        Returns:
            Cluster statistics
        """
        from src.services.clustering import get_clustering_service

        cluster_index = self._get_cluster_index(index_name)
        opensearch_index = self._get_opensearch_index(index_name)

        clustering_service = get_clustering_service()

        # Get FAISS stats
        faiss_stats = clustering_service.get_stats(cluster_index)

        # Get OpenSearch cluster aggregations
        os_clusters = await self.opensearch.get_cluster_stats(opensearch_index)

        return {
            'status': 'success',
            'index_name': index_name,
            'faiss': {
                'is_trained': faiss_stats.is_trained,
                'n_clusters': faiss_stats.n_clusters,
                'n_vectors': faiss_stats.n_vectors,
                'avg_cluster_size': faiss_stats.avg_cluster_size,
                'min_cluster_size': faiss_stats.min_cluster_size,
                'max_cluster_size': faiss_stats.max_cluster_size,
                'empty_clusters': faiss_stats.empty_clusters,
                'trained_at': faiss_stats.trained_at,
            },
            'opensearch_clusters': os_clusters[:20],  # Top 20 clusters
            'total_clusters_in_opensearch': len(os_clusters),
        }

    async def check_cluster_balance(self, index_name: str) -> dict:
        """
        Check if clusters need rebalancing.

        Args:
            index_name: Which index to check

        Returns:
            Balance assessment with recommendation
        """
        from src.services.clustering import get_clustering_service

        cluster_index = self._get_cluster_index(index_name)

        clustering_service = get_clustering_service()

        if not clustering_service.is_trained(cluster_index):
            return {
                'status': 'error',
                'error': f'Clusters not trained for {index_name}',
            }

        balance = await clustering_service.check_balance(cluster_index)

        return {
            'status': 'success',
            'index_name': balance.index_name,
            'is_balanced': balance.is_balanced,
            'imbalance_ratio': round(balance.imbalance_ratio, 2),
            'empty_ratio': round(balance.empty_ratio, 4),
            'vectors_since_training': balance.vectors_since_training,
            'needs_rebalance': balance.needs_rebalance,
            'reason': balance.reason,
        }

    async def rebalance_clusters(self, index_name: str) -> dict:
        """
        Force rebalance clusters by re-training from current data.

        Args:
            index_name: Which index to rebalance

        Returns:
            Rebalancing stats
        """
        # Just call train_clusters which does a full retrain
        return await self.train_clusters(index_name=index_name)

    async def get_cluster_members(
        self,
        index_name: str,
        cluster_id: int,
        page: int = 0,
        size: int = 50,
    ) -> dict:
        """
        Get members of a specific cluster (album view).

        Args:
            index_name: Which index
            cluster_id: Cluster ID to retrieve
            page: Page number
            size: Page size

        Returns:
            Cluster members sorted by distance to centroid
        """
        opensearch_index = self._get_opensearch_index(index_name)

        members = await self.opensearch.get_cluster_members(
            index_name=opensearch_index,
            cluster_id=cluster_id,
            page=page,
            size=size,
        )

        return {
            'status': 'success',
            'index_name': index_name,
            'cluster_id': cluster_id,
            'page': page,
            'size': size,
            'count': len(members),
            'members': members,
        }

    async def list_albums(self, min_size: int = 5) -> dict:
        """
        List auto-generated albums (clusters) from global index.

        Args:
            min_size: Minimum cluster size to include

        Returns:
            List of albums with metadata
        """
        from src.clients.opensearch import IndexName

        # Get cluster stats from global index
        clusters = await self.opensearch.get_cluster_stats(IndexName.GLOBAL)

        # Filter by minimum size
        albums = [c for c in clusters if c['count'] >= min_size]

        # Sort by size descending
        albums.sort(key=lambda x: x['count'], reverse=True)

        return {
            'status': 'success',
            'total_albums': len(albums),
            'albums': albums[:100],  # Top 100 albums
        }
