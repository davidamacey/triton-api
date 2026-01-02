"""
Track E: OpenSearch Client for Visual Search

Multi-Index Architecture with FAISS IVF Clustering:

Indexes:
1. visual_search_global - Whole image similarity (scene matching)
2. visual_search_vehicles - Vehicle detections (car, truck, motorcycle, bus, boat)
3. visual_search_people - Person appearance (clothing, pose)
4. visual_search_faces - Face identity matching (future: ArcFace embeddings)

Key Features:
- Category-specific indexes for faster, more accurate search
- Automatic class-to-category routing during ingestion
- FAISS IVF clustering for industry-standard similarity grouping
- Cluster-filtered search for optimized queries
- Independent HNSW tuning per category

Usage:
    # Initialize client with clustering
    from src.services.clustering import get_clustering_service

    client = OpenSearchClient(hosts=["http://localhost:9200"])
    clustering = get_clustering_service()

    # Create all indexes
    await client.create_all_indexes()

    # Ingest image with auto-routing and cluster assignment
    await client.ingest_image(
        image_id="img_001",
        image_path="/path/to/image.jpg",
        global_embedding=np.array([...]),  # 512-dim
        box_embeddings=np.array([[...]]),  # [N, 512]
        normalized_boxes=np.array([[...]]),  # [N, 4]
        det_classes=[0, 2, 7],  # person, car, truck
        det_scores=[0.95, 0.87, 0.76],
        clustering_service=clustering,  # Optional: enables cluster assignment
    )
    # Routes: person -> visual_search_people, car/truck -> visual_search_vehicles
    # Each document gets cluster_id and cluster_distance fields

    # Search with cluster optimization
    results = await client.search_vehicles(
        query_embedding,
        top_k=10,
        cluster_ids=[5, 12, 23],  # Optional: narrow search to specific clusters
    )
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
from opensearchpy import AsyncOpenSearch
from opensearchpy.helpers import async_bulk


if TYPE_CHECKING:
    from src.services.clustering import ClusterIndex, ClusteringService


logger = logging.getLogger(__name__)


# =============================================================================
# Index Constants and Category Mappings
# =============================================================================


class IndexName(str, Enum):
    """OpenSearch index names for visual search."""

    GLOBAL = 'visual_search_global'
    VEHICLES = 'visual_search_vehicles'
    PEOPLE = 'visual_search_people'
    FACES = 'visual_search_faces'


class DetectionCategory(str, Enum):
    """Detection categories for routing."""

    VEHICLE = 'vehicle'
    PERSON = 'person'
    FACE = 'face'
    OTHER = 'other'


# COCO class ID to category mapping
VEHICLE_CLASSES = {2, 3, 5, 7, 8}  # car, motorcycle, bus, truck, boat
PERSON_CLASS = 0

# Class ID to human-readable name (COCO subset)
CLASS_NAMES = {
    0: 'person',
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck',
    8: 'boat',
}


def get_category(class_id: int) -> DetectionCategory:
    """Map COCO class ID to detection category."""
    if class_id == PERSON_CLASS:
        return DetectionCategory.PERSON
    if class_id in VEHICLE_CLASSES:
        return DetectionCategory.VEHICLE
    return DetectionCategory.OTHER


def get_class_name(class_id: int) -> str:
    """Get human-readable class name from COCO class ID."""
    return CLASS_NAMES.get(class_id, f'class_{class_id}')


def get_cluster_index_name(index_name: IndexName) -> ClusterIndex:
    """Map OpenSearch IndexName to FAISS ClusterIndex."""
    # Import here to avoid circular imports
    from src.services.clustering import ClusterIndex

    mapping = {
        IndexName.GLOBAL: ClusterIndex.GLOBAL,
        IndexName.VEHICLES: ClusterIndex.VEHICLES,
        IndexName.PEOPLE: ClusterIndex.PEOPLE,
        IndexName.FACES: ClusterIndex.FACES,
    }
    return mapping[index_name]


class OpenSearchClient:
    """
    Async OpenSearch client for multi-index visual search with MobileCLIP embeddings.

    Supports four category-specific indexes:
    - visual_search_global: Whole image embeddings
    - visual_search_vehicles: Vehicle detections (car, truck, motorcycle, bus, boat)
    - visual_search_people: Person detections
    - visual_search_faces: Face identity embeddings (future: ArcFace)
    """

    def __init__(
        self,
        hosts: list[str] | None = None,
        http_auth: tuple | None = None,
        verify_certs: bool = False,
        ssl_show_warn: bool = False,
        timeout: int = 30,
    ):
        """
        Initialize OpenSearch async client.

        Args:
            hosts: List of OpenSearch node URLs
            http_auth: Tuple of (username, password) for authentication
            verify_certs: Whether to verify SSL certificates
            ssl_show_warn: Whether to show SSL warnings
            timeout: Request timeout in seconds
        """
        hosts = hosts or ['http://localhost:9200']
        client_kwargs = {
            'hosts': hosts,
            'use_ssl': False,
            'verify_certs': verify_certs,
            'ssl_show_warn': ssl_show_warn,
            'timeout': timeout,
        }

        if http_auth:
            client_kwargs['http_auth'] = http_auth

        self.client = AsyncOpenSearch(**client_kwargs)
        self.embedding_dim = 512  # MobileCLIP2-S2

    # =========================================================================
    # HNSW Configuration by Index Type
    # =========================================================================

    def _get_hnsw_config(self, index_type: IndexName) -> dict:
        """
        Get optimized HNSW parameters for each index type.

        Different use cases require different quality/speed tradeoffs:
        - Global: Balanced (general scene matching)
        - Vehicles: Smaller index, faster search
        - People: Common queries, higher quality
        - Faces: Identity matching requires highest precision
        """
        configs = {
            IndexName.GLOBAL: {
                'ef_construction': 512,
                'm': 16,
                'ef_search': 256,
            },
            IndexName.VEHICLES: {
                'ef_construction': 256,
                'm': 12,
                'ef_search': 128,
            },
            IndexName.PEOPLE: {
                'ef_construction': 512,
                'm': 16,
                'ef_search': 256,
            },
            IndexName.FACES: {
                'ef_construction': 1024,
                'm': 32,
                'ef_search': 512,
            },
        }
        return configs.get(index_type, configs[IndexName.GLOBAL])

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

    # =========================================================================
    # Multi-Index Creation Methods
    # =========================================================================

    async def create_all_indexes(self, force_recreate: bool = False) -> dict[str, bool]:
        """
        Create all category-specific indexes.

        Args:
            force_recreate: Delete existing indexes if present

        Returns:
            Dict mapping index name to creation success
        """
        results = {}
        results[IndexName.GLOBAL.value] = await self.create_global_index(force_recreate)
        results[IndexName.VEHICLES.value] = await self.create_vehicles_index(force_recreate)
        results[IndexName.PEOPLE.value] = await self.create_people_index(force_recreate)
        results[IndexName.FACES.value] = await self.create_faces_index(force_recreate)
        return results

    async def create_global_index(self, force_recreate: bool = False) -> bool:
        """
        Create visual_search_global index for whole image embeddings.

        Schema:
        - image_id (keyword): Unique identifier
        - image_path (keyword): File path or URL
        - global_embedding (knn_vector): 512-dim MobileCLIP embedding
        - cluster_id (integer): FAISS IVF cluster assignment
        - cluster_distance (float): Distance to cluster centroid
        - width, height (integer): Image dimensions
        - metadata (object): Flexible metadata
        - indexed_at (date): Ingestion timestamp
        - clustered_at (date): Cluster assignment timestamp
        """
        index_name = IndexName.GLOBAL.value
        hnsw = self._get_hnsw_config(IndexName.GLOBAL)

        index_body = {
            'settings': {
                'index': {
                    'number_of_shards': 1,
                    'number_of_replicas': 0,
                    'knn': True,
                },
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
                            'space_type': 'cosinesimil',
                            'engine': 'faiss',
                            'parameters': {
                                'ef_construction': hnsw['ef_construction'],
                                'm': hnsw['m'],
                            },
                        },
                    },
                    'cluster_id': {'type': 'integer'},
                    'cluster_distance': {'type': 'float'},
                    'width': {'type': 'integer'},
                    'height': {'type': 'integer'},
                    'metadata': {'type': 'object', 'enabled': True},
                    'indexed_at': {'type': 'date'},
                    'clustered_at': {'type': 'date'},
                }
            },
        }

        return await self._create_index(index_name, index_body, force_recreate)

    async def create_vehicles_index(self, force_recreate: bool = False) -> bool:
        """
        Create visual_search_vehicles index for vehicle detections.

        Classes: car(2), motorcycle(3), bus(5), truck(7), boat(8)

        Schema:
        - detection_id (keyword): Unique ID (image_id + box index)
        - image_id (keyword): Source image ID
        - image_path (keyword): File path or URL
        - embedding (knn_vector): 512-dim MobileCLIP embedding of cropped vehicle
        - cluster_id (integer): FAISS IVF cluster assignment
        - cluster_distance (float): Distance to cluster centroid
        - box (float[]): [x1, y1, x2, y2] normalized coordinates
        - class_id (integer): COCO class ID
        - class_name (keyword): Human-readable class name
        - confidence (float): Detection confidence
        - metadata (object): Flexible metadata
        - indexed_at (date): Ingestion timestamp
        - clustered_at (date): Cluster assignment timestamp
        """
        index_name = IndexName.VEHICLES.value
        hnsw = self._get_hnsw_config(IndexName.VEHICLES)

        index_body = {
            'settings': {
                'index': {
                    'number_of_shards': 1,
                    'number_of_replicas': 0,
                    'knn': True,
                },
            },
            'mappings': {
                'properties': {
                    'detection_id': {'type': 'keyword'},
                    'image_id': {'type': 'keyword'},
                    'image_path': {'type': 'keyword'},
                    'embedding': {
                        'type': 'knn_vector',
                        'dimension': self.embedding_dim,
                        'method': {
                            'name': 'hnsw',
                            'space_type': 'cosinesimil',
                            'engine': 'faiss',
                            'parameters': {
                                'ef_construction': hnsw['ef_construction'],
                                'm': hnsw['m'],
                            },
                        },
                    },
                    'cluster_id': {'type': 'integer'},
                    'cluster_distance': {'type': 'float'},
                    'box': {'type': 'float'},
                    'class_id': {'type': 'integer'},
                    'class_name': {'type': 'keyword'},
                    'confidence': {'type': 'float'},
                    'metadata': {'type': 'object', 'enabled': True},
                    'indexed_at': {'type': 'date'},
                    'clustered_at': {'type': 'date'},
                }
            },
        }

        return await self._create_index(index_name, index_body, force_recreate)

    async def create_people_index(self, force_recreate: bool = False) -> bool:
        """
        Create visual_search_people index for person detections.

        Schema:
        - detection_id (keyword): Unique ID (image_id + box index)
        - image_id (keyword): Source image ID
        - image_path (keyword): File path or URL
        - embedding (knn_vector): 512-dim MobileCLIP embedding (appearance)
        - cluster_id (integer): FAISS IVF cluster assignment
        - cluster_distance (float): Distance to cluster centroid
        - box (float[]): [x1, y1, x2, y2] normalized coordinates
        - confidence (float): Detection confidence
        - has_face (boolean): Whether face was detected in this person
        - face_id (keyword): Link to face in visual_search_faces (optional)
        - metadata (object): Flexible metadata
        - indexed_at (date): Ingestion timestamp
        - clustered_at (date): Cluster assignment timestamp
        """
        index_name = IndexName.PEOPLE.value
        hnsw = self._get_hnsw_config(IndexName.PEOPLE)

        index_body = {
            'settings': {
                'index': {
                    'number_of_shards': 1,
                    'number_of_replicas': 0,
                    'knn': True,
                },
            },
            'mappings': {
                'properties': {
                    'detection_id': {'type': 'keyword'},
                    'image_id': {'type': 'keyword'},
                    'image_path': {'type': 'keyword'},
                    'embedding': {
                        'type': 'knn_vector',
                        'dimension': self.embedding_dim,
                        'method': {
                            'name': 'hnsw',
                            'space_type': 'cosinesimil',
                            'engine': 'faiss',
                            'parameters': {
                                'ef_construction': hnsw['ef_construction'],
                                'm': hnsw['m'],
                            },
                        },
                    },
                    'cluster_id': {'type': 'integer'},
                    'cluster_distance': {'type': 'float'},
                    'box': {'type': 'float'},
                    'confidence': {'type': 'float'},
                    'has_face': {'type': 'boolean'},
                    'face_id': {'type': 'keyword'},
                    'metadata': {'type': 'object', 'enabled': True},
                    'indexed_at': {'type': 'date'},
                    'clustered_at': {'type': 'date'},
                }
            },
        }

        return await self._create_index(index_name, index_body, force_recreate)

    async def create_faces_index(self, force_recreate: bool = False) -> bool:
        """
        Create visual_search_faces index for face identity matching.

        Future: Will use ArcFace embeddings (512-dim, identity-based)
        Currently: Placeholder for face detection integration

        Schema:
        - face_id (keyword): Unique face ID
        - image_id (keyword): Source image ID
        - image_path (keyword): File path or URL
        - person_detection_id (keyword): Link to person in visual_search_people
        - embedding (knn_vector): 512-dim face embedding (ArcFace)
        - cluster_id (integer): FAISS IVF cluster assignment
        - cluster_distance (float): Distance to cluster centroid
        - box (float[]): [x1, y1, x2, y2] face bounding box
        - landmarks (object): Facial landmarks (5-point)
        - confidence (float): Face detection confidence
        - quality_score (float): Face quality assessment
        - person_id (keyword): Identity cluster ID (same person)
        - person_name (keyword): Optional name label
        - is_reference (boolean): Reference face for person
        - metadata (object): Flexible metadata
        - indexed_at (date): Ingestion timestamp
        - clustered_at (date): Cluster assignment timestamp
        """
        index_name = IndexName.FACES.value
        hnsw = self._get_hnsw_config(IndexName.FACES)

        index_body = {
            'settings': {
                'index': {
                    'number_of_shards': 1,
                    'number_of_replicas': 0,
                    'knn': True,
                },
            },
            'mappings': {
                'properties': {
                    'face_id': {'type': 'keyword'},
                    'image_id': {'type': 'keyword'},
                    'image_path': {'type': 'keyword'},
                    'person_detection_id': {'type': 'keyword'},
                    'embedding': {
                        'type': 'knn_vector',
                        'dimension': self.embedding_dim,
                        'method': {
                            'name': 'hnsw',
                            'space_type': 'cosinesimil',
                            'engine': 'faiss',
                            'parameters': {
                                'ef_construction': hnsw['ef_construction'],
                                'm': hnsw['m'],
                            },
                        },
                    },
                    'cluster_id': {'type': 'integer'},
                    'cluster_distance': {'type': 'float'},
                    'box': {'type': 'float'},
                    'landmarks': {
                        'type': 'object',
                        'properties': {
                            'left_eye': {'type': 'float'},
                            'right_eye': {'type': 'float'},
                            'nose': {'type': 'float'},
                            'left_mouth': {'type': 'float'},
                            'right_mouth': {'type': 'float'},
                        },
                    },
                    'confidence': {'type': 'float'},
                    'quality_score': {'type': 'float'},
                    'person_id': {'type': 'keyword'},
                    'person_name': {'type': 'keyword'},
                    'is_reference': {'type': 'boolean'},
                    'metadata': {'type': 'object', 'enabled': True},
                    'indexed_at': {'type': 'date'},
                    'clustered_at': {'type': 'date'},
                }
            },
        }

        return await self._create_index(index_name, index_body, force_recreate)

    async def _create_index(
        self, index_name: str, index_body: dict, force_recreate: bool = False
    ) -> bool:
        """
        Internal helper to create an index with error handling.
        """
        try:
            exists = await self.client.indices.exists(index=index_name)

            if exists:
                if force_recreate:
                    logger.info(f'Deleting existing index: {index_name}')
                    await self.client.indices.delete(index=index_name)
                else:
                    logger.info(f'Index already exists: {index_name}')
                    return True

            await self.client.indices.create(index=index_name, body=index_body)
            logger.info(f'Index created successfully: {index_name}')
            return True

        except Exception as e:
            logger.error(f'Failed to create index {index_name}: {e}')
            return False

    # =========================================================================
    # Multi-Index Ingestion (Category-Specific Routing)
    # =========================================================================

    async def ingest_image(
        self,
        image_id: str,
        image_path: str,
        global_embedding: np.ndarray,
        box_embeddings: np.ndarray | None = None,
        normalized_boxes: np.ndarray | None = None,
        det_classes: list[int] | None = None,
        det_scores: list[float] | None = None,
        image_width: int | None = None,
        image_height: int | None = None,
        metadata: dict[str, Any] | None = None,
        clustering_service: ClusteringService | None = None,
    ) -> dict[str, Any]:
        """
        Ingest image with auto-routing to category-specific indexes.

        Routes detections by class:
        - Person (class 0) -> visual_search_people
        - Vehicles (class 2,3,5,7,8) -> visual_search_vehicles
        - Global embedding -> visual_search_global

        When clustering_service is provided, each document gets:
        - cluster_id: FAISS IVF cluster assignment (~0.1ms overhead)
        - cluster_distance: Distance to cluster centroid

        Args:
            image_id: Unique identifier for the image
            image_path: File path or URL to the image
            global_embedding: 512-dim L2-normalized embedding for entire image
            box_embeddings: [N, 512] embeddings for detected objects
            normalized_boxes: [N, 4] boxes in [0, 1] range
            det_classes: [N] class IDs for detected objects
            det_scores: [N] detection confidence scores
            image_width: Image width in pixels
            image_height: Image height in pixels
            metadata: Optional dictionary for custom fields
            clustering_service: Optional ClusteringService for cluster assignment

        Returns:
            Dict with ingestion results per index
        """
        results = {
            'global': False,
            'vehicles': 0,
            'people': 0,
            'skipped': 0,
            'clustered': 0,
            'errors': [],
        }
        timestamp = datetime.now(UTC).isoformat()

        # 1. Ingest global embedding
        try:
            global_doc = {
                'image_id': image_id,
                'image_path': image_path,
                'global_embedding': global_embedding.tolist(),
                'indexed_at': timestamp,
            }

            # Assign to cluster if clustering is available
            if clustering_service is not None:
                try:
                    cluster_idx = get_cluster_index_name(IndexName.GLOBAL)
                    if clustering_service.is_trained(cluster_idx):
                        assignment = clustering_service.assign_cluster(
                            cluster_idx, global_embedding
                        )
                        global_doc['cluster_id'] = assignment.cluster_id
                        global_doc['cluster_distance'] = assignment.distance
                        global_doc['clustered_at'] = timestamp
                        results['clustered'] += 1
                except Exception as e:
                    logger.warning(f'Cluster assignment failed for global: {e}')

            if image_width:
                global_doc['width'] = image_width
            if image_height:
                global_doc['height'] = image_height
            if metadata:
                global_doc['metadata'] = metadata

            await self.client.index(
                index=IndexName.GLOBAL.value,
                id=image_id,
                body=global_doc,
                refresh=False,
            )
            results['global'] = True
        except Exception as e:
            results['errors'].append(f'Global: {e!s}')

        # 2. Route box embeddings to category-specific indexes
        if box_embeddings is not None and det_classes is not None:
            num_boxes = box_embeddings.shape[0]

            for i in range(num_boxes):
                class_id = int(det_classes[i])
                category = get_category(class_id)
                detection_id = f'{image_id}_box_{i}'

                try:
                    if category == DetectionCategory.VEHICLE:
                        # Index vehicle detection
                        vehicle_doc = {
                            'detection_id': detection_id,
                            'image_id': image_id,
                            'image_path': image_path,
                            'embedding': box_embeddings[i].tolist(),
                            'class_id': class_id,
                            'class_name': get_class_name(class_id),
                            'indexed_at': timestamp,
                        }

                        # Assign to vehicle cluster
                        if clustering_service is not None:
                            try:
                                cluster_idx = get_cluster_index_name(IndexName.VEHICLES)
                                if clustering_service.is_trained(cluster_idx):
                                    assignment = clustering_service.assign_cluster(
                                        cluster_idx, box_embeddings[i]
                                    )
                                    vehicle_doc['cluster_id'] = assignment.cluster_id
                                    vehicle_doc['cluster_distance'] = assignment.distance
                                    vehicle_doc['clustered_at'] = timestamp
                                    results['clustered'] += 1
                            except Exception as e:
                                logger.warning(f'Cluster assignment failed for vehicle: {e}')

                        if normalized_boxes is not None:
                            vehicle_doc['box'] = normalized_boxes[i].tolist()
                        if det_scores is not None:
                            vehicle_doc['confidence'] = float(det_scores[i])
                        if metadata:
                            vehicle_doc['metadata'] = metadata

                        await self.client.index(
                            index=IndexName.VEHICLES.value,
                            id=detection_id,
                            body=vehicle_doc,
                            refresh=False,
                        )
                        results['vehicles'] += 1

                    elif category == DetectionCategory.PERSON:
                        # Index person detection
                        person_doc = {
                            'detection_id': detection_id,
                            'image_id': image_id,
                            'image_path': image_path,
                            'embedding': box_embeddings[i].tolist(),
                            'has_face': False,  # Will be updated when face detection is added
                            'indexed_at': timestamp,
                        }

                        # Assign to people cluster
                        if clustering_service is not None:
                            try:
                                cluster_idx = get_cluster_index_name(IndexName.PEOPLE)
                                if clustering_service.is_trained(cluster_idx):
                                    assignment = clustering_service.assign_cluster(
                                        cluster_idx, box_embeddings[i]
                                    )
                                    person_doc['cluster_id'] = assignment.cluster_id
                                    person_doc['cluster_distance'] = assignment.distance
                                    person_doc['clustered_at'] = timestamp
                                    results['clustered'] += 1
                            except Exception as e:
                                logger.warning(f'Cluster assignment failed for person: {e}')

                        if normalized_boxes is not None:
                            person_doc['box'] = normalized_boxes[i].tolist()
                        if det_scores is not None:
                            person_doc['confidence'] = float(det_scores[i])
                        if metadata:
                            person_doc['metadata'] = metadata

                        await self.client.index(
                            index=IndexName.PEOPLE.value,
                            id=detection_id,
                            body=person_doc,
                            refresh=False,
                        )
                        results['people'] += 1

                    else:
                        # Skip non-vehicle, non-person classes
                        results['skipped'] += 1

                except Exception as e:
                    results['errors'].append(f'{detection_id}: {e!s}')

        logger.info(
            f'Multi-index ingestion: global={results["global"]}, '
            f'vehicles={results["vehicles"]}, people={results["people"]}, '
            f'skipped={results["skipped"]}, clustered={results["clustered"]}'
        )
        return results

    async def ingest_faces(
        self,
        image_id: str,
        image_path: str,
        faces: list[dict[str, Any]],
        embeddings: list[list[float]] | np.ndarray,
        person_name: str | None = None,
        metadata: dict[str, Any] | None = None,
        clustering_service: ClusteringService | None = None,
    ) -> dict[str, Any]:
        """
        Ingest face detections with ArcFace embeddings.

        Args:
            image_id: Unique identifier for the source image
            image_path: File path or URL to the image
            faces: List of face detections with box, landmarks, score
            embeddings: [N, 512] ArcFace embeddings
            person_name: Optional name/label for all faces in image
            metadata: Optional metadata dictionary
            clustering_service: Optional ClusteringService for cluster assignment

        Returns:
            Dict with face ingestion results
        """
        results = {
            'faces': 0,
            'clustered': 0,
            'errors': [],
        }
        timestamp = datetime.now(UTC).isoformat()

        if isinstance(embeddings, np.ndarray):
            embeddings = embeddings.tolist()

        for i, (face, embedding) in enumerate(zip(faces, embeddings, strict=False)):
            face_id = f'{image_id}_face_{i}'

            try:
                # Convert flat landmarks list to named object
                raw_landmarks = face.get('landmarks', [])
                if len(raw_landmarks) >= 10:
                    landmarks = {
                        'left_eye': [float(raw_landmarks[0]), float(raw_landmarks[1])],
                        'right_eye': [float(raw_landmarks[2]), float(raw_landmarks[3])],
                        'nose': [float(raw_landmarks[4]), float(raw_landmarks[5])],
                        'left_mouth': [float(raw_landmarks[6]), float(raw_landmarks[7])],
                        'right_mouth': [float(raw_landmarks[8]), float(raw_landmarks[9])],
                    }
                else:
                    landmarks = {}

                face_doc = {
                    'face_id': face_id,
                    'image_id': image_id,
                    'image_path': image_path,
                    'embedding': embedding,
                    'box': face.get('box', [0, 0, 1, 1]),
                    'landmarks': landmarks,
                    'confidence': face.get('score', 0.0),
                    'quality': face.get('quality', 0.0),
                    'indexed_at': timestamp,
                }

                if person_name:
                    face_doc['person_name'] = person_name

                # Assign to face cluster
                if clustering_service is not None:
                    try:
                        cluster_idx = get_cluster_index_name(IndexName.FACES)
                        if clustering_service.is_trained(cluster_idx):
                            emb_array = np.array(embedding, dtype=np.float32)
                            assignment = clustering_service.assign_cluster(cluster_idx, emb_array)
                            face_doc['cluster_id'] = assignment.cluster_id
                            face_doc['cluster_distance'] = assignment.distance
                            face_doc['clustered_at'] = timestamp
                            results['clustered'] += 1
                    except Exception as e:
                        logger.warning(f'Cluster assignment failed for face: {e}')

                if metadata:
                    face_doc['metadata'] = metadata

                await self.client.index(
                    index=IndexName.FACES.value,
                    id=face_id,
                    body=face_doc,
                    refresh=False,
                )
                results['faces'] += 1

            except Exception as e:
                results['errors'].append(f'{face_id}: {e!s}')

        logger.info(f'Face ingestion: faces={results["faces"]}, clustered={results["clustered"]}')
        return results

    async def bulk_ingest(
        self,
        documents: list[dict[str, Any]],
        refresh: bool = False,
    ) -> dict[str, Any]:
        """
        Bulk ingest multiple images with auto-routing to category-specific indexes.

        Args:
            documents: List of document dicts with:
                - image_id, image_path, global_embedding
                - Optional: box_embeddings, normalized_boxes, det_classes, det_scores
            refresh: Whether to refresh indexes after bulk operation

        Returns:
            Dict with counts per index
        """
        global_actions = []
        vehicle_actions = []
        people_actions = []
        timestamp = datetime.now(UTC).isoformat()

        for doc in documents:
            image_id = doc['image_id']
            image_path = doc['image_path']
            global_embedding = doc['global_embedding']
            metadata = doc.get('metadata')

            # Global document
            global_doc = {
                'image_id': image_id,
                'image_path': image_path,
                'global_embedding': (
                    global_embedding.tolist()
                    if hasattr(global_embedding, 'tolist')
                    else global_embedding
                ),
                'indexed_at': timestamp,
            }
            if 'width' in doc:
                global_doc['width'] = doc['width']
            if 'height' in doc:
                global_doc['height'] = doc['height']
            if metadata:
                global_doc['metadata'] = metadata

            global_actions.append(
                {
                    '_index': IndexName.GLOBAL.value,
                    '_id': image_id,
                    '_source': global_doc,
                }
            )

            # Route box embeddings
            if 'box_embeddings' in doc and 'det_classes' in doc:
                box_embeddings = doc['box_embeddings']
                det_classes = doc['det_classes']
                normalized_boxes = doc.get('normalized_boxes')
                det_scores = doc.get('det_scores')

                num_boxes = len(box_embeddings)
                for i in range(num_boxes):
                    class_id = int(det_classes[i])
                    category = get_category(class_id)
                    detection_id = f'{image_id}_box_{i}'

                    embedding = (
                        box_embeddings[i].tolist()
                        if hasattr(box_embeddings[i], 'tolist')
                        else box_embeddings[i]
                    )

                    if category == DetectionCategory.VEHICLE:
                        vehicle_doc = {
                            'detection_id': detection_id,
                            'image_id': image_id,
                            'image_path': image_path,
                            'embedding': embedding,
                            'class_id': class_id,
                            'class_name': get_class_name(class_id),
                            'indexed_at': timestamp,
                        }
                        if normalized_boxes is not None:
                            vehicle_doc['box'] = (
                                normalized_boxes[i].tolist()
                                if hasattr(normalized_boxes[i], 'tolist')
                                else normalized_boxes[i]
                            )
                        if det_scores is not None:
                            vehicle_doc['confidence'] = float(det_scores[i])
                        if metadata:
                            vehicle_doc['metadata'] = metadata

                        vehicle_actions.append(
                            {
                                '_index': IndexName.VEHICLES.value,
                                '_id': detection_id,
                                '_source': vehicle_doc,
                            }
                        )

                    elif category == DetectionCategory.PERSON:
                        person_doc = {
                            'detection_id': detection_id,
                            'image_id': image_id,
                            'image_path': image_path,
                            'embedding': embedding,
                            'has_face': False,
                            'indexed_at': timestamp,
                        }
                        if normalized_boxes is not None:
                            person_doc['box'] = (
                                normalized_boxes[i].tolist()
                                if hasattr(normalized_boxes[i], 'tolist')
                                else normalized_boxes[i]
                            )
                        if det_scores is not None:
                            person_doc['confidence'] = float(det_scores[i])
                        if metadata:
                            person_doc['metadata'] = metadata

                        people_actions.append(
                            {
                                '_index': IndexName.PEOPLE.value,
                                '_id': detection_id,
                                '_source': person_doc,
                            }
                        )

        # Execute bulk operations
        results = {'global': 0, 'vehicles': 0, 'people': 0, 'errors': []}

        try:
            if global_actions:
                success, _ = await async_bulk(
                    self.client, global_actions, refresh=refresh, raise_on_error=False
                )
                results['global'] = success

            if vehicle_actions:
                success, _ = await async_bulk(
                    self.client, vehicle_actions, refresh=refresh, raise_on_error=False
                )
                results['vehicles'] = success

            if people_actions:
                success, _ = await async_bulk(
                    self.client, people_actions, refresh=refresh, raise_on_error=False
                )
                results['people'] = success

        except Exception as e:
            results['errors'].append(str(e))

        logger.info(
            f'Bulk multi-index ingestion: global={results["global"]}, '
            f'vehicles={results["vehicles"]}, people={results["people"]}'
        )
        return results

    # =========================================================================
    # Category-Specific Search Methods
    # =========================================================================

    async def search_global(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        min_score: float | None = None,
        cluster_ids: list[int] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search for similar images by whole-image embedding.

        Args:
            query_embedding: 512-dim L2-normalized query embedding
            top_k: Number of results to return
            min_score: Minimum similarity score threshold
            cluster_ids: Optional list of cluster IDs to narrow search (from ClusteringService)

        Returns:
            List of image results with scores
        """
        return await self._search_index(
            index_name=IndexName.GLOBAL.value,
            embedding_field='global_embedding',
            query_embedding=query_embedding,
            top_k=top_k,
            min_score=min_score,
            cluster_ids=cluster_ids,
        )

    async def search_vehicles(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        min_score: float | None = None,
        class_filter: list[int] | None = None,
        cluster_ids: list[int] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search for similar vehicles by embedding.

        Args:
            query_embedding: 512-dim L2-normalized query embedding
            top_k: Number of results to return
            min_score: Minimum similarity score threshold
            class_filter: Filter by specific vehicle classes (e.g., [2] for cars only)
            cluster_ids: Optional list of cluster IDs to narrow search

        Returns:
            List of vehicle detection results with scores
        """
        return await self._search_index(
            index_name=IndexName.VEHICLES.value,
            embedding_field='embedding',
            query_embedding=query_embedding,
            top_k=top_k,
            min_score=min_score,
            class_filter=class_filter,
            cluster_ids=cluster_ids,
        )

    async def search_people(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        min_score: float | None = None,
        cluster_ids: list[int] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search for similar people by appearance embedding.

        Args:
            query_embedding: 512-dim L2-normalized query embedding
            top_k: Number of results to return
            min_score: Minimum similarity score threshold
            cluster_ids: Optional list of cluster IDs to narrow search

        Returns:
            List of person detection results with scores
        """
        return await self._search_index(
            index_name=IndexName.PEOPLE.value,
            embedding_field='embedding',
            query_embedding=query_embedding,
            top_k=top_k,
            min_score=min_score,
            cluster_ids=cluster_ids,
        )

    async def search_faces(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        min_score: float | None = 0.7,  # Higher threshold for identity matching
        cluster_ids: list[int] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search for same person by face embedding (identity matching).

        Args:
            query_embedding: 512-dim ArcFace embedding
            top_k: Number of results to return
            min_score: Minimum similarity score (default 0.7 for identity)
            cluster_ids: Optional list of cluster IDs to narrow search

        Returns:
            List of face results with identity info
        """
        return await self._search_index(
            index_name=IndexName.FACES.value,
            embedding_field='embedding',
            query_embedding=query_embedding,
            top_k=top_k,
            min_score=min_score,
            cluster_ids=cluster_ids,
        )

    async def _search_index(
        self,
        index_name: str,
        embedding_field: str,
        query_embedding: np.ndarray,
        top_k: int = 10,
        min_score: float | None = None,
        class_filter: list[int] | None = None,
        cluster_ids: list[int] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Internal helper for k-NN search on any index.

        Args:
            index_name: OpenSearch index name
            embedding_field: Field name containing the embedding vector
            query_embedding: Query embedding vector
            top_k: Number of results to return
            min_score: Minimum similarity score threshold
            class_filter: Filter by class IDs (for vehicle searches)
            cluster_ids: Filter by cluster IDs (for cluster-optimized search)

        When cluster_ids is provided, OpenSearch only searches documents
        in those clusters, typically providing 10-100x speedup for large indexes.
        """
        try:
            query = {
                'size': top_k,
                'query': {
                    'knn': {
                        embedding_field: {
                            'vector': query_embedding.tolist(),
                            'k': top_k,
                        }
                    }
                },
            }

            # Build filter conditions
            filters = []
            if class_filter:
                filters.append({'terms': {'class_id': class_filter}})
            if cluster_ids:
                filters.append({'terms': {'cluster_id': cluster_ids}})

            # Apply filters if any
            if filters:
                query['query'] = {
                    'bool': {
                        'must': [query['query']],
                        'filter': filters,
                    }
                }

            if min_score is not None:
                query['min_score'] = min_score

            response = await self.client.search(index=index_name, body=query)

            results = []
            for hit in response['hits']['hits']:
                result = {
                    'score': hit['_score'],
                    **hit['_source'],
                }
                # Remove embedding from results (too large)
                result.pop(embedding_field, None)
                result.pop('embedding', None)
                result.pop('global_embedding', None)
                results.append(result)

            return results

        except Exception as e:
            logger.error(f'Search failed on {index_name}: {e}')
            return []

    async def get_all_index_stats(self) -> dict[str, Any]:
        """Get statistics for all visual search indexes."""
        stats = {}
        for index_name in [IndexName.GLOBAL, IndexName.VEHICLES, IndexName.PEOPLE, IndexName.FACES]:
            try:
                exists = await self.client.indices.exists(index=index_name.value)
                if exists:
                    response = await self.client.indices.stats(index=index_name.value)
                    stats[index_name.value] = {
                        'total_documents': response['_all']['primaries']['docs']['count'],
                        'index_size_mb': round(
                            response['_all']['primaries']['store']['size_in_bytes'] / 1024 / 1024, 2
                        ),
                    }
                else:
                    stats[index_name.value] = {'exists': False}
            except Exception as e:
                stats[index_name.value] = {'error': str(e)}
        return stats

    # =========================================================================
    # Embedding Extraction (for Cluster Training/Rebalancing)
    # =========================================================================

    async def get_all_embeddings(
        self,
        index_name: IndexName,
        batch_size: int = 1000,
        max_docs: int | None = None,
    ) -> tuple[np.ndarray, list[str]]:
        """
        Extract all embeddings from an index for cluster training.

        Uses scroll API for efficient large-scale extraction.

        Args:
            index_name: Which index to extract from
            batch_size: Documents per scroll batch
            max_docs: Maximum documents to extract (None = all)

        Returns:
            Tuple of (embeddings array [N, 512], document IDs list)
        """
        embedding_field = 'global_embedding' if index_name == IndexName.GLOBAL else 'embedding'
        id_field = 'image_id' if index_name == IndexName.GLOBAL else 'detection_id'

        embeddings = []
        doc_ids = []

        try:
            # Initialize scroll
            response = await self.client.search(
                index=index_name.value,
                body={
                    'size': batch_size,
                    '_source': [embedding_field, id_field],
                    'query': {'match_all': {}},
                },
                scroll='5m',
            )

            scroll_id = response['_scroll_id']
            hits = response['hits']['hits']

            while hits:
                for hit in hits:
                    source = hit['_source']
                    if embedding_field in source:
                        embeddings.append(source[embedding_field])
                        doc_ids.append(source.get(id_field, hit['_id']))

                        if max_docs and len(embeddings) >= max_docs:
                            break

                if max_docs and len(embeddings) >= max_docs:
                    break

                # Get next batch
                response = await self.client.scroll(scroll_id=scroll_id, scroll='5m')
                scroll_id = response['_scroll_id']
                hits = response['hits']['hits']

            # Clear scroll
            await self.client.clear_scroll(scroll_id=scroll_id)

            logger.info(f'Extracted {len(embeddings)} embeddings from {index_name.value}')
            return np.array(embeddings, dtype=np.float32), doc_ids

        except Exception as e:
            logger.error(f'Failed to extract embeddings from {index_name.value}: {e}')
            return np.array([]), []

    async def get_unclustered_embeddings(
        self,
        index_name: IndexName,
        batch_size: int = 1000,
    ) -> tuple[np.ndarray, list[str]]:
        """
        Get embeddings that don't have cluster assignments yet.

        Useful for incremental clustering of newly ingested items.

        Returns:
            Tuple of (embeddings array, document IDs)
        """
        embedding_field = 'global_embedding' if index_name == IndexName.GLOBAL else 'embedding'
        id_field = 'image_id' if index_name == IndexName.GLOBAL else 'detection_id'

        embeddings = []
        doc_ids = []

        try:
            response = await self.client.search(
                index=index_name.value,
                body={
                    'size': batch_size,
                    '_source': [embedding_field, id_field],
                    'query': {'bool': {'must_not': {'exists': {'field': 'cluster_id'}}}},
                },
                scroll='5m',
            )

            scroll_id = response['_scroll_id']
            hits = response['hits']['hits']

            while hits:
                for hit in hits:
                    source = hit['_source']
                    if embedding_field in source:
                        embeddings.append(source[embedding_field])
                        doc_ids.append(source.get(id_field, hit['_id']))

                response = await self.client.scroll(scroll_id=scroll_id, scroll='5m')
                scroll_id = response['_scroll_id']
                hits = response['hits']['hits']

            await self.client.clear_scroll(scroll_id=scroll_id)

            logger.info(f'Found {len(embeddings)} unclustered items in {index_name.value}')
            return np.array(embeddings, dtype=np.float32), doc_ids

        except Exception as e:
            logger.error(f'Failed to get unclustered embeddings from {index_name.value}: {e}')
            return np.array([]), []

    async def update_cluster_assignments(
        self,
        index_name: IndexName,
        doc_ids: list[str],
        cluster_ids: list[int],
        cluster_distances: list[float],
    ) -> int:
        """
        Bulk update cluster assignments for documents.

        Args:
            index_name: Which index to update
            doc_ids: List of document IDs
            cluster_ids: Corresponding cluster IDs
            cluster_distances: Corresponding distances to centroids

        Returns:
            Number of successfully updated documents
        """
        if not doc_ids:
            return 0

        timestamp = datetime.now(UTC).isoformat()
        actions = []

        for doc_id, cluster_id, distance in zip(
            doc_ids, cluster_ids, cluster_distances, strict=True
        ):
            actions.append(
                {
                    '_op_type': 'update',
                    '_index': index_name.value,
                    '_id': doc_id,
                    'doc': {
                        'cluster_id': cluster_id,
                        'cluster_distance': distance,
                        'clustered_at': timestamp,
                    },
                }
            )

        try:
            success, errors = await async_bulk(self.client, actions, raise_on_error=False)
            if errors:
                logger.warning(f'Some cluster updates failed: {len(errors)} errors')
            logger.info(f'Updated {success} cluster assignments in {index_name.value}')
            return success
        except Exception as e:
            logger.error(f'Bulk cluster update failed: {e}')
            return 0

    # =========================================================================
    # Cluster Album Queries
    # =========================================================================

    async def get_cluster_members(
        self,
        index_name: IndexName,
        cluster_id: int,
        page: int = 0,
        size: int = 50,
        sort_by_distance: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Get all members of a specific cluster (like a Google Photos album).

        Args:
            index_name: Which index to query
            cluster_id: Cluster ID to retrieve
            page: Page number (0-indexed)
            size: Page size
            sort_by_distance: If True, sort by distance to centroid (most representative first)

        Returns:
            List of documents in the cluster
        """
        embedding_field = 'global_embedding' if index_name == IndexName.GLOBAL else 'embedding'

        try:
            query = {
                'query': {'term': {'cluster_id': cluster_id}},
                'from': page * size,
                'size': size,
                '_source': {'excludes': [embedding_field]},  # Don't return large embeddings
            }

            if sort_by_distance:
                query['sort'] = [{'cluster_distance': 'asc'}]

            response = await self.client.search(index=index_name.value, body=query)

            results = []
            for hit in response['hits']['hits']:
                result = {'score': hit.get('_score'), **hit['_source']}
                results.append(result)

            return results

        except Exception as e:
            logger.error(f'Failed to get cluster members: {e}')
            return []

    async def get_cluster_stats(
        self,
        index_name: IndexName,
    ) -> list[dict[str, Any]]:
        """
        Get statistics about clusters in an index.

        Returns:
            List of cluster stats with id, count, avg_distance
        """
        try:
            response = await self.client.search(
                index=index_name.value,
                body={
                    'size': 0,
                    'aggs': {
                        'clusters': {
                            'terms': {
                                'field': 'cluster_id',
                                'size': 10000,  # Get all clusters
                            },
                            'aggs': {
                                'avg_distance': {'avg': {'field': 'cluster_distance'}},
                                'min_distance': {'min': {'field': 'cluster_distance'}},
                            },
                        }
                    },
                },
            )

            return [
                {
                    'cluster_id': bucket['key'],
                    'count': bucket['doc_count'],
                    'avg_distance': bucket['avg_distance']['value'],
                    'min_distance': bucket['min_distance']['value'],
                }
                for bucket in response['aggregations']['clusters']['buckets']
            ]

        except Exception as e:
            logger.error(f'Failed to get cluster stats: {e}')
            return []

    async def delete_all_indexes(self) -> dict[str, bool]:
        """Delete all visual search indexes."""
        results = {}
        for index_name in [IndexName.GLOBAL, IndexName.VEHICLES, IndexName.PEOPLE, IndexName.FACES]:
            try:
                exists = await self.client.indices.exists(index=index_name.value)
                if exists:
                    await self.client.indices.delete(index=index_name.value)
                    results[index_name.value] = True
                    logger.info(f'Deleted index: {index_name.value}')
                else:
                    results[index_name.value] = True  # Already doesn't exist
            except Exception as e:
                logger.error(f'Failed to delete {index_name.value}: {e}')
                results[index_name.value] = False
        return results


# Convenience function for standalone usage
async def create_client(**kwargs) -> OpenSearchClient:
    """Create and return an OpenSearch client instance."""
    return OpenSearchClient(**kwargs)
