"""
Face Identity Service for 1:N face identification.

Provides face identity management and search capabilities:
- Face ingestion with automatic embedding extraction
- 1:N face identification against database
- Person management (grouping faces by identity)
- Face-to-person assignment

Uses OpenSearch 'visual_search_faces' index for storage and k-NN search.
"""

import base64
import io
import logging
import uuid
from datetime import UTC, datetime
from typing import Any

import numpy as np
from PIL import Image

from src.clients.opensearch import IndexName, OpenSearchClient
from src.config.settings import Settings, get_settings
from src.services.inference import InferenceService


logger = logging.getLogger(__name__)


class FaceIdentityService:
    """
    Service for face identity management and 1:N identification.

    Features:
    - Ingest faces with ArcFace embeddings
    - 1:N face identification (find matching persons in database)
    - Person management (assign faces to persons)
    - Face quality assessment and thumbnail generation

    Design:
    - Uses OpenSearchClient for vector storage and k-NN search
    - Uses InferenceService for face detection and embedding extraction
    - Stores face crops as base64 thumbnails for UI preview
    """

    def __init__(self, settings: Settings | None = None):
        """
        Initialize face identity service.

        Args:
            settings: Optional settings instance. If None, uses singleton.
        """
        self.settings = settings or get_settings()
        self.inference = InferenceService()
        self._opensearch: OpenSearchClient | None = None

    @property
    def opensearch(self) -> OpenSearchClient:
        """Lazy-load OpenSearch client."""
        if self._opensearch is None:
            self._opensearch = OpenSearchClient(
                hosts=[self.settings.opensearch_url],
                timeout=self.settings.opensearch_timeout,
            )
        return self._opensearch

    # =========================================================================
    # Face Ingestion
    # =========================================================================

    async def ingest_face(
        self,
        image_bytes: bytes,
        person_id: str | None = None,
        face_id: str | None = None,
        source_image_id: str | None = None,
        metadata: dict | None = None,
    ) -> dict:
        """
        Detect faces in image and store embeddings in OpenSearch.

        Pipeline:
        1. Run face detection (SCRFD or YOLO11-face)
        2. Extract ArcFace embeddings for each detected face
        3. Generate face crop thumbnails
        4. Index faces with embeddings to OpenSearch

        Args:
            image_bytes: Raw JPEG/PNG image bytes
            person_id: Optional person ID to assign faces to.
                      If None, auto-generates one or clusters later.
            face_id: Optional custom face ID. If None, auto-generates.
            source_image_id: Optional source image identifier for tracking.
            metadata: Optional metadata dictionary to store with faces.

        Returns:
            dict with:
                - status: 'success' or 'error'
                - num_faces: Number of faces detected
                - face_ids: List of ingested face IDs
                - person_id: Assigned or generated person ID
                - errors: List of any errors encountered
        """
        try:
            # 1. Run face detection and embedding extraction
            face_result = self.inference.infer_faces(image_bytes)

            if face_result.get('num_faces', 0) == 0:
                return {
                    'status': 'success',
                    'num_faces': 0,
                    'face_ids': [],
                    'person_id': None,
                    'message': 'No faces detected in image',
                }

            num_faces = face_result['num_faces']
            faces = face_result['faces']
            embeddings = face_result['embeddings']
            orig_h, orig_w = face_result['orig_shape']

            # 2. Decode image for thumbnail generation
            img = Image.open(io.BytesIO(image_bytes))
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # 3. Generate unique IDs if not provided
            timestamp = datetime.now(UTC).isoformat()
            generated_face_ids = []
            indexed_count = 0
            errors = []

            # Generate person_id if not provided (will be same for all faces in image)
            effective_person_id = person_id or f'person_{uuid.uuid4().hex[:12]}'

            for i in range(num_faces):
                face = faces[i]
                embedding = embeddings[i]

                # Generate face_id
                current_face_id = face_id if (face_id and i == 0) else f'face_{uuid.uuid4().hex[:12]}'
                generated_face_ids.append(current_face_id)

                try:
                    # Extract face crop and generate thumbnail
                    box = face['box']  # Already normalized [0, 1]
                    x1 = int(box[0] * orig_w)
                    y1 = int(box[1] * orig_h)
                    x2 = int(box[2] * orig_w)
                    y2 = int(box[3] * orig_h)

                    # Add padding for better thumbnails
                    pad_w = int((x2 - x1) * 0.1)
                    pad_h = int((y2 - y1) * 0.1)
                    x1 = max(0, x1 - pad_w)
                    y1 = max(0, y1 - pad_h)
                    x2 = min(orig_w, x2 + pad_w)
                    y2 = min(orig_h, y2 + pad_h)

                    face_crop = img.crop((x1, y1, x2, y2))

                    # Resize thumbnail to reasonable size (max 128px)
                    max_thumb_size = 128
                    face_crop.thumbnail((max_thumb_size, max_thumb_size), Image.Resampling.LANCZOS)

                    # Convert to base64
                    thumb_buffer = io.BytesIO()
                    face_crop.save(thumb_buffer, format='JPEG', quality=85)
                    thumbnail_b64 = base64.b64encode(thumb_buffer.getvalue()).decode('utf-8')

                    # Convert landmarks to named object format
                    raw_landmarks = face.get('landmarks', [])
                    landmarks = {}
                    if len(raw_landmarks) >= 10:
                        landmarks = {
                            'left_eye': [float(raw_landmarks[0]), float(raw_landmarks[1])],
                            'right_eye': [float(raw_landmarks[2]), float(raw_landmarks[3])],
                            'nose': [float(raw_landmarks[4]), float(raw_landmarks[5])],
                            'left_mouth': [float(raw_landmarks[6]), float(raw_landmarks[7])],
                            'right_mouth': [float(raw_landmarks[8]), float(raw_landmarks[9])],
                        }

                    # Build document
                    face_doc = {
                        'face_id': current_face_id,
                        'image_id': source_image_id or f'img_{uuid.uuid4().hex[:12]}',
                        'image_path': source_image_id or '',
                        'embedding': embedding if isinstance(embedding, list) else embedding.tolist(),
                        'box': box,
                        'landmarks': landmarks,
                        'confidence': face.get('score', 0.0),
                        'quality_score': face.get('quality', 0.0),
                        'person_id': effective_person_id,
                        'thumbnail_b64': thumbnail_b64,
                        'is_reference': (i == 0 and person_id is None),  # First face is reference if new person
                        'indexed_at': timestamp,
                    }

                    if metadata:
                        face_doc['metadata'] = metadata

                    # Index to OpenSearch
                    await self.opensearch.client.index(
                        index=IndexName.FACES.value,
                        id=current_face_id,
                        body=face_doc,
                        refresh=False,
                    )
                    indexed_count += 1

                except Exception as e:
                    logger.error(f'Failed to index face {current_face_id}: {e}')
                    errors.append(f'{current_face_id}: {str(e)}')

            return {
                'status': 'success' if indexed_count > 0 else 'error',
                'num_faces': num_faces,
                'indexed': indexed_count,
                'face_ids': generated_face_ids[:indexed_count],
                'person_id': effective_person_id if indexed_count > 0 else None,
                'errors': errors if errors else None,
            }

        except Exception as e:
            logger.error(f'Face ingestion failed: {e}')
            return {
                'status': 'error',
                'num_faces': 0,
                'face_ids': [],
                'person_id': None,
                'error': str(e),
            }

    # =========================================================================
    # Face Identification (1:N Search)
    # =========================================================================

    async def identify_faces(
        self,
        image_bytes: bytes,
        top_k: int = 5,
        threshold: float = 0.6,
        face_detector: str = 'scrfd',
    ) -> dict:
        """
        1:N face identification - find matching persons in database.

        Pipeline:
        1. Detect faces in query image
        2. Extract ArcFace embeddings
        3. For each face, search database for matches
        4. Return top-k matches with similarity scores

        Args:
            image_bytes: Raw JPEG/PNG image bytes
            top_k: Number of top matches to return per face
            threshold: Minimum similarity threshold (0.0-1.0).
                      Recommended: 0.5-0.6 for same person verification.
            face_detector: Face detector to use ('scrfd' or 'yolo11')

        Returns:
            dict with:
                - status: 'success' or 'error'
                - num_faces: Number of faces in query image
                - identifications: List of identification results per face
                  Each contains:
                    - query_face: Query face info (box, score)
                    - matches: List of matching faces with scores
                    - best_match_person_id: Person ID of best match (if above threshold)
        """
        try:
            # 1. Run face detection based on detector choice
            if face_detector == 'yolo11':
                face_result = self.inference.infer_faces_yolo11(image_bytes)
            else:
                face_result = self.inference.infer_faces(image_bytes)

            if face_result.get('num_faces', 0) == 0:
                return {
                    'status': 'success',
                    'num_faces': 0,
                    'identifications': [],
                    'message': 'No faces detected in query image',
                }

            num_faces = face_result['num_faces']
            faces = face_result['faces']
            embeddings = face_result['embeddings']
            orig_h, orig_w = face_result['orig_shape']

            identifications = []

            # 2. Search for each detected face
            for i in range(num_faces):
                face = faces[i]
                embedding = embeddings[i]

                # Convert embedding to numpy array if needed
                if isinstance(embedding, list):
                    embedding = np.array(embedding, dtype=np.float32)

                # Query face info
                query_face = {
                    'index': i,
                    'box': face['box'],
                    'score': face.get('score', 0.0),
                    'quality': face.get('quality'),
                }

                # 3. Search faces index
                matches = await self.search_faces_by_embedding(
                    embedding=embedding.tolist() if hasattr(embedding, 'tolist') else embedding,
                    top_k=top_k,
                    min_score=threshold,
                )

                # Determine best match
                best_match_person_id = None
                if matches and matches[0].get('score', 0) >= threshold:
                    best_match_person_id = matches[0].get('person_id')

                identifications.append({
                    'query_face': query_face,
                    'matches': matches,
                    'best_match_person_id': best_match_person_id,
                    'is_identified': best_match_person_id is not None,
                })

            return {
                'status': 'success',
                'num_faces': num_faces,
                'identifications': identifications,
                'threshold_used': threshold,
            }

        except Exception as e:
            logger.error(f'Face identification failed: {e}')
            return {
                'status': 'error',
                'num_faces': 0,
                'identifications': [],
                'error': str(e),
            }

    # =========================================================================
    # Person Management
    # =========================================================================

    async def get_person_faces(self, person_id: str) -> list:
        """
        Get all faces belonging to a person identity.

        Args:
            person_id: Person identifier

        Returns:
            List of face documents (without embeddings)
        """
        try:
            response = await self.opensearch.client.search(
                index=IndexName.FACES.value,
                body={
                    'query': {'term': {'person_id': person_id}},
                    'size': 1000,  # Get all faces for person
                    'sort': [
                        {'is_reference': {'order': 'desc'}},  # Reference face first
                        {'confidence': {'order': 'desc'}},
                    ],
                    '_source': {
                        'excludes': ['embedding'],  # Don't return large embedding
                    },
                },
            )

            faces = []
            for hit in response['hits']['hits']:
                face = hit['_source']
                face['_id'] = hit['_id']
                faces.append(face)

            return faces

        except Exception as e:
            logger.error(f'Failed to get faces for person {person_id}: {e}')
            return []

    async def assign_face_to_person(self, face_id: str, person_id: str) -> dict:
        """
        Assign or reassign a face to a person.

        Args:
            face_id: Face identifier to update
            person_id: Person identifier to assign to

        Returns:
            dict with:
                - status: 'success' or 'error'
                - face_id: Updated face ID
                - person_id: New person ID
                - previous_person_id: Previous person ID (if any)
        """
        try:
            # Get current face document to check previous person_id
            current_doc = await self.opensearch.client.get(
                index=IndexName.FACES.value,
                id=face_id,
                _source=['person_id'],
            )
            previous_person_id = current_doc['_source'].get('person_id')

            # Update the face document
            timestamp = datetime.now(UTC).isoformat()
            await self.opensearch.client.update(
                index=IndexName.FACES.value,
                id=face_id,
                body={
                    'doc': {
                        'person_id': person_id,
                        'is_reference': False,  # Reset reference status on reassignment
                        'updated_at': timestamp,
                    }
                },
                refresh=True,
            )

            logger.info(f'Assigned face {face_id} to person {person_id}')

            return {
                'status': 'success',
                'face_id': face_id,
                'person_id': person_id,
                'previous_person_id': previous_person_id,
            }

        except Exception as e:
            logger.error(f'Failed to assign face {face_id} to person {person_id}: {e}')
            return {
                'status': 'error',
                'face_id': face_id,
                'person_id': person_id,
                'error': str(e),
            }

    # =========================================================================
    # Face Search by Embedding
    # =========================================================================

    async def search_faces_by_embedding(
        self,
        embedding: list[float],
        top_k: int = 10,
        min_score: float = 0.0,
    ) -> list:
        """
        Search faces by embedding vector using k-NN.

        Args:
            embedding: 512-dim ArcFace embedding vector
            top_k: Number of top results to return
            min_score: Minimum similarity score threshold

        Returns:
            List of matching face documents with similarity scores
        """
        try:
            # Convert to numpy for OpenSearch
            query_embedding = np.array(embedding, dtype=np.float32)

            # Build k-NN query
            query = {
                'size': top_k,
                'query': {
                    'knn': {
                        'embedding': {
                            'vector': query_embedding.tolist(),
                            'k': top_k,
                        }
                    }
                },
                '_source': {
                    'excludes': ['embedding'],  # Don't return large embedding
                },
            }

            if min_score > 0:
                query['min_score'] = min_score

            response = await self.opensearch.client.search(
                index=IndexName.FACES.value,
                body=query,
            )

            results = []
            for hit in response['hits']['hits']:
                result = {
                    'score': hit['_score'],
                    **hit['_source'],
                }
                results.append(result)

            return results

        except Exception as e:
            logger.error(f'Face search by embedding failed: {e}')
            return []

    # =========================================================================
    # Utility Methods
    # =========================================================================

    async def get_face_by_id(self, face_id: str) -> dict | None:
        """
        Get a single face document by ID.

        Args:
            face_id: Face identifier

        Returns:
            Face document or None if not found
        """
        try:
            response = await self.opensearch.client.get(
                index=IndexName.FACES.value,
                id=face_id,
                _source={'excludes': ['embedding']},
            )
            return response['_source']
        except Exception as e:
            logger.error(f'Failed to get face {face_id}: {e}')
            return None

    async def delete_face(self, face_id: str) -> bool:
        """
        Delete a face from the index.

        Args:
            face_id: Face identifier to delete

        Returns:
            True if deleted, False otherwise
        """
        try:
            await self.opensearch.client.delete(
                index=IndexName.FACES.value,
                id=face_id,
                refresh=True,
            )
            logger.info(f'Deleted face {face_id}')
            return True
        except Exception as e:
            logger.error(f'Failed to delete face {face_id}: {e}')
            return False

    async def get_all_persons(self, limit: int = 100) -> list[dict]:
        """
        Get all unique person IDs with face counts.

        Args:
            limit: Maximum number of persons to return

        Returns:
            List of person summaries with face counts
        """
        try:
            response = await self.opensearch.client.search(
                index=IndexName.FACES.value,
                body={
                    'size': 0,
                    'aggs': {
                        'persons': {
                            'terms': {
                                'field': 'person_id',
                                'size': limit,
                            },
                            'aggs': {
                                'reference_face': {
                                    'top_hits': {
                                        'size': 1,
                                        'sort': [{'is_reference': {'order': 'desc'}}],
                                        '_source': ['face_id', 'thumbnail_b64', 'confidence'],
                                    }
                                }
                            }
                        }
                    },
                },
            )

            persons = []
            for bucket in response['aggregations']['persons']['buckets']:
                ref_hit = bucket['reference_face']['hits']['hits']
                ref_face = ref_hit[0]['_source'] if ref_hit else {}

                persons.append({
                    'person_id': bucket['key'],
                    'face_count': bucket['doc_count'],
                    'reference_face_id': ref_face.get('face_id'),
                    'reference_thumbnail': ref_face.get('thumbnail_b64'),
                })

            return persons

        except Exception as e:
            logger.error(f'Failed to get all persons: {e}')
            return []

    async def close(self):
        """Close OpenSearch connection."""
        if self._opensearch is not None:
            await self._opensearch.close()
            self._opensearch = None


# Singleton instance
_face_identity_service: FaceIdentityService | None = None


def get_face_identity_service() -> FaceIdentityService:
    """
    Get singleton FaceIdentityService instance.

    Returns:
        FaceIdentityService: Service instance
    """
    global _face_identity_service
    if _face_identity_service is None:
        _face_identity_service = FaceIdentityService()
    return _face_identity_service
