"""
Track E: Visual Search Service.

Orchestrates inference + OpenSearch operations for visual search.
Bridges InferenceService (Triton) and OpenSearchClient (multi-index vector search).

Multi-Index Architecture:
- visual_search_global: Whole image similarity (scene matching)
- visual_search_vehicles: Vehicle detections (car, truck, motorcycle, bus, boat)
- visual_search_people: Person appearance (clothing, pose)
- visual_search_faces: Face identity matching (future: ArcFace)
- visual_search_ocr: Text content (OCR with trigram search)

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
        skip_duplicates: bool = True,
        detect_near_duplicates: bool = True,
        near_duplicate_threshold: float = 0.99,
        enable_ocr: bool = True,
    ) -> dict[str, Any]:
        """
        Ingest single image with auto-routing to category indexes.

        Pipeline:
        1. Compute imohash for exact duplicate detection (if skip_duplicates=True)
        2. Check if image already exists by hash
        3. Run Track E full ensemble (YOLO + MobileCLIP)
        4. Extract global + box embeddings
        5. Route to appropriate indexes:
           - Global embedding -> visual_search_global
           - Person detections -> visual_search_people
           - Vehicle detections -> visual_search_vehicles
        6. Check for near-duplicates and auto-assign to groups (if enabled)
        7. Run OCR and index text content (if enable_ocr=True)

        Args:
            image_bytes: Raw JPEG/PNG bytes
            image_id: Unique identifier for the image
            image_path: Optional file path (for retrieval)
            metadata: Optional metadata dictionary
            skip_duplicates: If True, skip processing if image hash already exists.
                           Set to False for benchmarks. Default: True (production).
            detect_near_duplicates: If True, check for visually similar images
                           and auto-assign to duplicate groups. Default: True.
            near_duplicate_threshold: Similarity threshold for grouping.
                           Default: 0.99 (matches Immich's maxDistance=0.01).
                           Range: 0.90 (similar content) to 0.99 (near-identical).
            enable_ocr: If True, run OCR and index detected text.
                           Default: True.

        Returns:
            dict with status, routing info, and counts per category
        """
        try:
            # 1. Compute imohash for duplicate detection
            import io

            import imohash

            from src.clients.triton_client import get_triton_client
            from src.config import get_settings

            image_hash = imohash.hashfileobject(io.BytesIO(image_bytes)).hex()
            file_size = len(image_bytes)

            # 2. Check for duplicates if enabled
            if skip_duplicates:
                existing = await self.opensearch.check_duplicate_by_hash(image_hash)
                if existing:
                    logger.info(
                        f'Duplicate detected: {image_id} matches {existing.get("image_id")}'
                    )
                    return {
                        'status': 'duplicate',
                        'image_id': image_id,
                        'existing_image_id': existing.get('image_id'),
                        'existing_image_path': existing.get('image_path'),
                        'imohash': image_hash,
                        'message': 'Image already exists in index (same imohash)',
                    }

            settings = get_settings()
            client = get_triton_client(settings.triton_url)

            # Run Track E (YOLO + MobileCLIP) and YOLO11-face detection in parallel
            from concurrent.futures import ThreadPoolExecutor

            def run_track_e():
                return client.infer_track_e(image_bytes, full_pipeline=True)

            def run_face_detection():
                return client.infer_faces_yolo11(image_bytes, confidence=0.5)

            with ThreadPoolExecutor(max_workers=2) as executor:
                track_e_future = executor.submit(run_track_e)
                face_future = executor.submit(run_face_detection)
                result = track_e_future.result()
                face_result = face_future.result()

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
                imohash=image_hash,
                file_size_bytes=file_size,
            )

            # Index faces if detected
            num_faces_indexed = 0
            if face_result.get('num_faces', 0) > 0:
                face_embeddings = face_result.get('face_embeddings', [])
                face_boxes = face_result.get('face_boxes', [])
                face_scores = face_result.get('face_scores', [])
                face_quality = face_result.get('face_quality', [])
                face_landmarks = face_result.get('face_landmarks', [])

                for i in range(min(len(face_embeddings), len(face_boxes))):
                    if len(face_embeddings[i]) > 0:
                        face_id = f'{image_id}_face_{i}'
                        try:
                            await self.opensearch.index_face(
                                face_id=face_id,
                                image_id=image_id,
                                image_path=image_path or image_id,
                                embedding=face_embeddings[i],
                                box=face_boxes[i].tolist()
                                if hasattr(face_boxes[i], 'tolist')
                                else list(face_boxes[i]),
                                landmarks=face_landmarks[i].tolist()
                                if hasattr(face_landmarks[i], 'tolist')
                                else list(face_landmarks[i])
                                if len(face_landmarks) > i
                                else [],
                                confidence=float(face_scores[i]) if len(face_scores) > i else 0.0,
                                quality=float(face_quality[i]) if len(face_quality) > i else 0.0,
                            )
                            num_faces_indexed += 1
                        except Exception as e:
                            logger.debug(f'Failed to index face {face_id}: {e}')

            # 6. Auto-detect and assign near-duplicates (runs in background, non-blocking)
            duplicate_info = None
            if detect_near_duplicates and ingest_result['global']:
                duplicate_info = await self._assign_to_duplicate_group(
                    image_id=image_id,
                    embedding=global_embedding,
                    threshold=near_duplicate_threshold,
                )

            response = {
                'status': 'success' if ingest_result['global'] else 'failed',
                'image_id': image_id,
                'num_detections': result['num_dets'],
                'num_faces': face_result.get('num_faces', 0),
                'embedding_norm': float(np.linalg.norm(global_embedding)),
                'imohash': image_hash,
                'indexed': {
                    'global': ingest_result['global'],
                    'vehicles': ingest_result['vehicles'],
                    'people': ingest_result['people'],
                    'faces': num_faces_indexed,
                    'skipped': ingest_result['skipped'],
                },
                'errors': ingest_result.get('errors', []),
            }

            if duplicate_info:
                response['near_duplicate'] = duplicate_info

            # 7. Run OCR and index text content (if enabled)
            ocr_info = None
            if enable_ocr:
                ocr_info = await self._process_ocr(
                    image_bytes=image_bytes,
                    image_id=image_id,
                    image_path=image_path,
                )
                if ocr_info:
                    response['ocr'] = ocr_info

            return response

        except Exception as e:
            logger.error(f'Failed to ingest image {image_id}: {e}')
            return {
                'status': 'error',
                'image_id': image_id,
                'error': str(e),
            }

    async def _assign_to_duplicate_group(
        self,
        image_id: str,
        embedding: np.ndarray,
        threshold: float = 0.99,
    ) -> dict[str, Any] | None:
        """
        Check for near-duplicates and assign to existing group or create new one.

        This runs automatically during ingestion to keep duplicate groups current.
        No periodic retraining needed - groups are updated in real-time.

        Algorithm:
        1. Search for similar images above threshold
        2. If found with existing group -> join that group
        3. If found without group -> create new group with both images
        4. If not found -> no action (unique image)

        Args:
            image_id: ID of newly ingested image
            embedding: CLIP embedding of the image
            threshold: Similarity threshold (default 0.99, matches Immich)

        Returns:
            Dict with duplicate info if assigned, None if unique
        """
        try:
            from src.services.duplicate_detection import DuplicateDetectionService

            dup_service = DuplicateDetectionService(self.opensearch)

            # Find near-duplicates (excluding self)
            duplicates = await dup_service.find_duplicates_by_embedding(
                embedding=embedding,
                threshold=threshold,
                max_results=10,
                exclude_image_id=image_id,
            )

            if not duplicates:
                return None  # Unique image, no duplicates

            # Check if any duplicate is already in a group
            existing_group_id = None
            for dup in duplicates:
                if dup.duplicate_group_id:
                    existing_group_id = dup.duplicate_group_id
                    break

            if existing_group_id:
                # Join existing group
                await self.opensearch.client.update(
                    index='visual_search_global',
                    id=image_id,
                    body={
                        'doc': {
                            'duplicate_group_id': existing_group_id,
                            'is_duplicate_primary': False,
                            'duplicate_score': duplicates[0].similarity,
                        }
                    },
                )
                logger.info(f'Added {image_id} to existing duplicate group {existing_group_id}')
                return {
                    'action': 'joined_group',
                    'group_id': existing_group_id,
                    'similarity': duplicates[0].similarity,
                    'matched_image': duplicates[0].image_id,
                }

            # Create new group with this image as primary and first duplicate
            best_match = duplicates[0]
            group_id = await dup_service.create_duplicate_group(
                primary_image_id=image_id,
                duplicate_image_ids=[best_match.image_id],
                duplicate_scores=[best_match.similarity],
            )
            logger.info(f'Created new duplicate group {group_id} for {image_id}')
            return {
                'action': 'created_group',
                'group_id': group_id,
                'similarity': best_match.similarity,
                'matched_image': best_match.image_id,
            }

        except Exception as e:
            # Don't fail ingestion if duplicate detection fails
            logger.warning(f'Near-duplicate detection failed for {image_id}: {e}')
            return None

    async def _process_ocr(
        self,
        image_bytes: bytes,
        image_id: str,
        image_path: str | None = None,
    ) -> dict[str, Any] | None:
        """
        Process OCR for an image and index results.

        Non-blocking - failures don't affect main ingestion.

        Args:
            image_bytes: Raw image bytes
            image_id: Image identifier
            image_path: Optional file path

        Returns:
            OCR info dict or None if no text detected
        """
        try:
            from src.services.ocr_service import get_ocr_service

            ocr_service = get_ocr_service()
            ocr_result = ocr_service.extract_text(image_bytes, filter_by_score=True)

            if ocr_result.get('status') != 'success' or ocr_result.get('num_texts', 0) == 0:
                return None  # No text detected or OCR failed

            # Index OCR results to OpenSearch
            full_text = ocr_service.get_full_text(ocr_result)
            await self.opensearch.index_ocr_results(
                image_id=image_id,
                image_path=image_path or image_id,
                texts=ocr_result['texts'],
                boxes=ocr_result['boxes'],
                boxes_normalized=ocr_result['boxes_normalized'],
                det_scores=ocr_result['det_scores'],
                rec_scores=ocr_result['rec_scores'],
                full_text=full_text,
            )

            logger.info(f'OCR indexed {ocr_result["num_texts"]} text regions for {image_id}')

            return {
                'num_texts': ocr_result['num_texts'],
                'full_text': full_text[:200]
                if len(full_text) > 200
                else full_text,  # Truncate for response
                'indexed': True,
            }

        except Exception as e:
            # Don't fail main ingestion if OCR fails
            logger.warning(f'OCR processing failed for {image_id}: {e}')
            return None

    # =========================================================================
    # Batch Ingestion (HIGH THROUGHPUT - 300+ RPS)
    # =========================================================================

    async def ingest_batch(
        self,
        images_data: list[tuple[bytes, str, str | None]],
        skip_duplicates: bool = True,
        detect_near_duplicates: bool = True,
        near_duplicate_threshold: float = 0.99,
        enable_ocr: bool = True,
        max_workers: int = 64,
    ) -> dict[str, Any]:
        """
        Batch ingest multiple images with optimized parallel processing.

        Performance: 3-5x faster than individual /ingest calls.
        - Parallel hash computation
        - Batch Triton inference (dynamic batching: 16-48 avg)
        - OpenSearch bulk indexing
        - Reduced HTTP overhead

        Target throughput: 300+ RPS with batch sizes of 32-64.

        Args:
            images_data: List of (image_bytes, image_id, image_path) tuples
            skip_duplicates: Skip processing if image hash exists
            detect_near_duplicates: Auto-assign to duplicate groups
            near_duplicate_threshold: Similarity threshold for grouping
            enable_ocr: Run OCR (disabled by default for performance)
            max_workers: Max parallel threads for inference

        Returns:
            Summary with processed count, duplicates, and errors
        """
        import asyncio
        from concurrent.futures import ThreadPoolExecutor

        import imohash

        from src.clients.triton_client import get_triton_client
        from src.config import get_settings

        if not images_data:
            return {
                'status': 'success',
                'total': 0,
                'processed': 0,
                'duplicates': 0,
                'errors_count': 0,
                'indexed': {'global': 0, 'vehicles': 0, 'people': 0, 'faces': 0, 'ocr': 0},
                'near_duplicates': 0,
            }

        total = len(images_data)
        settings = get_settings()
        client = get_triton_client(settings.triton_url)

        # Step 1: Compute hashes in parallel
        def compute_hash(img_bytes: bytes) -> str:
            import io

            return imohash.hashfileobject(io.BytesIO(img_bytes)).hex()

        hashes = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            hashes = list(executor.map(compute_hash, [img[0] for img in images_data]))

        # Step 2: Check for duplicates if enabled
        duplicates = []
        to_process = []
        to_process_indices = []

        if skip_duplicates:
            for i, (img_data, image_hash) in enumerate(zip(images_data, hashes, strict=False)):
                existing = await self.opensearch.check_duplicate_by_hash(image_hash)
                if existing:
                    duplicates.append(
                        {
                            'image_id': img_data[1],
                            'existing_image_id': existing.get('image_id'),
                            'imohash': image_hash,
                        }
                    )
                else:
                    to_process.append(img_data)
                    to_process_indices.append(i)
        else:
            to_process = images_data
            to_process_indices = list(range(len(images_data)))

        if not to_process:
            return {
                'status': 'success',
                'total': total,
                'processed': 0,
                'duplicates': len(duplicates),
                'errors_count': 0,
                'indexed': {'global': 0, 'vehicles': 0, 'people': 0, 'faces': 0, 'ocr': 0},
                'near_duplicates': 0,
                'duplicate_details': duplicates,
            }

        # Step 3: Run unified batch inference (YOLO + CLIP + Face in single GPU pass)
        loop = asyncio.get_running_loop()
        from concurrent.futures import ThreadPoolExecutor

        def run_single_unified(img_bytes: bytes) -> dict:
            """Run unified pipeline on a single image (YOLO + CLIP + Face)."""
            try:
                return client.infer_unified(img_bytes)
            except Exception as e:
                logger.debug(f'Unified inference failed: {e}')
                return {'error': str(e), 'num_dets': 0, 'num_faces': 0}

        def run_unified_batch(images_bytes: list[bytes]) -> list[dict]:
            """Run unified pipeline on batch of images in parallel."""
            with ThreadPoolExecutor(max_workers=min(max_workers, len(images_bytes))) as executor:
                return list(executor.map(run_single_unified, images_bytes))

        # Single unified inference call - includes YOLO, CLIP, and Face detection
        inference_results = await loop.run_in_executor(
            None,
            run_unified_batch,
            [img[0] for img in to_process],
        )

        # Extract face results from unified response (already included)
        face_results = [
            {
                'num_faces': r.get('num_faces', 0),
                'face_embeddings': r.get('face_embeddings', []),
                'face_boxes': r.get('face_boxes', []),
                'face_scores': r.get('face_scores', []),
                'face_quality': r.get('face_quality', []),
                'face_landmarks': r.get('face_landmarks', []),
            }
            for r in inference_results
        ]

        # Step 4: Format documents for bulk indexing
        from datetime import UTC, datetime

        import numpy as np

        documents = []
        face_documents = []
        errors = []
        timestamp = datetime.now(UTC).isoformat()

        for _i, (result, face_result, data_tuple, orig_idx) in enumerate(
            zip(inference_results, face_results, to_process, to_process_indices, strict=False)
        ):
            img_bytes, image_id, image_path = data_tuple
            image_hash = hashes[orig_idx]

            if 'error' in result:
                errors.append(
                    {
                        'image_id': image_id,
                        'error': result['error'],
                    }
                )
                continue

            # Handle both unified (global_embedding) and track_e (image_embedding) responses
            global_embedding = np.array(
                result.get('global_embedding', result.get('image_embedding', []))
            )

            # Get image dimensions from response or parse from JPEG
            orig_shape = result.get('orig_shape')
            if orig_shape:
                width, height = orig_shape[1], orig_shape[0]
            else:
                # Parse from JPEG header if not in response
                from src.clients.triton_client import get_jpeg_dimensions_fast

                try:
                    width, height = get_jpeg_dimensions_fast(img_bytes)
                except Exception:
                    width, height = 0, 0

            doc = {
                'image_id': image_id,
                'image_path': image_path or image_id,
                'global_embedding': global_embedding,
                'imohash': image_hash,
                'file_size_bytes': len(img_bytes),
                'width': width,
                'height': height,
            }

            # Add box data if detections exist
            if result.get('num_dets', 0) > 0:
                doc['box_embeddings'] = np.array(result.get('box_embeddings', []))
                doc['normalized_boxes'] = np.array(result.get('normalized_boxes', []))
                doc['det_classes'] = result.get('classes', [])
                doc['det_scores'] = result.get('scores', [])

            documents.append(doc)

            # Collect face documents for bulk indexing
            if face_result.get('num_faces', 0) > 0:
                face_embeddings = face_result.get('face_embeddings', [])
                face_boxes = face_result.get('face_boxes', [])
                face_scores = face_result.get('face_scores', [])
                face_quality = face_result.get('face_quality', [])
                face_landmarks = face_result.get('face_landmarks', [])

                for j in range(min(len(face_embeddings), len(face_boxes))):
                    if len(face_embeddings[j]) > 0:
                        face_id = f'{image_id}_face_{j}'
                        face_documents.append(
                            {
                                'face_id': face_id,
                                'image_id': image_id,
                                'image_path': image_path or image_id,
                                'embedding': face_embeddings[j],
                                'box': face_boxes[j].tolist()
                                if hasattr(face_boxes[j], 'tolist')
                                else list(face_boxes[j]),
                                'landmarks': face_landmarks[j].tolist()
                                if hasattr(face_landmarks[j], 'tolist')
                                else list(face_landmarks[j])
                                if len(face_landmarks) > j
                                else [],
                                'confidence': float(face_scores[j])
                                if len(face_scores) > j
                                else 0.0,
                                'quality': float(face_quality[j]) if len(face_quality) > j else 0.0,
                                'indexed_at': timestamp,
                            }
                        )

        # Step 5: Bulk index to OpenSearch (global, vehicles, people)
        indexed = {'global': 0, 'vehicles': 0, 'people': 0, 'faces': 0}
        if documents:
            bulk_result = await self.opensearch.bulk_ingest(documents, refresh=False)
            indexed = {
                'global': bulk_result.get('global', 0),
                'vehicles': bulk_result.get('vehicles', 0),
                'people': bulk_result.get('people', 0),
                'faces': 0,
            }

        # Step 5b: Bulk index faces
        if face_documents:
            for face_doc in face_documents:
                try:
                    await self.opensearch.index_face(
                        face_id=face_doc['face_id'],
                        image_id=face_doc['image_id'],
                        image_path=face_doc['image_path'],
                        embedding=face_doc['embedding'],
                        box=face_doc['box'],
                        landmarks=face_doc.get('landmarks'),
                        confidence=face_doc.get('confidence', 0.0),
                        quality=face_doc.get('quality', 0.0),
                    )
                    indexed['faces'] += 1
                except Exception as e:
                    logger.debug(f'Failed to index face {face_doc["face_id"]}: {e}')

        # Step 5c: OCR processing (if enabled)
        indexed['ocr'] = 0
        if enable_ocr:
            for data_tuple in to_process:
                img_bytes, image_id, image_path = data_tuple
                try:
                    ocr_result = await self._process_ocr(
                        image_bytes=img_bytes,
                        image_id=image_id,
                        image_path=image_path or image_id,
                    )
                    if ocr_result:
                        indexed['ocr'] += 1
                except Exception as e:
                    logger.debug(f'OCR failed for {image_id}: {e}')

        # Step 6: Near-duplicate detection (optional, can be heavy)
        near_duplicates = []
        if detect_near_duplicates:
            for doc in documents:
                dup_info = await self._assign_to_duplicate_group(
                    image_id=doc['image_id'],
                    embedding=doc['global_embedding'],
                    threshold=near_duplicate_threshold,
                )
                if dup_info:
                    near_duplicates.append(
                        {
                            'image_id': doc['image_id'],
                            **dup_info,
                        }
                    )

        return {
            'status': 'success',
            'total': total,
            'processed': len(documents),
            'duplicates': len(duplicates),
            'errors_count': len(errors),
            'indexed': indexed,
            'near_duplicates': len(near_duplicates),
            'duplicate_details': duplicates if duplicates else None,
            'error_details': errors if errors else None,
            'near_duplicate_details': near_duplicates if near_duplicates else None,
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

    async def search_faces_by_image(
        self,
        image_bytes: bytes,
        face_index: int = 0,
        top_k: int = 10,
        min_score: float = 0.7,
    ) -> dict[str, Any]:
        """
        Face-to-face identity search.

        Pipeline:
        1. Run SCRFD face detection on query image
        2. Extract ArcFace embedding for selected face
        3. Search visual_search_faces index

        Args:
            image_bytes: Raw image bytes
            face_index: Which detected face to use as query (0-indexed)
            top_k: Number of results to return
            min_score: Minimum similarity score (0.7 recommended for identity)

        Returns:
            dict with query_face info and search results
        """
        # Run face detection + recognition to get embeddings via InferenceService
        result = self.inference.infer_faces(image_bytes)

        if result.get('num_faces', 0) == 0:
            return {
                'status': 'error',
                'error': 'No faces detected',
                'results': [],
                'query_face': None,
            }

        if face_index >= result['num_faces']:
            return {
                'status': 'error',
                'error': f'Face index {face_index} out of range (0-{result["num_faces"] - 1})',
                'results': [],
                'query_face': None,
            }

        # Get query face info and embedding
        query_embedding = np.array(result['embeddings'][face_index])
        face_data = result['faces'][face_index]
        query_face = {
            'box': face_data['box'],
            'landmarks': face_data['landmarks'],
            'score': float(face_data['score']),
            'quality': face_data.get('quality'),
        }

        # Search faces index
        results = await self.opensearch.search_faces(
            query_embedding=query_embedding,
            top_k=top_k,
            min_score=min_score,
        )

        return {
            'status': 'success',
            'query_face': query_face,
            'results': results,
        }

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
