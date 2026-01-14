"""
Track E: Visual Search API Router.

Architecture:
- SYNC endpoints for inference-only operations (predict, embed, detect)
  FastAPI runs sync endpoints in thread pool with automatic backpressure.
- ASYNC endpoints for OpenSearch operations (ingest, search, index)
  OpenSearch client is async for proper concurrent handling.

Endpoints:
- /predict: YOLO detection + global embedding (SYNC)
- /predict_full: YOLO + global + per-box embeddings (SYNC)
- /embed/image: Image embedding only (SYNC)
- /embed/text: Text embedding only (SYNC)
- /detect: Detection only (SYNC)
- /ingest: Ingest image to OpenSearch (ASYNC)
- /search/*: Visual search endpoints (ASYNC)
- /index/*: Index management (ASYNC)
- /cache/*: Cache management (SYNC)
"""

import json
import logging
import time
import uuid

import numpy as np
from fastapi import APIRouter, Body, File, HTTPException, Path, Query, UploadFile
from fastapi.responses import ORJSONResponse

from src.core.dependencies import OpenSearchDep, VisualSearchDep
from src.schemas.detection import ImageMetadata, ModelMetadata
from src.schemas.track_e import (
    DetectOnlyResponse,
    FaceDetection,
    FaceDetectResponse,
    FaceFullResponse,
    FaceIdentifyResponse,
    FaceRecognizeResponse,
    FaceSearchResponse,
    FaceSearchResult,
    ImageEmbeddingResponse,
    PersonFacesResponse,
    PredictFullResponse,
    PredictResponse,
    TextEmbeddingResponse,
    VisualSearchResponse,
)
from src.services.face_identity import get_face_identity_service
from src.services.inference import InferenceService
from src.utils.cache import get_image_cache, get_text_cache


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix='/track_e', tags=['Track E: Visual Search'], default_response_class=ORJSONResponse
)


# =============================================================================
# Detection + Embedding Endpoints (SYNC - like Track D)
# =============================================================================


@router.post('/predict', response_model=PredictResponse, tags=['Track E: Detection + Embedding'])
def predict_track_e(
    image: UploadFile = File(..., description='Image file (JPEG/PNG)'),
):
    """
    Track E: YOLO Detection + Global Image Embedding (Simple Ensemble).

    100% GPU pipeline via DALI preprocessing.
    Uses SYNC endpoint for proper backpressure (like Track D).

    Response format matches other tracks (A, C, D) with additional embedding info.
    Timing is injected by middleware.
    """
    try:
        image_bytes = image.file.read()
        inference_service = InferenceService()

        # Use unified InferenceService (SYNC)
        response = inference_service.infer_track_e(image_bytes, full_pipeline=False)

        return PredictResponse(
            detections=response['detections'],
            num_detections=response['num_detections'],
            image=ImageMetadata(
                width=response['image']['width'], height=response['image']['height']
            ),
            model=ModelMetadata(
                name=response['model']['name'],
                backend=response['model']['backend'],
                device=response['model']['device'],
            ),
            embedding_norm=response.get('embedding_norm'),
            # total_time_ms injected by middleware
        )

    except Exception as e:
        logger.error(f'Track E prediction failed: {e}')
        raise HTTPException(500, f'Prediction failed: {e!s}') from e


@router.post(
    '/predict_full', response_model=PredictFullResponse, tags=['Track E: Detection + Embedding']
)
def predict_track_e_full(
    image: UploadFile = File(..., description='Image file (JPEG/PNG)'),
):
    """
    Track E Full: YOLO Detection + Global + Per-Box Embeddings.

    Full ensemble with box-level embeddings for object search.
    Response format matches other tracks with additional embedding data.
    Timing is injected by middleware.
    """
    try:
        image_bytes = image.file.read()
        inference_service = InferenceService()

        # Use unified InferenceService (SYNC full pipeline)
        response = inference_service.infer_track_e(image_bytes, full_pipeline=True)

        return PredictFullResponse(
            detections=response['detections'],
            num_detections=response['num_detections'],
            image=ImageMetadata(
                width=response['image']['width'], height=response['image']['height']
            ),
            model=ModelMetadata(
                name=response['model']['name'],
                backend=response['model']['backend'],
                device=response['model']['device'],
            ),
            normalized_boxes=response.get('normalized_boxes', []),
            box_embeddings=response.get('box_embeddings', []),
            embedding_norm=response.get('embedding_norm'),
            # total_time_ms injected by middleware
        )

    except Exception as e:
        logger.error(f'Track E full prediction failed: {e}')
        raise HTTPException(500, f'Prediction failed: {e!s}') from e


# =============================================================================
# Batch Processing Endpoints (SYNC - for large photo libraries)
# =============================================================================


@router.post('/predict_batch', tags=['Track E: Batch Processing'])
def predict_track_e_batch(
    images: list[UploadFile] = File(..., description='Multiple image files (JPEG/PNG, max 64)'),
):
    """
    Track E Batch: Process multiple images in a single request.

    **Optimized for large photo libraries (50K+ images)**

    Sends batch of images to DALI ensemble for parallel GPU processing.
    Batch sizes of 16-64 images significantly improve throughput by:
    - Reducing HTTP round-trip overhead
    - Ensuring full DALI/TRT batch utilization (avg batch 8-16 vs 1.5)
    - Maximizing GPU parallelism

    **Performance comparison (RTX A6000):**
    - Single-image requests: ~130 RPS @ 64 clients
    - Batch-32 requests: ~200+ RPS expected

    Args:
        images: List of image files (max 64 per batch)

    Returns:
        List of detection + embedding results
    """
    try:
        if len(images) > 64:
            raise HTTPException(400, f'Max 64 images per batch, got {len(images)}')

        if len(images) == 0:
            raise HTTPException(400, 'At least 1 image required')

        # Read all image bytes
        start_time = time.time()
        images_bytes = [img.file.read() for img in images]

        inference_service = InferenceService()
        results = inference_service.infer_track_e_batch(images_bytes)

        batch_time_ms = (time.time() - start_time) * 1000
        per_image_ms = batch_time_ms / len(images)

        return {
            'status': 'success',
            'batch_size': len(images),
            'results': results,
            'batch_time_ms': round(batch_time_ms, 2),
            'per_image_ms': round(per_image_ms, 2),
            'throughput_ips': round(len(images) / (batch_time_ms / 1000), 1),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Track E batch prediction failed: {e}')
        raise HTTPException(500, f'Batch prediction failed: {e!s}') from e


# =============================================================================
# Individual Model Endpoints (SYNC)
# =============================================================================


@router.post(
    '/embed/image', response_model=ImageEmbeddingResponse, tags=['Track E: Individual Models']
)
def embed_image(
    image: UploadFile = File(..., description='Image file (JPEG/PNG)'),
    index: bool = Query(False, description='Store for later retrieval'),
    image_id: str | None = Query(None, description='Custom ID (auto-generated if not provided)'),
):
    """
    MobileCLIP Image Embedding Only (100% GPU via DALI).

    Uses InferenceService for image encoding.
    """
    try:
        image_bytes = image.file.read()
        inference_service = InferenceService()

        # Use unified InferenceService for image encoding
        embedding = inference_service.encode_image_sync(image_bytes, use_cache=True)
        embedding_norm = float(np.linalg.norm(embedding))

        # Optional indexing (still needs async for OpenSearch)
        stored_id = None
        if index:
            stored_id = image_id or f'img_{uuid.uuid4().hex[:12]}'
            # OpenSearch indexing deferred - return without waiting
            logger.info(f'Index requested for {stored_id} (deferred)')

        return ImageEmbeddingResponse(
            embedding=embedding.tolist(),
            embedding_norm=embedding_norm,
            indexed=index,
            image_id=stored_id,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Image embedding failed: {e}')
        raise HTTPException(500, f'Embedding failed: {e!s}') from e


@router.post(
    '/embed/text', response_model=TextEmbeddingResponse, tags=['Track E: Individual Models']
)
def embed_text(
    text: str = Body(..., embed=True, description='Text to encode'),
    use_cache: bool = Query(True, description='Use embedding cache'),
):
    """
    MobileCLIP Text Embedding Only.

    Pipeline:
    1. Tokenization (CPU - cached singleton tokenizer)
    2. MobileCLIP text encoder (TensorRT)
    """
    try:
        inference_service = InferenceService()
        # Use unified InferenceService for text encoding
        embedding = inference_service.encode_text_sync(text, use_cache)

        return TextEmbeddingResponse(
            embedding=embedding.tolist(), embedding_norm=float(np.linalg.norm(embedding)), text=text
        )

    except Exception as e:
        logger.error(f'Text embedding failed: {e}')
        raise HTTPException(500, f'Embedding failed: {e!s}') from e


@router.post('/detect', response_model=DetectOnlyResponse, tags=['Track E: Individual Models'])
def detect_only(
    image: UploadFile = File(..., description='Image file (JPEG/PNG)'),
):
    """
    YOLO Detection Only (100% GPU via DALI).

    Uses full Track E ensemble but returns only detections.
    Response format matches other tracks (A, C, D).
    Timing is injected by middleware.
    """
    try:
        image_bytes = image.file.read()
        inference_service = InferenceService()

        # Use unified InferenceService (SYNC)
        response = inference_service.infer_track_e(image_bytes, full_pipeline=False)

        return DetectOnlyResponse(
            detections=response['detections'],
            num_detections=response['num_detections'],
            image=ImageMetadata(
                width=response['image']['width'], height=response['image']['height']
            ),
            model=ModelMetadata(
                name=response['model']['name'],
                backend=response['model']['backend'],
                device=response['model']['device'],
            ),
            # total_time_ms injected by middleware
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Detection failed: {e}')
        raise HTTPException(500, f'Detection failed: {e!s}') from e


# =============================================================================
# Ingestion Endpoint (SYNC inference, async OpenSearch deferred)
# =============================================================================


@router.post('/ingest')
async def ingest_image(
    search_service: VisualSearchDep,
    file: UploadFile = File(..., description='Image file (JPEG/PNG)'),
    image_id: str | None = Query(
        None, description='Unique identifier (auto-generated if not provided)'
    ),
    image_path: str | None = Query(None, description='File path for retrieval'),
    metadata: str | None = Query(None, description='JSON string for metadata'),
    skip_duplicates: bool = Query(
        True,
        description='Skip processing if image hash already exists. '
        'Set to False for benchmarks. Default: True (production).',
    ),
    detect_near_duplicates: bool = Query(
        True,
        description='Check for visually similar images and auto-assign to '
        'duplicate groups. Default: True.',
    ),
    near_duplicate_threshold: float = Query(
        0.99,
        ge=0.5,
        le=1.0,
        description='Similarity threshold for near-duplicate grouping. '
        '0.99 = near-identical (default, matches Immich), '
        '0.95 = very similar, 0.90 = similar content.',
    ),
    enable_ocr: bool = Query(True, description='Enable OCR text extraction'),
):
    """
    Ingest image into visual search indexes with auto-routing.

    Pipeline:
    1. Compute imohash for fast duplicate detection (if skip_duplicates=True)
    2. Check if image already exists by hash
    3. Run Track E ensemble (YOLO + MobileCLIP) for embeddings
    4. Route to appropriate indexes:
       - Global embedding → visual_search_global
       - Person detections → visual_search_people
       - Vehicle detections → visual_search_vehicles
    5. Auto-detect near-duplicates and assign to groups (if enabled)

    Args:
        file: Image file (JPEG/PNG)
        image_id: Unique identifier (auto-generated if not provided)
        image_path: File path for retrieval (defaults to image_id)
        metadata: JSON string for custom metadata
        skip_duplicates: If True, skip processing if image hash exists.
                        Set to False for benchmarks. Default: True.
        detect_near_duplicates: If True, check for similar images and
                               auto-assign to duplicate groups. Default: True.
        near_duplicate_threshold: Similarity threshold for grouping (0.5-1.0).
                                 Default: 0.99 (matches Immich).

    Returns:
        Ingestion result with counts per category, or duplicate info if skipped
    """
    try:
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(400, 'File must be an image')

        image_bytes = file.file.read()

        if image_id is None:
            image_id = f'img_{uuid.uuid4().hex[:12]}'

        metadata_dict = json.loads(metadata) if metadata else {}
        metadata_dict['filename'] = file.filename

        result = await search_service.ingest_image(
            image_bytes=image_bytes,
            image_id=image_id,
            image_path=image_path,
            metadata=metadata_dict,
            skip_duplicates=skip_duplicates,
            detect_near_duplicates=detect_near_duplicates,
            near_duplicate_threshold=near_duplicate_threshold,
            enable_ocr=enable_ocr,
        )

        # Handle duplicate detection response
        if result['status'] == 'duplicate':
            return {
                'status': 'duplicate',
                'image_id': image_id,
                'existing_image_id': result.get('existing_image_id'),
                'existing_image_path': result.get('existing_image_path'),
                'imohash': result.get('imohash'),
                'message': result.get('message', 'Image already exists'),
            }

        if result['status'] == 'error':
            raise HTTPException(500, result.get('error', 'Ingestion failed'))

        indexed = result.get('indexed', {})
        response = {
            'status': 'success',
            'image_id': image_id,
            'num_detections': result['num_detections'],
            'embedding_norm': result.get('embedding_norm', 0.0),
            'imohash': result.get('imohash'),
            'indexed': {
                'global': indexed.get('global', False),
                'vehicles': indexed.get('vehicles', 0),
                'people': indexed.get('people', 0),
                'skipped': indexed.get('skipped', 0),
            },
            'message': (
                f'Indexed: global + {indexed.get("vehicles", 0)} vehicles + '
                f'{indexed.get("people", 0)} people'
            ),
        }

        # Include near-duplicate info if found
        if result.get('near_duplicate'):
            response['near_duplicate'] = result['near_duplicate']

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Ingestion failed: {e}')
        raise HTTPException(500, f'Ingestion failed: {e!s}') from e


@router.post('/ingest_batch')
async def ingest_batch(
    search_service: VisualSearchDep,
    files: list[UploadFile] = File(..., description='Image files (JPEG/PNG), max 64'),
    image_ids: str | None = Query(
        None,
        description='Comma-separated IDs matching file order (auto-generated if not provided)',
    ),
    image_paths: str | None = Query(
        None,
        description='Comma-separated file paths matching file order',
    ),
    skip_duplicates: bool = Query(
        True,
        description='Skip processing if image hash exists. Set to False for benchmarks.',
    ),
    detect_near_duplicates: bool = Query(
        False,
        description='Check for visually similar images. Disabled by default for performance.',
    ),
    near_duplicate_threshold: float = Query(
        0.99,
        ge=0.5,
        le=1.0,
        description='Similarity threshold for near-duplicate grouping.',
    ),
    enable_ocr: bool = Query(True, description='Enable OCR text extraction'),
):
    """
    Batch ingest up to 64 images with optimized parallel processing.

    **Performance**: 3-5x faster than individual /ingest calls.
    - HTTP overhead: 1 request vs N requests
    - GPU batching: Triton dynamic batching (16-48 avg batch size)
    - OpenSearch: Bulk indexing vs individual

    **Target throughput**: 300+ RPS with batch sizes of 32-64.

    **Usage** (curl):
    ```bash
    curl -X POST "http://localhost:4603/track_e/ingest_batch" \\
      -F "files=@image1.jpg" \\
      -F "files=@image2.jpg" \\
      -F "files=@image3.jpg"
    ```

    **Usage** (Python):
    ```python
    import httpx
    files = [('files', open(f, 'rb')) for f in image_paths]
    response = httpx.post(url, files=files)
    ```

    Args:
        files: Image files (JPEG/PNG), max 64 per request
        image_ids: Comma-separated IDs (auto-generated if not provided)
        image_paths: Comma-separated file paths
        skip_duplicates: Skip if image hash exists (default: True)
        detect_near_duplicates: Check for similar images (default: False for speed)
        near_duplicate_threshold: Similarity threshold (0.5-1.0)

    Returns:
        Summary with processed count, duplicates, and errors
    """
    import time

    start_time = time.time()

    try:
        # Validate file count
        if len(files) > 64:
            raise HTTPException(400, f'Maximum 64 files per request, got {len(files)}')

        if len(files) == 0:
            raise HTTPException(400, 'No files provided')

        # Parse comma-separated IDs and paths
        ids_list = image_ids.split(',') if image_ids else []
        paths_list = image_paths.split(',') if image_paths else []

        # Validate file types and read bytes
        images_data = []
        for i, file in enumerate(files):
            if not file.content_type or not file.content_type.startswith('image/'):
                raise HTTPException(400, f'File {i} ({file.filename}) must be an image')

            image_bytes = file.file.read()

            # Get ID (from list, or auto-generate)
            image_id = ids_list[i].strip() if i < len(ids_list) else f'img_{uuid.uuid4().hex[:12]}'

            # Get path (from list, filename, or ID)
            image_path = (
                paths_list[i].strip() if i < len(paths_list) else (file.filename or image_id)
            )

            images_data.append((image_bytes, image_id, image_path))

        # Call batch ingestion service
        result = await search_service.ingest_batch(
            images_data=images_data,
            skip_duplicates=skip_duplicates,
            detect_near_duplicates=detect_near_duplicates,
            near_duplicate_threshold=near_duplicate_threshold,
            enable_ocr=enable_ocr,
        )

        elapsed_ms = (time.time() - start_time) * 1000
        images_per_sec = len(files) / (elapsed_ms / 1000) if elapsed_ms > 0 else 0

        return {
            'status': result['status'],
            'total_images': result['total'],
            'processed': result['processed'],
            'duplicates': result['duplicates'],
            'errors': result.get('errors_count', 0),
            'indexed': result['indexed'],
            'near_duplicates': result.get('near_duplicates', 0),
            'timing': {
                'total_ms': round(elapsed_ms, 1),
                'per_image_ms': round(elapsed_ms / len(files), 2) if files else 0,
                'images_per_sec': round(images_per_sec, 1),
            },
            'details': {
                'duplicates': result.get('duplicate_details'),
                'errors': result.get('error_details'),
                'near_duplicates': result.get('near_duplicate_details'),
            }
            if any(
                [
                    result.get('duplicate_details'),
                    result.get('error_details'),
                    result.get('near_duplicate_details'),
                ]
            )
            else None,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Batch ingestion failed: {e}')
        raise HTTPException(500, f'Batch ingestion failed: {e!s}') from e


# =============================================================================
# Search Endpoints (SYNC inference, sync response)
# =============================================================================


@router.post('/search/image', response_model=VisualSearchResponse)
async def search_by_image(
    search_service: VisualSearchDep,
    image: UploadFile = File(..., description='Query image'),
    top_k: int = Query(10, ge=1, le=100, description='Number of results'),
    min_score: float | None = Query(None, ge=0.0, le=1.0, description='Minimum score'),
):
    """
    Image-to-image similarity search.

    Pipeline:
    1. Extract global embedding from query image via MobileCLIP
    2. k-NN search on global_embedding field in OpenSearch

    Args:
        image: Query image file
        top_k: Number of results to return
        min_score: Minimum similarity score threshold
    """
    try:
        start_time = time.time()
        image_bytes = image.file.read()

        # Use VisualSearchService (inference + OpenSearch search)
        results = await search_service.search_by_image(
            image_bytes=image_bytes,
            top_k=top_k,
            min_score=min_score,
        )

        search_time = (time.time() - start_time) * 1000

        return VisualSearchResponse(
            status='success',
            query_type='image',
            results=results,
            total_results=len(results),
            search_time_ms=search_time,
        )

    except Exception as e:
        logger.error(f'Image search failed: {e}')
        raise HTTPException(500, f'Search failed: {e!s}') from e


@router.post('/search/text', response_model=VisualSearchResponse)
async def search_by_text(
    search_service: VisualSearchDep,
    text: str = Body(..., embed=True, description='Text query'),
    top_k: int = Query(10, ge=1, le=100, description='Number of results'),
    min_score: float | None = Query(None, ge=0.0, le=1.0, description='Minimum score'),
    use_cache: bool = Query(True, description='Use text embedding cache'),
):
    """
    Text-to-image search using MobileCLIP text encoder.

    Pipeline:
    1. Encode text to embedding via MobileCLIP text encoder
    2. k-NN search on global_embedding field in OpenSearch

    Args:
        text: Text query string (e.g., "red car on highway")
        top_k: Number of results to return
        min_score: Minimum similarity score threshold
        use_cache: Whether to use text embedding cache
    """
    try:
        start_time = time.time()

        # Use VisualSearchService (inference + OpenSearch search)
        results = await search_service.search_by_text(
            text=text,
            top_k=top_k,
            min_score=min_score,
            use_cache=use_cache,
        )

        search_time = (time.time() - start_time) * 1000

        return VisualSearchResponse(
            status='success',
            query_type='text',
            results=results,
            total_results=len(results),
            search_time_ms=search_time,
        )

    except Exception as e:
        logger.error(f'Text search failed: {e}')
        raise HTTPException(500, f'Search failed: {e!s}') from e


@router.post('/search/object', response_model=VisualSearchResponse)
async def search_by_object(
    search_service: VisualSearchDep,
    image: UploadFile = File(..., description='Query image'),
    box_index: int = Query(0, ge=0, description='Index of detection to search'),
    top_k: int = Query(10, ge=1, le=100, description='Number of results'),
    min_score: float | None = Query(None, ge=0.0, le=1.0, description='Minimum score'),
    class_filter: str | None = Query(None, description='Comma-separated class IDs to filter'),
):
    """
    Object-to-object search with auto-routing to category index.

    Automatically routes to the appropriate index based on detected class:
    - Person (class 0) → visual_search_people
    - Vehicles (2,3,5,7,8) → visual_search_vehicles
    - Other → visual_search_global (fallback)

    Args:
        image: Query image file
        box_index: Index of detected object to use for search (0 = first detection)
        top_k: Number of results to return
        min_score: Minimum similarity score threshold
        class_filter: Comma-separated class IDs to filter (e.g., "2,7" for cars and trucks)
    """
    try:
        start_time = time.time()
        image_bytes = image.file.read()

        class_ids = None
        if class_filter:
            class_ids = [int(c.strip()) for c in class_filter.split(',')]

        result = await search_service.search_by_object(
            image_bytes=image_bytes,
            box_index=box_index,
            top_k=top_k,
            min_score=min_score,
            class_filter=class_ids,
        )

        if result['status'] == 'error':
            raise HTTPException(400, result.get('error', 'Search failed'))

        search_time = (time.time() - start_time) * 1000

        return VisualSearchResponse(
            status='success',
            query_type='object',
            results=result['results'],
            total_results=len(result['results']),
            search_time_ms=search_time,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Object search failed: {e}')
        raise HTTPException(500, f'Search failed: {e!s}') from e


@router.post('/search/vehicles', response_model=VisualSearchResponse)
async def search_vehicles(
    search_service: VisualSearchDep,
    image: UploadFile = File(..., description='Query image with vehicle'),
    vehicle_index: int = Query(0, ge=0, description='Which vehicle to search (0 = first)'),
    top_k: int = Query(10, ge=1, le=100, description='Number of results'),
    min_score: float | None = Query(None, ge=0.0, le=1.0, description='Minimum score'),
    vehicle_type: str | None = Query(
        None, description='Filter: car,motorcycle,bus,truck,boat (comma-separated)'
    ),
):
    """
    Find similar vehicles across all indexed images.

    Like "Find all red cars" or "Show me motorcycles like this one".

    Vehicle classes (COCO):
    - 2 = car
    - 3 = motorcycle
    - 5 = bus
    - 7 = truck
    - 8 = boat

    Args:
        image: Query image containing a vehicle
        vehicle_index: Which detected vehicle to use (0 = first vehicle found)
        top_k: Number of results to return
        min_score: Minimum similarity score threshold
        vehicle_type: Filter by vehicle type (e.g., "car,truck")
    """
    try:
        start_time = time.time()
        image_bytes = image.file.read()

        # Parse vehicle type filter
        class_filter = None
        if vehicle_type:
            type_to_class = {'car': 2, 'motorcycle': 3, 'bus': 5, 'truck': 7, 'boat': 8}
            class_filter = [
                type_to_class[t.strip().lower()]
                for t in vehicle_type.split(',')
                if t.strip().lower() in type_to_class
            ]

        result = await search_service.search_vehicles(
            image_bytes=image_bytes,
            box_index=vehicle_index,
            top_k=top_k,
            min_score=min_score,
            class_filter=class_filter,
        )

        if result['status'] == 'error':
            raise HTTPException(400, result.get('error', 'Search failed'))

        search_time = (time.time() - start_time) * 1000

        return VisualSearchResponse(
            status='success',
            query_type='vehicle',
            results=result['results'],
            total_results=len(result['results']),
            search_time_ms=search_time,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Vehicle search failed: {e}')
        raise HTTPException(500, f'Search failed: {e!s}') from e


@router.post('/search/people', response_model=VisualSearchResponse)
async def search_people(
    search_service: VisualSearchDep,
    image: UploadFile = File(..., description='Query image with person'),
    person_index: int = Query(0, ge=0, description='Which person to search (0 = first)'),
    top_k: int = Query(10, ge=1, le=100, description='Number of results'),
    min_score: float | None = Query(None, ge=0.0, le=1.0, description='Minimum score'),
):
    """
    Find similar people by appearance (clothing, pose, context).

    Like "Find people wearing similar outfits" - matches visual appearance.
    NOT identity matching (use /search/faces for that - requires ArcFace).

    Args:
        image: Query image containing a person
        person_index: Which detected person to use (0 = first person found)
        top_k: Number of results to return
        min_score: Minimum similarity score threshold
    """
    try:
        start_time = time.time()
        image_bytes = image.file.read()

        result = await search_service.search_people(
            image_bytes=image_bytes,
            box_index=person_index,
            top_k=top_k,
            min_score=min_score,
        )

        if result['status'] == 'error':
            raise HTTPException(400, result.get('error', 'Search failed'))

        search_time = (time.time() - start_time) * 1000

        return VisualSearchResponse(
            status='success',
            query_type='person',
            results=result['results'],
            total_results=len(result['results']),
            search_time_ms=search_time,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'People search failed: {e}')
        raise HTTPException(500, f'Search failed: {e!s}') from e


# =============================================================================
# Index Management Endpoints (lightweight, keep sync)
# =============================================================================


@router.get('/index/stats')
async def get_index_stats(search_service: VisualSearchDep):
    """
    Get statistics for all visual search indexes.

    Returns counts and sizes for:
    - visual_search_global: Whole image embeddings
    - visual_search_vehicles: Vehicle detections
    - visual_search_people: Person detections
    - visual_search_faces: Face embeddings (future)
    """
    try:
        return await search_service.get_index_stats()
    except Exception as e:
        logger.error(f'Failed to get index stats: {e}')
        return {'status': 'error', 'error': str(e)}


@router.post('/index/create')
async def create_index(
    search_service: VisualSearchDep,
    force_recreate: bool = Query(False, description='Delete existing indexes first'),
):
    """
    Create all visual search indexes.

    Creates:
    - visual_search_global: Whole image similarity
    - visual_search_vehicles: Vehicle detections
    - visual_search_people: Person detections
    - visual_search_faces: Face identity matching (future)

    Args:
        force_recreate: Whether to delete existing indexes before creating
    """
    try:
        results = await search_service.setup_index(force_recreate=force_recreate)
        all_success = all(results.values())
        return {
            'status': 'success' if all_success else 'partial',
            'indexes': results,
            'message': 'All indexes created' if all_success else 'Some indexes failed',
        }
    except Exception as e:
        logger.error(f'Failed to create indexes: {e}')
        raise HTTPException(500, f'Failed to create indexes: {e!s}') from e


@router.delete('/index')
async def delete_index(search_service: VisualSearchDep):
    """Delete all visual search indexes."""
    try:
        results = await search_service.delete_index()
        all_success = all(results.values())
        return {
            'status': 'success' if all_success else 'partial',
            'indexes': results,
            'message': 'All indexes deleted' if all_success else 'Some indexes failed',
        }
    except Exception as e:
        logger.error(f'Failed to delete indexes: {e}')
        raise HTTPException(500, f'Failed to delete indexes: {e!s}') from e


# =============================================================================
# Clustering Endpoints (FAISS IVF - Industry Standard)
# =============================================================================


@router.post('/clusters/train/{index_name}', tags=['Track E: Clustering'])
async def train_clusters(
    search_service: VisualSearchDep,
    index_name: str,
    n_clusters: int | None = Query(None, description='Number of clusters (uses default if None)'),
    max_samples: int | None = Query(None, description='Max embeddings for training (None = all)'),
):
    """
    Train FAISS IVF clustering for an index.

    This is typically a one-time operation or run periodically for rebalancing.
    Training time scales with embedding count:
    - 100K: ~2s
    - 1M: ~15s
    - 10M: ~120s

    Args:
        index_name: Which index to train (global, vehicles, people, faces)
        n_clusters: Number of clusters (default: 1024 for global/faces, 512 for people, 256 for vehicles)
        max_samples: Max samples for training (None = use all)
    """
    try:
        return await search_service.train_clusters(
            index_name=index_name,
            n_clusters=n_clusters,
            max_samples=max_samples,
        )
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:
        logger.error(f'Cluster training failed: {e}')
        raise HTTPException(500, f'Training failed: {e!s}') from e


@router.post('/clusters/assign/{index_name}', tags=['Track E: Clustering'])
async def assign_unclustered(
    search_service: VisualSearchDep,
    index_name: str,
):
    """
    Assign clusters to unclustered documents in an index.

    Finds all documents without cluster_id and assigns them to the nearest
    centroid. This is useful after ingesting new images without clustering enabled.

    Time complexity: ~0.1ms per embedding (very fast).

    Args:
        index_name: Which index to assign (global, vehicles, people, faces)
    """
    try:
        return await search_service.assign_unclustered(index_name=index_name)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:
        logger.error(f'Cluster assignment failed: {e}')
        raise HTTPException(500, f'Assignment failed: {e!s}') from e


@router.get('/clusters/stats/{index_name}', tags=['Track E: Clustering'])
async def get_cluster_stats(
    search_service: VisualSearchDep,
    index_name: str,
):
    """
    Get detailed cluster statistics for an index.

    Returns:
    - Per-cluster counts and distances
    - Cluster balance metrics
    - Training metadata (when trained, n_vectors)

    Args:
        index_name: Which index (global, vehicles, people, faces)
    """
    try:
        return await search_service.get_cluster_stats(index_name=index_name)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:
        logger.error(f'Failed to get cluster stats: {e}')
        raise HTTPException(500, f'Failed to get stats: {e!s}') from e


@router.get('/clusters/balance/{index_name}', tags=['Track E: Clustering'])
async def check_cluster_balance(
    search_service: VisualSearchDep,
    index_name: str,
):
    """
    Check if clusters need rebalancing.

    Returns recommendation based on:
    - Imbalance ratio (max cluster / min cluster size)
    - Empty cluster percentage
    - New data since last training

    Args:
        index_name: Which index to check
    """
    try:
        return await search_service.check_cluster_balance(index_name=index_name)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:
        logger.error(f'Failed to check balance: {e}')
        raise HTTPException(500, f'Failed to check balance: {e!s}') from e


@router.post('/clusters/rebalance/{index_name}', tags=['Track E: Clustering'])
async def rebalance_clusters(
    search_service: VisualSearchDep,
    index_name: str,
):
    """
    Force rebalance clusters by re-training from current data.

    This extracts all embeddings from OpenSearch and re-trains the FAISS index.
    Use when check_balance indicates rebalancing is needed.

    Args:
        index_name: Which index to rebalance
    """
    try:
        return await search_service.rebalance_clusters(index_name=index_name)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:
        logger.error(f'Cluster rebalance failed: {e}')
        raise HTTPException(500, f'Rebalance failed: {e!s}') from e


@router.get('/clusters/{index_name}/{cluster_id}', tags=['Track E: Clustering'])
async def get_cluster_members(
    search_service: VisualSearchDep,
    index_name: str,
    cluster_id: int,
    page: int = Query(0, ge=0, description='Page number'),
    size: int = Query(50, ge=1, le=200, description='Page size'),
):
    """
    Get members of a specific cluster (like a Google Photos album).

    Returns documents sorted by distance to centroid (most representative first).

    Args:
        index_name: Which index (global, vehicles, people, faces)
        cluster_id: The cluster ID to retrieve
        page: Page number (0-indexed)
        size: Page size (max 200)
    """
    try:
        return await search_service.get_cluster_members(
            index_name=index_name,
            cluster_id=cluster_id,
            page=page,
            size=size,
        )
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:
        logger.error(f'Failed to get cluster members: {e}')
        raise HTTPException(500, f'Failed to get members: {e!s}') from e


@router.get('/albums', tags=['Track E: Clustering'])
async def list_albums(
    search_service: VisualSearchDep,
    min_size: int = Query(5, ge=1, description='Minimum cluster size to include'),
):
    """
    List auto-generated albums (clusters) from global index.

    Like Google Photos "Things" or "Places" - automatically grouped similar images.

    Args:
        min_size: Only include clusters with at least this many images
    """
    try:
        return await search_service.list_albums(min_size=min_size)
    except Exception as e:
        logger.error(f'Failed to list albums: {e}')
        raise HTTPException(500, f'Failed to list albums: {e!s}') from e


# =============================================================================
# Cluster Maintenance Endpoints (Automatic Rebalancing)
# =============================================================================


@router.get('/maintenance/status', tags=['Track E: Maintenance'])
async def get_maintenance_status(search_service: VisualSearchDep):
    """
    Check rebalancing status for all indexes.

    Returns which indexes need rebalancing based on:
    - Amount of new data since training
    - Cluster size imbalance
    - Empty cluster ratio

    Use this to monitor cluster health and decide when to rebalance.
    """
    try:
        from src.services.cluster_maintenance import ClusterMaintenanceService

        service = ClusterMaintenanceService(search_service.opensearch)
        return await service.check_all_indexes()
    except Exception as e:
        logger.error(f'Failed to get maintenance status: {e}')
        raise HTTPException(500, f'Failed to get status: {e!s}') from e


@router.post('/maintenance/run', tags=['Track E: Maintenance'])
async def run_maintenance(
    search_service: VisualSearchDep,
    force: bool = Query(False, description='Force rebalance even if not needed'),
    pattern: str = Query(
        'medium_volume',
        description='Ingestion pattern: low_volume, medium_volume, high_volume, very_high_volume',
    ),
):
    """
    Run maintenance check and rebalance indexes that need it.

    This checks all indexes and automatically rebalances any that exceed
    the configured thresholds. Use this as a scheduled task or cron job.

    Ingestion patterns determine rebalance thresholds:
    - low_volume: < 1K images/day, rebalance when 50% new data
    - medium_volume: 1K-10K images/day, rebalance when 40% new data
    - high_volume: 10K-100K images/day, rebalance when 30% new data
    - very_high_volume: 100K+ images/day, rebalance when 20% new data

    Args:
        force: Force rebalance all indexes regardless of thresholds
        pattern: Ingestion pattern for determining thresholds
    """
    try:
        from src.services.cluster_maintenance import ClusterMaintenanceService, IngestionPattern

        # Parse pattern
        pattern_map = {
            'low_volume': IngestionPattern.LOW_VOLUME,
            'medium_volume': IngestionPattern.MEDIUM_VOLUME,
            'high_volume': IngestionPattern.HIGH_VOLUME,
            'very_high_volume': IngestionPattern.VERY_HIGH_VOLUME,
        }

        if pattern.lower() not in pattern_map:
            raise HTTPException(400, f'Invalid pattern. Must be one of: {list(pattern_map.keys())}')

        ingestion_pattern = pattern_map[pattern.lower()]
        service = ClusterMaintenanceService(search_service.opensearch, pattern=ingestion_pattern)

        return await service.check_and_rebalance_all(force=force)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Maintenance run failed: {e}')
        raise HTTPException(500, f'Maintenance failed: {e!s}') from e


@router.post('/maintenance/rebalance-after-bulk', tags=['Track E: Maintenance'])
async def rebalance_after_bulk_ingestion(
    search_service: VisualSearchDep,
    indexes: str = Query(
        'global',
        description='Comma-separated indexes to rebalance: global,vehicles,people,faces',
    ),
):
    """
    Rebalance specific indexes after a bulk ingestion.

    Call this after ingesting a large batch of images (e.g., 10K+ photos).
    Only rebalances if thresholds are exceeded.

    Args:
        indexes: Which indexes to check and rebalance

    Example workflow:
    1. POST /track_e/ingest (many times for bulk upload)
    2. POST /track_e/maintenance/rebalance-after-bulk?indexes=global,vehicles
    """
    try:
        from src.clients.opensearch import IndexName
        from src.services.cluster_maintenance import ClusterMaintenanceService
        from src.services.clustering import ClusterIndex

        # Parse indexes
        index_list = [idx.strip().lower() for idx in indexes.split(',')]

        name_map = {
            'global': (ClusterIndex.GLOBAL, IndexName.GLOBAL),
            'vehicles': (ClusterIndex.VEHICLES, IndexName.VEHICLES),
            'people': (ClusterIndex.PEOPLE, IndexName.PEOPLE),
            'faces': (ClusterIndex.FACES, IndexName.FACES),
        }

        invalid = [idx for idx in index_list if idx not in name_map]
        if invalid:
            raise HTTPException(
                400, f'Invalid indexes: {invalid}. Must be: {list(name_map.keys())}'
            )

        service = ClusterMaintenanceService(search_service.opensearch)
        results = {}

        for idx in index_list:
            cluster_idx, os_idx = name_map[idx]
            result = await service.check_and_rebalance(cluster_idx, os_idx)
            results[idx] = result

        rebalanced = sum(1 for r in results.values() if r.get('action') == 'rebalanced')

        return {
            'status': 'complete',
            'indexes_checked': len(index_list),
            'indexes_rebalanced': rebalanced,
            'results': results,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Post-bulk rebalance failed: {e}')
        raise HTTPException(500, f'Rebalance failed: {e!s}') from e


# =============================================================================
# Cache Management Endpoints (sync, lightweight)
# =============================================================================


@router.get('/cache/stats')
def get_cache_stats():
    """Get embedding cache statistics."""
    try:
        image_cache = get_image_cache()
        text_cache = get_text_cache()

        return {
            'status': 'success',
            'image_cache': {'size': len(image_cache), 'max_size': image_cache.maxsize},
            'text_cache': {'size': len(text_cache), 'max_size': text_cache.maxsize},
        }

    except Exception as e:
        logger.error(f'Failed to get cache stats: {e}')
        raise HTTPException(500, f'Failed to get stats: {e!s}') from e


@router.post('/cache/clear')
def clear_caches():
    """Clear all embedding caches."""
    try:
        image_cache = get_image_cache()
        text_cache = get_text_cache()

        image_cache.clear()
        text_cache.clear()

        return {'status': 'success', 'message': 'All caches cleared'}

    except Exception as e:
        logger.error(f'Failed to clear caches: {e}')
        raise HTTPException(500, f'Failed to clear caches: {e!s}') from e


# =============================================================================
# Face Detection & Recognition Endpoints (SYNC - GPU Pipeline)
# =============================================================================


@router.post('/faces/detect', response_model=FaceDetectResponse, tags=['Track E: Face Recognition'])
def detect_faces(
    image: UploadFile = File(..., description='Image file (JPEG/PNG)'),
    detector: str = Query(
        'yolo11',
        pattern='^(scrfd|yolo11)$',
        description='Face detection model: yolo11 (default, batching enabled) or scrfd (higher accuracy)',
    ),
    confidence: float = Query(
        0.5,
        ge=0.1,
        le=0.99,
        description='Minimum face detection confidence threshold',
    ),
):
    """
    Detect faces in image using configurable face detector.

    100% GPU pipeline via DALI preprocessing:
    1. GPU JPEG decode (nvJPEG via DALI)
    2. GPU preprocessing for face detection (640x640)
    3. Face detection with GPU NMS (End2End TensorRT)
    4. Returns face boxes, 5-point landmarks, and confidence scores

    Args:
        image: Image file (JPEG/PNG)
        detector: Face detection model to use:
            - 'yolo11' (default): YOLO11-face End2End with batching support
            - 'scrfd': SCRFD-10G, higher accuracy, better for small faces
        confidence: Minimum detection confidence threshold (0.1-0.99)

    Response includes normalized [0,1] coordinates for boxes and landmarks.
    """
    try:
        image_bytes = image.file.read()
        inference_service = InferenceService()

        if detector == 'yolo11':
            result = inference_service.infer_faces_yolo11(image_bytes, confidence=confidence)
        else:
            result = inference_service.infer_faces(image_bytes)
        orig_h, orig_w = result['orig_shape']

        faces = [
            FaceDetection(
                box=f['box'],
                landmarks=f['landmarks'],
                score=f['score'],
                quality=f.get('quality'),
            )
            for f in result['faces']
        ]

        return FaceDetectResponse(
            num_faces=result['num_faces'],
            faces=faces,
            image=ImageMetadata(width=orig_w, height=orig_h),
        )

    except Exception as e:
        logger.error(f'Face detection failed: {e}')
        raise HTTPException(500, f'Face detection failed: {e!s}') from e


@router.post(
    '/faces/recognize', response_model=FaceRecognizeResponse, tags=['Track E: Face Recognition']
)
def recognize_faces(
    image: UploadFile = File(..., description='Image file (JPEG/PNG)'),
    detector: str = Query(
        'yolo11',
        pattern='^(scrfd|yolo11)$',
        description='Face detection model: yolo11 (default, batching enabled) or scrfd (higher accuracy)',
    ),
    confidence: float = Query(
        0.5,
        ge=0.1,
        le=0.99,
        description='Minimum face detection confidence threshold',
    ),
):
    """
    Detect faces and extract ArcFace identity embeddings.

    100% GPU pipeline:
    1. GPU JPEG decode (nvJPEG via DALI)
    2. Face detection + GPU NMS (End2End TensorRT)
    3. Face alignment using MTCNN-style cropping
    4. ArcFace embedding extraction (512-dim L2-normalized)

    Args:
        image: Image file (JPEG/PNG)
        detector: Face detection model to use:
            - 'yolo11' (default): YOLO11-face End2End with batching support
            - 'scrfd': SCRFD-10G, higher accuracy, better for small faces
        confidence: Minimum detection confidence threshold (0.1-0.99)

    Use embeddings for:
    - Face verification (1:1 matching) - cosine similarity > 0.6
    - Face identification (1:N search) - OpenSearch k-NN

    Response includes normalized [0,1] coordinates and 512-dim embeddings per face.
    """
    try:
        image_bytes = image.file.read()
        inference_service = InferenceService()

        if detector == 'yolo11':
            result = inference_service.infer_faces_yolo11(image_bytes, confidence=confidence)
        else:
            result = inference_service.infer_faces(image_bytes)
        orig_h, orig_w = result['orig_shape']

        faces = [
            FaceDetection(
                box=f['box'],
                landmarks=f['landmarks'],
                score=f['score'],
                quality=f.get('quality'),
            )
            for f in result['faces']
        ]

        return FaceRecognizeResponse(
            num_faces=result['num_faces'],
            faces=faces,
            embeddings=result['embeddings'],
            image=ImageMetadata(width=orig_w, height=orig_h),
        )

    except Exception as e:
        logger.error(f'Face recognition failed: {e}')
        raise HTTPException(500, f'Face recognition failed: {e!s}') from e


@router.post('/faces/full', response_model=FaceFullResponse, tags=['Track E: Face Recognition'])
def predict_faces_full(
    image: UploadFile = File(..., description='Image file (JPEG/PNG)'),
    face_detector: str = Query(
        'yolo11',
        pattern='^(scrfd|yolo11)$',
        description='Face detection model: yolo11 (default, batching enabled) or scrfd (higher accuracy)',
    ),
    confidence: float = Query(
        0.5,
        ge=0.1,
        le=0.99,
        description='Minimum face detection confidence',
    ),
):
    """
    Unified pipeline: YOLO + Face Detection + MobileCLIP + ArcFace.

    All processing happens in Triton via quad-branch ensemble:
    1. GPU JPEG decode (nvJPEG via DALI)
    2. Quad-branch preprocessing (YOLO 640, CLIP 256, face detector input, HD original)
    3. YOLO object detection (parallel with 4, 5)
    4. MobileCLIP global embedding (parallel with 3, 5)
    5. Face detection + GPU NMS (End2End TensorRT, parallel with 3, 4)
    6. ArcFace face embeddings (depends on 5)

    Args:
        image: Image file (JPEG/PNG)
        face_detector: Face detection model to use:
            - 'yolo11' (default): YOLO11-face End2End with batching support
            - 'scrfd': Higher accuracy, better for small faces
        confidence: Minimum face detection confidence threshold (0.1-0.99)

    Returns:
    - YOLO detections (objects in image)
    - Face detections with landmarks
    - ArcFace 512-dim embeddings per face
    - MobileCLIP 512-dim global image embedding
    """
    try:
        image_bytes = image.file.read()
        inference_service = InferenceService()

        if face_detector == 'yolo11':
            result = inference_service.infer_faces_full_yolo11(image_bytes, confidence)
        else:
            result = inference_service.infer_faces_full(image_bytes, confidence)
        orig_h, orig_w = result['orig_shape']

        faces = [
            FaceDetection(
                box=f['box'],
                landmarks=f['landmarks'],
                score=f['score'],
                quality=f.get('quality'),
            )
            for f in result['faces']
        ]

        # Convert image_embedding to list if it's a numpy array
        image_embedding = result.get('image_embedding')
        if image_embedding is not None and hasattr(image_embedding, 'tolist'):
            image_embedding = image_embedding.tolist()

        return FaceFullResponse(
            detections=result['detections'],
            num_detections=result['num_detections'],
            num_faces=result['num_faces'],
            faces=faces,
            face_embeddings=result['face_embeddings'],
            image_embedding=image_embedding,
            embedding_norm=result['embedding_norm'],
            image=ImageMetadata(width=orig_w, height=orig_h),
            model=ModelMetadata(
                name='yolo_face_clip_ensemble',
                backend='triton',
                device='gpu',
            ),
        )

    except Exception as e:
        logger.error(f'Full face pipeline failed: {e}')
        raise HTTPException(500, f'Full face pipeline failed: {e!s}') from e


@router.post('/faces/full_batch', response_model=None, tags=['Track E: Face Recognition'])
async def predict_faces_full_batch(
    files: list[UploadFile] = File(..., description='Multiple image files (max 64)'),
    face_detector: str = Query(
        'yolo11',
        pattern='^(scrfd|yolo11)$',
        description='Face detection model: yolo11 (default, batching enabled) or scrfd',
    ),
    confidence: float = Query(
        0.5, ge=0.1, le=0.99, description='Minimum face detection confidence'
    ),
):
    """
    Batch unified pipeline: YOLO + Face + MobileCLIP + ArcFace.

    Process up to 64 images in a single request for maximum throughput.
    Uses YOLO11-Face End2End for optimal batching performance.

    Args:
        files: Multiple image files (JPEG/PNG), max 64 per request
        face_detector: Face detection model (yolo11 default with batching, or scrfd)
        confidence: Minimum face detection confidence

    Returns:
        Batch results with per-image YOLO detections, faces, and embeddings
    """
    if len(files) > 64:
        raise HTTPException(400, 'Maximum 64 images per batch request')

    results = []
    inference_service = InferenceService()

    for file in files:
        try:
            # Validate content type
            if file.content_type not in ('image/jpeg', 'image/png'):
                results.append(
                    {
                        'status': 'error',
                        'error': f'Invalid content type: {file.content_type}',
                        'filename': file.filename,
                    }
                )
                continue

            image_bytes = await file.read()

            if face_detector == 'yolo11':
                result = inference_service.infer_faces_full_yolo11(image_bytes, confidence)
            else:
                result = inference_service.infer_faces_full(image_bytes, confidence)

            result['filename'] = file.filename
            results.append(result)

        except Exception as e:
            results.append({'status': 'error', 'error': str(e), 'filename': file.filename})

    return {
        'status': 'success',
        'total_images': len(files),
        'processed': len([r for r in results if r.get('status') != 'error']),
        'errors': len([r for r in results if r.get('status') == 'error']),
        'results': results,
    }


@router.post('/faces/verify', tags=['Track E: Face Recognition'])
def verify_faces(
    image1: UploadFile = File(..., description='First image with face'),
    image2: UploadFile = File(..., description='Second image with face'),
    threshold: float = Query(0.6, ge=0.0, le=1.0, description='Similarity threshold for match'),
    detector: str = Query(
        'scrfd',
        pattern='^(scrfd|yolo11)$',
        description='Face detection model: scrfd (default, higher accuracy) or yolo11 (faster)',
    ),
):
    """
    Verify if two images contain the same person (1:1 verification).

    Extracts ArcFace embeddings from both images and compares using cosine similarity.

    Args:
        image1: First image with face
        image2: Second image with face
        threshold: Similarity threshold for match decision
        detector: Face detection model to use:
            - 'scrfd' (default): SCRFD-10G, higher accuracy, better for small faces
            - 'yolo11': YOLO11-face, faster inference, good for real-time applications

    Threshold guidelines:
    - 0.6: High confidence (recommended for security)
    - 0.5: Balanced precision/recall
    - 0.4: More permissive (may have false positives)

    Returns match decision, similarity score, and face info from both images.
    """
    try:
        image1_bytes = image1.file.read()
        image2_bytes = image2.file.read()
        inference_service = InferenceService()

        if detector == 'yolo11':
            result1 = inference_service.infer_faces_yolo11(image1_bytes)
            result2 = inference_service.infer_faces_yolo11(image2_bytes)
        else:
            result1 = inference_service.infer_faces(image1_bytes)
            result2 = inference_service.infer_faces(image2_bytes)

        if result1['num_faces'] == 0:
            raise HTTPException(400, 'No face detected in first image')
        if result2['num_faces'] == 0:
            raise HTTPException(400, 'No face detected in second image')

        # Use first face from each image
        emb1 = np.array(result1['embeddings'][0])
        emb2 = np.array(result2['embeddings'][0])

        # Cosine similarity (embeddings are L2-normalized)
        similarity = float(np.dot(emb1, emb2))
        is_match = similarity >= threshold

        return {
            'status': 'success',
            'match': is_match,
            'similarity': round(similarity, 4),
            'threshold': threshold,
            'image1': {
                'num_faces': result1['num_faces'],
                'face_used': result1['faces'][0],
            },
            'image2': {
                'num_faces': result2['num_faces'],
                'face_used': result2['faces'][0],
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Face verification failed: {e}')
        raise HTTPException(500, f'Face verification failed: {e!s}') from e


@router.post('/faces/ingest', tags=['Track E: Face Recognition'])
async def ingest_faces(
    search_service: VisualSearchDep,
    image: UploadFile = File(..., description='Image file (JPEG/PNG)'),
    image_id: str | None = Query(
        None, description='Unique identifier (auto-generated if not provided)'
    ),
    image_path: str | None = Query(None, description='File path for retrieval'),
    person_name: str | None = Query(None, description='Optional name/label for faces'),
):
    """
    Ingest faces from image into face identity database.

    Pipeline:
    1. Run SCRFD face detection
    2. Extract ArcFace embeddings for each face
    3. Index to visual_search_faces

    Args:
        image: Image file (JPEG/PNG)
        image_id: Unique identifier (auto-generated if not provided)
        image_path: File path for retrieval (defaults to image_id)
        person_name: Optional name/label for all faces in image

    Returns:
        Ingestion result with face count
    """
    try:
        if not image.content_type or not image.content_type.startswith('image/'):
            raise HTTPException(400, 'File must be an image')

        image_bytes = image.file.read()

        if image_id is None:
            image_id = f'img_{uuid.uuid4().hex[:12]}'

        result = await search_service.ingest_faces(
            image_bytes=image_bytes,
            image_id=image_id,
            image_path=image_path,
            person_name=person_name,
        )

        if result['status'] == 'error':
            raise HTTPException(500, result.get('error', 'Face ingestion failed'))

        return {
            'status': 'success',
            'image_id': image_id,
            'num_faces': result['num_faces'],
            'indexed': result['indexed'],
            'message': f'Indexed {result["indexed"]} faces',
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Face ingestion failed: {e}')
        raise HTTPException(500, f'Face ingestion failed: {e!s}') from e


@router.post('/unified', tags=['Track E: Unified Pipeline'])
def predict_unified(
    image: UploadFile = File(..., description='Image file (JPEG/PNG)'),
):
    """
    Unified pipeline: YOLO + MobileCLIP + person-only face detection.

    **More efficient than /faces/full** - runs SCRFD only on person bounding
    box crops instead of full image. Returns:

    - YOLO object detections (all classes)
    - Global MobileCLIP embedding (512-dim)
    - Per-box MobileCLIP embeddings
    - Face detections ONLY from person crops
    - Face ArcFace embeddings (512-dim)
    - Which person box each face belongs to

    Benefits:
    - ~2x faster than full-image face detection
    - Fewer false positives (faces only from person regions)
    - Face-to-person association included
    """
    try:
        image_bytes = image.file.read()
        inference_service = InferenceService()

        result = inference_service.infer_unified(image_bytes)

        return {
            'status': 'success',
            'num_detections': result['num_dets'],
            'detections': [
                {
                    'box': result['normalized_boxes'][i].tolist(),
                    'score': float(result['scores'][i]),
                    'class_id': int(result['classes'][i]),
                }
                for i in range(result['num_dets'])
            ],
            'global_embedding_norm': float(np.linalg.norm(result['global_embedding'])),
            'num_faces': result['num_faces'],
            'faces': [
                {
                    'box': result['face_boxes'][i].tolist(),
                    'landmarks': result['face_landmarks'][i].tolist(),
                    'score': float(result['face_scores'][i]),
                    'person_idx': int(result['face_person_idx'][i]),
                }
                for i in range(result['num_faces'])
            ],
            'image': {
                'width': result['orig_shape'][1],
                'height': result['orig_shape'][0],
            },
            'track': 'E_unified',
            'preprocessing': 'gpu_dali',
            'pipeline': 'person_only_face_detection',
        }

    except Exception as e:
        logger.error(f'Unified pipeline failed: {e}')
        raise HTTPException(500, f'Unified pipeline failed: {e!s}') from e


@router.post('/analyze', tags=['Track E: Unified Pipeline'])
def analyze_complete(
    image: UploadFile = File(..., description='Image file (JPEG/PNG)'),
    face_model: str = Query(
        'scrfd',
        description='Face detection model: "scrfd" (person crops) or "yolo11" (full image)',
        enum=['scrfd', 'yolo11'],
    ),
):
    """
    Complete unified analysis: YOLO + MobileCLIP + Faces + OCR.

    **All-in-one endpoint** that returns every analysis result in a single request:

    - **Object Detection**: YOLO bounding boxes (all COCO classes)
    - **Image Embeddings**: MobileCLIP global (512-dim) and per-box embeddings
    - **Face Detection**: Faces with landmarks and ArcFace embeddings (512-dim)
    - **OCR**: Text detection and recognition (PP-OCRv5)

    **Face Model Selection**:
    - `scrfd` (default): SCRFD detection on person crops only (faster, fewer false positives)
    - `yolo11`: YOLO11-face detection on full image (may detect more faces in crowds)

    **GPU Pipeline**: All processing runs on GPU via Triton ensembles and DALI preprocessing.
    """
    try:
        from src.clients.triton_client import get_triton_client
        from src.config import get_settings

        image_bytes = image.file.read()
        settings = get_settings()
        client = get_triton_client(settings.triton_url)

        result = client.infer_unified_complete(image_bytes, face_model=face_model)

        # Format detection results
        detections = [
            {
                'box': result['normalized_boxes'][i].tolist(),
                'score': float(result['scores'][i]),
                'class_id': int(result['classes'][i]),
            }
            for i in range(result['num_dets'])
        ]

        # Format face results
        faces = [
            {
                'box': result['face_boxes'][i].tolist(),
                'landmarks': result['face_landmarks'][i].tolist(),
                'score': float(result['face_scores'][i]),
                'person_idx': int(result['face_person_idx'][i])
                if result['face_person_idx'][i] >= 0
                else None,
            }
            for i in range(result['num_faces'])
        ]

        # Format OCR results
        ocr_results = [
            {
                'text': result['texts'][i] if i < len(result['texts']) else '',
                'box_normalized': result['text_boxes_normalized'][i].tolist(),
                'box_quad': result['text_boxes'][i].tolist()
                if i < len(result['text_boxes'])
                else [],
                'det_score': float(result['text_det_scores'][i]),
                'rec_score': float(result['text_rec_scores'][i]),
            }
            for i in range(result['num_texts'])
        ]

        return {
            'status': 'success',
            # Detection
            'num_detections': result['num_dets'],
            'detections': detections,
            # Embeddings
            'global_embedding_norm': float(np.linalg.norm(result['global_embedding'])),
            'global_embedding_dim': len(result['global_embedding']),
            'num_box_embeddings': result['num_dets'],
            # Faces
            'num_faces': result['num_faces'],
            'faces': faces,
            'face_model_used': result['face_model_used'],
            # OCR
            'num_texts': result['num_texts'],
            'texts': ocr_results,
            # Metadata
            'image': {
                'width': result['orig_shape'][1],
                'height': result['orig_shape'][0],
            },
            'track': 'E_analyze',
            'preprocessing': 'gpu_dali',
            'pipeline': 'unified_complete',
        }

    except Exception as e:
        logger.error(f'Complete analysis failed: {e}')
        raise HTTPException(500, f'Complete analysis failed: {e!s}') from e


@router.post('/analyze/ingest', tags=['Track E: Unified Pipeline'])
async def analyze_and_ingest(
    search_service: VisualSearchDep,
    image: UploadFile = File(..., description='Image file (JPEG/PNG)'),
    image_id: str | None = Query(
        None, description='Custom image ID (auto-generated if not provided)'
    ),
    image_path: str | None = Query(None, description='Original file path for reference'),
    face_model: str = Query(
        'scrfd',
        description='Face detection model: "scrfd" (person crops) or "yolo11" (full image)',
        enum=['scrfd', 'yolo11'],
    ),
    index_name: str = Query('track_e_visual', description='Index for global embeddings'),
    face_index_name: str = Query('track_e_faces', description='Index for face embeddings'),
):
    """
    Analyze image completely and ingest all embeddings into OpenSearch.

    **Ingests into multiple indices**:
    - Global image embedding → main visual search index
    - Per-face embeddings → face search index

    Returns the full analysis results plus ingestion confirmation.
    """
    try:
        import hashlib
        import time

        from src.clients.triton_client import get_triton_client
        from src.config import get_settings

        image_bytes = image.file.read()
        settings = get_settings()
        client = get_triton_client(settings.triton_url)

        # Run complete analysis
        result = client.infer_unified_complete(image_bytes, face_model=face_model)

        # Generate image_id if not provided
        if image_id is None:
            image_id = hashlib.sha256(image_bytes).hexdigest()[:16]

        # Ingest global embedding
        global_doc = {
            'image_id': image_id,
            'image_path': image_path or image.filename,
            'embedding': result['global_embedding'].tolist(),
            'num_detections': result['num_dets'],
            'num_faces': result['num_faces'],
            'num_texts': result['num_texts'],
            'texts': result['texts'],
            'image_width': result['orig_shape'][1],
            'image_height': result['orig_shape'][0],
            'ingested_at': int(time.time() * 1000),
        }
        await search_service.ingest_document(index_name, image_id, global_doc)

        # Ingest face embeddings
        face_ids = []
        for i in range(result['num_faces']):
            if len(result['face_embeddings']) > i:
                face_id = f'{image_id}_face_{i}'
                face_doc = {
                    'face_id': face_id,
                    'image_id': image_id,
                    'image_path': image_path or image.filename,
                    'embedding': result['face_embeddings'][i].tolist(),
                    'box': result['face_boxes'][i].tolist(),
                    'landmarks': result['face_landmarks'][i].tolist(),
                    'score': float(result['face_scores'][i]),
                    'person_idx': int(result['face_person_idx'][i])
                    if result['face_person_idx'][i] >= 0
                    else None,
                    'ingested_at': int(time.time() * 1000),
                }
                await search_service.ingest_document(face_index_name, face_id, face_doc)
                face_ids.append(face_id)

        return {
            'status': 'success',
            'image_id': image_id,
            'ingested': {
                'global_embedding': True,
                'face_embeddings': len(face_ids),
                'face_ids': face_ids,
            },
            'analysis': {
                'num_detections': result['num_dets'],
                'num_faces': result['num_faces'],
                'num_texts': result['num_texts'],
                'texts': result['texts'],
                'face_model_used': result['face_model_used'],
            },
            'image': {
                'width': result['orig_shape'][1],
                'height': result['orig_shape'][0],
            },
        }

    except Exception as e:
        logger.error(f'Analyze and ingest failed: {e}')
        raise HTTPException(500, f'Analyze and ingest failed: {e!s}') from e


# =============================================================================
# Duplicate Detection Endpoints (Near-Duplicate via CLIP Similarity)
# =============================================================================


@router.post('/duplicates/find')
async def find_duplicates(
    search_service: VisualSearchDep,
    image_id: str = Query(..., description='Image ID to find duplicates for'),
    threshold: float = Query(
        0.90, ge=0.5, le=1.0, description='Similarity threshold (0.90=more, 0.99=strict)'
    ),
    max_results: int = Query(50, ge=1, le=200, description='Maximum number of duplicates'),
):
    """
    Find near-duplicates for a specific image using CLIP similarity.

    Searches for images with similar visual content (same scene, different
    crops/angles/lighting). Does NOT check imohash (use /ingest for exact duplicates).

    **For GUI "show similar" feature**: Use threshold=0.90 to show more results.
    **For strict duplicate detection**: Use threshold=0.99 (Immich default).

    Threshold guide:
    - 0.99: Nearly identical (Immich default - crops, resizes, compression)
    - 0.95-0.98: Very similar (same scene, slight variations)
    - 0.90-0.95: Similar content (same subject, different angle)
    - <0.90: Related but distinct images

    Args:
        image_id: ID of image to find duplicates for
        threshold: Minimum similarity score (default 0.95)
        max_results: Maximum duplicates to return

    Returns:
        List of duplicate matches with similarity scores
    """
    from src.services.duplicate_detection import DuplicateDetectionService

    try:
        start_time = time.time()

        dup_service = DuplicateDetectionService(search_service.opensearch)
        duplicates = await dup_service.find_duplicates(
            image_id=image_id,
            threshold=threshold,
            max_results=max_results,
        )

        search_time = (time.time() - start_time) * 1000

        return {
            'status': 'success',
            'image_id': image_id,
            'threshold': threshold,
            'duplicates': [
                {
                    'image_id': d.image_id,
                    'image_path': d.image_path,
                    'similarity': round(d.similarity, 4),
                    'duplicate_group_id': d.duplicate_group_id,
                }
                for d in duplicates
            ],
            'count': len(duplicates),
            'search_time_ms': round(search_time, 2),
        }

    except Exception as e:
        logger.error(f'Find duplicates failed: {e}')
        raise HTTPException(500, f'Find duplicates failed: {e!s}') from e


@router.post('/duplicates/find_by_image')
async def find_duplicates_by_image(
    search_service: VisualSearchDep,
    image: UploadFile = File(..., description='Image to find duplicates for'),
    threshold: float = Query(
        0.90, ge=0.5, le=1.0, description='Similarity threshold (0.90=more, 0.99=strict)'
    ),
    max_results: int = Query(50, ge=1, le=200, description='Maximum results'),
):
    """
    Find near-duplicates by uploading an image (doesn't need to be indexed).

    **Perfect for GUI "find similar" feature** - upload any image to find matches.
    Use threshold=0.90 for "show similar", threshold=0.99 for strict duplicate check.

    Args:
        image: Image file to check
        threshold: Minimum similarity score
        max_results: Maximum duplicates to return

    Returns:
        List of similar images in the index
    """
    from src.services.duplicate_detection import DuplicateDetectionService

    try:
        start_time = time.time()
        image_bytes = image.file.read()

        # Get embedding for the uploaded image
        inference_service = InferenceService()
        embedding = inference_service.encode_image_sync(image_bytes)

        dup_service = DuplicateDetectionService(search_service.opensearch)
        duplicates = await dup_service.find_duplicates_by_embedding(
            embedding=embedding,
            threshold=threshold,
            max_results=max_results,
        )

        search_time = (time.time() - start_time) * 1000

        return {
            'status': 'success',
            'threshold': threshold,
            'duplicates': [
                {
                    'image_id': d.image_id,
                    'image_path': d.image_path,
                    'similarity': round(d.similarity, 4),
                    'duplicate_group_id': d.duplicate_group_id,
                }
                for d in duplicates
            ],
            'count': len(duplicates),
            'search_time_ms': round(search_time, 2),
        }

    except Exception as e:
        logger.error(f'Find duplicates by image failed: {e}')
        raise HTTPException(500, f'Find duplicates failed: {e!s}') from e


@router.post('/duplicates/scan')
async def scan_for_duplicates(
    search_service: VisualSearchDep,
    threshold: float = Query(
        0.99, ge=0.5, le=1.0, description='Similarity threshold (0.99=Immich default)'
    ),
    max_images: int | None = Query(None, ge=1, description='Max images to scan (None = all)'),
):
    """
    Scan the entire index and create duplicate groups.

    This is a potentially long-running operation. For large indexes,
    consider running with max_images to process in batches.

    Algorithm:
    1. Get all ungrouped images
    2. For each, find near-duplicates above threshold
    3. Create groups with first image as primary
    4. Skip images already in a group

    Args:
        threshold: Similarity threshold for grouping
        max_images: Maximum images to scan (None = all)

    Returns:
        Scan statistics
    """
    from src.services.duplicate_detection import DuplicateDetectionService

    try:
        dup_service = DuplicateDetectionService(search_service.opensearch)
        stats = await dup_service.scan_and_group(
            threshold=threshold,
            max_images=max_images,
        )

        return {
            'status': 'success',
            'threshold': threshold,
            'total_images': stats.total_images,
            'images_scanned': stats.images_scanned,
            'groups_created': stats.groups_created,
            'duplicates_found': stats.duplicates_found,
            'already_grouped': stats.already_grouped,
            'scan_time_seconds': round(stats.scan_time_seconds, 2),
        }

    except Exception as e:
        logger.error(f'Duplicate scan failed: {e}')
        raise HTTPException(500, f'Duplicate scan failed: {e!s}') from e


# =============================================================================
# Duplicate Group Management Endpoints
# =============================================================================


@router.get('/duplicates/groups', tags=['Track E: Duplicate Management'])
async def list_duplicate_groups(
    search_service: VisualSearchDep,
    min_size: int = Query(2, ge=2, description='Minimum group size'),
    limit: int = Query(100, ge=1, le=1000, description='Maximum groups to return'),
):
    """
    List all duplicate groups with 2+ images.

    Returns groups sorted by size (largest first).

    Args:
        min_size: Minimum members per group (default 2)
        limit: Maximum number of groups to return (default 100, max 1000)

    Returns:
        List of duplicate groups with metadata including primary image info
    """
    from src.services.duplicate_detection import DuplicateDetectionService

    try:
        dup_service = DuplicateDetectionService(search_service.opensearch)
        groups = await dup_service.get_duplicate_groups(
            min_size=min_size,
            page=0,
            size=limit,
        )

        return {
            'status': 'success',
            'total_groups': len(groups),
            'min_size': min_size,
            'groups': [
                {
                    'group_id': g.group_id,
                    'primary_image_id': g.primary_image_id,
                    'primary_image_path': g.primary_image_path,
                    'member_count': g.member_count,
                }
                for g in groups
            ],
        }

    except Exception as e:
        logger.error(f'List duplicate groups failed: {e}')
        raise HTTPException(500, f'List groups failed: {e!s}') from e


@router.get('/duplicates/group/{group_id}', tags=['Track E: Duplicate Management'])
async def get_duplicate_group(
    search_service: VisualSearchDep,
    group_id: str,
    include_embeddings: bool = Query(False, description='Include embedding vectors'),
):
    """
    Get all images in a duplicate group.

    Returns image metadata, thumbnails, and optionally embeddings.

    Args:
        group_id: Duplicate group ID
        include_embeddings: Whether to include 512-dim embedding vectors (default False)

    Returns:
        List of group members with metadata (primary image first)
    """
    from src.clients.opensearch import IndexName

    try:
        # Build source fields list
        source_fields = [
            'image_id',
            'image_path',
            'duplicate_group_id',
            'is_duplicate_primary',
            'duplicate_score',
            'width',
            'height',
            'indexed_at',
        ]
        if include_embeddings:
            source_fields.append('global_embedding')

        # Query directly to control source fields
        response = await search_service.opensearch.client.search(
            index=IndexName.GLOBAL.value,
            body={
                'size': 1000,
                'query': {'term': {'duplicate_group_id': group_id}},
                '_source': source_fields,
                'sort': [
                    {'is_duplicate_primary': {'order': 'desc'}},
                    {'duplicate_score': {'order': 'desc'}},
                ],
            },
        )

        members = [hit['_source'] for hit in response['hits']['hits']]

        if not members:
            raise HTTPException(404, f'Group not found: {group_id}')

        # Find primary image
        primary = next(
            (m for m in members if m.get('is_duplicate_primary')), members[0] if members else None
        )

        return {
            'status': 'success',
            'group_id': group_id,
            'primary_image_id': primary.get('image_id') if primary else None,
            'primary_image_path': primary.get('image_path') if primary else None,
            'member_count': len(members),
            'members': members,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Get duplicate group failed: {e}')
        raise HTTPException(500, f'Get duplicate group failed: {e!s}') from e


@router.delete('/duplicates/group/{group_id}', tags=['Track E: Duplicate Management'])
async def delete_duplicate_group(
    search_service: VisualSearchDep,
    group_id: str,
    keep_first: bool = Query(True, description='Keep the first/primary image'),
):
    """
    Delete images in a duplicate group.

    If keep_first=True (default), keeps the primary image and deletes others.
    If keep_first=False, deletes ALL images in the group.

    **WARNING**: This permanently deletes images from the index. Use with caution.

    Args:
        group_id: Duplicate group ID
        keep_first: If True, keeps the primary/first image (default True)

    Returns:
        Deletion result with counts
    """
    from src.clients.opensearch import IndexName
    from src.services.duplicate_detection import DuplicateDetectionService

    try:
        dup_service = DuplicateDetectionService(search_service.opensearch)

        # Get all members first
        members = await dup_service.get_group_members(group_id=group_id, include_primary=True)

        if not members:
            raise HTTPException(404, f'Group not found: {group_id}')

        # Determine which images to delete
        if keep_first:
            # Find primary and keep it, delete the rest
            primary = next((m for m in members if m.get('is_duplicate_primary')), None)
            if primary:
                images_to_delete = [m for m in members if m['image_id'] != primary['image_id']]
                kept_image_id = primary['image_id']
            else:
                # No primary marked, keep first by sort order
                images_to_delete = members[1:] if len(members) > 1 else []
                kept_image_id = members[0]['image_id'] if members else None
        else:
            # Delete all images in the group
            images_to_delete = members
            kept_image_id = None

        # Delete images
        deleted_count = 0
        deleted_ids = []
        errors = []

        for member in images_to_delete:
            image_id = member['image_id']
            try:
                await search_service.opensearch.client.delete(
                    index=IndexName.GLOBAL.value,
                    id=image_id,
                )
                deleted_count += 1
                deleted_ids.append(image_id)
            except Exception as e:
                errors.append({'image_id': image_id, 'error': str(e)})

        # If we kept an image, clear its duplicate group fields
        if kept_image_id:
            try:
                await search_service.opensearch.client.update(
                    index=IndexName.GLOBAL.value,
                    id=kept_image_id,
                    body={
                        'doc': {
                            'duplicate_group_id': None,
                            'is_duplicate_primary': None,
                            'duplicate_score': None,
                        }
                    },
                )
            except Exception as e:
                logger.warning(
                    f'Failed to clear duplicate fields for kept image {kept_image_id}: {e}'
                )

        # Refresh index
        await search_service.opensearch.client.indices.refresh(index=IndexName.GLOBAL.value)

        return {
            'status': 'success',
            'group_id': group_id,
            'deleted_count': deleted_count,
            'deleted_ids': deleted_ids,
            'kept_image_id': kept_image_id,
            'errors': errors if errors else None,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Delete duplicate group failed: {e}')
        raise HTTPException(500, f'Delete duplicate group failed: {e!s}') from e


@router.delete('/duplicates/group/{group_id}/member/{image_id}')
async def remove_from_duplicate_group(
    search_service: VisualSearchDep,
    group_id: str,
    image_id: str,
):
    """
    Remove an image from a duplicate group.

    If removing the primary image, the next highest-scored member becomes primary.

    Args:
        group_id: Duplicate group ID
        image_id: Image ID to remove

    Returns:
        Success status
    """
    from src.services.duplicate_detection import DuplicateDetectionService

    try:
        dup_service = DuplicateDetectionService(search_service.opensearch)
        success = await dup_service.remove_from_group(image_id)

        if not success:
            raise HTTPException(404, f'Image not found or not in group: {image_id}')

        return {
            'status': 'success',
            'message': f'Removed {image_id} from group {group_id}',
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Remove from group failed: {e}')
        raise HTTPException(500, f'Remove from group failed: {e!s}') from e


@router.post('/duplicates/groups/merge')
async def merge_duplicate_groups(
    search_service: VisualSearchDep,
    group_ids: list[str] = Body(..., description='List of group IDs to merge'),
):
    """
    Merge multiple duplicate groups into one.

    The primary of the first group becomes the new primary.

    Args:
        group_ids: List of group IDs to merge (minimum 2)

    Returns:
        The merged group ID
    """
    from src.services.duplicate_detection import DuplicateDetectionService

    try:
        if len(group_ids) < 2:
            raise HTTPException(400, 'Need at least 2 groups to merge')

        dup_service = DuplicateDetectionService(search_service.opensearch)
        merged_id = await dup_service.merge_groups(group_ids)

        return {
            'status': 'success',
            'merged_group_id': merged_id,
            'groups_merged': len(group_ids),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Merge groups failed: {e}')
        raise HTTPException(500, f'Merge groups failed: {e!s}') from e


@router.get('/duplicates/stats')
async def get_duplicate_stats(
    search_service: VisualSearchDep,
):
    """
    Get duplicate detection statistics.

    Returns:
        Statistics about duplicate groups and images
    """
    from src.services.duplicate_detection import DuplicateDetectionService

    try:
        dup_service = DuplicateDetectionService(search_service.opensearch)
        stats = await dup_service.get_stats()

        return {
            'status': 'success',
            **stats,
        }

    except Exception as e:
        logger.error(f'Get duplicate stats failed: {e}')
        raise HTTPException(500, f'Get stats failed: {e!s}') from e


# =============================================================================
# OCR Endpoints (PP-OCRv5 Text Detection and Recognition)
# =============================================================================


@router.post('/ocr/predict', tags=['Track E: OCR'])
def predict_ocr(
    image: UploadFile = File(..., description='Image file (JPEG/PNG)'),
    min_det_score: float = Query(0.5, description='Minimum detection confidence'),
    min_rec_score: float = Query(0.8, description='Minimum recognition confidence'),
):
    """
    OCR: Extract text from an image using PP-OCRv5.

    Pipeline:
    1. GPU preprocessing (resize, normalize)
    2. Text detection (DB++ architecture)
    3. Text recognition (SVTR-LCNet)

    Returns:
        texts: List of detected text strings
        boxes: Quadrilateral box coordinates
        boxes_normalized: Axis-aligned boxes [x1,y1,x2,y2] normalized
        det_scores: Detection confidence scores
        rec_scores: Recognition confidence scores
    """
    from src.services.ocr_service import OcrService

    try:
        image_bytes = image.file.read()
        ocr_service = OcrService(min_det_score=min_det_score, min_rec_score=min_rec_score)
        return ocr_service.extract_text(image_bytes)

    except Exception as e:
        logger.error(f'OCR prediction failed: {e}')
        raise HTTPException(500, f'OCR prediction failed: {e!s}') from e


@router.post('/ocr/predict_batch', tags=['Track E: OCR'])
async def predict_ocr_batch(
    images: list[UploadFile] = File(..., description='Image files (JPEG/PNG)'),
    min_det_score: float = Query(0.5, description='Minimum detection confidence'),
    min_rec_score: float = Query(0.8, description='Minimum recognition confidence'),
):
    """
    OCR Batch: Extract text from multiple images.

    More efficient for processing large numbers of images.
    """
    from src.services.ocr_service import OcrService

    try:
        image_bytes_list = [await img.read() for img in images]
        ocr_service = OcrService(min_det_score=min_det_score, min_rec_score=min_rec_score)
        results = ocr_service.extract_text_batch(image_bytes_list)

        return {
            'status': 'success',
            'results': results,
            'total_images': len(results),
        }

    except Exception as e:
        logger.error(f'OCR batch prediction failed: {e}')
        raise HTTPException(500, f'OCR batch prediction failed: {e!s}') from e


@router.post('/search/ocr', tags=['Track E: OCR Search'])
async def search_by_ocr_text(
    search_service: VisualSearchDep,
    query: str = Body(..., description='Text to search for'),
    top_k: int = Body(10, description='Maximum results'),
    exact_match: bool = Body(False, description='Use exact keyword match'),
):
    """
    Search images by text content.

    Uses trigram analyzer for fuzzy text matching.
    """
    try:
        results = await search_service.opensearch.search_by_text(
            query_text=query,
            top_k=top_k,
            exact_match=exact_match,
        )

        return {
            'status': 'success',
            'query': query,
            'results': results,
            'total_results': len(results),
        }

    except Exception as e:
        logger.error(f'OCR text search failed: {e}')
        raise HTTPException(500, f'Text search failed: {e!s}') from e


@router.get('/ocr/{image_id}', tags=['Track E: OCR'])
async def get_ocr_for_image(
    search_service: VisualSearchDep,
    image_id: str,
):
    """
    Get all OCR results for a specific image.

    Returns all detected text regions with boxes and confidence scores.
    """
    try:
        results = await search_service.opensearch.get_ocr_for_image(image_id)

        return {
            'status': 'success',
            'image_id': image_id,
            'ocr_results': results,
            'total_texts': len(results),
        }

    except Exception as e:
        logger.error(f'Get OCR for image failed: {e}')
        raise HTTPException(500, f'Get OCR failed: {e!s}') from e


@router.delete('/ocr/{image_id}', tags=['Track E: OCR'])
async def delete_ocr_for_image(
    search_service: VisualSearchDep,
    image_id: str,
):
    """
    Delete all OCR results for a specific image.
    """
    try:
        deleted = await search_service.opensearch.delete_ocr_for_image(image_id)

        return {
            'status': 'success',
            'image_id': image_id,
            'deleted': deleted,
        }

    except Exception as e:
        logger.error(f'Delete OCR for image failed: {e}')
        raise HTTPException(500, f'Delete OCR failed: {e!s}') from e


@router.post('/index/ocr/create', tags=['Track E: OCR Index'])
async def create_ocr_index(
    search_service: VisualSearchDep,
    force_recreate: bool = Query(False, description='Delete and recreate if exists'),
):
    """
    Create or recreate the OCR text index.

    Uses trigram analyzer for fuzzy text search.
    """
    try:
        success = await search_service.opensearch.create_ocr_index(force_recreate)

        return {
            'status': 'success' if success else 'failed',
            'index': 'visual_search_ocr',
            'force_recreate': force_recreate,
        }

    except Exception as e:
        logger.error(f'Create OCR index failed: {e}')
        raise HTTPException(500, f'Create index failed: {e!s}') from e


# =============================================================================
# Face Search Endpoints
# =============================================================================


@router.post('/faces/search', response_model=FaceSearchResponse, tags=['Track E: Face Search'])
async def search_faces_by_image(
    search_service: VisualSearchDep,
    file: UploadFile = File(...),
    face_index: int = Query(0, ge=0, description='Which detected face to use as query (0-indexed)'),
    top_k: int = Query(10, ge=1, le=100, description='Number of results to return'),
    min_score: float = Query(
        0.7, ge=0.0, le=1.0, description='Minimum similarity score (0.7 recommended for identity)'
    ),
):
    """
    Find similar faces in the database (1:N identity search).

    Upload an image containing a face. The system will:
    1. Detect faces using SCRFD
    2. Extract ArcFace embedding for the selected face
    3. Search the faces database for matching identities

    Returns faces sorted by similarity score.

    Recommended thresholds:
    - 0.7+: Same person (high confidence)
    - 0.5-0.7: Possibly same person
    - <0.5: Different people
    """
    start_time = time.time()

    image_bytes = await file.read()

    result = await search_service.search_faces_by_image(
        image_bytes=image_bytes,
        face_index=face_index,
        top_k=top_k,
        min_score=min_score,
    )

    if result['status'] == 'error':
        raise HTTPException(status_code=400, detail=result['error'])

    search_time_ms = (time.time() - start_time) * 1000

    return FaceSearchResponse(
        status='success',
        query_face=FaceDetection(**result['query_face']),
        results=[FaceSearchResult(**r) for r in result['results']],
        total_results=len(result['results']),
        search_time_ms=search_time_ms,
    )


@router.post('/faces/identify', response_model=FaceIdentifyResponse, tags=['Track E: Face Search'])
async def identify_face(
    search_service: VisualSearchDep,
    opensearch: OpenSearchDep,
    file: UploadFile = File(...),
    face_index: int = Query(0, ge=0, description='Which detected face to identify'),
    threshold: float = Query(0.7, ge=0.0, le=1.0, description='Identity match threshold'),
):
    """
    Identify who a face belongs to (person_id lookup).

    Searches the database for the best matching face. If a match is found
    above the threshold, returns the person_id and all associated faces.
    """
    image_bytes = await file.read()

    result = await search_service.search_faces_by_image(
        image_bytes=image_bytes,
        face_index=face_index,
        top_k=1,
        min_score=threshold,
    )

    if result['status'] == 'error':
        raise HTTPException(status_code=400, detail=result['error'])

    query_face = FaceDetection(**result['query_face'])

    if not result['results']:
        return FaceIdentifyResponse(
            status='success',
            query_face=query_face,
            identified=False,
            person_id=None,
            person_name=None,
            match_score=None,
            associated_faces=[],
        )

    # Found a match
    best_match = result['results'][0]
    person_id = best_match.get('person_id')
    person_name = best_match.get('person_name')

    # Get all faces for this person if person_id exists
    associated_faces = []
    if person_id:
        faces = await opensearch.get_faces_by_person_id(person_id, limit=20)
        associated_faces = [
            FaceSearchResult(
                face_id=f.get('face_id', ''),
                image_id=f.get('image_id', ''),
                image_path=f.get('image_path'),
                score=1.0,  # These are known matches
                person_id=f.get('person_id'),
                person_name=f.get('person_name'),
                box=f.get('box', [0, 0, 0, 0]),
                confidence=f.get('confidence', 0.0),
            )
            for f in faces
        ]

    return FaceIdentifyResponse(
        status='success',
        query_face=query_face,
        identified=True,
        person_id=person_id,
        person_name=person_name,
        match_score=best_match.get('score'),
        associated_faces=associated_faces,
    )


@router.get(
    '/faces/person/{person_id}', response_model=PersonFacesResponse, tags=['Track E: Face Search']
)
async def get_person_faces(
    opensearch: OpenSearchDep,
    person_id: str = Path(..., description='Person identity ID'),
    limit: int = Query(50, ge=1, le=500, description='Maximum faces to return'),
):
    """
    Get all faces belonging to a person identity.

    Returns all indexed faces that have been assigned to this person_id,
    sorted by confidence score (highest first).
    """
    faces = await opensearch.get_faces_by_person_id(person_id, limit=limit)

    if not faces:
        raise HTTPException(status_code=404, detail=f'No faces found for person_id: {person_id}')

    # Get person_name from first face (should be consistent)
    person_name = faces[0].get('person_name') if faces else None

    return PersonFacesResponse(
        person_id=person_id,
        person_name=person_name,
        face_count=len(faces),
        faces=[
            FaceSearchResult(
                face_id=f.get('face_id', ''),
                image_id=f.get('image_id', ''),
                image_path=f.get('image_path'),
                score=1.0,
                person_id=f.get('person_id'),
                person_name=f.get('person_name'),
                box=f.get('box', [0, 0, 0, 0]),
                confidence=f.get('confidence', 0.0),
            )
            for f in faces
        ],
    )


# =============================================================================
# YOLO11-Face Endpoints (Alternative to SCRFD)
# =============================================================================


@router.post(
    '/faces/yolo11/detect', response_model=FaceDetectResponse, tags=['Track E: YOLO11-Face']
)
def yolo11_face_detect(
    file: UploadFile = File(...),
    confidence: float = Query(0.5, ge=0.1, le=0.99, description='Minimum confidence threshold'),
):
    """
    Detect faces using YOLO11-face model.

    **Alternative to /faces/detect (SCRFD)**. Use for benchmark comparison.

    YOLO11-face is a pose-based face detector that outputs:
    - Face bounding boxes
    - 5-point facial landmarks (eyes, nose, mouth corners)
    - Detection confidence scores

    This endpoint uses the yolo11_face_pipeline Triton model which:
    1. Runs YOLO11-face TensorRT for detection
    2. Applies GPU-accelerated NMS
    3. Returns normalized [0,1] coordinates

    Args:
        file: Image file (JPEG/PNG)
        confidence: Minimum detection confidence (default 0.5)

    Returns:
        Face detections with boxes, landmarks, and scores
    """
    try:
        image_bytes = file.file.read()
        inference_service = InferenceService()

        # Use YOLO11-face pipeline
        result = inference_service.infer_faces_yolo11(image_bytes, confidence=confidence)

        if result['status'] == 'error':
            raise HTTPException(400, result.get('error', 'Face detection failed'))

        orig_h, orig_w = result['orig_shape']

        faces = [
            FaceDetection(
                box=f['box'],
                landmarks=f['landmarks'],
                score=f['score'],
                quality=f.get('quality'),
            )
            for f in result['faces']
        ]

        return FaceDetectResponse(
            num_faces=result['num_faces'],
            faces=faces,
            image=ImageMetadata(width=orig_w, height=orig_h),
            track='E_faces_yolo11',
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'YOLO11 face detection failed: {e}')
        raise HTTPException(500, f'Face detection failed: {e!s}') from e


@router.post(
    '/faces/yolo11/recognize', response_model=FaceRecognizeResponse, tags=['Track E: YOLO11-Face']
)
def yolo11_face_recognize(
    file: UploadFile = File(...),
    confidence: float = Query(0.5, ge=0.1, le=0.99, description='Minimum confidence threshold'),
):
    """
    Detect faces + extract ArcFace embeddings using YOLO11-face.

    **Alternative to /faces/recognize (SCRFD)**. Use for benchmark comparison.

    Uses HD cropping for face alignment (industry standard):
    1. YOLO11-face detects faces with landmarks on 640x640 input
    2. Landmarks are mapped back to original HD coordinates
    3. Face alignment crops from HD original image (not detection input)
    4. ArcFace extracts 512-dim identity embeddings

    This ensures maximum accuracy for face recognition by:
    - Preserving full resolution facial details
    - Avoiding compression artifacts from detection preprocessing

    Args:
        file: Image file (JPEG/PNG)
        confidence: Minimum detection confidence (default 0.5)

    Returns:
        Face detections with boxes, landmarks, scores, and ArcFace embeddings
    """
    try:
        image_bytes = file.file.read()
        inference_service = InferenceService()

        # Use YOLO11-face pipeline with ArcFace embeddings
        result = inference_service.infer_faces_yolo11(image_bytes, confidence=confidence)

        if result['status'] == 'error':
            raise HTTPException(400, result.get('error', 'Face recognition failed'))

        orig_h, orig_w = result['orig_shape']

        faces = [
            FaceDetection(
                box=f['box'],
                landmarks=f['landmarks'],
                score=f['score'],
                quality=f.get('quality'),
            )
            for f in result['faces']
        ]

        return FaceRecognizeResponse(
            num_faces=result['num_faces'],
            faces=faces,
            embeddings=result.get('embeddings', []),
            image=ImageMetadata(width=orig_w, height=orig_h),
            track='E_faces_yolo11',
            model='yolo11_face + arcface_w600k_r50',
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'YOLO11 face recognition failed: {e}')
        raise HTTPException(500, f'Face recognition failed: {e!s}') from e


@router.post('/faces/yolo11/verify', tags=['Track E: YOLO11-Face'])
def yolo11_face_verify(
    image1: UploadFile = File(..., description='First image with face'),
    image2: UploadFile = File(..., description='Second image with face'),
    threshold: float = Query(0.6, ge=0.0, le=1.0, description='Similarity threshold for match'),
):
    """
    Verify if two images contain the same person using YOLO11-face.

    **Alternative to /faces/verify (SCRFD)**. Use for benchmark comparison.

    Extracts ArcFace embeddings from both images using YOLO11-face detection
    and compares using cosine similarity.

    Threshold guidelines:
    - 0.6: High confidence (recommended for security)
    - 0.5: Balanced precision/recall
    - 0.4: More permissive (may have false positives)

    Args:
        image1: First image with face
        image2: Second image with face
        threshold: Similarity threshold (default 0.6)

    Returns:
        Match decision, similarity score, and face info from both images
    """
    try:
        image1_bytes = image1.file.read()
        image2_bytes = image2.file.read()
        inference_service = InferenceService()

        result1 = inference_service.infer_faces_yolo11(image1_bytes)
        result2 = inference_service.infer_faces_yolo11(image2_bytes)

        if result1['num_faces'] == 0:
            raise HTTPException(400, 'No face detected in first image')
        if result2['num_faces'] == 0:
            raise HTTPException(400, 'No face detected in second image')

        # Use first face from each image
        emb1 = np.array(result1['embeddings'][0])
        emb2 = np.array(result2['embeddings'][0])

        # Cosine similarity (embeddings are L2-normalized)
        similarity = float(np.dot(emb1, emb2))
        is_match = similarity >= threshold

        return {
            'status': 'success',
            'match': is_match,
            'similarity': round(similarity, 4),
            'threshold': threshold,
            'detector': 'yolo11_face',
            'image1': {
                'num_faces': result1['num_faces'],
                'face_used': result1['faces'][0],
            },
            'image2': {
                'num_faces': result2['num_faces'],
                'face_used': result2['faces'][0],
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'YOLO11 face verification failed: {e}')
        raise HTTPException(500, f'Face verification failed: {e!s}') from e


@router.post('/faces/yolo11/benchmark', tags=['Track E: YOLO11-Face'])
def benchmark_face_detectors(
    file: UploadFile = File(...),
    iterations: int = Query(10, ge=1, le=100, description='Number of iterations per detector'),
):
    """
    Benchmark YOLO11-face vs SCRFD face detection.

    Runs both detectors on the same image multiple times and compares:
    - Detection count
    - Inference time
    - Detection consistency

    Use this endpoint to compare performance characteristics:
    - YOLO11-face: Faster single-stage detection
    - SCRFD: More accurate, multi-scale anchor-based

    Args:
        file: Image file (JPEG/PNG)
        iterations: Number of iterations for timing (default 10)

    Returns:
        Benchmark results for both detectors
    """
    import time

    try:
        image_bytes = file.file.read()
        inference_service = InferenceService()

        results = {
            'status': 'success',
            'iterations': iterations,
            'detectors': {},
        }

        # Benchmark YOLO11-face
        yolo11_times = []
        yolo11_faces = None
        for i in range(iterations):
            start = time.time()
            result = inference_service.infer_faces_yolo11(image_bytes)
            yolo11_times.append((time.time() - start) * 1000)
            if i == 0:
                yolo11_faces = result['num_faces']

        results['detectors']['yolo11_face'] = {
            'num_faces': yolo11_faces,
            'mean_time_ms': round(np.mean(yolo11_times), 2),
            'std_time_ms': round(np.std(yolo11_times), 2),
            'min_time_ms': round(np.min(yolo11_times), 2),
            'max_time_ms': round(np.max(yolo11_times), 2),
        }

        # Benchmark SCRFD
        scrfd_times = []
        scrfd_faces = None
        for i in range(iterations):
            start = time.time()
            result = inference_service.infer_faces(image_bytes)
            scrfd_times.append((time.time() - start) * 1000)
            if i == 0:
                scrfd_faces = result['num_faces']

        results['detectors']['scrfd'] = {
            'num_faces': scrfd_faces,
            'mean_time_ms': round(np.mean(scrfd_times), 2),
            'std_time_ms': round(np.std(scrfd_times), 2),
            'min_time_ms': round(np.min(scrfd_times), 2),
            'max_time_ms': round(np.max(scrfd_times), 2),
        }

        # Calculate speedup
        yolo11_mean = np.mean(yolo11_times)
        scrfd_mean = np.mean(scrfd_times)
        if yolo11_mean > 0:
            results['speedup'] = {
                'yolo11_vs_scrfd': round(scrfd_mean / yolo11_mean, 2),
                'faster_detector': 'yolo11_face' if yolo11_mean < scrfd_mean else 'scrfd',
            }

        return results

    except Exception as e:
        logger.error(f'Face detector benchmark failed: {e}')
        raise HTTPException(500, f'Benchmark failed: {e!s}') from e


# =============================================================================
# Face Identity Management Endpoints (using FaceIdentityService)
# =============================================================================


@router.post('/faces/identity/ingest', tags=['Track E: Face Identity'])
async def ingest_faces_identity(
    file: UploadFile = File(...),
    person_id: str | None = Query(None, description='Person ID to assign faces to'),
    source_image_id: str | None = Query(None, description='Source image ID'),
):
    """
    Detect faces and ingest into OpenSearch with embeddings.

    This endpoint uses FaceIdentityService for face identity management.
    It detects faces in the uploaded image, extracts ArcFace embeddings,
    and stores them in the visual_search_faces index.

    Args:
        file: Image file (JPEG/PNG) containing faces to ingest
        person_id: Optional person ID to assign faces to. If not provided,
                   a new person_id will be auto-generated.
        source_image_id: Optional source image ID for tracking provenance

    Returns:
        Ingestion result with face IDs and person ID assignment
    """
    try:
        # Validate file is an image (check content-type or filename extension)
        valid_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
        filename = file.filename or ''
        file_ext = filename.lower()[filename.rfind('.') :] if '.' in filename else ''
        is_valid_extension = file_ext in valid_extensions
        is_valid_content_type = file.content_type and file.content_type.startswith('image/')

        if not is_valid_extension and not is_valid_content_type:
            raise HTTPException(400, 'File must be an image (JPEG, PNG, WebP, BMP)')

        image_bytes = await file.read()
        face_service = get_face_identity_service()

        result = await face_service.ingest_face(
            image_bytes=image_bytes,
            person_id=person_id,
            source_image_id=source_image_id,
        )

        if result['status'] == 'error':
            raise HTTPException(500, result.get('error', 'Face ingestion failed'))

        return {
            'status': 'success',
            'num_faces': result['num_faces'],
            'indexed': result.get('indexed', 0),
            'face_ids': result['face_ids'],
            'person_id': result['person_id'],
            'errors': result.get('errors'),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Face identity ingestion failed: {e}')
        raise HTTPException(500, f'Face ingestion failed: {e!s}') from e


@router.post('/faces/identity/ingest_batch', tags=['Track E: Face Identity'])
async def ingest_faces_batch(
    files: list[UploadFile] = File(..., description='Multiple image files (max 64)'),
    person_id: str | None = Query(None, description='Person ID to assign all faces to'),
):
    """
    Batch ingest faces into OpenSearch with embeddings.

    Process up to 64 images in a single request. Each image is processed
    for face detection and embedding extraction, then indexed.

    Args:
        files: Multiple image files (JPEG/PNG), max 64 per request
        person_id: Optional person ID to assign all detected faces to

    Returns:
        Batch results with per-image face counts and status
    """
    if len(files) > 64:
        raise HTTPException(400, 'Maximum 64 images per batch request')

    results = []
    face_identity_service = get_face_identity_service()

    valid_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

    for file in files:
        try:
            # Validate file is an image (check content-type or filename extension)
            filename = file.filename or ''
            file_ext = filename.lower()[filename.rfind('.') :] if '.' in filename else ''
            is_valid_extension = file_ext in valid_extensions
            is_valid_content_type = file.content_type and file.content_type.startswith('image/')

            if not is_valid_extension and not is_valid_content_type:
                results.append(
                    {
                        'status': 'error',
                        'error': f'Invalid file type: {file.content_type or filename}',
                        'filename': file.filename,
                    }
                )
                continue

            image_bytes = await file.read()

            result = await face_identity_service.ingest_face(
                image_bytes=image_bytes,
                person_id=person_id,
                source_image_id=file.filename,
            )

            result['filename'] = file.filename
            results.append(result)

        except Exception as e:
            logger.error(f'Face ingestion failed for {file.filename}: {e}')
            results.append({'status': 'error', 'error': str(e), 'filename': file.filename})

    total_faces = sum(r.get('num_faces', 0) for r in results if r.get('status') == 'success')

    return {
        'status': 'success',
        'total_images': len(files),
        'processed': len([r for r in results if r.get('status') == 'success']),
        'errors': len([r for r in results if r.get('status') == 'error']),
        'total_faces': total_faces,
        'results': results,
    }


@router.post('/faces/identity/identify', tags=['Track E: Face Identity'])
async def identify_faces_1n(
    file: UploadFile = File(...),
    top_k: int = Query(5, ge=1, le=100, description='Number of top matches per face'),
    threshold: float = Query(0.6, ge=0.0, le=1.0, description='Minimum similarity threshold'),
    face_detector: str = Query(
        'yolo11', pattern='^(scrfd|yolo11)$', description='Face detector: yolo11 (default) or scrfd'
    ),
):
    """
    1:N face identification - find matching persons in database.

    Detects faces in the uploaded image and searches the face database
    for matching identities. For each detected face, returns the top-k
    most similar faces from the database.

    Args:
        file: Image file (JPEG/PNG) containing faces to identify
        top_k: Number of top matches to return per face (1-100)
        threshold: Minimum similarity threshold for matches (0.0-1.0)
                   Recommended: 0.6 for same person verification
        face_detector: Face detector to use ('yolo11' default with batching, or 'scrfd')

    Returns:
        Identification results for each detected face, including
        best match person_id if above threshold
    """
    try:
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(400, 'File must be an image')

        image_bytes = await file.read()
        face_service = get_face_identity_service()

        result = await face_service.identify_faces(
            image_bytes=image_bytes,
            top_k=top_k,
            threshold=threshold,
            face_detector=face_detector,
        )

        if result['status'] == 'error':
            raise HTTPException(500, result.get('error', 'Face identification failed'))

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Face identification failed: {e}')
        raise HTTPException(500, f'Face identification failed: {e!s}') from e


@router.get('/faces/identity/person/{person_id}', tags=['Track E: Face Identity'])
async def get_person_faces_identity(person_id: str):
    """
    Get all faces for a person.

    Retrieves all face records associated with a specific person_id
    from the face identity database.

    Args:
        person_id: Person identifier to look up

    Returns:
        List of face documents (without embeddings) for the person,
        sorted by reference status and confidence
    """
    try:
        face_service = get_face_identity_service()
        faces = await face_service.get_person_faces(person_id)

        if not faces:
            raise HTTPException(404, f'No faces found for person_id: {person_id}')

        return {
            'status': 'success',
            'person_id': person_id,
            'face_count': len(faces),
            'faces': faces,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Failed to get person faces: {e}')
        raise HTTPException(500, f'Failed to get person faces: {e!s}') from e


@router.put('/faces/identity/{face_id}/assign', tags=['Track E: Face Identity'])
async def assign_face_to_person(
    face_id: str,
    person_id: str = Query(..., description='Person ID to assign to'),
):
    """
    Assign or reassign a face to a person.

    Updates the person_id field for a specific face record.
    Useful for correcting identity assignments or grouping
    faces under a known identity.

    Args:
        face_id: Face identifier to update
        person_id: Person identifier to assign the face to

    Returns:
        Assignment result with previous and new person_id
    """
    try:
        face_service = get_face_identity_service()
        result = await face_service.assign_face_to_person(
            face_id=face_id,
            person_id=person_id,
        )

        if result['status'] == 'error':
            raise HTTPException(500, result.get('error', 'Failed to assign face'))

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Failed to assign face to person: {e}')
        raise HTTPException(500, f'Failed to assign face: {e!s}') from e


@router.post('/faces/identity/search', tags=['Track E: Face Identity'])
async def search_faces_by_image_identity(
    file: UploadFile = File(...),
    top_k: int = Query(10, ge=1, le=100, description='Number of results to return'),
    min_score: float = Query(0.0, ge=0.0, le=1.0, description='Minimum similarity score'),
    face_detector: str = Query(
        'yolo11', pattern='^(scrfd|yolo11)$', description='Face detector: yolo11 (default) or scrfd'
    ),
):
    """
    Search for similar faces by uploading an image.

    Detects faces in the uploaded image, extracts embeddings,
    and searches the face database for similar faces.

    Args:
        file: Image file (JPEG/PNG) containing a face to search
        top_k: Number of similar faces to return (1-100)
        min_score: Minimum similarity score threshold (0.0-1.0)
        face_detector: Face detector to use ('yolo11' default with batching, or 'scrfd')

    Returns:
        Search results with similar faces sorted by similarity score
    """
    try:
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(400, 'File must be an image')

        image_bytes = await file.read()
        inference_service = InferenceService()
        face_service = get_face_identity_service()

        # Detect faces based on detector choice
        if face_detector == 'yolo11':
            face_result = inference_service.infer_faces_yolo11(image_bytes)
        else:
            face_result = inference_service.infer_faces(image_bytes)

        if face_result.get('num_faces', 0) == 0:
            return {
                'status': 'success',
                'num_faces': 0,
                'results': [],
                'message': 'No faces detected in query image',
            }

        # Use first detected face for search
        embedding = face_result['embeddings'][0]
        if hasattr(embedding, 'tolist'):
            embedding = embedding.tolist()

        # Search for similar faces
        results = await face_service.search_faces_by_embedding(
            embedding=embedding,
            top_k=top_k,
            min_score=min_score,
        )

        return {
            'status': 'success',
            'num_faces': face_result['num_faces'],
            'query_face': {
                'box': face_result['faces'][0]['box'],
                'score': face_result['faces'][0].get('score', 0.0),
            },
            'results': results,
            'total_results': len(results),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Face search failed: {e}')
        raise HTTPException(500, f'Face search failed: {e!s}') from e
