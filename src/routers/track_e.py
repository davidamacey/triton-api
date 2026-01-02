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
from fastapi import APIRouter, Body, File, HTTPException, Query, UploadFile
from fastapi.responses import ORJSONResponse

from src.core.dependencies import VisualSearchDep
from src.schemas.detection import ImageMetadata, ModelMetadata
from src.schemas.track_e import (
    DetectOnlyResponse,
    FaceDetection,
    FaceDetectResponse,
    FaceFullResponse,
    FaceRecognizeResponse,
    ImageEmbeddingResponse,
    PredictFullResponse,
    PredictResponse,
    TextEmbeddingResponse,
    VisualSearchResponse,
)
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
):
    """
    Ingest image into visual search indexes with auto-routing.

    Pipeline:
    1. Run Track E ensemble (YOLO + MobileCLIP) for embeddings
    2. Route to appropriate indexes:
       - Global embedding → visual_search_global
       - Person detections → visual_search_people
       - Vehicle detections → visual_search_vehicles

    Args:
        file: Image file (JPEG/PNG)
        image_id: Unique identifier (auto-generated if not provided)
        image_path: File path for retrieval (defaults to image_id)
        metadata: JSON string for custom metadata

    Returns:
        Ingestion result with counts per category
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
        )

        if result['status'] == 'error':
            raise HTTPException(500, result.get('error', 'Ingestion failed'))

        indexed = result.get('indexed', {})
        return {
            'status': 'success',
            'image_id': image_id,
            'num_detections': result['num_detections'],
            'embedding_norm': result.get('embedding_norm', 0.0),
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

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Ingestion failed: {e}')
        raise HTTPException(500, f'Ingestion failed: {e!s}') from e


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
):
    """
    Detect faces in image using SCRFD-10G.

    100% GPU pipeline via DALI preprocessing:
    1. GPU JPEG decode (nvJPEG via DALI)
    2. GPU preprocessing for SCRFD (640x640)
    3. SCRFD face detection with GPU NMS
    4. Returns face boxes, 5-point landmarks, and confidence scores

    Response includes normalized [0,1] coordinates for boxes and landmarks.
    """
    try:
        image_bytes = image.file.read()
        inference_service = InferenceService()

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
):
    """
    Detect faces and extract ArcFace identity embeddings.

    100% GPU pipeline:
    1. GPU JPEG decode (nvJPEG via DALI)
    2. SCRFD face detection + NMS
    3. Face alignment using landmarks (Umeyama transform)
    4. ArcFace embedding extraction (512-dim L2-normalized)

    Use embeddings for:
    - Face verification (1:1 matching) - cosine similarity > 0.6
    - Face identification (1:N search) - OpenSearch k-NN

    Response includes normalized [0,1] coordinates and 512-dim embeddings per face.
    """
    try:
        image_bytes = image.file.read()
        inference_service = InferenceService()

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
):
    """
    Unified pipeline: YOLO + SCRFD + MobileCLIP + ArcFace.

    All processing happens in Triton via quad-branch ensemble:
    1. GPU JPEG decode (nvJPEG via DALI)
    2. Quad-branch preprocessing (YOLO 640, CLIP 256, SCRFD 640, HD original)
    3. YOLO object detection (parallel with 4, 5)
    4. MobileCLIP global embedding (parallel with 3, 5)
    5. SCRFD face detection + NMS (parallel with 3, 4)
    6. ArcFace face embeddings (depends on 5)

    Returns:
    - YOLO detections (objects in image)
    - Face detections with landmarks
    - ArcFace 512-dim embeddings per face
    - MobileCLIP 512-dim global image embedding
    """
    try:
        image_bytes = image.file.read()
        inference_service = InferenceService()

        result = inference_service.infer_faces_full(image_bytes)
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

        return FaceFullResponse(
            detections=result['detections'],
            num_detections=result['num_detections'],
            num_faces=result['num_faces'],
            faces=faces,
            face_embeddings=result['face_embeddings'],
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


@router.post('/faces/verify', tags=['Track E: Face Recognition'])
def verify_faces(
    image1: UploadFile = File(..., description='First image with face'),
    image2: UploadFile = File(..., description='Second image with face'),
    threshold: float = Query(0.6, ge=0.0, le=1.0, description='Similarity threshold for match'),
):
    """
    Verify if two images contain the same person (1:1 verification).

    Extracts ArcFace embeddings from both images and compares using cosine similarity.

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
