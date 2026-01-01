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
    ImageEmbeddingResponse,
    ImageIngestResponse,
    IndexStatsResponse,
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


@router.post('/ingest', response_model=ImageIngestResponse)
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
    Ingest image into visual search index.

    Pipeline:
    1. Run Track E ensemble (YOLO + MobileCLIP) for embeddings
    2. Index global + per-box embeddings in OpenSearch

    Args:
        file: Image file (JPEG/PNG)
        image_id: Unique identifier (auto-generated if not provided)
        image_path: File path for retrieval (defaults to image_id)
        metadata: JSON string for custom metadata
    """
    try:
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(400, 'File must be an image')

        image_bytes = file.file.read()

        # Generate image_id if not provided
        if image_id is None:
            image_id = f'img_{uuid.uuid4().hex[:12]}'

        # Parse metadata
        metadata_dict = json.loads(metadata) if metadata else {}
        metadata_dict['filename'] = file.filename

        # Use VisualSearchService (inference + OpenSearch indexing)
        result = await search_service.ingest_image(
            image_bytes=image_bytes,
            image_id=image_id,
            image_path=image_path,
            metadata=metadata_dict,
        )

        if result['status'] == 'error':
            raise HTTPException(500, result.get('error', 'Ingestion failed'))

        return ImageIngestResponse(
            status='success',
            image_id=image_id,
            message=f'Ingested with {result["num_detections"]} detections',
            num_detections=result['num_detections'],
            global_embedding_norm=result.get('embedding_norm', 0.0),
        )

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
    Object-to-object search using per-box embeddings.

    Pipeline:
    1. Run Track E full ensemble to get per-box embeddings
    2. Use specified box's embedding for nested k-NN search

    Args:
        image: Query image file
        box_index: Index of detected object to use for search (0 = first detection)
        top_k: Number of results to return
        min_score: Minimum similarity score threshold
        class_filter: Comma-separated class IDs to filter (e.g., "0,15,62" for person, cat, tv)
    """
    try:
        start_time = time.time()
        image_bytes = image.file.read()

        # Parse class filter
        class_ids = None
        if class_filter:
            class_ids = [int(c.strip()) for c in class_filter.split(',')]

        # Use VisualSearchService (inference + OpenSearch search)
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


# =============================================================================
# Index Management Endpoints (lightweight, keep sync)
# =============================================================================


@router.get('/index/stats', response_model=IndexStatsResponse)
async def get_index_stats(search_service: VisualSearchDep):
    """Get visual search index statistics."""
    try:
        stats = await search_service.get_index_stats()
        return IndexStatsResponse(
            status=stats.get('status', 'success'),
            total_documents=stats.get('total_documents', 0),
            index_size_mb=stats.get('index_size_mb', 0.0),
        )
    except Exception as e:
        logger.error(f'Failed to get index stats: {e}')
        return IndexStatsResponse(status='error', total_documents=0, index_size_mb=0.0)


@router.post('/index/create')
async def create_index(
    search_service: VisualSearchDep,
    force_recreate: bool = Query(False, description='Delete existing index first'),
):
    """
    Create or recreate the visual search index.

    Args:
        force_recreate: Whether to delete existing index before creating
    """
    try:
        success = await search_service.setup_index(force_recreate=force_recreate)
        if success:
            return {'status': 'success', 'message': 'Index created successfully'}
        return {'status': 'error', 'message': 'Failed to create index'}
    except Exception as e:
        logger.error(f'Failed to create index: {e}')
        raise HTTPException(500, f'Failed to create index: {e!s}') from e


@router.delete('/index')
async def delete_index(search_service: VisualSearchDep):
    """Delete the visual search index."""
    try:
        success = await search_service.delete_index()
        if success:
            return {'status': 'success', 'message': 'Index deleted successfully'}
        return {'status': 'error', 'message': 'Failed to delete index'}
    except Exception as e:
        logger.error(f'Failed to delete index: {e}')
        raise HTTPException(500, f'Failed to delete index: {e!s}') from e


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
