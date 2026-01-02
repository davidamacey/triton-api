"""
Unified YOLO Inference FastAPI Service
All 5 Performance Tracks in One Application

This unified service provides all tracks through a single endpoint structure:
- Track A: PyTorch Direct (baseline) - /pytorch/predict/{model_name}
- Track B: Standard TRT + CPU NMS - /predict/{model_name}
- Track C: End2End TRT + GPU NMS - /predict/{model_name}_end2end
- Track D: DALI + TRT (Full GPU) - /predict/{model_name}_gpu_e2e_*
- Track E: Visual Search - /track_e/*

Simplified deployment: One Docker container, all endpoints available.
"""

import logging
import time
from contextlib import asynccontextmanager

import numpy as np
import orjson
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import ORJSONResponse, Response
from ultralytics import YOLO

from src.config import get_settings
from src.core.dependencies import OpenSearchClientFactory, TritonClientFactory, app_state
from src.routers import (
    health_router,
    models_router,
    track_a_router,
    track_e_router,
    track_f_router,
    triton_router,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001 - Required by FastAPI lifespan API contract
    """
    Application lifecycle manager.

    Args:
        app: FastAPI application instance (required by API contract, not used in implementation)

    Startup:
    - Load PyTorch models for Track A (if enabled)
    - Shared Triton gRPC client auto-created on first use

    Shutdown:
    - Clean up PyTorch models
    - Close all Triton gRPC connections
    - Close OpenSearch connections
    """
    settings = get_settings()

    # =========================================================================
    # STARTUP
    # =========================================================================
    logger.info('=== STARTUP: Loading Models ===')

    if settings.enable_pytorch:
        logger.info('Loading PyTorch models for Track A...')
        for model_name, model_path in settings.models.PYTORCH_MODELS.items():
            try:
                logger.info(f'Loading {model_name} from {model_path}')
                model = YOLO(model_path, task='detect')

                # Warmup with FP16 if GPU available
                dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
                if torch.cuda.is_available():
                    _ = model(dummy_img, verbose=False, half=True)
                    logger.info('  Model uses FP16 inference')
                else:
                    _ = model(dummy_img, verbose=False)

                app_state.pytorch_models[model_name] = model
                logger.info(f'  {model_name} loaded successfully')

            except Exception as e:
                logger.error(f'Failed to load {model_name}: {e}')
                raise

        logger.info(f'PyTorch models loaded: {list(app_state.pytorch_models.keys())}')
    else:
        logger.info('Track A (PyTorch) DISABLED - GPU memory reserved for Triton')

    logger.info('=== SERVICE READY ===')
    logger.info(f'Active Tracks: {"A, B, C, D, E" if settings.enable_pytorch else "B, C, D, E"}')
    logger.info(f'Triton URL: {settings.triton_url}')

    yield

    # =========================================================================
    # SHUTDOWN
    # =========================================================================
    logger.info('=== SHUTDOWN: Cleaning Up ===')

    # Clean up PyTorch models
    app_state.pytorch_models.clear()
    logger.info('PyTorch models cleaned up')

    # Close Triton connections
    try:
        await TritonClientFactory.close_all()
    except Exception as e:
        logger.warning(f'Error closing Triton clients: {e}')

    # Close OpenSearch connections
    try:
        await OpenSearchClientFactory.close()
    except Exception as e:
        logger.warning(f'Error closing OpenSearch client: {e}')

    logger.info('=== SHUTDOWN COMPLETE ===')


# =============================================================================
# FastAPI Application Factory
# =============================================================================
def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    application = FastAPI(
        title=settings.api_title,
        description=settings.api_description,
        version=settings.api_version,
        lifespan=lifespan,
        default_response_class=ORJSONResponse,
    )

    # Performance Middleware
    @application.middleware('http')
    async def performance_middleware(request: Request, call_next):
        """
        Monitor request performance, validate file size, and inject timing into response.

        Industry standard: timing included in both header (X-Process-Time) and response body.
        """
        start_time = time.time()

        # Validate file size for upload endpoints
        if request.method == 'POST' and 'predict' in request.url.path:
            content_length = request.headers.get('content-length')
            # Batch endpoints allow up to 10GB for large photo library processing
            # Single image endpoints use default limit (50MB)
            is_batch_endpoint = 'predict_batch' in request.url.path
            max_size = (
                10 * 1024 * 1024 * 1024 if is_batch_endpoint else settings.max_file_size_bytes
            )
            max_size_label = '10GB' if is_batch_endpoint else f'{settings.max_file_size_mb}MB'
            if content_length and int(content_length) > max_size:
                raise HTTPException(
                    status_code=413,
                    detail=f'File too large. Maximum: {max_size_label}',
                )

        response = await call_next(request)

        # Calculate timing
        duration_ms = (time.time() - start_time) * 1000

        # Add timing header (always)
        response.headers['X-Process-Time'] = f'{duration_ms:.2f}ms'

        # Inject timing into JSON response body for inference endpoints
        content_type = response.headers.get('content-type', '')
        is_inference_endpoint = any(
            path in request.url.path
            for path in ['/predict', '/pytorch/predict', '/track_e/', '/track_f/']
        )

        if 'application/json' in content_type and is_inference_endpoint:
            # Read response body
            body_chunks = [chunk async for chunk in response.body_iterator]
            body = b''.join(body_chunks)

            try:
                # Parse and inject timing
                data = orjson.loads(body)
                if isinstance(data, dict):
                    data['total_time_ms'] = round(duration_ms, 2)
                body = orjson.dumps(data)

                # Build new headers without content-length (will be recalculated)
                new_headers = {
                    k: v for k, v in response.headers.items() if k.lower() != 'content-length'
                }
                new_headers['X-Process-Time'] = f'{duration_ms:.2f}ms'

                # Create new response with modified body
                return Response(
                    content=body,
                    status_code=response.status_code,
                    headers=new_headers,
                    media_type='application/json',
                )
            except Exception:
                # If parsing fails, return original response
                return Response(
                    content=body,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    media_type=content_type,
                )

        # Log slow requests
        if duration_ms > settings.slow_request_threshold_ms:
            logger.warning(
                f'Slow request: {request.method} {request.url.path} - {duration_ms:.2f}ms'
            )

        return response

    # Include Routers
    application.include_router(health_router)
    application.include_router(track_a_router)
    application.include_router(triton_router)
    application.include_router(track_e_router)
    application.include_router(track_f_router)
    application.include_router(models_router)

    return application


# Create application instance
app = create_app()
