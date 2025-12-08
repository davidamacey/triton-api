"""
Health and Monitoring Router

Provides health checks, service info, and connection pool statistics.
"""

import logging
import os

import psutil
import torch
from fastapi import APIRouter

from src.clients.triton_pool import get_client_pool_stats
from src.config import get_settings
from src.core.dependencies import get_pytorch_models


logger = logging.getLogger(__name__)

router = APIRouter(
    tags=['Health & Monitoring'],
)


@router.get('/')
def root():
    """
    Service information endpoint.

    Returns available tracks, models, and backend configuration.
    """
    settings = get_settings()
    models = settings.models
    pytorch_models = get_pytorch_models()

    return {
        'service': 'Unified YOLO Inference API',
        'status': 'running',
        'tracks': {
            'track_a': {
                'endpoint': '/pytorch/predict/{model_name}',
                'description': 'PyTorch baseline',
                'models': list(pytorch_models.keys()),
                'enabled': settings.enable_pytorch,
            },
            'track_b': {
                'endpoint': '/predict/{model_name}',
                'description': 'Standard TRT + CPU NMS',
                'models': list(models.STANDARD_MODELS.keys()),
            },
            'track_c': {
                'endpoint': '/predict/{model_name}_end2end',
                'description': 'End2End TRT + GPU NMS',
                'models': [f'{k}_end2end' for k in models.END2END_MODELS],
            },
            'track_d': {
                'endpoints': [
                    '/predict/{model_name}_gpu_e2e_streaming',
                    '/predict/{model_name}_gpu_e2e',
                    '/predict/{model_name}_gpu_e2e_batch',
                ],
                'description': 'DALI + TRT (Full GPU)',
                'models': {
                    'streaming': [
                        f'{k}_gpu_e2e_streaming' for k in models.GPU_E2E_STREAMING_MODELS
                    ],
                    'balanced': [f'{k}_gpu_e2e' for k in models.GPU_E2E_MODELS],
                    'batch': [f'{k}_gpu_e2e_batch' for k in models.GPU_E2E_BATCH_MODELS],
                },
            },
            'track_e': {
                'endpoint': '/track_e/*',
                'description': 'Visual Search with MobileCLIP',
                'ensembles': list(models.ENSEMBLE_MODELS.keys()),
            },
        },
        'triton_backend': f'grpc://{settings.triton_url}',
        'gpu_available': torch.cuda.is_available(),
    }


@router.get('/health')
def health():
    """
    Enhanced health check with performance metrics.

    Returns:
    - Service status
    - Track availability
    - Memory and GPU usage
    - Optimization flags
    """
    settings = get_settings()
    pytorch_models = get_pytorch_models()

    # Process metrics
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()

    health_data = {
        'status': 'healthy',
        'tracks': {
            'track_a_pytorch': {
                'enabled': settings.enable_pytorch,
                'models': dict.fromkeys(pytorch_models.keys(), 'loaded'),
                'gpu_available': torch.cuda.is_available(),
            },
            'track_b_c_d_triton': {
                'backend': 'triton',
                'protocol': 'gRPC',
                'url': settings.triton_url,
                'client_mode': 'shared_pool',
            },
            'track_e_visual_search': {
                'opensearch_url': settings.opensearch_url,
                'embedding_dim': 512,
            },
        },
        'performance': {
            'memory_mb': round(memory_info.rss / 1024 / 1024, 2),
            'cpu_percent': process.cpu_percent(),
            'max_file_size_mb': settings.max_file_size_mb,
            'slow_request_threshold_ms': settings.slow_request_threshold_ms,
            'optimizations': {
                'orjson_enabled': True,
                'opencv_image_processing': True,
                'shared_grpc_client': True,
                'embedding_cache': True,
                'affine_cache': True,
                'performance_middleware': True,
            },
        },
    }

    # Add GPU metrics if available
    if torch.cuda.is_available():
        health_data['performance']['gpu_memory_allocated_mb'] = round(
            torch.cuda.memory_allocated() / 1024 / 1024, 2
        )
        health_data['performance']['gpu_memory_reserved_mb'] = round(
            torch.cuda.memory_reserved() / 1024 / 1024, 2
        )
        health_data['performance']['gpu_name'] = torch.cuda.get_device_name(0)

    return health_data


@router.get('/connection_pool_info')
def connection_pool_info():
    """
    Triton gRPC connection pool statistics.

    Useful for monitoring shared client usage and A/B testing.
    """
    stats = get_client_pool_stats()

    return {
        'shared_client_pool': {
            'active': stats['active_connections'] > 0,
            'connection_count': stats['active_connections'],
            'triton_urls': stats['urls'],
        },
        'usage_info': {
            'shared_client_enabled': 'Use ?shared_client=true (default)',
            'per_request_client': 'Use ?shared_client=false for testing',
            'performance_impact': {
                'shared_mode': 'Enables batching, 400-600 RPS (recommended)',
                'per_request_mode': 'No batching, 50-100 RPS (testing only)',
            },
        },
        'testing': {
            'example_shared': "curl 'http://localhost:9600/predict/small?shared_client=true' -F 'image=@test.jpg'",
            'example_per_request': "curl 'http://localhost:9600/predict/small?shared_client=false' -F 'image=@test.jpg'",
            'check_batching': "docker compose logs triton-api | grep 'batch size'",
        },
    }
