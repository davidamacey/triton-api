"""
FastAPI dependency injection for shared resources.

Uses FastAPI's Depends() pattern for proper lifecycle management.
Resources are created once and reused across requests.
"""

import logging
from typing import Annotated, Any

from fastapi import Depends

from src.clients.opensearch import OpenSearchClient
from src.clients.triton_pool import TritonClientManager
from src.config.settings import Settings, get_settings


logger = logging.getLogger(__name__)


# =============================================================================
# Type Aliases for Dependency Injection
# =============================================================================
SettingsDep = Annotated[Settings, Depends(get_settings)]


# =============================================================================
# Application State (managed by lifespan context)
# =============================================================================
class AppState:
    """
    Application state container for shared resources.

    Resources are initialized in lifespan and accessed via dependencies.
    This avoids global state while providing singleton-like behavior.
    """

    def __init__(self):
        self.pytorch_models: dict[str, Any] = {}
        self._triton_sync_client = None
        self._triton_async_client = None
        self._opensearch_client = None

    @property
    def triton_url(self) -> str:
        return get_settings().triton_url

    @property
    def opensearch_url(self) -> str:
        return get_settings().opensearch_url


# Global app state - initialized in lifespan
app_state = AppState()


# =============================================================================
# Triton Client Factory
# =============================================================================
class TritonClientFactory:
    """
    Factory for creating Triton inference clients.

    Provides both sync and async clients with connection pooling.
    """

    @staticmethod
    def get_sync_client():
        """Get sync Triton gRPC client with connection pooling."""
        return TritonClientManager.get_sync_client(app_state.triton_url)

    @staticmethod
    async def get_async_client():
        """Get async Triton gRPC client (lazy initialization)."""
        if app_state._triton_async_client is None:
            logger.info(f'Initializing async Triton client ({app_state.triton_url})...')
            app_state._triton_async_client = await TritonClientManager.get_async_client(
                app_state.triton_url
            )
            logger.info('Async Triton client ready (batching enabled)')
        return app_state._triton_async_client

    @staticmethod
    async def close_all():
        """Close all Triton connections."""
        await TritonClientManager.close_all()
        app_state._triton_async_client = None
        logger.info('All Triton connections closed')


# =============================================================================
# OpenSearch Client Factory
# =============================================================================
class OpenSearchClientFactory:
    """Factory for OpenSearch client with lazy initialization."""

    @staticmethod
    async def get_client():
        """Get OpenSearch client (lazy initialization)."""
        if app_state._opensearch_client is None:
            settings = get_settings()

            logger.info(f'Initializing OpenSearch client ({settings.opensearch_url})...')
            app_state._opensearch_client = OpenSearchClient(
                hosts=[settings.opensearch_url], http_auth=None, timeout=settings.opensearch_timeout
            )

            if not await app_state._opensearch_client.ping():
                raise RuntimeError(f'OpenSearch connection failed: {settings.opensearch_url}')

            logger.info('OpenSearch client ready')
        return app_state._opensearch_client

    @staticmethod
    async def close():
        """Close OpenSearch client."""
        if app_state._opensearch_client is not None:
            await app_state._opensearch_client.close()
            app_state._opensearch_client = None
            logger.info('OpenSearch client closed')


# =============================================================================
# VisualSearchService Factory
# =============================================================================
class VisualSearchServiceFactory:
    """Factory for VisualSearchService with lazy initialization."""

    _instance = None

    @classmethod
    async def get_service(cls):
        """Get VisualSearchService (lazy initialization)."""
        if cls._instance is None:
            from src.services.visual_search import VisualSearchService

            opensearch_client = await OpenSearchClientFactory.get_client()
            cls._instance = VisualSearchService(opensearch_client)
            logger.info('VisualSearchService initialized')
        return cls._instance

    @classmethod
    async def close(cls):
        """Close service resources."""
        cls._instance = None


# =============================================================================
# FastAPI Dependencies (use with Depends())
# =============================================================================
async def get_async_triton():
    """Dependency for async Triton client."""
    return await TritonClientFactory.get_async_client()


async def get_opensearch():
    """Dependency for OpenSearch client."""
    return await OpenSearchClientFactory.get_client()


async def get_visual_search_service():
    """Dependency for VisualSearchService."""
    return await VisualSearchServiceFactory.get_service()


def get_pytorch_models() -> dict[str, Any]:
    """Dependency for PyTorch models."""
    return app_state.pytorch_models


# Type aliases for cleaner endpoint signatures
AsyncTritonDep = Annotated[Any, Depends(get_async_triton)]
OpenSearchDep = Annotated[Any, Depends(get_opensearch)]
PyTorchModelsDep = Annotated[dict[str, Any], Depends(get_pytorch_models)]

# Import VisualSearchService for type annotation
from src.services.visual_search import VisualSearchService  # noqa: E402


VisualSearchDep = Annotated[VisualSearchService, Depends(get_visual_search_service)]
