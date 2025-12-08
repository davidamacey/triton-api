"""
Unified Triton gRPC Client Pool.

Provides both sync and async client management with connection pooling.
This is the canonical implementation - use this instead of the legacy
triton_async_client.py and triton_shared_client.py in utils/.

Features:
- Singleton pattern for connection reuse
- Thread-safe sync client access
- Async-safe async client access
- Connection pooling enables Triton dynamic batching (5-10x throughput)
- Monitoring and statistics

Usage:
    # Sync client (for background tasks, legacy code)
    client = TritonClientManager.get_sync_client("triton-api:8001")

    # Async client (for FastAPI endpoints - RECOMMENDED)
    client = await TritonClientManager.get_async_client("triton-api:8001")

    # Cleanup on shutdown
    await TritonClientManager.close_all()
"""

import asyncio
import logging
import threading
from typing import Any

import tritonclient.grpc as grpcclient
import tritonclient.grpc.aio as grpcclient_aio


logger = logging.getLogger(__name__)


class TritonClientManager:
    """
    Unified manager for Triton gRPC client connections.

    Provides both synchronous and asynchronous client access with
    connection pooling for optimal Triton batching performance.

    Global state justification:
    - Class-level dictionaries store shared client connections per URL
    - Required for connection pooling and enabling Triton's dynamic batching
    - Thread-safe locks prevent race conditions during client creation
    - Alternative (per-request clients) would disable batching and exhaust connections
    """

    # Global sync client pool (thread-safe)
    # Dictionary maps triton_url -> InferenceServerClient instance
    _sync_clients: dict[str, Any] = {}
    _sync_lock = threading.Lock()

    # Global async client pool (async-safe)
    # Dictionary maps triton_url -> AsyncInferenceServerClient instance
    _async_clients: dict[str, Any] = {}
    _async_lock: asyncio.Lock | None = None

    @classmethod
    def _get_async_lock(cls) -> asyncio.Lock:
        """Get or create async lock (must be created in event loop context)."""
        if cls._async_lock is None:
            cls._async_lock = asyncio.Lock()
        return cls._async_lock

    # =========================================================================
    # Sync Client Pool
    # =========================================================================
    @classmethod
    def get_sync_client(
        cls,
        triton_url: str = 'triton-api:8001',
        verbose: bool = False,
        ssl: bool = False,
        root_certificates: str | None = None,
        private_key: str | None = None,
        certificate_chain: str | None = None,
    ):
        """
        Get sync gRPC client with connection pooling.

        Thread-safe singleton pattern. Multiple callers share the same
        connection, enabling Triton's dynamic batching.

        Args:
            triton_url: Triton gRPC endpoint
            verbose: Enable verbose logging
            ssl: Use SSL/TLS
            root_certificates: SSL root certs path
            private_key: SSL private key path
            certificate_chain: SSL cert chain path

        Returns:
            tritonclient.grpc.InferenceServerClient instance
        """
        with cls._sync_lock:
            if triton_url not in cls._sync_clients:
                try:
                    client = grpcclient.InferenceServerClient(
                        url=triton_url,
                        verbose=verbose,
                        ssl=ssl,
                        root_certificates=root_certificates,
                        private_key=private_key,
                        certificate_chain=certificate_chain,
                    )

                    cls._sync_clients[triton_url] = client
                    logger.info(f'Created sync Triton client: {triton_url}')

                except Exception as e:
                    logger.error(f'Failed to create sync client for {triton_url}: {e}')
                    raise

            return cls._sync_clients[triton_url]

    @classmethod
    def close_sync_clients(cls):
        """Close all sync client connections."""
        with cls._sync_lock:
            for url, client in cls._sync_clients.items():
                try:
                    client.close()
                    logger.debug(f'Closed sync client: {url}')
                except Exception as e:
                    logger.warning(f'Error closing sync client {url}: {e}')
            cls._sync_clients.clear()
            logger.info('All sync Triton clients closed')

    @classmethod
    def get_sync_stats(cls) -> dict[str, Any]:
        """Get sync client pool statistics."""
        with cls._sync_lock:
            return {
                'active_connections': len(cls._sync_clients),
                'urls': list(cls._sync_clients.keys()),
                'type': 'sync',
            }

    # =========================================================================
    # Async Client Pool
    # =========================================================================
    @classmethod
    async def get_async_client(
        cls,
        triton_url: str = 'triton-api:8001',
        verbose: bool = False,
        ssl: bool = False,
        root_certificates: str | None = None,
        private_key: str | None = None,
        certificate_chain: str | None = None,
    ):
        """
        Get async gRPC client with connection pooling.

        Async-safe singleton pattern. Multiple concurrent requests share
        the same connection, enabling Triton's dynamic batching.

        RECOMMENDED for FastAPI endpoints - native async without thread overhead.

        Args:
            triton_url: Triton gRPC endpoint
            verbose: Enable verbose logging
            ssl: Use SSL/TLS
            root_certificates: SSL root certs path
            private_key: SSL private key path
            certificate_chain: SSL cert chain path

        Returns:
            tritonclient.grpc.aio.InferenceServerClient instance
        """
        async with cls._get_async_lock():
            if triton_url not in cls._async_clients:
                try:
                    client = grpcclient_aio.InferenceServerClient(
                        url=triton_url,
                        verbose=verbose,
                        ssl=ssl,
                        root_certificates=root_certificates,
                        private_key=private_key,
                        certificate_chain=certificate_chain,
                    )

                    cls._async_clients[triton_url] = client
                    logger.info(f'Created async Triton client: {triton_url} (batching enabled)')

                except Exception as e:
                    logger.error(f'Failed to create async client for {triton_url}: {e}')
                    raise

            return cls._async_clients[triton_url]

    @classmethod
    async def close_async_clients(cls):
        """Close all async client connections."""
        async with cls._get_async_lock():
            for url, client in cls._async_clients.items():
                try:
                    await client.close()
                    logger.debug(f'Closed async client: {url}')
                except Exception as e:
                    logger.warning(f'Error closing async client {url}: {e}')
            cls._async_clients.clear()
            logger.info('All async Triton clients closed')

    @classmethod
    async def get_async_stats(cls) -> dict[str, Any]:
        """Get async client pool statistics."""
        async with cls._get_async_lock():
            return {
                'active_connections': len(cls._async_clients),
                'urls': list(cls._async_clients.keys()),
                'type': 'async',
            }

    # =========================================================================
    # Combined Operations
    # =========================================================================
    @classmethod
    async def close_all(cls):
        """Close all clients (both sync and async)."""
        cls.close_sync_clients()
        await cls.close_async_clients()
        logger.info('All Triton clients closed')

    @classmethod
    async def get_all_stats(cls) -> dict[str, Any]:
        """Get combined statistics for all client pools."""
        sync_stats = cls.get_sync_stats()
        async_stats = await cls.get_async_stats()

        return {
            'sync': sync_stats,
            'async': async_stats,
            'total_connections': (
                sync_stats['active_connections'] + async_stats['active_connections']
            ),
        }


# =============================================================================
# Factory Functions (for backward compatibility)
# =============================================================================
def get_triton_client(triton_url: str = 'triton-api:8001', **kwargs):
    """
    Get sync Triton client (backward compatible).

    Prefer TritonClientManager.get_sync_client() for new code.
    """
    return TritonClientManager.get_sync_client(triton_url, **kwargs)


async def get_async_triton_client(triton_url: str = 'triton-api:8001', **kwargs):
    """
    Get async Triton client (backward compatible).

    Prefer TritonClientManager.get_async_client() for new code.
    """
    return await TritonClientManager.get_async_client(triton_url, **kwargs)


def close_all_clients():
    """Close sync clients (backward compatible)."""
    TritonClientManager.close_sync_clients()


async def close_async_clients():
    """Close async clients (backward compatible)."""
    await TritonClientManager.close_async_clients()


def get_client_pool_stats() -> dict[str, Any]:
    """Get sync pool stats (backward compatible)."""
    return TritonClientManager.get_sync_stats()


async def get_async_client_pool_stats() -> dict[str, Any]:
    """Get async pool stats (backward compatible)."""
    return await TritonClientManager.get_async_stats()
