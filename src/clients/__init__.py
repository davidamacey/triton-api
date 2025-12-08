"""
Client modules for external services.

Provides unified interfaces for Triton Inference Server and OpenSearch.

Triton Clients:
- TritonClient: Unified client for all tracks (C, D, E)
- TritonClientManager: Connection pool manager (from triton_pool)

Legacy clients (deprecated, use TritonClient instead):
- triton.py: Old Track D/E client
- triton_end2end_client.py: Old Track C client
- triton_track_e_client.py: Old Track E client
"""

from src.clients.opensearch import OpenSearchClient
from src.clients.triton_client import TritonClient, get_triton_client
from src.clients.triton_pool import TritonClientManager


__all__ = [
    # OpenSearch
    'OpenSearchClient',
    # Unified Triton client (NEW)
    'TritonClient',
    'TritonClientManager',
    'get_triton_client',
]
