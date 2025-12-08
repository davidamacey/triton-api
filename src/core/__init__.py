"""
Core module with shared dependencies and exception handling.

Provides FastAPI dependency injection and custom exceptions.
"""

from src.core.dependencies import (
    AsyncTritonDep,
    OpenSearchClientFactory,
    OpenSearchDep,
    PyTorchModelsDep,
    TritonClientFactory,
    app_state,
    get_async_triton,
    get_opensearch,
    get_pytorch_models,
)
from src.core.exceptions import (
    ClientConnectionError,
    InferenceError,
    InvalidImageError,
    ModelNotFoundError,
)


__all__ = [
    # Type aliases
    'AsyncTritonDep',
    # Exceptions
    'ClientConnectionError',
    'InferenceError',
    'InvalidImageError',
    'ModelNotFoundError',
    # Client factories
    'OpenSearchClientFactory',
    'OpenSearchDep',
    'PyTorchModelsDep',
    'TritonClientFactory',
    # App state
    'app_state',
    # FastAPI dependencies
    'get_async_triton',
    'get_opensearch',
    'get_pytorch_models',
]
