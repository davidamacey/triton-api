"""
Retry utilities for guaranteed request processing.

Implements exponential backoff with jitter for resilient inference.
Used to ensure requests are processed even under high load.
"""

import asyncio
import logging
import random
from functools import wraps
from typing import Callable, TypeVar

import grpc

logger = logging.getLogger(__name__)

T = TypeVar('T')

# Triton gRPC error codes that are retryable
RETRYABLE_GRPC_CODES = {
    grpc.StatusCode.UNAVAILABLE,      # Server overloaded
    grpc.StatusCode.RESOURCE_EXHAUSTED,  # Queue full
    grpc.StatusCode.DEADLINE_EXCEEDED,   # Timeout
    grpc.StatusCode.ABORTED,          # Request aborted
}


class RetryExhaustedError(Exception):
    """Raised when all retry attempts are exhausted."""

    def __init__(self, message: str, last_error: Exception | None = None):
        super().__init__(message)
        self.last_error = last_error


def is_retryable_error(error: Exception) -> bool:
    """
    Check if an error is retryable.

    Args:
        error: The exception to check

    Returns:
        True if the error is retryable
    """
    # Check for gRPC errors
    if hasattr(error, 'code') and callable(error.code):
        return error.code() in RETRYABLE_GRPC_CODES

    # Check error message for common retryable patterns
    error_str = str(error).lower()
    retryable_patterns = [
        'queue full',
        'resource exhausted',
        'unavailable',
        'timeout',
        'deadline exceeded',
        'connection refused',
        'server overloaded',
    ]
    return any(pattern in error_str for pattern in retryable_patterns)


async def retry_async(
    func: Callable[..., T],
    *args,
    max_retries: int = 3,
    base_delay: float = 0.1,
    max_delay: float = 5.0,
    exponential_base: float = 2.0,
    jitter: float = 0.1,
    **kwargs,
) -> T:
    """
    Retry an async function with exponential backoff and jitter.

    Args:
        func: Async function to retry
        *args: Positional arguments for func
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay between retries
        exponential_base: Base for exponential backoff
        jitter: Random jitter factor (0-1)
        **kwargs: Keyword arguments for func

    Returns:
        Result from successful function call

    Raises:
        RetryExhaustedError: If all retries are exhausted
    """
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            last_error = e

            if not is_retryable_error(e):
                # Non-retryable error, raise immediately
                raise

            if attempt == max_retries:
                # Last attempt failed
                break

            # Calculate delay with exponential backoff
            delay = min(base_delay * (exponential_base ** attempt), max_delay)

            # Add jitter to prevent thundering herd
            jitter_amount = delay * jitter * random.random()
            delay += jitter_amount

            logger.warning(
                f'Retry {attempt + 1}/{max_retries} after {delay:.2f}s: {e}'
            )

            await asyncio.sleep(delay)

    raise RetryExhaustedError(
        f'All {max_retries} retries exhausted',
        last_error=last_error,
    )


def retry_sync(
    func: Callable[..., T],
    *args,
    max_retries: int = 3,
    base_delay: float = 0.1,
    max_delay: float = 5.0,
    exponential_base: float = 2.0,
    jitter: float = 0.1,
    **kwargs,
) -> T:
    """
    Retry a sync function with exponential backoff and jitter.

    Same as retry_async but for synchronous functions.
    """
    import time

    last_error = None

    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_error = e

            if not is_retryable_error(e):
                raise

            if attempt == max_retries:
                break

            delay = min(base_delay * (exponential_base ** attempt), max_delay)
            jitter_amount = delay * jitter * random.random()
            delay += jitter_amount

            logger.warning(
                f'Retry {attempt + 1}/{max_retries} after {delay:.2f}s: {e}'
            )

            time.sleep(delay)

    raise RetryExhaustedError(
        f'All {max_retries} retries exhausted',
        last_error=last_error,
    )


def with_retry(
    max_retries: int = 3,
    base_delay: float = 0.1,
    max_delay: float = 5.0,
):
    """
    Decorator to add retry logic to async functions.

    Usage:
        @with_retry(max_retries=5)
        async def my_inference_function():
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            return await retry_async(
                func, *args,
                max_retries=max_retries,
                base_delay=base_delay,
                max_delay=max_delay,
                **kwargs,
            )
        return wrapper
    return decorator
