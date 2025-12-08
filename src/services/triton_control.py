"""
Triton Model Control Service.

Provides interface for dynamic model loading/unloading via Triton's HTTP API.
"""

import logging
from typing import Any

import httpx

from src.config import get_settings


logger = logging.getLogger(__name__)


class TritonControlService:
    """Service for controlling Triton model repository."""

    def __init__(self):
        settings = get_settings()
        host = settings.triton_host
        self.http_url = f'http://{host}:8000'
        self.timeout = 60.0

    async def load_model(self, model_name: str) -> tuple[bool, str]:
        """Load a model into Triton."""
        endpoint = f'{self.http_url}/v2/repository/models/{model_name}/load'

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(endpoint)

                if response.status_code == 200:
                    logger.info(f'Loaded model {model_name} into Triton')
                    return True, f'Model {model_name} loaded successfully'

                logger.warning(f'Failed to load {model_name}: {response.text}')
                return False, f'Load failed: {response.text}'

        except httpx.TimeoutException:
            return False, f'Timeout loading model {model_name}'
        except httpx.ConnectError:
            return False, 'Cannot connect to Triton server'
        except Exception as e:
            return False, f'Load error: {e}'

    async def unload_model(self, model_name: str) -> tuple[bool, str]:
        """Unload a model from Triton."""
        endpoint = f'{self.http_url}/v2/repository/models/{model_name}/unload'

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(endpoint)

                if response.status_code == 200:
                    logger.info(f'Unloaded model {model_name} from Triton')
                    return True, f'Model {model_name} unloaded successfully'

                return False, f'Unload failed: {response.text}'

        except Exception as e:
            return False, f'Unload error: {e}'

    async def get_repository_index(self) -> list[dict[str, Any]]:
        """Get list of all models in repository."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(f'{self.http_url}/v2/repository/index')
                if response.status_code == 200:
                    return response.json()
        except Exception as e:
            logger.warning(f'Failed to get repository index: {e}')
        return []

    async def get_model_config(self, model_name: str) -> dict[str, Any] | None:
        """Get model configuration from Triton."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f'{self.http_url}/v2/models/{model_name}/config')
                if response.status_code == 200:
                    return response.json()
        except Exception:
            pass
        return None

    async def server_ready(self) -> bool:
        """Check if Triton server is ready."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f'{self.http_url}/v2/health/ready')
                return response.status_code == 200
        except Exception:
            return False
