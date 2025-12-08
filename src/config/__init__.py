"""
Configuration module for the YOLO inference service.

Provides centralized configuration using Pydantic Settings with environment variable support.
"""

from src.config.settings import Settings, get_settings


__all__ = ['Settings', 'get_settings']
