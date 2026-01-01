"""
FastAPI routers for different tracks.

Each router handles a specific performance track:
- track_a: PyTorch direct inference (baseline)
- triton: Tracks B/C/D via Triton Inference Server
- track_e: Visual search with MobileCLIP (DALI preprocessing)
- track_f: CPU preprocessing + direct TRT (no DALI)
- models: Model upload, export, and management
- health: Health checks and monitoring
"""

from src.routers.health import router as health_router
from src.routers.models import router as models_router
from src.routers.track_a import router as track_a_router
from src.routers.track_e import router as track_e_router
from src.routers.track_f import router as track_f_router
from src.routers.triton import router as triton_router


__all__ = [
    'health_router',
    'models_router',
    'track_a_router',
    'track_e_router',
    'track_f_router',
    'triton_router',
]
