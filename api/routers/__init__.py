"""
API routers package.
"""

from .health import router as health_router
from .prediction import router as prediction_router
from .evaluation import router as evaluation_router

__all__ = ['health_router', 'prediction_router', 'evaluation_router']
