"""
Health check and system information endpoints.
"""

from fastapi import APIRouter, Depends
from datetime import datetime
from ..schemas import HealthResponse, ModelInfo
from ..config import settings
from typing import Optional

router = APIRouter(tags=["Health"])

# Global model service reference (will be injected)
_model_service = None


def set_model_service(service):
    """Set the model service for health checks."""
    global _model_service
    _model_service = service


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns system status and model availability.
    """
    model_loaded = _model_service.is_loaded() if _model_service else False
    
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        timestamp=datetime.now(),
        version=settings.APP_VERSION,
        model_loaded=model_loaded
    )


@router.get("/info", response_model=ModelInfo)
async def model_info():
    """
    Get model information.
    
    Returns model architecture and configuration details.
    """
    if not _model_service or not _model_service.is_loaded():
        return ModelInfo(
            model_type="Not Loaded",
            num_nodes=0,
            input_window=0,
            forecast_horizon=0,
            hidden_dim=0,
            num_basis=0,
            edge_types=[]
        )
    
    info = _model_service.get_model_info()
    return ModelInfo(**info)


@router.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "description": "Heterogeneous Graph-KAN for Wind Power Forecasting",
        "endpoints": {
            "health": f"{settings.API_V1_PREFIX}/health",
            "info": f"{settings.API_V1_PREFIX}/info",
            "predict": f"{settings.API_V1_PREFIX}/predict",
            "evaluate": f"{settings.API_V1_PREFIX}/evaluate",
        },
        "documentation": "/docs"
    }
