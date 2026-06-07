"""
FastAPI main application for HG-KAN Wind Forecasting.
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from .config import settings, MODEL_CONFIG, GRAPH_CONFIG
from .services.model_service import ModelService
from .routers import health_router, prediction_router, evaluation_router
from .routers.health import set_model_service as set_health_model_service
from .routers.prediction import set_model_service as set_prediction_model_service

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format=settings.LOG_FORMAT
)
logger = logging.getLogger(__name__)

# Global model service
model_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    logger.info("="*60)
    logger.info("Starting HG-KAN Wind Forecasting API")
    logger.info("="*60)
    
    global model_service
    
    # Initialize model service
    logger.info("Initializing model service...")
    model_service = ModelService(MODEL_CONFIG, device=settings.DEVICE)
    
    # Load model
    checkpoint_path = settings.MODEL_CHECKPOINT
    if checkpoint_path:
        logger.info(f"Loading model from checkpoint: {checkpoint_path}")
    else:
        logger.warning("No checkpoint specified, using random initialization")
    
    success = model_service.load_model(checkpoint_path)
    
    if success:
        logger.info("✓ Model loaded successfully")
        model_info = model_service.get_model_info()
        logger.info(f"  - Model type: {model_info['model_type']}")
        logger.info(f"  - Parameters: {model_info.get('parameters', 'N/A')}")
        logger.info(f"  - Device: {model_info['device']}")
    else:
        logger.error("✗ Model loading failed")
    
    # Inject model service into routers
    set_health_model_service(model_service)
    set_prediction_model_service(model_service)
    
    logger.info(f"API server ready on {settings.HOST}:{settings.PORT}")
    logger.info(f"Documentation available at: http://{settings.HOST}:{settings.PORT}/docs")
    logger.info("="*60)
    
    yield
    
    # Shutdown
    logger.info("Shutting down HG-KAN API...")
    logger.info("Goodbye!")


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="""
    **Heterogeneous Graph-KAN Wind Power Forecasting API**
    
    This API provides endpoints for wind power forecasting using a novel
    deep learning architecture that combines Kolmogorov-Arnold Networks (KAN)
    with heterogeneous graph neural networks.
    
    ## Features
    
    - 🔮 **Multi-horizon forecasting**: Predict 1-6 hours ahead
    - 🌐 **Graph-based**: Leverages spatial and temporal relationships
    - 🧠 **Interpretable**: KAN layers provide adaptive transformations
    - ⚡ **Fast inference**: Optimized for real-time predictions
    
    ## Quick Start
    
    1. Upload time series data (24-48 hours of historical power data)
    2. Call `/predict` endpoint with your data
    3. Receive forecasts for all turbines
    
    ## Model Architecture
    
    The model uses three types of edges to capture wind farm dynamics:
    - **Spatial**: Geographic proximity between turbines
    - **Wake**: Directional wake effects based on wind direction
    - **Correlation**: Power output correlation patterns
    """,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health_router, prefix=settings.API_V1_PREFIX)
app.include_router(prediction_router, prefix=settings.API_V1_PREFIX)
app.include_router(evaluation_router, prefix=settings.API_V1_PREFIX)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.DEBUG else "An unexpected error occurred"
        }
    )


# Root endpoint (redirects to docs)
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "online",
        "docs": "/docs",
        "health": f"{settings.API_V1_PREFIX}/health",
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        log_level=settings.LOG_LEVEL.lower(),
    )
