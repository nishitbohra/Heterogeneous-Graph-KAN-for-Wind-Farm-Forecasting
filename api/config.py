"""
Configuration management for FastAPI backend.
"""

from pydantic_settings import BaseSettings
from pathlib import Path
from typing import List, Optional


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Application
    APP_NAME: str = "HG-KAN Wind Forecasting API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # API Configuration
    API_V1_PREFIX: str = "/api/v1"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    RELOAD: bool = False
    
    # CORS
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:8501",  # Streamlit default
        "http://localhost:3000",
        "http://localhost:8000",
    ]
    
    # File Upload
    MAX_UPLOAD_SIZE_MB: int = 100
    ALLOWED_EXTENSIONS: List[str] = [".csv", ".npz", ".npy"]
    
    # Model Configuration
    MODEL_TYPE: str = "HG-KAN"
    NUM_NODES: int = 200
    INPUT_WINDOW: int = 24
    FORECAST_HORIZON: int = 6
    HIDDEN_DIM: int = 64
    NUM_BASIS: int = 5
    
    # Graph Configuration
    EDGE_TYPES: List[str] = ["spatial", "wake", "correlation"]
    KNN_K: int = 8
    WAKE_MAX_DISTANCE_KM: float = 5.0
    CORRELATION_THRESHOLD: float = 0.7
    
    # Paths (relative to project root)
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    CHECKPOINT_DIR: Path = PROJECT_ROOT / "checkpoints"
    RESULTS_DIR: Path = PROJECT_ROOT / "results"
    CACHE_DIR: Path = PROJECT_ROOT / "data" / "processed"
    
    # Model Checkpoint
    MODEL_CHECKPOINT: Optional[str] = None  # Set to path if available
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Performance
    DEVICE: str = "cpu"  # "cuda" if GPU available
    NUM_WORKERS: int = 4
    BATCH_SIZE: int = 32
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()


# Model configuration dictionary
MODEL_CONFIG = {
    "num_nodes": settings.NUM_NODES,
    "input_window": settings.INPUT_WINDOW,
    "forecast_horizon": settings.FORECAST_HORIZON,
    "node_features": 1,
    "hidden_dim": settings.HIDDEN_DIM,
    "edge_types": settings.EDGE_TYPES,
    "num_basis": settings.NUM_BASIS,
    "use_temporal_conv": True,
    "use_attention": True,
}


# Graph construction configuration
GRAPH_CONFIG = {
    "knn_k": settings.KNN_K,
    "wake_max_distance_km": settings.WAKE_MAX_DISTANCE_KM,
    "correlation_threshold": settings.CORRELATION_THRESHOLD,
    "include_spatial": True,
    "include_wake": True,
    "include_correlation": True,
}
