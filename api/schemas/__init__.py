"""
Pydantic schemas for API requests and responses.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
import numpy as np


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    timestamp: datetime = Field(default_factory=datetime.now)
    version: str
    model_loaded: bool


class ModelInfo(BaseModel):
    """Model information response."""
    model_type: str
    num_nodes: int
    input_window: int
    forecast_horizon: int
    hidden_dim: int
    num_basis: int
    edge_types: List[str]
    parameters: Optional[int] = None


class PredictionRequest(BaseModel):
    """Request for wind power prediction."""
    
    # Input time series data
    input_data: List[List[float]] = Field(
        ..., 
        description="Input time series [num_nodes, input_window]"
    )
    
    # Optional: pre-computed graph
    edge_indices: Optional[Dict[str, List[List[int]]]] = Field(
        None,
        description="Pre-computed edge indices by type"
    )
    
    edge_weights: Optional[Dict[str, List[float]]] = Field(
        None,
        description="Pre-computed edge weights by type"
    )
    
    # Metadata (optional)
    node_metadata: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Node metadata (coordinates, types) for graph construction"
    )
    
    # Configuration overrides
    forecast_horizon: Optional[int] = Field(
        None,
        description="Override default forecast horizon"
    )
    
    @validator('input_data')
    def validate_input_shape(cls, v):
        if not v or not v[0]:
            raise ValueError("input_data cannot be empty")
        
        num_nodes = len(v)
        input_window = len(v[0])
        
        if num_nodes <= 0:
            raise ValueError("Number of nodes must be positive")
        if input_window <= 0:
            raise ValueError("Input window must be positive")
        
        # Check consistency
        for i, node_data in enumerate(v):
            if len(node_data) != input_window:
                raise ValueError(f"Node {i} has inconsistent input window length")
        
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "input_data": [
                    [0.5, 0.6, 0.55, 0.58],  # Node 0
                    [0.45, 0.52, 0.48, 0.51],  # Node 1
                ],
                "forecast_horizon": 6
            }
        }


class PredictionResponse(BaseModel):
    """Response with predictions."""
    
    predictions: List[List[float]] = Field(
        ...,
        description="Predicted values [num_nodes, forecast_horizon]"
    )
    
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the prediction"
    )
    
    model_type: str
    forecast_horizon: int
    num_nodes: int
    
    class Config:
        json_schema_extra = {
            "example": {
                "predictions": [
                    [0.57, 0.59, 0.61, 0.60, 0.58, 0.56],  # Node 0
                    [0.49, 0.51, 0.53, 0.52, 0.50, 0.48],  # Node 1
                ],
                "metadata": {
                    "inference_time_ms": 45.2,
                    "input_window": 24,
                },
                "model_type": "HG-KAN",
                "forecast_horizon": 6,
                "num_nodes": 2
            }
        }


class EvaluationRequest(BaseModel):
    """Request for model evaluation."""
    
    predictions: List[List[float]] = Field(
        ...,
        description="Predicted values [num_samples, num_nodes, forecast_horizon]"
    )
    
    ground_truth: List[List[float]] = Field(
        ...,
        description="Ground truth values [num_samples, num_nodes, forecast_horizon]"
    )
    
    @validator('predictions', 'ground_truth')
    def validate_shape(cls, v):
        if not v:
            raise ValueError("Data cannot be empty")
        return v


class MetricsResponse(BaseModel):
    """Response with evaluation metrics."""
    
    mae: float = Field(..., description="Mean Absolute Error")
    rmse: float = Field(..., description="Root Mean Squared Error")
    mape: float = Field(..., description="Mean Absolute Percentage Error")
    r2: float = Field(..., description="R-squared score")
    nrmse: float = Field(..., description="Normalized RMSE")
    
    per_horizon: Optional[List[Dict[str, float]]] = Field(
        None,
        description="Metrics per forecast horizon"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "mae": 0.142,
                "rmse": 0.178,
                "mape": 15.3,
                "r2": 0.298,
                "nrmse": 0.962,
                "per_horizon": [
                    {"horizon": 1, "mae": 0.125, "rmse": 0.156},
                    {"horizon": 2, "mae": 0.138, "rmse": 0.172},
                ]
            }
        }


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class BatchPredictionRequest(BaseModel):
    """Request for batch predictions."""
    
    batch_data: List[List[List[float]]] = Field(
        ...,
        description="Batch of input data [batch_size, num_nodes, input_window]"
    )
    
    @validator('batch_data')
    def validate_batch(cls, v):
        if not v:
            raise ValueError("Batch cannot be empty")
        
        # Check consistency
        batch_size = len(v)
        if batch_size == 0:
            raise ValueError("Batch size must be positive")
        
        num_nodes = len(v[0])
        input_window = len(v[0][0])
        
        for i, sample in enumerate(v):
            if len(sample) != num_nodes:
                raise ValueError(f"Sample {i} has inconsistent number of nodes")
            for j, node in enumerate(sample):
                if len(node) != input_window:
                    raise ValueError(f"Sample {i}, node {j} has inconsistent input window")
        
        return v


class BatchPredictionResponse(BaseModel):
    """Response for batch predictions."""
    
    predictions: List[List[List[float]]] = Field(
        ...,
        description="Batch predictions [batch_size, num_nodes, forecast_horizon]"
    )
    
    batch_size: int
    inference_time_ms: float
