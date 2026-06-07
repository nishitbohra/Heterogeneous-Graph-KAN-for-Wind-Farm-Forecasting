"""
Prediction endpoints for wind power forecasting.
"""

from fastapi import APIRouter, HTTPException, status
from ..schemas import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ErrorResponse
)
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Prediction"])

# Global model service reference
_model_service = None


def set_model_service(service):
    """Set the model service for predictions."""
    global _model_service
    _model_service = service


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Generate wind power predictions.
    
    Takes input time series data and returns forecasted values.
    """
    try:
        # Check if model is loaded
        if not _model_service or not _model_service.is_loaded():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded. Please wait for model initialization."
            )
        
        # Convert input data to numpy array
        input_data = np.array(request.input_data)  # [num_nodes, input_window]
        
        # Validate shape
        num_nodes, input_window = input_data.shape
        logger.info(f"Received prediction request: {num_nodes} nodes, {input_window} timesteps")
        
        # Convert edge indices and weights if provided
        edge_indices = None
        edge_weights = None
        
        if request.edge_indices:
            edge_indices = {
                k: torch.LongTensor(v) for k, v in request.edge_indices.items()
            }
        
        if request.edge_weights:
            edge_weights = {
                k: torch.FloatTensor(v) for k, v in request.edge_weights.items()
            }
        
        # Build graph from metadata if provided but no edges given
        if request.node_metadata and not edge_indices:
            from ..config import GRAPH_CONFIG
            edge_indices, edge_weights = _model_service.build_graph_from_metadata(
                request.node_metadata,
                GRAPH_CONFIG
            )
        
        # Run prediction
        predictions, metadata = _model_service.predict(
            input_data,
            edge_indices=edge_indices,
            edge_weights=edge_weights,
            forecast_horizon=request.forecast_horizon
        )
        
        # Convert to list for JSON serialization
        predictions_list = predictions.tolist()
        
        logger.info(f"Prediction completed in {metadata['inference_time_ms']:.2f}ms")
        
        return PredictionResponse(
            predictions=predictions_list,
            metadata=metadata,
            model_type="HG-KAN",
            forecast_horizon=predictions.shape[1],
            num_nodes=predictions.shape[0]
        )
    
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Generate predictions for a batch of inputs.
    
    Useful for processing multiple time windows efficiently.
    """
    try:
        if not _model_service or not _model_service.is_loaded():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded"
            )
        
        # Convert to numpy array
        batch_data = np.array(request.batch_data)  # [batch, nodes, window]
        
        batch_size, num_nodes, input_window = batch_data.shape
        logger.info(f"Batch prediction: {batch_size} samples, {num_nodes} nodes, {input_window} steps")
        
        # Run batch prediction
        predictions, metadata = _model_service.predict_batch(batch_data)
        
        # Convert to list
        predictions_list = predictions.tolist()
        
        return BatchPredictionResponse(
            predictions=predictions_list,
            batch_size=batch_size,
            inference_time_ms=metadata['inference_time_ms']
        )
    
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


def __init__():
    pass
