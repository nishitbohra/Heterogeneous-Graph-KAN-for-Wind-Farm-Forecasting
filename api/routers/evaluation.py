"""
Evaluation endpoints for computing metrics.
"""

from fastapi import APIRouter, HTTPException, status
from ..schemas import EvaluationRequest, MetricsResponse, ErrorResponse
from ..services import EvaluationService
import numpy as np
import logging

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Evaluation"])


@router.post("/evaluate", response_model=MetricsResponse)
async def evaluate(request: EvaluationRequest):
    """
    Evaluate predictions against ground truth.
    
    Computes MAE, RMSE, MAPE, R², and NRMSE metrics.
    """
    try:
        # Convert to numpy arrays
        predictions = np.array(request.predictions)
        ground_truth = np.array(request.ground_truth)
        
        logger.info(f"Evaluating predictions: shape {predictions.shape}")
        
        # Validate shapes match
        if predictions.shape != ground_truth.shape:
            raise ValueError(
                f"Shape mismatch: predictions {predictions.shape} vs "
                f"ground truth {ground_truth.shape}"
            )
        
        # Compute overall metrics
        metrics = EvaluationService.compute_metrics(ground_truth, predictions)
        
        # Compute per-horizon metrics if 3D data
        per_horizon = None
        if predictions.ndim == 3:
            per_horizon = EvaluationService.compute_metrics_per_horizon(
                ground_truth, predictions
            )
        
        logger.info(f"Evaluation complete: MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}")
        
        return MetricsResponse(
            mae=metrics['mae'],
            rmse=metrics['rmse'],
            mape=metrics['mape'],
            r2=metrics['r2'],
            nrmse=metrics['nrmse'],
            per_horizon=per_horizon
        )
    
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    except Exception as e:
        logger.error(f"Evaluation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


def __init__():
    pass
