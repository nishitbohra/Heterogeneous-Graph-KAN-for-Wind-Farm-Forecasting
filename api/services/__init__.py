"""
Evaluation service for computing metrics.
"""

import numpy as np
from typing import Dict, List
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging

logger = logging.getLogger(__name__)


class EvaluationService:
    """Service for evaluation metrics computation."""
    
    @staticmethod
    def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Compute forecasting metrics.
        
        Args:
            y_true: Ground truth array
            y_pred: Predictions array
            
        Returns:
            Dictionary of metrics
        """
        # Flatten arrays
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        # Remove NaN values
        mask = ~(np.isnan(y_true_flat) | np.isnan(y_pred_flat))
        y_true_flat = y_true_flat[mask]
        y_pred_flat = y_pred_flat[mask]
        
        if len(y_true_flat) == 0:
            logger.warning("No valid data points for metric computation")
            return {
                "mae": float('nan'),
                "rmse": float('nan'),
                "mape": float('nan'),
                "r2": float('nan'),
                "nrmse": float('nan'),
            }
        
        # Compute metrics
        mae = mean_absolute_error(y_true_flat, y_pred_flat)
        rmse = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
        
        # MAPE (avoid division by zero)
        mask_nonzero = y_true_flat != 0
        if mask_nonzero.sum() > 0:
            mape = np.mean(np.abs((y_true_flat[mask_nonzero] - y_pred_flat[mask_nonzero]) / 
                                  y_true_flat[mask_nonzero])) * 100
        else:
            mape = float('nan')
        
        # R-squared
        try:
            r2 = r2_score(y_true_flat, y_pred_flat)
        except:
            r2 = float('nan')
        
        # Normalized RMSE
        mean_true = y_true_flat.mean()
        nrmse = rmse / (mean_true + 1e-8) if mean_true != 0 else float('nan')
        
        return {
            "mae": round(float(mae), 6),
            "rmse": round(float(rmse), 6),
            "mape": round(float(mape), 6),
            "r2": round(float(r2), 6),
            "nrmse": round(float(nrmse), 6),
        }
    
    @staticmethod
    def compute_metrics_per_horizon(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> List[Dict[str, float]]:
        """
        Compute metrics for each forecast horizon.
        
        Args:
            y_true: [num_samples, num_nodes, horizon]
            y_pred: [num_samples, num_nodes, horizon]
            
        Returns:
            List of metric dictionaries, one per horizon
        """
        if y_true.ndim != 3 or y_pred.ndim != 3:
            logger.warning("Expected 3D arrays for per-horizon metrics")
            return []
        
        horizon = y_true.shape[2]
        metrics_list = []
        
        for h in range(horizon):
            y_true_h = y_true[:, :, h]
            y_pred_h = y_pred[:, :, h]
            
            metrics = EvaluationService.compute_metrics(y_true_h, y_pred_h)
            metrics['horizon'] = h + 1
            metrics_list.append(metrics)
        
        return metrics_list


def __init__():
    """Initialize services package."""
    pass
