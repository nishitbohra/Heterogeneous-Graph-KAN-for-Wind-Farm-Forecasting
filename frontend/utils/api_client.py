"""
API client for communicating with FastAPI backend.
"""

import requests
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class APIClient:
    """Client for HG-KAN API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize API client.
        
        Args:
            base_url: Base URL of the FastAPI backend
        """
        self.base_url = base_url.rstrip('/')
        self.api_v1 = f"{self.base_url}/api/v1"
    
    def health_check(self) -> Dict:
        """Check API health status."""
        try:
            response = requests.get(f"{self.api_v1}/health", timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {"status": "unavailable", "error": str(e)}
    
    def get_model_info(self) -> Dict:
        """Get model information."""
        try:
            response = requests.get(f"{self.api_v1}/info", timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get model info: {str(e)}")
            return {}
    
    def predict(
        self,
        input_data: np.ndarray,
        edge_indices: Optional[Dict] = None,
        edge_weights: Optional[Dict] = None,
        forecast_horizon: Optional[int] = None
    ) -> Tuple[Optional[np.ndarray], Optional[Dict], Optional[str]]:
        """
        Send prediction request.
        
        Args:
            input_data: Input array [num_nodes, input_window]
            edge_indices: Optional edge indices
            edge_weights: Optional edge weights
            forecast_horizon: Optional horizon override
            
        Returns:
            predictions: Numpy array or None
            metadata: Response metadata or None
            error: Error message or None
        """
        try:
            # Prepare request payload
            payload = {
                "input_data": input_data.tolist(),
            }
            
            if edge_indices:
                payload["edge_indices"] = edge_indices
            
            if edge_weights:
                payload["edge_weights"] = edge_weights
            
            if forecast_horizon:
                payload["forecast_horizon"] = forecast_horizon
            
            # Send request
            response = requests.post(
                f"{self.api_v1}/predict",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                predictions = np.array(data['predictions'])
                metadata = data.get('metadata', {})
                return predictions, metadata, None
            else:
                error_detail = response.json().get('detail', 'Unknown error')
                return None, None, f"API Error: {error_detail}"
        
        except requests.exceptions.Timeout:
            return None, None, "Request timeout - server may be busy"
        except requests.exceptions.ConnectionError:
            return None, None, "Connection error - is the API server running?"
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return None, None, f"Error: {str(e)}"
    
    def evaluate(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray
    ) -> Tuple[Optional[Dict], Optional[str]]:
        """
        Send evaluation request.
        
        Args:
            predictions: Predictions array
            ground_truth: Ground truth array
            
        Returns:
            metrics: Metrics dictionary or None
            error: Error message or None
        """
        try:
            payload = {
                "predictions": predictions.tolist(),
                "ground_truth": ground_truth.tolist()
            }
            
            response = requests.post(
                f"{self.api_v1}/evaluate",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json(), None
            else:
                error_detail = response.json().get('detail', 'Unknown error')
                return None, f"API Error: {error_detail}"
        
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            return None, f"Error: {str(e)}"
