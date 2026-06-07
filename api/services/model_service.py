"""
Model service for loading and running inference.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import logging
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from hg_kan_model import HeterogeneousGraphKAN
from graph_building import build_heterogeneous_graph
from data_loading import extract_node_metadata
import pandas as pd

logger = logging.getLogger(__name__)


class ModelService:
    """Service for model loading and inference."""
    
    def __init__(self, config: dict, device: str = "cpu"):
        """
        Initialize model service.
        
        Args:
            config: Model configuration dictionary
            device: Device to run model on ("cpu" or "cuda")
        """
        self.config = config
        self.device = torch.device(device)
        self.model = None
        self.graph_cache = {}
        
        logger.info(f"Initializing ModelService on {device}")
    
    def load_model(self, checkpoint_path: Optional[str] = None) -> bool:
        """
        Load model from checkpoint or initialize new model.
        
        Args:
            checkpoint_path: Path to model checkpoint (optional)
            
        Returns:
            Success status
        """
        try:
            # Initialize model
            self.model = HeterogeneousGraphKAN(
                num_nodes=self.config['num_nodes'],
                input_window=self.config['input_window'],
                forecast_horizon=self.config['forecast_horizon'],
                node_features=self.config['node_features'],
                hidden_dim=self.config['hidden_dim'],
                edge_types=self.config['edge_types'],
                num_basis=self.config['num_basis'],
                use_temporal_conv=self.config.get('use_temporal_conv', True),
                use_attention=self.config.get('use_attention', True),
            )
            
            # Load checkpoint if provided
            if checkpoint_path and Path(checkpoint_path).exists():
                logger.info(f"Loading checkpoint from {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                
                # Handle different checkpoint formats
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                
                logger.info("Checkpoint loaded successfully")
            else:
                logger.warning("No checkpoint provided, using randomly initialized model")
            
            # Move to device and set to eval mode
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("Model loaded and ready for inference")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return False
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None
    
    def get_model_info(self) -> dict:
        """Get model information."""
        if not self.is_loaded():
            return {}
        
        # Count parameters
        num_params = sum(p.numel() for p in self.model.parameters())
        
        return {
            "model_type": "HG-KAN",
            "num_nodes": self.config['num_nodes'],
            "input_window": self.config['input_window'],
            "forecast_horizon": self.config['forecast_horizon'],
            "hidden_dim": self.config['hidden_dim'],
            "num_basis": self.config['num_basis'],
            "edge_types": self.config['edge_types'],
            "parameters": num_params,
            "device": str(self.device),
        }
    
    @torch.no_grad()
    def predict(
        self,
        input_data: np.ndarray,
        edge_indices: Optional[Dict[str, torch.Tensor]] = None,
        edge_weights: Optional[Dict[str, torch.Tensor]] = None,
        forecast_horizon: Optional[int] = None,
    ) -> Tuple[np.ndarray, dict]:
        """
        Run inference on input data.
        
        Args:
            input_data: Input array [num_nodes, input_window] or [num_nodes, input_window, features]
            edge_indices: Optional pre-computed edge indices
            edge_weights: Optional pre-computed edge weights
            forecast_horizon: Optional override for forecast horizon
            
        Returns:
            predictions: [num_nodes, forecast_horizon]
            metadata: Dictionary with inference info
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")
        
        import time
        start_time = time.time()
        
        # Prepare input tensor
        if input_data.ndim == 2:
            # Add feature dimension: [num_nodes, input_window, 1]
            input_data = input_data[:, :, np.newaxis]
        
        # Convert to tensor
        x = torch.FloatTensor(input_data).to(self.device)
        
        # Add batch dimension: [1, num_nodes, input_window, features]
        x = x.unsqueeze(0)
        
        # Use default graph if not provided
        if edge_indices is None or edge_weights is None:
            edge_indices, edge_weights = self._get_default_graph()
        
        # Move graph to device
        edge_indices = {k: v.to(self.device) for k, v in edge_indices.items()}
        if edge_weights:
            edge_weights = {k: v.to(self.device) for k, v in edge_weights.items()}
        
        # Run inference
        predictions = self.model(x, edge_indices, edge_weights)
        
        # Remove batch dimension and convert to numpy
        predictions = predictions.squeeze(0).cpu().numpy()
        
        inference_time = (time.time() - start_time) * 1000  # ms
        
        metadata = {
            "inference_time_ms": round(inference_time, 2),
            "input_shape": list(input_data.shape),
            "output_shape": list(predictions.shape),
            "device": str(self.device),
        }
        
        return predictions, metadata
    
    @torch.no_grad()
    def predict_batch(
        self,
        batch_data: np.ndarray,
        edge_indices: Optional[Dict[str, torch.Tensor]] = None,
        edge_weights: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[np.ndarray, dict]:
        """
        Run batch inference.
        
        Args:
            batch_data: [batch_size, num_nodes, input_window] or 
                       [batch_size, num_nodes, input_window, features]
            edge_indices: Optional pre-computed edge indices
            edge_weights: Optional pre-computed edge weights
            
        Returns:
            predictions: [batch_size, num_nodes, forecast_horizon]
            metadata: Dictionary with inference info
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")
        
        import time
        start_time = time.time()
        
        # Prepare input
        if batch_data.ndim == 3:
            batch_data = batch_data[:, :, :, np.newaxis]
        
        x = torch.FloatTensor(batch_data).to(self.device)
        
        # Use default graph if not provided
        if edge_indices is None or edge_weights is None:
            edge_indices, edge_weights = self._get_default_graph()
        
        # Move graph to device
        edge_indices = {k: v.to(self.device) for k, v in edge_indices.items()}
        if edge_weights:
            edge_weights = {k: v.to(self.device) for k, v in edge_weights.items()}
        
        # Run inference
        predictions = self.model(x, edge_indices, edge_weights)
        predictions = predictions.cpu().numpy()
        
        inference_time = (time.time() - start_time) * 1000
        
        metadata = {
            "inference_time_ms": round(inference_time, 2),
            "batch_size": batch_data.shape[0],
            "time_per_sample_ms": round(inference_time / batch_data.shape[0], 2),
        }
        
        return predictions, metadata
    
    def _get_default_graph(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Get default graph structure (cached or dummy).
        
        Returns:
            edge_indices, edge_weights
        """
        # Check cache
        if 'default' in self.graph_cache:
            return self.graph_cache['default']
        
        # Create dummy fully-connected graph
        num_nodes = self.config['num_nodes']
        
        # Simple spatial edges (k-NN-like pattern)
        k = min(8, num_nodes - 1)
        edge_list = []
        for i in range(num_nodes):
            for j in range(max(0, i-k//2), min(num_nodes, i+k//2+1)):
                if i != j:
                    edge_list.append([i, j])
        
        edge_index = torch.LongTensor(edge_list).T
        edge_weight = torch.ones(edge_index.shape[1])
        
        edge_indices = {'spatial': edge_index}
        edge_weights = {'spatial': edge_weight}
        
        # Cache it
        self.graph_cache['default'] = (edge_indices, edge_weights)
        
        logger.warning("Using dummy graph structure (k-NN pattern)")
        
        return edge_indices, edge_weights
    
    def build_graph_from_metadata(
        self,
        metadata: List[Dict],
        graph_config: dict,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Build graph from node metadata.
        
        Args:
            metadata: List of node metadata dicts with lat/lon
            graph_config: Graph construction configuration
            
        Returns:
            edge_indices, edge_weights
        """
        try:
            # Convert to dataframe
            metadata_df = pd.DataFrame(metadata)
            
            # Build graph using existing utilities
            from graph_building import (
                build_spatial_edges_knn,
                build_wake_edges,
                build_correlation_edges
            )
            
            edge_indices = {}
            edge_weights = {}
            
            # Spatial edges
            if graph_config.get('include_spatial', True):
                spatial_edges, spatial_weights = build_spatial_edges_knn(
                    metadata_df,
                    k=graph_config.get('knn_k', 8)
                )
                edge_indices['spatial'] = torch.LongTensor(spatial_edges)
                edge_weights['spatial'] = torch.FloatTensor(spatial_weights)
            
            return edge_indices, edge_weights
            
        except Exception as e:
            logger.error(f"Failed to build graph: {str(e)}")
            return self._get_default_graph()
