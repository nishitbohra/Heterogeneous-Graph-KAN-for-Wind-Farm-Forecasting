"""
Graph construction utilities for heterogeneous wind farm graph.
Builds spatial, wake, and correlation-based edges.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors


def compute_distance_matrix(metadata: pd.DataFrame) -> np.ndarray:
    """
    Compute pairwise distances between nodes using Haversine formula.
    
    Args:
        metadata: DataFrame with latitude, longitude columns
        
    Returns:
        Distance matrix (km) of shape [num_nodes, num_nodes]
    """
    coords = metadata[['latitude', 'longitude']].values
    
    # Haversine formula for great circle distance
    lat1, lon1 = coords[:, 0:1], coords[:, 1:2]
    lat2, lon2 = coords[:, 0], coords[:, 1]
    
    # Convert to radians
    lat1_rad = np.deg2rad(lat1)
    lat2_rad = np.deg2rad(lat2)
    dlat = np.deg2rad(lat2 - lat1)
    dlon = np.deg2rad(lon2 - lon1)
    
    # Haversine formula
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Earth radius in km
    R = 6371.0
    distances = R * c
    
    return distances


def build_spatial_edges_knn(metadata: pd.DataFrame, 
                            k: int = 8,
                            max_distance_km: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build spatial proximity edges using k-NN.
    
    Args:
        metadata: Node metadata with coordinates
        k: Number of nearest neighbors
        max_distance_km: Maximum distance threshold (optional)
        
    Returns:
        edge_index: [2, num_edges] array of edge indices
        edge_weights: Edge weights (inverse distance)
    """
    coords = metadata[['latitude', 'longitude']].values
    
    # Fit k-NN
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='haversine')
    nbrs.fit(np.deg2rad(coords))
    
    # Find neighbors
    distances, indices = nbrs.kneighbors(np.deg2rad(coords))
    
    # Convert distances to km
    distances_km = distances * 6371.0
    
    # Build edges (exclude self-loops)
    edge_list = []
    edge_weights = []
    
    for i in range(len(coords)):
        for j in range(1, k+1):  # Skip first (self)
            neighbor_idx = indices[i, j]
            dist = distances_km[i, j]
            
            # Apply distance threshold if specified
            if max_distance_km is None or dist <= max_distance_km:
                edge_list.append([i, neighbor_idx])
                # Inverse distance weighting (add small epsilon to avoid division by zero)
                edge_weights.append(1.0 / (dist + 0.01))
    
    edge_index = np.array(edge_list).T  # Shape: [2, num_edges]
    edge_weights = np.array(edge_weights)
    
    return edge_index, edge_weights


def build_spatial_edges_radius(metadata: pd.DataFrame,
                               radius_km: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build spatial edges within a distance radius.
    
    Args:
        metadata: Node metadata with coordinates
        radius_km: Distance threshold in kilometers
        
    Returns:
        edge_index: [2, num_edges] array
        edge_weights: Edge weights (inverse distance)
    """
    distance_matrix = compute_distance_matrix(metadata)
    
    # Find edges within radius
    edge_list = []
    edge_weights = []
    
    for i in range(len(metadata)):
        for j in range(i+1, len(metadata)):  # Upper triangle only
            dist = distance_matrix[i, j]
            if dist <= radius_km:
                # Add undirected edge
                edge_list.append([i, j])
                edge_list.append([j, i])
                
                weight = 1.0 / (dist + 0.01)
                edge_weights.append(weight)
                edge_weights.append(weight)
    
    edge_index = np.array(edge_list).T
    edge_weights = np.array(edge_weights)
    
    return edge_index, edge_weights


def build_wake_edges(metadata: pd.DataFrame,
                    wind_direction: float,
                    wake_angle: float = 30.0,
                    max_distance_km: float = 5.0) -> np.ndarray:
    """
    Build directed wake edges based on wind direction.
    
    Args:
        metadata: Node metadata with coordinates
        wind_direction: Dominant wind direction in degrees (0-360)
        wake_angle: Angular tolerance for wake effect (degrees)
        max_distance_km: Maximum wake propagation distance
        
    Returns:
        edge_index: [2, num_edges] directed edges (upstream -> downstream)
    """
    coords = metadata[['latitude', 'longitude']].values
    n_nodes = len(coords)
    
    # Convert to radians
    wind_dir_rad = np.deg2rad(wind_direction)
    wake_angle_rad = np.deg2rad(wake_angle)
    
    edge_list = []
    
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i == j:
                continue
            
            # Vector from i to j
            dx = coords[j, 1] - coords[i, 1]  # longitude diff
            dy = coords[j, 0] - coords[i, 0]  # latitude diff
            
            # Distance
            dist_km = np.sqrt((dx * 111)**2 + (dy * 111)**2)  # Approximate km
            
            if dist_km > max_distance_km:
                continue
            
            # Angle from i to j
            angle_to_j = np.arctan2(dy, dx)
            
            # Angle difference with wind direction
            angle_diff = np.abs(angle_to_j - wind_dir_rad)
            angle_diff = np.minimum(angle_diff, 2*np.pi - angle_diff)  # Wrap around
            
            # Check if j is downstream of i (within wake cone)
            if angle_diff <= wake_angle_rad:
                edge_list.append([i, j])  # Directed: i influences j
    
    if len(edge_list) == 0:
        return np.array([[], []])
    
    edge_index = np.array(edge_list).T
    return edge_index


def build_correlation_edges(timeseries_df: pd.DataFrame,
                           power_cols: List[str],
                           threshold: float = 0.7,
                           max_edges_per_node: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build edges based on power time series correlation.
    
    Args:
        timeseries_df: Time series data
        power_cols: List of power column names
        threshold: Correlation threshold (0-1)
        max_edges_per_node: Maximum edges per node to limit graph density
        
    Returns:
        edge_index: [2, num_edges] array
        edge_weights: Correlation values as edge weights
    """
    # Compute correlation matrix
    power_data = timeseries_df[power_cols].fillna(0)
    corr_matrix = power_data.corr().values
    
    # Set diagonal to 0 (no self-loops)
    np.fill_diagonal(corr_matrix, 0)
    
    edge_list = []
    edge_weights = []
    
    n_nodes = len(power_cols)
    
    for i in range(n_nodes):
        # Get correlations for node i
        correlations = corr_matrix[i, :]
        
        # Find nodes with correlation above threshold
        high_corr_indices = np.where(correlations >= threshold)[0]
        
        # Limit number of edges per node (keep highest correlations)
        if len(high_corr_indices) > max_edges_per_node:
            sorted_indices = high_corr_indices[np.argsort(-correlations[high_corr_indices])]
            high_corr_indices = sorted_indices[:max_edges_per_node]
        
        for j in high_corr_indices:
            if i < j:  # Avoid duplicates (undirected)
                edge_list.append([i, j])
                edge_list.append([j, i])
                edge_weights.append(correlations[j])
                edge_weights.append(correlations[j])
    
    if len(edge_list) == 0:
        return np.array([[], []]), np.array([])
    
    edge_index = np.array(edge_list).T
    edge_weights = np.array(edge_weights)
    
    return edge_index, edge_weights


def build_heterogeneous_graph(metadata: pd.DataFrame,
                              timeseries_df: pd.DataFrame,
                              power_cols: List[str],
                              wind_direction: float,
                              config: Dict) -> Dict:
    """
    Build complete heterogeneous graph with multiple edge types.
    
    Args:
        metadata: Node metadata
        timeseries_df: Time series data
        power_cols: Power column names
        wind_direction: Dominant wind direction
        config: Configuration dict with graph building parameters
        
    Returns:
        Dictionary with edge_index and edge_weights for each edge type
    """
    # Spatial edges (k-NN)
    spatial_edges, spatial_weights = build_spatial_edges_knn(
        metadata, 
        k=config.get('spatial_k', 8),
        max_distance_km=config.get('spatial_max_dist', None)
    )
    
    # Wake edges (directional)
    wake_edges = build_wake_edges(
        metadata,
        wind_direction,
        wake_angle=config.get('wake_angle', 30.0),
        max_distance_km=config.get('wake_max_dist', 5.0)
    )
    
    # Correlation edges
    corr_edges, corr_weights = build_correlation_edges(
        timeseries_df,
        power_cols,
        threshold=config.get('corr_threshold', 0.7),
        max_edges_per_node=config.get('corr_max_edges', 20)
    )
    
    graph_dict = {
        'spatial': {
            'edge_index': spatial_edges,
            'edge_weight': spatial_weights
        },
        'wake': {
            'edge_index': wake_edges,
            'edge_weight': np.ones(wake_edges.shape[1]) if wake_edges.size > 0 else np.array([])
        },
        'correlation': {
            'edge_index': corr_edges,
            'edge_weight': corr_weights
        },
        'metadata': metadata,
        'num_nodes': len(metadata)
    }
    
    return graph_dict


def print_graph_statistics(graph_dict: Dict):
    """Print statistics about the constructed graph."""
    print("=" * 60)
    print("HETEROGENEOUS GRAPH STATISTICS")
    print("=" * 60)
    print(f"\nNumber of nodes: {graph_dict['num_nodes']}")
    
    for edge_type in ['spatial', 'wake', 'correlation']:
        edge_index = graph_dict[edge_type]['edge_index']
        num_edges = edge_index.shape[1] if edge_index.size > 0 else 0
        print(f"\n{edge_type.upper()} EDGES:")
        print(f"  Total edges: {num_edges}")
        if num_edges > 0:
            print(f"  Average degree: {num_edges / graph_dict['num_nodes']:.2f}")
    
    print("=" * 60)


if __name__ == "__main__":
    # Example usage
    from data_loading import load_wind_dataset, extract_node_metadata, get_column_groups
    from pathlib import Path
    
    data_path = Path("data/raw/Wind Spatio-Temporal Dataset2.csv")
    
    # Load data
    coords_df, timeseries_df = load_wind_dataset(data_path)
    metadata = extract_node_metadata(coords_df)
    col_groups = get_column_groups(timeseries_df)
    
    # Build graph
    config = {
        'spatial_k': 8,
        'wake_angle': 30.0,
        'wake_max_dist': 5.0,
        'corr_threshold': 0.75,
        'corr_max_edges': 15
    }
    
    # Use mean wind direction from Mast1
    wind_dir = timeseries_df['Mast1_Direction'].mean()
    
    graph = build_heterogeneous_graph(
        metadata,
        timeseries_df,
        col_groups['turbine_power'][:200],  # Turbines only
        wind_dir,
        config
    )
    
    print_graph_statistics(graph)
