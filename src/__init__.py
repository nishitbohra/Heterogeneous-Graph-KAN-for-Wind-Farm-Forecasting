"""
Heterogeneous Graph-KAN Wind Forecasting Package

This package provides tools for wind power forecasting using
Heterogeneous Graph Neural Networks with Kolmogorov-Arnold Network layers.

Modules:
    data_loading: Dataset loading and preprocessing
    graph_building: Heterogeneous graph construction
    kan_layers: KAN layer implementations
    hg_kan_model: HG-KAN model architecture
    baselines: Baseline model implementations
    training: Training utilities and loops
    evaluation: Evaluation metrics and visualizations
"""

__version__ = "1.0.0"
__author__ = "Wind Forecasting Research Team"

# Import key functions for easy access
from .data_loading import (
    load_wind_dataset,
    extract_node_metadata,
    get_column_groups,
    handle_missing_values,
    split_data_chronological
)

from .graph_building import (
    build_spatial_edges_knn,
    build_wake_edges,
    build_correlation_edges,
    build_heterogeneous_graph
)

from .kan_layers import (
    KANLayer,
    KANLinear,
    MultiLayerKAN
)

from .hg_kan_model import (
    HeterogeneousGraphKAN
)

from .baselines import (
    PersistenceModel,
    LSTMBaseline,
    GraphConvLSTM,
    STGATBaseline
)

from .training import (
    WindDataset,
    create_dataloaders,
    train_model,
    load_checkpoint
)

from .evaluation import (
    compute_metrics,
    evaluate_model,
    plot_training_history,
    plot_predictions,
    compare_models
)

__all__ = [
    # Data loading
    'load_wind_dataset',
    'extract_node_metadata',
    'get_column_groups',
    'handle_missing_values',
    'split_data_chronological',
    
    # Graph building
    'build_spatial_edges_knn',
    'build_wake_edges',
    'build_correlation_edges',
    'build_heterogeneous_graph',
    
    # KAN layers
    'KANLayer',
    'KANLinear',
    'MultiLayerKAN',
    
    # Models
    'HeterogeneousGraphKAN',
    'PersistenceModel',
    'LSTMBaseline',
    'GraphConvLSTM',
    'STGATBaseline',
    
    # Training
    'WindDataset',
    'create_dataloaders',
    'train_model',
    'load_checkpoint',
    
    # Evaluation
    'compute_metrics',
    'evaluate_model',
    'plot_training_history',
    'plot_predictions',
    'compare_models',
]
