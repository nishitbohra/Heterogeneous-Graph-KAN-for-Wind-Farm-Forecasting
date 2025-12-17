# Heterogeneous Graph-KAN Wind Forecasting Project

This directory contains the source code for the wind power forecasting models.

## Module Overview

### `data_loading.py`
Data loading and preprocessing utilities:
- `load_wind_dataset()`: Load CSV with special coordinate structure
- `extract_node_metadata()`: Parse node coordinates and types
- `handle_missing_values()`: Handle NaN values in power data
- `create_temporal_features()`: Generate cyclical time features
- `split_data_chronological()`: Split into train/val/test sets

### `graph_building.py`
Heterogeneous graph construction:
- `build_spatial_edges_knn()`: k-NN spatial proximity edges
- `build_wake_edges()`: Directional wake effect edges
- `build_correlation_edges()`: Power correlation-based edges
- `build_heterogeneous_graph()`: Combine all edge types

### `kan_layers.py`
Kolmogorov-Arnold Network layer implementations:
- `BSplineBasis`: B-spline basis functions
- `KANLayer`: Basic KAN layer with learnable coefficients
- `KANLinear`: KAN with residual connections and normalization
- `MultiLayerKAN`: Stacked KAN layers

### `hg_kan_model.py`
Main HG-KAN model architecture:
- `HeterogeneousGraphConv`: Multi-edge-type graph convolution
- `TemporalConvModule`: 1D CNN for temporal patterns
- `TemporalAttention`: Multi-head attention for temporal modeling
- `HeterogeneousGraphKAN`: Complete end-to-end model

### `baselines.py`
Baseline model implementations:
- `PersistenceModel`: Naive persistence baseline
- `LSTMBaseline`: Vanilla LSTM without graph structure
- `GraphConvLSTM`: GCN + LSTM combination
- `STGATBaseline`: Spatio-temporal graph attention network

### `training.py`
Training utilities:
- `WindDataset`: PyTorch dataset for sliding windows
- `create_dataloaders()`: Create train/val/test loaders
- `train_epoch()`: Single epoch training loop
- `train_model()`: Full training with early stopping and checkpointing
- `load_checkpoint()`: Load saved model weights

### `evaluation.py`
Evaluation and visualization utilities:
- `compute_metrics()`: MAE, RMSE, MAPE, R², NRMSE
- `compute_metrics_per_horizon()`: Metrics for each forecast step
- `evaluate_model()`: Generate predictions on test set
- `plot_training_history()`: Training curves
- `plot_predictions()`: Time series predictions
- `plot_scatter()`: Prediction vs ground truth scatter
- `compare_models()`: Side-by-side model comparison
- `save_results()`: Save metrics and predictions

## Usage Example

```python
import sys
sys.path.append('..')

from src.data_loading import load_wind_dataset, extract_node_metadata
from src.graph_building import build_heterogeneous_graph
from src.hg_kan_model import HeterogeneousGraphKAN
from src.training import create_dataloaders, train_model
from src.evaluation import evaluate_model, compute_metrics

# Load data
coords_df, timeseries_df = load_wind_dataset('data/raw/Wind Spatio-Temporal Dataset2.csv')
metadata = extract_node_metadata(coords_df)

# Build graph
graph = build_heterogeneous_graph(metadata, timeseries_df, power_cols, wind_dir, config)

# Create model
model = HeterogeneousGraphKAN(
    num_nodes=200,
    input_window=24,
    forecast_horizon=6,
    node_features=1,
    edge_types=['spatial', 'wake', 'correlation']
)

# Train
history = train_model(model, train_loader, val_loader, config, edge_indices, edge_weights)

# Evaluate
y_true, y_pred = evaluate_model(model, test_loader, device, edge_indices, edge_weights)
metrics = compute_metrics(y_true, y_pred)
```

## Key Design Principles

1. **Modularity**: Each module has clear responsibilities and interfaces
2. **Flexibility**: Models support various configurations and hyperparameters
3. **Efficiency**: Memory-conscious implementations for large graphs
4. **Reproducibility**: Consistent random seeding and checkpointing
5. **Extensibility**: Easy to add new edge types, models, or metrics

## Dependencies

See `requirements.txt` in project root:
- PyTorch 2.0+
- PyTorch Geometric 2.3+
- NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn

## Testing

Each module includes a `__main__` block for quick testing:

```bash
python src/data_loading.py
python src/kan_layers.py
python src/hg_kan_model.py
python src/baselines.py
```

## Notes

- Graph edges are stored as PyTorch LongTensor [2, num_edges]
- Edge weights are FloatTensor [num_edges]
- All models use batch-first convention: [batch, nodes, time, features]
- Device handling is automatic (CUDA if available, else CPU)
- Early stopping monitors validation loss with patience
- Checkpoints saved every 10 epochs + best model

For detailed usage, see notebooks in `../notebooks/`.
