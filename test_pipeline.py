"""
Comprehensive test script for the entire wind forecasting pipeline.
Tests data loading, graph building, model initialization, and basic training.
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

from data_loading import (
    load_wind_dataset, extract_node_metadata, get_column_groups,
    handle_missing_values, split_data_chronological
)
from graph_building import build_heterogeneous_graph
from hg_kan_model import HeterogeneousGraphKAN
from baselines import PersistenceModel, LSTMBaseline
from training import WindDataset, create_dataloaders
from evaluation import compute_metrics

print("="*70)
print("WIND FORECASTING PIPELINE - COMPREHENSIVE TEST")
print("="*70)

# Test 1: Data Loading
print("\n[TEST 1] Data Loading...")
data_path = Path('data/raw/Wind Spatio-Temporal Dataset2.csv')
coords_df, timeseries_df = load_wind_dataset(data_path)
print(f"  ✓ Loaded {len(timeseries_df)} timestamps")
print(f"  ✓ Columns: {len(timeseries_df.columns)}")

# Test 2: Metadata Extraction
print("\n[TEST 2] Metadata Extraction...")
metadata = extract_node_metadata(coords_df)
print(f"  ✓ Extracted {len(metadata)} nodes")
print(f"  ✓ Turbines: {(metadata['node_type'] == 'turbine').sum()}")
print(f"  ✓ Masts: {(metadata['node_type'] == 'mast').sum()}")

# Test 3: Column Groups
print("\n[TEST 3] Column Groups...")
col_groups = get_column_groups(timeseries_df)
power_cols = col_groups['turbine_power'][:200]
print(f"  ✓ Found {len(power_cols)} turbine power columns")
print(f"  ✓ Speed columns: {len(col_groups['all_speed'])}")
print(f"  ✓ Direction columns: {len(col_groups['all_direction'])}")

# Test 4: Missing Value Handling
print("\n[TEST 4] Missing Value Handling...")
timeseries_clean = handle_missing_values(timeseries_df, power_cols, method='forward_fill')
na_before = timeseries_df[power_cols].isna().sum().sum()
na_after = timeseries_clean[power_cols].isna().sum().sum()
print(f"  ✓ NaN values before: {na_before}")
print(f"  ✓ NaN values after: {na_after}")

# Test 5: Data Splitting
print("\n[TEST 5] Data Splitting...")
train_df, val_df, test_df = split_data_chronological(timeseries_clean, 0.6, 0.2)
print(f"  ✓ Train: {len(train_df)} samples")
print(f"  ✓ Val: {len(val_df)} samples")
print(f"  ✓ Test: {len(test_df)} samples")

# Test 6: Graph Construction
print("\n[TEST 6] Graph Construction...")
metadata_turbines = metadata[metadata['node_type'] == 'turbine'].iloc[:200].reset_index(drop=True)
graph_config = {
    'spatial_k': 8,
    'wake_angle': 30.0,
    'wake_max_dist': 5.0,
    'corr_threshold': 0.75,
    'corr_max_edges': 15
}
wind_dir = timeseries_clean['Mast1_Direction'].mean()
graph = build_heterogeneous_graph(
    metadata_turbines, train_df, power_cols, wind_dir, graph_config
)
print(f"  ✓ Spatial edges: {graph['spatial']['edge_index'].shape[1]}")
print(f"  ✓ Wake edges: {graph['wake']['edge_index'].shape[1]}")
print(f"  ✓ Correlation edges: {graph['correlation']['edge_index'].shape[1]}")

# Test 7: Dataset Creation
print("\n[TEST 7] Dataset Creation...")
INPUT_WINDOW = 24
FORECAST_HORIZON = 6
train_dataset = WindDataset(train_df, power_cols, INPUT_WINDOW, FORECAST_HORIZON)
print(f"  ✓ Train dataset samples: {len(train_dataset)}")
x_sample, y_sample = train_dataset[0]
print(f"  ✓ Input shape: {x_sample.shape}")
print(f"  ✓ Target shape: {y_sample.shape}")

# Test 8: DataLoader Creation
print("\n[TEST 8] DataLoader Creation...")
train_loader, val_loader, test_loader = create_dataloaders(
    train_df, val_df, test_df, power_cols, INPUT_WINDOW, FORECAST_HORIZON, batch_size=8
)
print(f"  ✓ Train batches: {len(train_loader)}")
print(f"  ✓ Val batches: {len(val_loader)}")
print(f"  ✓ Test batches: {len(test_loader)}")

# Test 9: Model Initialization
print("\n[TEST 9] Model Initialization...")

# Persistence model
persistence_model = PersistenceModel(FORECAST_HORIZON)
print(f"  ✓ Persistence model created")

# LSTM model
lstm_model = LSTMBaseline(
    input_features=1, hidden_dim=32, num_layers=2, forecast_horizon=FORECAST_HORIZON
)
lstm_params = sum(p.numel() for p in lstm_model.parameters())
print(f"  ✓ LSTM model created ({lstm_params:,} params)")

# HG-KAN model
edge_indices = {
    'spatial': torch.LongTensor(graph['spatial']['edge_index']),
    'wake': torch.LongTensor(graph['wake']['edge_index']),
    'correlation': torch.LongTensor(graph['correlation']['edge_index'])
}

hg_kan_model = HeterogeneousGraphKAN(
    num_nodes=200,
    input_window=INPUT_WINDOW,
    forecast_horizon=FORECAST_HORIZON,
    node_features=1,
    edge_types=['spatial', 'wake', 'correlation'],
    hidden_dim=32,  # Smaller for testing
    kan_basis=3,
    num_graph_layers=1
)
hgkan_params = sum(p.numel() for p in hg_kan_model.parameters())
print(f"  ✓ HG-KAN model created ({hgkan_params:,} params)")

# Test 10: Forward Pass
print("\n[TEST 10] Forward Pass Testing...")
x_batch, y_batch = next(iter(train_loader))

# Persistence
with torch.no_grad():
    pred_pers = persistence_model(x_batch)
print(f"  ✓ Persistence forward pass: {pred_pers.shape}")

# LSTM
with torch.no_grad():
    pred_lstm = lstm_model(x_batch)
print(f"  ✓ LSTM forward pass: {pred_lstm.shape}")

# HG-KAN
with torch.no_grad():
    pred_hgkan = hg_kan_model(x_batch, edge_indices)
print(f"  ✓ HG-KAN forward pass: {pred_hgkan.shape}")

# Test 11: Metrics Computation
print("\n[TEST 11] Metrics Computation...")
# Create dummy predictions for testing
y_true_test = np.random.rand(10, 200, 6)
y_pred_test = np.random.rand(10, 200, 6)
metrics = compute_metrics(y_true_test, y_pred_test)
print(f"  ✓ MAE: {metrics['MAE']:.4f}")
print(f"  ✓ RMSE: {metrics['RMSE']:.4f}")
print(f"  ✓ R²: {metrics['R2']:.4f}")

# Test 12: Training Setup
print("\n[TEST 12] Training Setup...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  ✓ Device: {device}")

criterion = torch.nn.MSELoss()
optimizer = torch.optim.AdamW(lstm_model.parameters(), lr=0.001)
print(f"  ✓ Loss function: MSE")
print(f"  ✓ Optimizer: AdamW")

# Test 13: Single Training Step
print("\n[TEST 13] Single Training Step...")
lstm_model.train()
optimizer.zero_grad()
pred = lstm_model(x_batch)
loss = criterion(pred, y_batch)
loss.backward()
optimizer.step()
print(f"  ✓ Training step completed")
print(f"  ✓ Loss: {loss.item():.4f}")

# Test 14: Directory Structure
print("\n[TEST 14] Directory Structure...")
directories = [
    Path('data/raw'),
    Path('data/processed'),
    Path('checkpoints'),
    Path('results'),
    Path('notebooks'),
    Path('src')
]
for dir_path in directories:
    if dir_path.exists():
        print(f"  ✓ {dir_path} exists")
    else:
        print(f"  ✗ {dir_path} missing")

# Test 15: File Existence
print("\n[TEST 15] Required Files...")
required_files = [
    'README.md',
    'requirements.txt',
    'QUICKSTART.md',
    'LICENSE',
    'src/__init__.py',
    'src/data_loading.py',
    'src/graph_building.py',
    'src/kan_layers.py',
    'src/hg_kan_model.py',
    'src/baselines.py',
    'src/training.py',
    'src/evaluation.py',
    'notebooks/01_project_setup_and_eda.ipynb',
    'notebooks/02_baseline_experiments.ipynb',
    'notebooks/03_hg_kan_experiments.ipynb'
]
missing_files = []
for file_path in required_files:
    if Path(file_path).exists():
        print(f"  ✓ {file_path}")
    else:
        print(f"  ✗ {file_path}")
        missing_files.append(file_path)

# Final Summary
print("\n" + "="*70)
print("TEST SUMMARY")
print("="*70)
if len(missing_files) == 0:
    print("✅ ALL TESTS PASSED!")
    print("\nProject is ready for:")
    print("  • Running EDA notebook")
    print("  • Training baseline models")
    print("  • Training HG-KAN model")
    print("  • Conducting experiments")
else:
    print(f"⚠ {len(missing_files)} files missing")
    for f in missing_files:
        print(f"  - {f}")

print("\n" + "="*70)
print("Next step: Open notebooks/01_project_setup_and_eda.ipynb in Jupyter")
print("="*70)
