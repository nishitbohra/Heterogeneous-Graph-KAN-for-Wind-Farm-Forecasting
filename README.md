# Heterogeneous Graph-KAN for Wind Farm Forecasting

## Project Overview

Novel deep learning framework combining **Kolmogorov-Arnold Networks (KAN)** with **heterogeneous graph neural networks** for wind power forecasting.

### Key Innovation
Replace traditional MLP-based graph convolutions with interpretable KAN layers for adaptive spatio-temporal transformations in wind farm power prediction.

## Dataset

- **Source**: Wind Spatio-Temporal Dataset2.csv
- **Scale**: 200 turbines + 3 meteorological masts
- **Duration**: 8,764 hourly measurements (Sept 2010 - Aug 2011)
- **Features**: 606 columns (speed, power, direction)

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run EDA Notebook

```bash
jupyter notebook notebooks/01_project_setup_and_eda.ipynb
```

### 3. Train Model

```python
python src/training.py
```

## Project Structure

```
Wind Spatio/
├── data/
│   ├── raw/
│   │   └── Wind Spatio-Temporal Dataset2.csv
│   └── processed/
│       └── (generated preprocessed data)
├── src/
│   ├── data_loading.py          # Data parsing and loading
│   ├── graph_building.py         # Graph construction utilities
│   ├── kan_layers.py             # KAN layer implementations
│   ├── hg_kan_model.py           # HG-KAN model
│   ├── baselines.py              # Baseline models
│   ├── training.py               # Training utilities
│   └── evaluation.py             # Evaluation and visualization
├── notebooks/
│   ├── 01_project_setup_and_eda.ipynb
│   ├── 02_baseline_experiments.ipynb
│   └── 03_hg_kan_experiments.ipynb
├── checkpoints/
│   └── (saved model checkpoints)
├── results/
│   └── (figures and metrics)
├── README.md
├── requirements.txt
└── Dataset_Information.md
```

## Model Architecture

### HG-KAN Components

1. **Node Encoders**: Type-specific KAN encoders for turbines and masts
2. **Heterogeneous Graph Convolution**: Edge-type-specific KAN message passing
   - Spatial proximity edges (k-NN based)
   - Wake edges (directional, wind-based)
   - Correlation edges (power correlation)
3. **Temporal Module**: Conv1D/Attention over time sequences
4. **KAN Decoder**: Per-turbine multi-horizon forecasting

### KAN Layer

```python
KAN(x) = Σ_i c_i * φ_i(x)
```

Where:
- `φ_i(x)`: B-spline or Chebyshev basis functions
- `c_i`: Learnable coefficients

## Training Configuration

- **Input Window (L)**: 24-48 hours
- **Prediction Horizon (H)**: 1, 3, or 6 hours
- **Train/Val/Test**: 60% / 20% / 20% (chronological)
- **Loss**: MAE on turbine power
- **Optimizer**: AdamW with ReduceLROnPlateau
- **Hidden Dim**: 32-64
- **Basis Functions**: 3-5

## Baseline Models

- Persistence (last value)
- Univariate LSTM
- GCN-GRU
- ST-GAT

## Evaluation Metrics

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)
- R² Score

## Key Features

✅ Spatio-temporal graph modeling  
✅ Interpretable KAN transformations  
✅ Heterogeneous node/edge types  
✅ Multi-horizon forecasting  
✅ Wake effect modeling  
✅ Correlation-based connectivity  

## Citation

*Manuscript in preparation for Q1 2026 journal submission*

## License

Research code - see license file for details.

## Contact

For questions about the implementation, please refer to the documentation in the notebooks.
