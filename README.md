# Heterogeneous Graph-KAN for Wind Farm Forecasting

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-orange.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Project Overview

Novel deep learning framework combining **Kolmogorov-Arnold Networks (KAN)** with **heterogeneous graph neural networks** for wind power forecasting. This repository includes both the research codebase and a **production-ready web application** with FastAPI backend and Streamlit frontend.

### Key Innovation
Replace traditional MLP-based graph convolutions with interpretable KAN layers for adaptive spatio-temporal transformations in wind farm power prediction.

### 🌟 New: Web Application
- 🚀 **FastAPI REST API** for model serving
- 🎨 **Streamlit Dashboard** for interactive predictions
- 🐳 **Docker Support** for easy deployment
- ☁️ **Cloud-Ready** for AWS/GCP/Azure

## Dataset

- **Source**: Wind Spatio-Temporal Dataset2.csv
- **Scale**: 200 turbines + 3 meteorological masts
- **Duration**: 8,764 hourly measurements (Sept 2010 - Aug 2011)
- **Features**: 606 columns (speed, power, direction)

## Quick Start

### Option 1: Web Application (Recommended for End Users)

```bash
# 1. Install dependencies
pip install -r requirements.txt
pip install -r requirements-api.txt
pip install -r requirements-frontend.txt

# 2. Start API server (Terminal 1)
python -m api.main

# 3. Start Streamlit frontend (Terminal 2)
streamlit run frontend/streamlit_app.py

# 4. Open browser
# Frontend: http://localhost:8501
# API Docs: http://localhost:8000/docs
```

### Option 2: Docker Deployment

```bash
# Build and start all services
docker-compose up --build

# Access:
# Frontend: http://localhost:8501
# API: http://localhost:8000
```

### Option 3: Research/Development

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run EDA Notebook
jupyter notebook notebooks/01_project_setup_and_eda.ipynb

# 3. Train Model
python src/training.py
```

## Project Structure

```
Heterogeneous-Graph-KAN-for-Wind-Farm-Forecasting/
├── api/                          # FastAPI Backend
│   ├── main.py                   # API entry point
│   ├── config.py                 # Configuration management
│   ├── routers/                  # API endpoints
│   │   ├── health.py            # Health check
│   │   ├── prediction.py        # Prediction endpoints
│   │   └── evaluation.py        # Evaluation endpoints
│   ├── schemas/                  # Pydantic models
│   └── services/                 # Business logic
│       ├── model_service.py     # Model management
│       └── __init__.py          # Evaluation service
├── frontend/                     # Streamlit Frontend
│   ├── streamlit_app.py         # Main UI application
│   └── utils/
│       └── api_client.py        # Backend client
├── src/                          # Core ML Models
│   ├── data_loading.py          # Data parsing
│   ├── graph_building.py         # Graph construction
│   ├── kan_layers.py             # KAN implementations
│   ├── hg_kan_model.py           # HG-KAN model
│   ├── baselines.py              # Baseline models
│   ├── training.py               # Training utilities
│   └── evaluation.py             # Evaluation tools
├── notebooks/                    # Research notebooks
│   ├── 01_project_setup_and_eda.ipynb
│   ├── 02_baseline_experiments.ipynb
│   └── 03_hg_kan_experiments.ipynb
├── data/                         # Datasets
│   ├── raw/                      # Original data
│   └── processed/                # Preprocessed data
├── checkpoints/                  # Model checkpoints
├── results/                      # Predictions & metrics
├── docs/                         # Documentation
│   ├── API.md                    # API documentation
│   ├── DEPLOYMENT.md             # Deployment guide
│   └── ARCHITECTURE.md           # Architecture docs
├── Dockerfile                    # Docker configuration
├── docker-compose.yml            # Multi-container setup
├── requirements.txt              # Core dependencies
├── requirements-api.txt          # API dependencies
├── requirements-frontend.txt     # Frontend dependencies
├── .env.example                  # Environment template
└── README.md                     # This file
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

### Research Features
✅ Spatio-temporal graph modeling  
✅ Interpretable KAN transformations  
✅ Heterogeneous node/edge types  
✅ Multi-horizon forecasting  
✅ Wake effect modeling  
✅ Correlation-based connectivity  

### Deployment Features
✅ **FastAPI REST API** - Production-ready backend  
✅ **Streamlit Dashboard** - User-friendly interface  
✅ **Docker Support** - Containerized deployment  
✅ **Pydantic Validation** - Type-safe API  
✅ **Interactive Visualization** - Real-time charts  
✅ **Batch Predictions** - Efficient processing  
✅ **Model Evaluation** - Comprehensive metrics  
✅ **Cloud-Ready** - Deploy to AWS/GCP/Azure  

## Web Application Features

### 🎨 User Interface
- **File Upload**: CSV, NPY, NPZ formats
- **Sample Data Generation**: Test without data
- **Real-time Predictions**: Fast inference
- **Interactive Visualizations**: Charts and heatmaps
- **Model Evaluation**: Compare predictions
- **Results Export**: Download predictions

### 🚀 API Endpoints

#### Health & Info
- `GET /api/v1/health` - System status
- `GET /api/v1/info` - Model information

#### Predictions
- `POST /api/v1/predict` - Single prediction
- `POST /api/v1/predict/batch` - Batch predictions

#### Evaluation
- `POST /api/v1/evaluate` - Compute metrics

**Full API Documentation**: `http://localhost:8000/docs`

## Usage Examples

### Python API Client

```python
import requests
import numpy as np

# API endpoint
API_URL = "http://localhost:8000/api/v1"

# Prepare data (200 turbines, 24 hours)
input_data = np.random.rand(200, 24).tolist()

# Make prediction
response = requests.post(
    f"{API_URL}/predict",
    json={"input_data": input_data}
)

result = response.json()
predictions = np.array(result['predictions'])
print(f"Shape: {predictions.shape}")  # (200, 6)
```

### Streamlit Interface

1. Upload CSV or generate sample data
2. Click "Generate Prediction"
3. View results in interactive dashboard
4. Download predictions as CSV/NPY

## Deployment

### Local Development

```bash
# Terminal 1: API
python -m api.main

# Terminal 2: Frontend
streamlit run frontend/streamlit_app.py
```

### Docker

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Cloud Deployment

#### AWS EC2
```bash
# SSH into instance
ssh -i key.pem ubuntu@<instance-ip>

# Clone and deploy
git clone <repo-url>
cd Heterogeneous-Graph-KAN-for-Wind-Farm-Forecasting
docker-compose up -d
```

#### Heroku
```bash
heroku create hg-kan-app
heroku container:push web
heroku container:release web
```

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for detailed instructions.

## Documentation

- 📖 [API Documentation](docs/API.md) - REST API reference
- 🏗️ [Architecture](docs/ARCHITECTURE.md) - System design
- 🚀 [Deployment Guide](docs/DEPLOYMENT.md) - Deployment instructions
- 📔 [Notebooks](notebooks/) - Research experiments  

## Citation

*Manuscript in preparation for Q1 2026 journal submission*

## License

Research code - see license file for details.

## Contact

For questions about the implementation, please refer to the documentation in the notebooks.
