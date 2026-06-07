# Software Architecture

## HG-KAN Wind Forecasting System Architecture

This document describes the software architecture of the Heterogeneous Graph-KAN Wind Power Forecasting application.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture Diagram](#architecture-diagram)
3. [Component Description](#component-description)
4. [Data Flow](#data-flow)
5. [Design Patterns](#design-patterns)
6. [Technology Stack](#technology-stack)
7. [Scalability & Performance](#scalability--performance)

---

## Overview

The system follows a **three-tier architecture** with clear separation of concerns:

1. **Presentation Layer**: Streamlit frontend for user interaction
2. **Application Layer**: FastAPI backend for business logic
3. **Data/Model Layer**: PyTorch models and data processing

### Key Design Principles

- **Modularity**: Each component has well-defined responsibilities
- **Separation of Concerns**: Frontend, backend, and models are decoupled
- **RESTful API**: Standard HTTP/JSON communication
- **Stateless Services**: Enables horizontal scaling
- **Container-Ready**: Docker-based deployment
- **Production-Grade**: Logging, error handling, validation

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                          USER LAYER                             │
│  - Web Browser                                                  │
│  - HTTP/HTTPS Client                                            │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                           │
│                    (Streamlit Frontend)                         │
├─────────────────────────────────────────────────────────────────┤
│  Components:                                                    │
│  ├─ streamlit_app.py          Main application                 │
│  ├─ utils/api_client.py       Backend communication           │
│  └─ components/               UI components                    │
│      ├─ upload.py             File upload handling             │
│      ├─ visualization.py      Plotting and charts              │
│      └─ results.py            Results display                  │
│                                                                 │
│  Responsibilities:                                              │
│  - User interaction                                            │
│  - Data upload/download                                        │
│  - Visualization rendering                                     │
│  - API communication                                           │
└────────────────────────┬───────────────────────────────────────┘
                         │ REST API (HTTP/JSON)
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                    APPLICATION LAYER                            │
│                    (FastAPI Backend)                            │
├─────────────────────────────────────────────────────────────────┤
│  API Layer (api/main.py):                                      │
│  ├─ FastAPI app initialization                                 │
│  ├─ Middleware (CORS, logging)                                 │
│  ├─ Exception handlers                                         │
│  └─ Lifespan events (startup/shutdown)                         │
│                                                                 │
│  Routers (api/routers/):                                       │
│  ├─ health.py          Health & info endpoints                 │
│  ├─ prediction.py      Prediction endpoints                    │
│  └─ evaluation.py      Evaluation endpoints                    │
│                                                                 │
│  Schemas (api/schemas/):                                       │
│  ├─ Pydantic models for validation                             │
│  ├─ Request/response contracts                                 │
│  └─ Type safety                                                │
│                                                                 │
│  Services (api/services/):                                     │
│  ├─ model_service.py   Model loading & inference               │
│  └─ EvaluationService  Metrics computation                     │
│                                                                 │
│  Configuration (api/config.py):                                │
│  ├─ Environment variables                                      │
│  ├─ Model parameters                                           │
│  └─ Graph configuration                                        │
└────────────────────────┬───────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                    DATA/MODEL LAYER                             │
│                    (PyTorch Models)                             │
├─────────────────────────────────────────────────────────────────┤
│  Core Models (src/):                                           │
│  ├─ hg_kan_model.py                                            │
│  │   ├─ HeterogeneousGraphKAN    Main model                    │
│  │   ├─ HeterogeneousGraphConv   Graph convolution             │
│  │   ├─ TemporalConvModule       Temporal processing           │
│  │   └─ TemporalAttention        Attention mechanism           │
│  │                                                              │
│  ├─ kan_layers.py                                              │
│  │   ├─ BSplineBasis             Basis functions               │
│  │   ├─ KANLayer                 KAN transformation            │
│  │   ├─ KANLinear                Linear KAN                    │
│  │   └─ MultiLayerKAN            Stacked KAN                   │
│  │                                                              │
│  ├─ graph_building.py                                          │
│  │   ├─ build_spatial_edges_knn  Proximity graph               │
│  │   ├─ build_wake_edges         Wake effect graph             │
│  │   └─ build_correlation_edges  Correlation graph             │
│  │                                                              │
│  ├─ data_loading.py                                            │
│  │   ├─ load_wind_dataset        CSV parsing                   │
│  │   ├─ extract_node_metadata    Coordinate extraction         │
│  │   └─ WindDataset               PyTorch dataset              │
│  │                                                              │
│  ├─ training.py                                                │
│  │   ├─ train_model               Training loop                │
│  │   └─ create_dataloaders        Data preparation             │
│  │                                                              │
│  └─ evaluation.py                                              │
│      ├─ compute_metrics           Metric calculation           │
│      └─ plot_* functions          Visualization                │
└────────────────────────┬───────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                     STORAGE LAYER                               │
├─────────────────────────────────────────────────────────────────┤
│  ├─ checkpoints/       Trained model weights                   │
│  ├─ data/              Input datasets                          │
│  │   ├─ raw/           Original CSV data                       │
│  │   └─ processed/     Preprocessed/cached data                │
│  └─ results/           Predictions and metrics                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component Description

### 1. Presentation Layer (Frontend)

**Technology**: Streamlit 1.28+

**Purpose**: Provide user-friendly interface for non-technical users

**Key Features**:
- File upload (CSV, NPY, NPZ)
- Interactive configuration
- Real-time prediction display
- Visualization dashboard
- Results export

**Communication**: HTTP requests to backend API

### 2. Application Layer (Backend)

**Technology**: FastAPI 0.104+ with Uvicorn

**Purpose**: Business logic, model management, API gateway

**Key Components**:

#### 2.1 API Gateway (main.py)
- Request routing
- CORS configuration
- Global exception handling
- Lifespan management (model loading)
- OpenAPI documentation

#### 2.2 Routers
- **Health Router**: System status endpoints
- **Prediction Router**: Inference endpoints
- **Evaluation Router**: Metrics computation

#### 2.3 Services
- **ModelService**: 
  - Model initialization
  - Checkpoint loading
  - Inference execution
  - Graph management
  - Caching

- **EvaluationService**:
  - Metrics computation (MAE, RMSE, MAPE, R², NRMSE)
  - Per-horizon analysis

#### 2.4 Schemas
- Pydantic models for:
  - Input validation
  - Output serialization
  - Type safety
  - API documentation

### 3. Data/Model Layer

**Technology**: PyTorch 2.0+, PyTorch Geometric

**Purpose**: Core ML models and data processing

**Key Components**:

#### 3.1 HG-KAN Model
- **Architecture**: Encoder → Graph Conv → Temporal → Decoder
- **Innovation**: KAN layers instead of MLPs
- **Edge Types**: Spatial, Wake, Correlation
- **Input**: [batch, nodes, time, features]
- **Output**: [batch, nodes, horizon]

#### 3.2 KAN Layers
- B-spline basis functions
- Learnable coefficients
- Adaptive transformations
- Interpretable activations

#### 3.3 Graph Construction
- k-NN spatial edges
- Directional wake edges
- Correlation-based edges
- Dynamic graph building

#### 3.4 Data Processing
- CSV parsing
- Metadata extraction
- Sliding window dataset
- Normalization
- Train/val/test splitting

---

## Data Flow

### Prediction Flow

```
1. User Upload
   ↓
2. Frontend → API Request
   {input_data: [[...], [...], ...]}
   ↓
3. API Router (prediction.py)
   - Validate request (Pydantic)
   - Extract parameters
   ↓
4. Model Service
   - Convert to PyTorch tensor
   - Build/retrieve graph structure
   - Add batch dimension
   ↓
5. HG-KAN Model Forward Pass
   a. Node Encoders (KAN)
   b. Heterogeneous Graph Conv
   c. Temporal Module (Conv1D/Attention)
   d. KAN Decoder
   ↓
6. Post-processing
   - Remove batch dimension
   - Convert to numpy/list
   - Package metadata
   ↓
7. API Response
   {predictions: [[...]], metadata: {...}}
   ↓
8. Frontend Visualization
   - Plot time series
   - Show heatmaps
   - Display statistics
```

### Evaluation Flow

```
1. User Upload (Predictions + Ground Truth)
   ↓
2. Frontend → API Request
   ↓
3. Evaluation Service
   - Flatten arrays
   - Remove NaN values
   - Compute MAE, RMSE, MAPE, R², NRMSE
   - Per-horizon analysis
   ↓
4. API Response
   {mae: X, rmse: Y, ...}
   ↓
5. Frontend Display
   - Metrics table
   - Scatter plot
   - Horizon trends
```

---

## Design Patterns

### 1. Model-View-Controller (MVC)
- **Model**: PyTorch models (src/)
- **View**: Streamlit UI (frontend/)
- **Controller**: FastAPI routers (api/routers/)

### 2. Service Layer Pattern
- Business logic encapsulated in services
- Reusable across multiple endpoints
- Testable in isolation

### 3. Repository Pattern
- ModelService abstracts model access
- Caching layer for performance
- Checkpoint management

### 4. Dependency Injection
- Services injected into routers
- Configuration via environment
- Testable components

### 5. Factory Pattern
- Model creation in ModelService
- Dynamic graph construction
- Dataset builders

### 6. Singleton Pattern
- Global settings instance
- Shared model service
- Cached graph structures

---

## Technology Stack

### Backend
- **Framework**: FastAPI 0.104+
- **ASGI Server**: Uvicorn
- **Validation**: Pydantic 2.0+
- **Async**: asyncio, aiofiles

### Frontend
- **Framework**: Streamlit 1.28+
- **HTTP Client**: requests
- **Visualization**: Matplotlib, Seaborn

### ML/AI
- **Deep Learning**: PyTorch 2.0+
- **Graph**: PyTorch Geometric 2.3+
- **Scientific**: NumPy, Pandas, SciPy
- **ML Utilities**: scikit-learn

### Deployment
- **Containerization**: Docker
- **Orchestration**: Docker Compose
- **Configuration**: python-dotenv

---

## Scalability & Performance

### Horizontal Scaling

#### Stateless Design
- No session state in backend
- Model loaded at startup
- Each instance independent

#### Load Balancing
```
        Load Balancer
              ↓
    ┌─────────┼─────────┐
    ↓         ↓         ↓
  API-1     API-2     API-3
    ↓         ↓         ↓
      Shared Storage
```

### Vertical Scaling

- Multi-core CPU utilization
- GPU acceleration (optional)
- Memory optimization

### Caching Strategy

```python
# Model cached at startup
# Graph structures cached per configuration
# Results cached (optional Redis integration)

cache = {
    'model': model_instance,
    'graphs': {
        'config_hash_1': (edge_indices, edge_weights),
        'config_hash_2': (edge_indices, edge_weights),
    }
}
```

### Performance Metrics

| Operation | Single | Batch (32) | Notes |
|-----------|--------|------------|-------|
| Prediction | ~50ms | ~800ms | CPU, 200 nodes |
| Graph Build | ~100ms | Cached | First time |
| Evaluation | ~20ms | - | NumPy ops |

### Optimization Techniques

1. **Model Quantization**: Reduce model size
2. **ONNX Export**: Faster inference
3. **Batch Processing**: Amortize overhead
4. **Graph Caching**: Precompute structures
5. **Async I/O**: Non-blocking operations
6. **Connection Pooling**: Reuse connections

---

## Security Considerations

### Input Validation
- Pydantic schemas enforce types
- Array shape validation
- Value range checks
- File size limits

### API Security
- CORS configuration
- Rate limiting (TODO)
- Authentication (optional)
- HTTPS in production

### Data Security
- No persistent user data storage
- Temporary file cleanup
- Environment-based secrets
- Sanitized error messages

---

## Extensibility

### Adding New Models

```python
# 1. Implement model in src/
class NewModel(nn.Module):
    ...

# 2. Add to ModelService
if config['model_type'] == 'NewModel':
    self.model = NewModel(...)

# 3. Update schemas
# 4. Add router if needed
```

### Adding New Endpoints

```python
# 1. Create router file
@router.post("/new-endpoint")
async def new_endpoint(request: NewRequest):
    ...

# 2. Include in main.py
app.include_router(new_router, prefix="/api/v1")
```

### Adding New Visualizations

```python
# frontend/components/new_viz.py
def plot_new_visualization(data):
    fig, ax = plt.subplots()
    # ... plotting code
    return fig
```

---

## Testing Strategy

### Unit Tests
```bash
# Test individual components
pytest tests/test_models.py
pytest tests/test_services.py
pytest tests/test_routers.py
```

### Integration Tests
```bash
# Test API endpoints
pytest tests/test_api.py
pytest tests/test_integration.py
```

### End-to-End Tests
```bash
# Test full pipeline
pytest tests/test_e2e.py
```

---

## Monitoring & Logging

### Logging Levels
- **DEBUG**: Detailed diagnostic info
- **INFO**: General operational events
- **WARNING**: Unexpected situations
- **ERROR**: Error events

### Metrics to Monitor
- Request latency
- Error rate
- Model inference time
- Memory usage
- CPU/GPU utilization

### Recommended Tools
- **Logging**: Python logging + ELK stack
- **Metrics**: Prometheus + Grafana
- **Tracing**: Jaeger, OpenTelemetry
- **Alerting**: PagerDuty, Slack

---

## Future Enhancements

1. **Database Integration**: Store predictions and metrics
2. **User Authentication**: Multi-user support
3. **Model Versioning**: A/B testing, rollback
4. **Real-time Updates**: WebSocket for live predictions
5. **Advanced Caching**: Redis for distributed caching
6. **Auto-scaling**: Kubernetes deployment
7. **Model Training API**: Online learning
8. **Advanced Analytics**: Time-series analysis dashboard

---

## References

- FastAPI Documentation: https://fastapi.tiangolo.com/
- Streamlit Documentation: https://docs.streamlit.io/
- PyTorch Documentation: https://pytorch.org/docs/
- Docker Documentation: https://docs.docker.com/
