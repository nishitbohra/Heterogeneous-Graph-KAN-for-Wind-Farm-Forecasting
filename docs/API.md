# API Documentation

## HG-KAN Wind Forecasting API

FastAPI-based REST API for wind power forecasting using Heterogeneous Graph-KAN model.

---

## Quick Start

### 1. Start the API Server

```bash
# Install dependencies
pip install -r requirements.txt
pip install -r requirements-api.txt

# Run the server
python -m api.main
```

The API will be available at `http://localhost:8000`

Interactive documentation: `http://localhost:8000/docs`

### 2. Using Docker

```bash
# Build and run
docker-compose up api

# Or build manually
docker build -t hg-kan-api --target api .
docker run -p 8000:8000 hg-kan-api
```

---

## API Endpoints

### Health Check

**GET** `/api/v1/health`

Check API status and model availability.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-06-07T10:30:00",
  "version": "1.0.0",
  "model_loaded": true
}
```

### Model Information

**GET** `/api/v1/info`

Get model configuration and architecture details.

**Response:**
```json
{
  "model_type": "HG-KAN",
  "num_nodes": 200,
  "input_window": 24,
  "forecast_horizon": 6,
  "hidden_dim": 64,
  "num_basis": 5,
  "edge_types": ["spatial", "wake", "correlation"],
  "parameters": 45320
}
```

### Generate Prediction

**POST** `/api/v1/predict`

Generate wind power forecasts.

**Request Body:**
```json
{
  "input_data": [
    [0.5, 0.6, 0.55, 0.58, ...],  // Node 0 (24 timesteps)
    [0.45, 0.52, 0.48, 0.51, ...], // Node 1
    ...
  ],
  "forecast_horizon": 6  // Optional override
}
```

**Response:**
```json
{
  "predictions": [
    [0.57, 0.59, 0.61, 0.60, 0.58, 0.56],  // Node 0 (6 forecasts)
    [0.49, 0.51, 0.53, 0.52, 0.50, 0.48],  // Node 1
    ...
  ],
  "metadata": {
    "inference_time_ms": 45.2,
    "input_shape": [200, 24, 1],
    "output_shape": [200, 6]
  },
  "model_type": "HG-KAN",
  "forecast_horizon": 6,
  "num_nodes": 200
}
```

### Batch Prediction

**POST** `/api/v1/predict/batch`

Process multiple time windows in a single request.

**Request Body:**
```json
{
  "batch_data": [
    [[...], [...], ...],  // Sample 1
    [[...], [...], ...],  // Sample 2
    ...
  ]
}
```

### Evaluate Predictions

**POST** `/api/v1/evaluate`

Compute evaluation metrics.

**Request Body:**
```json
{
  "predictions": [...],
  "ground_truth": [...]
}
```

**Response:**
```json
{
  "mae": 0.142,
  "rmse": 0.178,
  "mape": 15.3,
  "r2": 0.298,
  "nrmse": 0.962,
  "per_horizon": [
    {"horizon": 1, "mae": 0.125, "rmse": 0.156, ...},
    {"horizon": 2, "mae": 0.138, "rmse": 0.172, ...},
    ...
  ]
}
```

---

## Python Client Example

```python
import requests
import numpy as np

# API URL
API_URL = "http://localhost:8000/api/v1"

# Check health
response = requests.get(f"{API_URL}/health")
print(response.json())

# Prepare input data (200 nodes, 24 timesteps)
input_data = np.random.rand(200, 24).tolist()

# Make prediction
payload = {
    "input_data": input_data,
    "forecast_horizon": 6
}

response = requests.post(f"{API_URL}/predict", json=payload)
result = response.json()

predictions = np.array(result['predictions'])
print(f"Predictions shape: {predictions.shape}")
print(f"Inference time: {result['metadata']['inference_time_ms']:.2f}ms")
```

---

## Configuration

### Environment Variables

Create a `.env` file from `.env.example`:

```bash
cp .env.example .env
```

Key settings:

- `MODEL_CHECKPOINT`: Path to trained model checkpoint
- `NUM_NODES`: Number of turbines (default: 200)
- `INPUT_WINDOW`: Input sequence length (default: 24)
- `FORECAST_HORIZON`: Prediction horizon (default: 6)
- `DEVICE`: "cpu" or "cuda"
- `LOG_LEVEL`: "DEBUG", "INFO", "WARNING", "ERROR"

### Model Loading

To use a trained model checkpoint:

1. Place checkpoint file in `checkpoints/` directory
2. Set `MODEL_CHECKPOINT` in `.env`:
   ```
   MODEL_CHECKPOINT=checkpoints/best_model.pth
   ```

---

## Error Handling

The API returns standard HTTP status codes:

- `200`: Success
- `400`: Bad Request (invalid input)
- `422`: Validation Error (Pydantic validation failed)
- `500`: Internal Server Error
- `503`: Service Unavailable (model not loaded)

**Error Response Format:**
```json
{
  "error": "Error type",
  "detail": "Detailed error message",
  "timestamp": "2024-06-07T10:30:00"
}
```

---

## Performance

### Benchmarks (CPU)

- Single prediction (200 nodes): ~40-60ms
- Batch prediction (32 samples): ~800ms (~25ms/sample)

### Optimization Tips

1. **Use batch endpoints** for multiple samples
2. **Enable GPU** by setting `DEVICE=cuda` (if available)
3. **Cache graph structures** when making multiple predictions
4. **Increase workers** for concurrent requests

---

## API Testing

```bash
# Using curl
curl -X GET http://localhost:8000/api/v1/health

# Using httpie
http GET localhost:8000/api/v1/info

# Using pytest
pytest tests/test_api.py
```

---

## Deployment

### Production Considerations

1. **Use Gunicorn/Uvicorn** with multiple workers
2. **Enable HTTPS** with SSL certificates
3. **Add rate limiting** to prevent abuse
4. **Set up monitoring** (logs, metrics)
5. **Use reverse proxy** (Nginx, Traefik)

### Example Production Command

```bash
uvicorn api.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --log-level info \
  --no-access-log
```

---

## Support

- **Documentation**: `/docs` (Swagger UI)
- **Alternative Docs**: `/redoc` (ReDoc)
- **Issues**: GitHub Issues
- **Email**: support@example.com
