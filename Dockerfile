# Multi-stage Docker build for HG-KAN Wind Forecasting

# Base stage with Python and dependencies
FROM python:3.10-slim as base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt requirements-api.txt requirements-frontend.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r requirements-api.txt \
    && pip install --no-cache-dir -r requirements-frontend.txt

# API stage
FROM base as api

WORKDIR /app

# Copy source code
COPY src/ ./src/
COPY api/ ./api/
COPY data/ ./data/
COPY checkpoints/ ./checkpoints/

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/api/v1/health')"

# Run API server
CMD ["python", "-m", "api.main"]

# Frontend stage
FROM base as frontend

WORKDIR /app

# Copy frontend code
COPY frontend/ ./frontend/

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run Streamlit
CMD ["streamlit", "run", "frontend/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# All-in-one stage (for development)
FROM base as full

WORKDIR /app

# Copy all code
COPY . .

EXPOSE 8000 8501

# Default to API server
CMD ["python", "-m", "api.main"]
