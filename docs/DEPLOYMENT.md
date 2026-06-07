# Deployment Guide

## HG-KAN Wind Forecasting - Deployment Instructions

This guide covers local and cloud deployment options.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Deployment](#local-deployment)
3. [Docker Deployment](#docker-deployment)
4. [Cloud Deployment](#cloud-deployment)
5. [Production Checklist](#production-checklist)

---

## Prerequisites

### System Requirements

- **OS**: Linux, macOS, or Windows
- **Python**: 3.9 or higher
- **RAM**: 4GB minimum (8GB recommended)
- **Storage**: 5GB for dependencies and model
- **Optional**: NVIDIA GPU with CUDA for faster inference

### Software Requirements

- Python 3.9+
- pip or conda
- Docker (optional, for containerized deployment)
- Git

---

## Local Deployment

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd Heterogeneous-Graph-KAN-for-Wind-Farm-Forecasting
```

### Step 2: Install Dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows PowerShell

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-api.txt
pip install -r requirements-frontend.txt
```

### Step 3: Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your settings
# Set MODEL_CHECKPOINT if you have a trained model
```

### Step 4: Start API Server

```bash
# Terminal 1: Start API
python -m api.main
```

API will be available at: `http://localhost:8000`

### Step 5: Start Frontend

```bash
# Terminal 2: Start Streamlit
streamlit run frontend/streamlit_app.py
```

Frontend will be available at: `http://localhost:8501`

### Step 6: Verify

- Open browser: `http://localhost:8501`
- Check API status indicator
- Try sample prediction

---

## Docker Deployment

### Option 1: Docker Compose (Recommended)

```bash
# Build and start all services
docker-compose up --build

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

Services:
- API: `http://localhost:8000`
- Frontend: `http://localhost:8501`

### Option 2: Individual Containers

#### Build Images

```bash
# API only
docker build -t hg-kan-api --target api .

# Frontend only
docker build -t hg-kan-frontend --target frontend .

# Full image
docker build -t hg-kan-full --target full .
```

#### Run Containers

```bash
# API
docker run -d \
  --name hg-kan-api \
  -p 8000:8000 \
  -v $(pwd)/checkpoints:/app/checkpoints:ro \
  -v $(pwd)/data:/app/data:ro \
  --env-file .env \
  hg-kan-api

# Frontend
docker run -d \
  --name hg-kan-frontend \
  -p 8501:8501 \
  --link hg-kan-api:api \
  -e API_URL=http://api:8000 \
  hg-kan-frontend
```

### Docker Best Practices

1. **Use multi-stage builds** (already implemented)
2. **Mount volumes** for data and checkpoints
3. **Set resource limits**:
   ```bash
   docker run --memory="2g" --cpus="2.0" ...
   ```
4. **Enable logging**:
   ```bash
   docker run --log-driver json-file --log-opt max-size=10m ...
   ```

---

## Cloud Deployment

### AWS Deployment

#### EC2 Instance

1. **Launch EC2 Instance**
   - AMI: Ubuntu 22.04 LTS
   - Instance Type: t3.medium or larger
   - Security Group: Open ports 8000, 8501

2. **Setup**
   ```bash
   # SSH into instance
   ssh -i your-key.pem ubuntu@<instance-ip>
   
   # Update system
   sudo apt update && sudo apt upgrade -y
   
   # Install Docker
   sudo apt install docker.io docker-compose -y
   sudo usermod -aG docker ubuntu
   
   # Clone repository
   git clone <repo-url>
   cd Heterogeneous-Graph-KAN-for-Wind-Farm-Forecasting
   
   # Deploy
   docker-compose up -d
   ```

3. **Access**
   - API: `http://<instance-ip>:8000`
   - Frontend: `http://<instance-ip>:8501`

#### ECS (Elastic Container Service)

1. Push images to ECR
2. Create ECS cluster
3. Define task definitions
4. Create services
5. Configure load balancer

#### Elastic Beanstalk

```bash
# Initialize
eb init -p docker hg-kan-app

# Deploy
eb create hg-kan-env
eb deploy
```

### Google Cloud Platform

#### Cloud Run

```bash
# Build and push to Container Registry
gcloud builds submit --tag gcr.io/PROJECT_ID/hg-kan-api

# Deploy
gcloud run deploy hg-kan-api \
  --image gcr.io/PROJECT_ID/hg-kan-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

#### Compute Engine

Similar to AWS EC2 setup.

### Microsoft Azure

#### Container Instances

```bash
# Create resource group
az group create --name hg-kan-rg --location eastus

# Deploy container
az container create \
  --resource-group hg-kan-rg \
  --name hg-kan-api \
  --image your-registry/hg-kan-api \
  --ports 8000 \
  --dns-name-label hg-kan-api
```

#### App Service

```bash
# Create app service plan
az appservice plan create --name hg-kan-plan --resource-group hg-kan-rg --is-linux

# Create web app
az webapp create \
  --resource-group hg-kan-rg \
  --plan hg-kan-plan \
  --name hg-kan-app \
  --deployment-container-image-name your-registry/hg-kan-api
```

### Heroku

```bash
# Login
heroku login
heroku container:login

# Create app
heroku create hg-kan-app

# Deploy
heroku container:push web --app hg-kan-app
heroku container:release web --app hg-kan-app

# Open
heroku open --app hg-kan-app
```

---

## Production Checklist

### Security

- [ ] Enable HTTPS/TLS
- [ ] Set strong secrets in `.env`
- [ ] Configure firewall rules
- [ ] Enable CORS properly
- [ ] Add rate limiting
- [ ] Implement authentication (if needed)
- [ ] Regular security updates

### Performance

- [ ] Use production-grade ASGI server (Gunicorn + Uvicorn)
- [ ] Configure worker processes
- [ ] Enable caching
- [ ] Set up CDN for static assets
- [ ] Optimize model loading
- [ ] Monitor resource usage

### Reliability

- [ ] Set up health checks
- [ ] Configure auto-restart on failure
- [ ] Implement logging
- [ ] Set up monitoring (Prometheus, Grafana)
- [ ] Configure alerts
- [ ] Regular backups
- [ ] Disaster recovery plan

### Scalability

- [ ] Horizontal scaling capability
- [ ] Load balancer configuration
- [ ] Database for persistent storage (if needed)
- [ ] Caching layer (Redis)
- [ ] Queue system for async tasks (Celery)

### Monitoring

```bash
# Example: Prometheus + Grafana

# Install exporters
pip install prometheus-fastapi-instrumentator

# Add to API
from prometheus_fastapi_instrumentator import Instrumentator

Instrumentator().instrument(app).expose(app)
```

### Production ASGI Command

```bash
# Using Gunicorn with Uvicorn workers
gunicorn api.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 120 \
  --keep-alive 5 \
  --log-level info \
  --access-logfile - \
  --error-logfile -
```

### Nginx Reverse Proxy

```nginx
# /etc/nginx/sites-available/hg-kan

upstream api_backend {
    server localhost:8000;
}

upstream frontend_backend {
    server localhost:8501;
}

server {
    listen 80;
    server_name yourdomain.com;

    # API
    location /api {
        proxy_pass http://api_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Frontend
    location / {
        proxy_pass http://frontend_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```

### SSL with Let's Encrypt

```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx

# Get certificate
sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com

# Auto-renewal
sudo certbot renew --dry-run
```

---

## Troubleshooting

### API Not Starting

```bash
# Check logs
docker-compose logs api

# Verify dependencies
pip list | grep fastapi

# Test manually
python -c "from api.main import app; print('OK')"
```

### Frontend Can't Connect

```bash
# Check API URL in frontend
# Ensure API is accessible from frontend container

# Test connection
curl http://localhost:8000/api/v1/health
```

### Model Loading Issues

```bash
# Verify checkpoint path
ls -l checkpoints/

# Check permissions
chmod 644 checkpoints/*.pth

# Verify in .env
cat .env | grep MODEL_CHECKPOINT
```

### Memory Issues

```bash
# Monitor memory
docker stats

# Increase container limits
docker run --memory="4g" ...

# Or in docker-compose.yml:
# services:
#   api:
#     mem_limit: 4g
```

---

## Support

- **Documentation**: See `docs/` directory
- **Issues**: GitHub Issues
- **Email**: support@example.com
