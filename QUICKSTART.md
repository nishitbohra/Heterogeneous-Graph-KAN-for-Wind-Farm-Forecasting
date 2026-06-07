# Quick Start Guide

## Installation

```bash
# Install all dependencies
pip install -r requirements.txt
```

## Starting the Application

### Option 1: Windows Batch Scripts

**Terminal 1 - Start API:**
```bash
start_api.bat
```

**Terminal 2 - Start Frontend:**
```bash
start_frontend.bat
```

### Option 2: Manual Commands

**Terminal 1 - Start API:**
```bash
python -m api.main
```

**Terminal 2 - Start Frontend:**
```bash
streamlit run frontend/streamlit_app.py
```

### Option 3: Docker

```bash
docker-compose up
```

## Access the Application

- **Frontend (User Interface)**: http://localhost:8501
- **API Documentation**: http://localhost:8000/docs
- **API Health Check**: http://localhost:8000/api/v1/health

## Testing the Application

1. Open http://localhost:8501 in your browser
2. The interface will check if the API is running
3. Try the "Generate Sample Data" option
4. Click "Generate Prediction"
5. View results in the Visualization tab

## Troubleshooting

### API won't start
- Check if port 8000 is already in use
- Verify all dependencies are installed: `pip list`

### Frontend can't connect to API
- Make sure API is running on port 8000
- Check the API health: http://localhost:8000/api/v1/health

### Import errors
- Reinstall dependencies: `pip install -r requirements.txt --upgrade`

## Quick Test

```bash
# Test API
curl http://localhost:8000/api/v1/health

# Or in PowerShell
Invoke-WebRequest http://localhost:8000/api/v1/health
```
