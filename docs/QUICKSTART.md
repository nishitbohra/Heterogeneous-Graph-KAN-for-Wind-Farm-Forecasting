# Quick Start Guide - Deployment

## Prerequisites

- Python 3.8+
- pip package manager
- 4GB+ RAM recommended

## Installation

1. **Clone/Navigate to the repository**
   ```bash
   cd Heterogeneous-Graph-KAN-for-Wind-Farm-Forecasting
   ```

2. **Create virtual environment (recommended)**
   ```bash
   python -m venv venv
   ```

3. **Activate virtual environment**
   
   Windows PowerShell:
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```
   
   Windows CMD:
   ```cmd
   venv\Scripts\activate.bat
   ```
   
   Linux/Mac:
   ```bash
   source venv/bin/activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

### Option 1: Start Both Services (Recommended for first-time users)

**Windows PowerShell:**
```powershell
.\start_app.ps1
```

This will:
- Start the FastAPI backend on port 8000
- Start the Streamlit frontend on port 8501
- Open your browser automatically

### Option 2: Start Services Individually

**Terminal 1 - Backend:**
```powershell
.\start_backend.ps1
```
Or manually:
```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 - Frontend:**
```powershell
.\start_frontend.ps1
```
Or manually:
```bash
streamlit run frontend/streamlit_app.py
```

## Accessing the Application

Once both services are running:

- **Frontend UI**: http://localhost:8501
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/v1/health

## Quick Test

1. Open http://localhost:8501 in your browser
2. Check that the sidebar shows "✅ API Connected"
3. Go to "Upload & Predict" tab
4. Select "Generate Sample Data"
5. Click "🔮 Generate Prediction"
6. View results in "Visualization" tab

## Configuration

Edit `.env` file to customize:
- Model parameters (input window, forecast horizon)
- Graph construction settings
- API port and host
- Logging level

## Troubleshooting

### Backend won't start
- Check if port 8000 is already in use
- Verify all dependencies are installed: `pip list`
- Check Python version: `python --version` (need 3.8+)

### Frontend shows "API Unavailable"
- Ensure backend is running first
- Check backend health: http://localhost:8000/api/v1/health
- Verify CORS settings in `.env`

### Import errors
- Reinstall dependencies: `pip install -r requirements.txt --upgrade`
- Check PyTorch installation: `python -c "import torch; print(torch.__version__)"`

### Memory errors
- Reduce `NUM_NODES` in `.env`
- Reduce `BATCH_SIZE` in `.env`
- Close other applications

## File Structure

```
.
├── api/                    # FastAPI backend
│   ├── main.py            # Entry point
│   ├── config.py          # Configuration
│   ├── routers/           # API endpoints
│   └── services/          # Business logic
├── frontend/              # Streamlit UI
│   ├── streamlit_app.py   # Main app
│   └── utils/             # Helper utilities
├── src/                   # Core ML models
│   ├── hg_kan_model.py    # Main model
│   ├── kan_layers.py      # KAN implementation
│   └── ...
├── .env                   # Configuration
├── requirements.txt       # Dependencies
├── start_app.ps1          # Startup script (both)
├── start_backend.ps1      # Startup script (API)
└── start_frontend.ps1     # Startup script (UI)
```

## Next Steps

- Upload your own wind farm data (CSV format)
- Customize model parameters in `.env`
- Train your own model using notebooks
- Deploy to cloud (see DEPLOYMENT.md)

## Support

- Documentation: See `docs/` folder
- Issues: GitHub Issues
- Architecture: `docs/ARCHITECTURE.md`
