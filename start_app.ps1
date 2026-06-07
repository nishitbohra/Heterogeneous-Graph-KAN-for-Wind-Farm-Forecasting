# Quick Start Script for HG-KAN Wind Forecasting Application
# Run this script to start both backend and frontend

Write-Host "================================" -ForegroundColor Cyan
Write-Host "HG-KAN Wind Forecasting Startup" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

# Check if virtual environment exists
if (Test-Path "venv") {
    Write-Host "[1/5] Activating virtual environment..." -ForegroundColor Green
    .\venv\Scripts\Activate.ps1
} else {
    Write-Host "[WARNING] No virtual environment found. Using system Python." -ForegroundColor Yellow
    Write-Host "         Recommended: Create venv with 'python -m venv venv'" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "[2/5] Checking dependencies..." -ForegroundColor Green

# Check critical packages
$packages = @("fastapi", "uvicorn", "streamlit", "torch", "pandas")
$missing = @()

foreach ($pkg in $packages) {
    $result = python -c "import $pkg" 2>$null
    if ($LASTEXITCODE -ne 0) {
        $missing += $pkg
    }
}

if ($missing.Count -gt 0) {
    Write-Host "[WARNING] Missing packages: $($missing -join ', ')" -ForegroundColor Yellow
    Write-Host "          Installing dependencies..." -ForegroundColor Yellow
    pip install -r requirements.txt
} else {
    Write-Host "          All dependencies installed!" -ForegroundColor Green
}

Write-Host ""
Write-Host "[3/5] Starting FastAPI Backend..." -ForegroundColor Green
Write-Host "      API will be available at: http://localhost:8000" -ForegroundColor Cyan
Write-Host "      API Docs at: http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host ""

# Start backend in background
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload"

# Wait for backend to start
Write-Host "[4/5] Waiting for backend to initialize (10 seconds)..." -ForegroundColor Green
Start-Sleep -Seconds 10

Write-Host ""
Write-Host "[5/5] Starting Streamlit Frontend..." -ForegroundColor Green
Write-Host "      UI will be available at: http://localhost:8501" -ForegroundColor Cyan
Write-Host ""
Write-Host "================================" -ForegroundColor Cyan
Write-Host "Both services are starting!" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Backend API: http://localhost:8000/docs" -ForegroundColor Yellow
Write-Host "Frontend UI: http://localhost:8501" -ForegroundColor Yellow
Write-Host ""
Write-Host "Press Ctrl+C to stop the frontend." -ForegroundColor Gray
Write-Host "Close the other PowerShell window to stop the backend." -ForegroundColor Gray
Write-Host ""

# Start frontend (blocking)
streamlit run frontend/streamlit_app.py
