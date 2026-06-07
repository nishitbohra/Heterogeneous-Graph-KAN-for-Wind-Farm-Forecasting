# Start FastAPI Backend Only

Write-Host "================================" -ForegroundColor Cyan
Write-Host "Starting HG-KAN API Backend" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "API Server: http://localhost:8000" -ForegroundColor Green
Write-Host "API Docs:   http://localhost:8000/docs" -ForegroundColor Green
Write-Host "Health:     http://localhost:8000/api/v1/health" -ForegroundColor Green
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

# Start uvicorn
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
