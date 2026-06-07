# Start Streamlit Frontend Only

Write-Host "================================" -ForegroundColor Cyan
Write-Host "Starting HG-KAN Frontend UI" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Make sure the backend is running at: http://localhost:8000" -ForegroundColor Yellow
Write-Host ""
Write-Host "Frontend: http://localhost:8501" -ForegroundColor Green
Write-Host ""
Write-Host "Press Ctrl+C to stop the UI" -ForegroundColor Yellow
Write-Host ""

# Start streamlit
streamlit run frontend/streamlit_app.py
