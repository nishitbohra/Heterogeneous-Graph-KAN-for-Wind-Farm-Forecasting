@echo off
REM Start FastAPI Backend Server

echo ========================================
echo Starting HG-KAN API Backend Server
echo ========================================
echo.
echo API will be available at: http://localhost:8000
echo Documentation at: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.

python -m api.main
