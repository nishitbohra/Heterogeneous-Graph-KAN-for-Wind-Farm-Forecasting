@echo off
REM Start Streamlit Frontend

echo ========================================
echo Starting HG-KAN Streamlit Frontend
echo ========================================
echo.
echo Frontend will be available at: http://localhost:8501
echo.
echo Make sure the API is running on port 8000!
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.

streamlit run frontend/streamlit_app.py
