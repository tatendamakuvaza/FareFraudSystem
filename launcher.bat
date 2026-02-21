@echo off
echo ==========================================
echo FRAUD DETECTION SYSTEM LAUNCHER
echo ==========================================
echo.

cd /d "%~dp0"

if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run setup_complete.bat first.
    pause
    exit /b 1
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Starting Streamlit Dashboard...
echo.
echo The app will open in your browser at: http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo ==========================================

streamlit run app.py

echo.
echo Server stopped.
pause