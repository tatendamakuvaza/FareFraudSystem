@echo off
echo ==========================================
echo FARE FRAUD DETECTION SYSTEM - AUTO SETUP
echo Mukumba Brothers (Pvt) Ltd
echo ==========================================
echo.

:: Create project folder on Desktop
set PROJECT_DIR=%USERPROFILE%\Desktop\FareFraudSystem
echo Creating project at: %PROJECT_DIR%
mkdir "%PROJECT_DIR%"
cd /d "%PROJECT_DIR%"

:: Create folders
echo Creating folder structure...
mkdir data
mkdir models
mkdir models\saved
mkdir static
mkdir templates
mkdir exports

:: Create virtual environment
echo Creating Python virtual environment...
python -m venv venv

:: Activate and install packages
echo Installing required packages...
call venv\Scripts\activate.bat
pip install pandas numpy scikit-learn streamlit plotly flask openpyxl xlsxwriter

echo.
echo ==========================================
echo SETUP COMPLETE!
echo ==========================================
echo.
echo Next steps:
echo 1. Save the Python files in: %PROJECT_DIR%
echo 2. Run: launcher.bat
echo.
pause