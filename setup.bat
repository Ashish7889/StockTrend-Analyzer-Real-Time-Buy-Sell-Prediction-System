@echo off
setlocal enabledelayedexpansion

:: Check Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed. Please install Python 3.8-3.11.
    exit /b 1
)

:: Navigate to project directory
cd /d "%~dp0"

:: Check if virtual environment exists
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
)

:: Activate virtual environment
call venv\Scripts\activate

:: Upgrade pip
python -m pip install --upgrade pip

:: Install dependencies
pip install -r requirements.txt

:: Verify Streamlit installation
streamlit --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Streamlit installation failed. Retrying...
    pip install streamlit
)

:: Run the application
echo Starting Stock Trend Analyzer...
streamlit run stock_trend_analyzer.py

:: Keep console open if something goes wrong
pause
