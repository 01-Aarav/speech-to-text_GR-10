@echo off
setlocal enabledelayedexpansion
title Whisper Speech-to-Text Studio
cd /d "%~dp0"

echo ========================================================
echo   Whisper Speech-to-Text Studio Launcher
echo ========================================================
echo.

:: 1. Verify Python is installed and accessible in PATH
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not added to system PATH!
    echo.
    echo Please install Python 3.10 or higher from https://www.python.org/
    echo IMPORTANT: Ensure you check "Add python.exe to PATH" during installation.
    echo.
    pause
    exit /b 1
)

:: 2. Check if virtual environment exists; if not, automatically configure it
if not exist ".venv\Scripts\activate.bat" (
    echo [FIRST-TIME SETUP] No virtual environment found.
    echo Configuring project environment automatically...
    echo.
    echo Step 1/3: Creating virtual environment (.venv)...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment (.venv).
        pause
        exit /b 1
    )

    echo Step 2/3: Activating environment...
    call .venv\Scripts\activate.bat

    echo Step 3/3: Installing dependencies from requirements.txt...
    echo (This may take 2-4 minutes to download PyTorch, Whisper, and Streamlit)
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to install dependencies from requirements.txt.
        pause
        exit /b 1
    )
    echo.
    echo [SUCCESS] Environment successfully set up!
    echo.
) else (
    call .venv\Scripts\activate.bat
)

:: 3. Launch Streamlit Application
echo Launching Streamlit Web App...
echo If your browser does not open automatically, visit: http://localhost:8501
echo.
streamlit run app.py

if %errorlevel% neq 0 (
    echo.
    echo [NOTICE] Streamlit process exited.
)

pause
