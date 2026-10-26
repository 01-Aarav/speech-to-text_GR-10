@echo off
title Whisper Speech-to-Text Studio
echo ========================================================
echo  Launching Whisper Speech-to-Text Web Application...
echo ========================================================
cd /d "%~dp0"
call .venv\Scripts\activate.bat
streamlit run app.py
pause
