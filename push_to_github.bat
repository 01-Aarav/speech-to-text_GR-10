@echo off
title Push to GitHub
cd /d "%~dp0"
echo ========================================================
echo  Pushing Whisper Speech-to-Text Studio to GitHub...
echo ========================================================
git push -u origin main
echo.
pause
