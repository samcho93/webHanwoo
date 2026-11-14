@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] Virtual environment not found. Run setup.bat first.
  pause
  exit /b 1
)
rem Hanwoo real-time tracking app (file / RTSP camera / webcam, live parameter tuning)
rem   live.bat                : open the app window
rem   live.bat rtsp://address : open with the given input
if "%~1"=="" (".venv\Scripts\python.exe" live_app.py) else (".venv\Scripts\python.exe" live_app.py --source "%~1")
if errorlevel 1 pause
