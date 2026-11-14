@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo [오류] 가상환경이 없습니다. 먼저 setup.bat 을 실행하세요.
  pause
  exit /b 1
)
rem 한우 실시간 추적 프로그램 (파일 / RTSP 카메라 / 웹캠, 파라미터 실시간 튜닝)
rem   live.bat                : 프로그램 창 열기
rem   live.bat rtsp://주소    : 입력을 지정해서 열기
if "%~1"=="" (".venv\Scripts\python.exe" live_app.py) else (".venv\Scripts\python.exe" live_app.py --source "%~1")
if errorlevel 1 pause
