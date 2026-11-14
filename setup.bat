@echo off
cd /d "%~dp0"
echo ==============================================================
echo    webHanwoo environment setup  (Python 3.10+, NVIDIA GPU recommended)
echo ==============================================================

set "BASE=python"
where py >nul 2>nul && set "BASE=py -3"

if exist ".venv\Scripts\python.exe" (
  echo [1/4] Virtual environment already exists
) else (
  echo [1/4] Creating virtual environment .venv
  %BASE% -m venv .venv
  if errorlevel 1 goto fail
)
set "PY=.venv\Scripts\python.exe"
"%PY%" -m pip install --upgrade pip

echo [2/4] Installing PyTorch (CUDA 12.8 / CPU build if that fails)
"%PY%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 (
  "%PY%" -m pip install torch torchvision
  if errorlevel 1 goto fail
)

echo [3/4] Remaining packages
"%PY%" -m pip install -r requirements.txt
if errorlevel 1 goto fail

echo [4/4] Verifying installation
"%PY%" -c "import torch, ultralytics, cv2; print('torch', torch.__version__, 'CUDA', torch.cuda.is_available()); print('ultralytics', ultralytics.__version__, ' opencv', cv2.__version__)"
where ffmpeg >nul 2>nul
if errorlevel 1 echo [WARNING] ffmpeg is not on PATH. It is required to save result videos (lesson 01).
if not exist "models\hanwoo_yolo11s.pt" echo [WARNING] models\hanwoo_yolo11s.pt missing: running with pretrained yolo11x.pt (training: lessons 04-05).
if not exist "models\hanwoo_reid.pt" echo [WARNING] models\hanwoo_reid.pt missing: using two-stage linking instead of ReID (training: lesson 07b).
echo.
echo Setup complete. Run run.bat.
pause
exit /b 0

:fail
echo [ERROR] Setup failed. Check the messages above.
pause
exit /b 1
