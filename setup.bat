@echo off
cd /d "%~dp0"
echo ==============================================================
echo    webHanwoo 환경 구축  (Python 3.10 이상, NVIDIA GPU 권장)
echo ==============================================================

set "BASE=python"
where py >nul 2>nul && set "BASE=py -3"

if exist ".venv\Scripts\python.exe" (
  echo [1/4] 가상환경 이미 있음
) else (
  echo [1/4] 가상환경 생성 .venv
  %BASE% -m venv .venv
  if errorlevel 1 goto fail
)
set "PY=.venv\Scripts\python.exe"
"%PY%" -m pip install --upgrade pip

echo [2/4] PyTorch 설치 (CUDA 12.8 / 실패하면 CPU 버전)
"%PY%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 (
  "%PY%" -m pip install torch torchvision
  if errorlevel 1 goto fail
)

echo [3/4] 나머지 패키지
"%PY%" -m pip install -r requirements.txt
if errorlevel 1 goto fail

echo [4/4] 설치 확인
"%PY%" -c "import torch, ultralytics, cv2; print('torch', torch.__version__, 'CUDA', torch.cuda.is_available()); print('ultralytics', ultralytics.__version__, ' opencv', cv2.__version__)"
where ffmpeg >nul 2>nul
if errorlevel 1 echo [주의] ffmpeg 가 PATH 에 없습니다. 결과 영상 저장에 필요합니다 (강좌 01).
if not exist "models\hanwoo_yolo11s.pt" echo [주의] models\hanwoo_yolo11s.pt 없음: 사전학습 yolo11x.pt 로 동작 (학습은 강좌 04-05).
if not exist "models\hanwoo_reid.pt" echo [주의] models\hanwoo_reid.pt 없음: ReID 대신 2단계 정합 사용 (학습은 강좌 08).
echo.
echo 설치 완료. run.bat 을 실행하세요.
pause
exit /b 0

:fail
echo [오류] 설치 실패. 위 메시지를 확인하세요.
pause
exit /b 1
