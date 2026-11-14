"""run.bat / setup.bat 생성기.

Windows cmd 는 배치 파일을 'ANSI 코드페이지(한국어 Windows = CP949)' 로 읽는다.
UTF-8 + chcp 65001 조합은 한글이 섞이면 줄을 잘못 해석하는 cmd 버그가 있으므로,
배치 파일은 CP949 + CRLF 로 저장한다.  (수정은 이 파일에서 하고  python scripts/make_bat.py 실행)
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RUN = r'''@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"
set "PY=.venv\Scripts\python.exe"
set "PORT=8791"
set "DEFAULT_SRC=E:\Video\cow"

if not exist "%PY%" (
  echo [오류] 가상환경이 없습니다. 먼저 setup.bat 을 실행하세요.
  pause
  exit /b 1
)

rem ---- 인자 실행 (메뉴 없이)
rem   run.bat all   [폴더/파일] [마릿수]
rem   run.bat reuse [폴더/파일] [마릿수]
rem   run.bat serve
rem   run.bat build
rem   run.bat live  [파일 또는 rtsp://주소]
set "MODE=%~1"
if /i "%MODE%"=="all" goto arg_run
if /i "%MODE%"=="reuse" goto arg_run
if /i "%MODE%"=="serve" goto serve_only
if /i "%MODE%"=="build" goto arg_build
if /i "%MODE%"=="live" goto arg_live

set EMPTY=0
:menu
cls
echo ==============================================================
echo    한우 개체 추적 / 활동량 측정  (webHanwoo)
echo ==============================================================
echo    1. 폴더의 모든 영상 처리   (검출 + 추적 + ReID + 활동량)
echo    2. 영상 1개 처리
echo    3. 빠른 재계산 (--reuse)   검출/추적 캐시 재사용
echo    4. 결과 요약 / 강좌 사이트 열기  (http://localhost:%PORT%)
echo    5. ID 콘택트 시트 만들기   (ID 유지 확인용)
echo    6. 강좌 사이트 다시 빌드
echo    7. 실시간 추적 프로그램  (파일 / RTSP 카메라, 파라미터 튜닝)
echo    0. 종료
echo ==============================================================
set "SEL="
set /p "SEL=번호 선택: "
if not defined SEL (
  set /a EMPTY+=1
  if !EMPTY! geq 3 exit /b 0
  goto menu
)
set EMPTY=0
if "%SEL%"=="1" goto folder
if "%SEL%"=="2" goto single
if "%SEL%"=="3" goto reuse
if "%SEL%"=="4" goto serve
if "%SEL%"=="5" goto sheet
if "%SEL%"=="6" goto build
if "%SEL%"=="7" goto live
if "%SEL%"=="0" exit /b 0
goto menu

:ask_herd
set "HERD="
set /p "HERD=축사 마릿수 N (모르면 Enter = 자동 추정): "
set "EXTRA="
if defined HERD set "EXTRA=--herd-size %HERD%"
exit /b 0

:folder
set "SRC=%DEFAULT_SRC%"
set /p "SRC=영상 폴더 [Enter = %DEFAULT_SRC%]: "
call :ask_herd
"%PY%" run_activity.py "%SRC%" %EXTRA%
goto done

:single
set "SRC="
set /p "SRC=영상 파일 경로 (예: E:\Video\cow\cow03.mp4): "
if not defined SRC goto menu
call :ask_herd
"%PY%" run_activity.py "%SRC%" %EXTRA%
goto done

:reuse
set "SRC=%DEFAULT_SRC%"
set /p "SRC=영상 폴더 또는 파일 [Enter = %DEFAULT_SRC%]: "
call :ask_herd
"%PY%" run_activity.py "%SRC%" --reuse %EXTRA%
goto done

:serve
echo 웹서버를 새 창에서 시작합니다. (그 창을 닫으면 서버 종료)
start "webHanwoo server" cmd /k ""%PY%" -m http.server %PORT%"
timeout /t 2 >nul
start "" "http://localhost:%PORT%/outputs/index.html"
start "" "http://localhost:%PORT%/site/"
goto menu

:sheet
set "V=cow03"
set /p "V=영상 이름 [Enter = cow03]: "
"%PY%" scripts\identity_sheet.py "outputs\%V%"
if exist "outputs\%V%\identity_sheet.jpg" start "" "outputs\%V%\identity_sheet.jpg"
goto done

:build
"%PY%" scripts\build_site.py
goto done

:live
echo 실시간 추적 프로그램을 실행합니다. (창을 닫으면 메뉴로 돌아옴)
"%PY%" live_app.py
goto menu

:done
echo.
echo 완료. 결과 폴더: %~dp0outputs
pause
goto menu

:arg_run
set "SRC=%~2"
if not defined SRC set "SRC=%DEFAULT_SRC%"
set "EXTRA="
if not "%~3"=="" set "EXTRA=--herd-size %~3"
if /i "%MODE%"=="reuse" set "EXTRA=--reuse %EXTRA%"
"%PY%" run_activity.py "%SRC%" %EXTRA%
exit /b %errorlevel%

:arg_build
"%PY%" scripts\build_site.py
exit /b %errorlevel%

:arg_live
if "%~2"=="" ("%PY%" live_app.py) else ("%PY%" live_app.py --source "%~2")
exit /b %errorlevel%

:serve_only
start "" "http://localhost:%PORT%/site/"
"%PY%" -m http.server %PORT%
exit /b 0
'''

SETUP = r'''@echo off
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
'''

LIVE = r'''@echo off
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
'''

for name, text in (("run.bat", RUN), ("setup.bat", SETUP), ("live.bat", LIVE)):
    (ROOT / name).write_bytes(text.replace("\r\n", "\n").replace("\n", "\r\n").encode("cp949"))
    print("wrote", name)
