@echo off
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
set "MODE=%~1"
if /i "%MODE%"=="all" goto arg_run
if /i "%MODE%"=="reuse" goto arg_run
if /i "%MODE%"=="serve" goto serve_only
if /i "%MODE%"=="build" goto arg_build

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

:serve_only
start "" "http://localhost:%PORT%/site/"
"%PY%" -m http.server %PORT%
exit /b 0
