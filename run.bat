@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"
set "PY=.venv\Scripts\python.exe"
set "PORT=8791"
set "DEFAULT_SRC=E:\Video\cow"

if not exist "%PY%" (
  echo [ERROR] Virtual environment not found. Run setup.bat first.
  pause
  exit /b 1
)

rem ---- Run with arguments (no menu)
rem   run.bat all   [folder/file] [herd size]
rem   run.bat reuse [folder/file] [herd size]
rem   run.bat serve
rem   run.bat build
rem   run.bat live  [file or rtsp://address]
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
echo    Hanwoo individual tracking / activity measurement  (webHanwoo)
echo ==============================================================
echo    1. Process all videos in a folder  (detection + tracking + ReID + activity)
echo    2. Process a single video
echo    3. Quick recompute (--reuse)  reuse detection/tracking cache
echo    4. Open result summary / course site  (http://localhost:%PORT%)
echo    5. Make ID contact sheet  (check ID consistency)
echo    6. Rebuild course site
echo    7. Real-time tracking app  (file / RTSP camera, parameter tuning)
echo    0. Exit
echo ==============================================================
set "SEL="
set /p "SEL=Select a number: "
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
set /p "HERD=Herd size N (Enter = estimate automatically): "
set "EXTRA="
if defined HERD set "EXTRA=--herd-size %HERD%"
exit /b 0

:folder
set "SRC=%DEFAULT_SRC%"
set /p "SRC=Video folder [Enter = %DEFAULT_SRC%]: "
call :ask_herd
"%PY%" run_activity.py "%SRC%" %EXTRA%
goto done

:single
set "SRC="
set /p "SRC=Video file path (e.g. E:\Video\cow\cow03.mp4): "
if not defined SRC goto menu
call :ask_herd
"%PY%" run_activity.py "%SRC%" %EXTRA%
goto done

:reuse
set "SRC=%DEFAULT_SRC%"
set /p "SRC=Video folder or file [Enter = %DEFAULT_SRC%]: "
call :ask_herd
"%PY%" run_activity.py "%SRC%" --reuse %EXTRA%
goto done

:serve
echo Starting the web server in a new window. (Close that window to stop the server)
start "webHanwoo server" cmd /k ""%PY%" -m http.server %PORT%"
timeout /t 2 >nul
start "" "http://localhost:%PORT%/outputs/index.html"
start "" "http://localhost:%PORT%/site/"
goto menu

:sheet
set "V=cow03"
set /p "V=Video name [Enter = cow03]: "
"%PY%" scripts\identity_sheet.py "outputs\%V%"
if exist "outputs\%V%\identity_sheet.jpg" start "" "outputs\%V%\identity_sheet.jpg"
goto done

:build
"%PY%" scripts\build_site.py
goto done

:live
echo Starting the real-time tracking app. (Close its window to return to the menu)
"%PY%" live_app.py
goto menu

:done
echo.
echo Done. Output folder: %~dp0outputs
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
