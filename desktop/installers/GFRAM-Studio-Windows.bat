@echo off
rem =====================================================================
rem  GFRAM Studio - o'rnatish va ishga tushirish / ustanovka i zapusk (Windows 10/11)
rem  Ikki marta bosing. Keyingi safar shunchaki ochiladi.
rem =====================================================================
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul
title GFRAM Studio
set "HERE=%~dp0"
set "APP=%LOCALAPPDATA%\GFRAM\StudioApp"
set "ENV=%APP%\env"
set "PYVER=3.12.10"
set "PYURL=https://www.python.org/ftp/python/%PYVER%/python-%PYVER%-amd64.exe"

echo.
echo    GFRAM Studio - gibrid yuz tanish tizimi / gibridnoe raspoznavanie lic
echo.

if /I not "%~1"=="--reinstall" if exist "%ENV%\Scripts\gfram-studio.exe" goto :run

rem ---------- 1. Python 3.10-3.14
echo ==^> 1/4  Python tekshirilmoqda / proverka Python
set "PY="
for %%V in (3.12 3.11 3.13 3.10 3.14) do (
  if not defined PY (
    for /f "delims=" %%P in ('py -%%V -c "import sys;print(sys.executable)" 2^>nul') do set "PY=%%P"
  )
)
for %%D in (Python312 Python311 Python313 Python310) do (
  if not defined PY if exist "%LOCALAPPDATA%\Programs\Python\%%D\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\%%D\python.exe"
)
if not defined PY call :install_python
if not defined PY (
  echo [X] Python o'rnatilmadi / Python ne ustanovlen.
  goto :fail
)
echo [OK] Python: %PY%

rem ---------- 2. Private environment + GFRAM
echo.
echo ==^> 2/4  GFRAM Studio o'rnatilmoqda (birinchi marta 3-10 daqiqa) / ustanovka (vpervye 3-10 minut)
if not exist "%APP%" mkdir "%APP%"
"%PY%" -m venv "%ENV%" || goto :fail
"%ENV%\Scripts\python.exe" -m pip install --upgrade --disable-pip-version-check pip >nul
set "SPEC=gfram[studio]>=3.4.0"
for %%W in ("%HERE%gfram-*.whl") do set "SPEC=%%~fW[studio]"
echo Paket / paket: !SPEC!
"%ENV%\Scripts\python.exe" -m pip install --upgrade --disable-pip-version-check "!SPEC!" || goto :fail

echo.
echo ==^> Modellar tayyorlanmoqda / podgotovka modelej...
"%ENV%\Scripts\python.exe" -c "import logging,tempfile;logging.disable(30);from gfram.api.simple_recognizer import SimpleRecognizer as R;r=R(db_path=tempfile.mkdtemp(),contribute=False);r.hybrid;print('models ready:',r.embedder.version)" || goto :fail

rem ---------- 3. Camera
echo.
echo ==^> 3/4  Kamera / kamera
"%ENV%\Scripts\python.exe" -c "import cv2,time;c=cv2.VideoCapture(0);t=time.time();ok=False;exec('while time.time()-t<6 and not ok:\n ok,_=c.read()');c.release();print('camera:', 'OK' if ok else 'NO ACCESS')" > "%TEMP%\gfram_cam.txt" 2>nul
type "%TEMP%\gfram_cam.txt"
findstr /C:"camera: OK" "%TEMP%\gfram_cam.txt" >nul
if errorlevel 1 (
  echo Kameraga ruxsat yoqilmagan bo'lishi mumkin. Sozlamalar ochiladi:
  echo "Ilovalarga kameradan foydalanishga ruxsat" va "Ish stoli ilovalari" ni YOQING.
  echo Vozmozhno, kamera zapreshchena. Otkroyutsya nastrojki: vklyuchite dostup dlya prilozhenij.
  start "" ms-settings:privacy-webcam
  pause
)

rem ---------- 4. Shortcuts
echo.
echo ==^> 4/4  Yorliq / yarlyk
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$t='%ENV%\Scripts\gfram-studio.exe';" ^
  "foreach($d in @([Environment]::GetFolderPath('Desktop'), (Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs'))){" ^
  " $s=(New-Object -ComObject WScript.Shell).CreateShortcut((Join-Path $d 'GFRAM Studio.lnk'));" ^
  " $s.TargetPath=$t; $s.WorkingDirectory=$env:USERPROFILE; $s.Description='GFRAM Studio'; $s.Save() }"
echo [OK] Ish stolida "GFRAM Studio" / na rabochem stole "GFRAM Studio"

:run
echo.
echo ==^> GFRAM Studio ishga tushirilmoqda / zapusk...
start "" "%ENV%\Scripts\gfram-studio.exe"
exit /b 0

:install_python
echo Python 3.10+ topilmadi - Python %PYVER% o'rnatiladi / Python ne najden - ustanavlivaetsya Python %PYVER%
where winget >nul 2>nul
if not errorlevel 1 (
  winget install -e --id Python.Python.3.12 --scope user --silent --accept-package-agreements --accept-source-agreements
  if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
)
if defined PY exit /b 0
set "PYEXE=%TEMP%\python-%PYVER%-amd64.exe"
powershell -NoProfile -Command "[Net.ServicePointManager]::SecurityProtocol='Tls12'; Invoke-WebRequest -UseBasicParsing '%PYURL%' -OutFile '%PYEXE%'" || exit /b 1
"%PYEXE%" /quiet InstallAllUsers=0 PrependPath=0 Include_launcher=0 Include_test=0
if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
exit /b 0

:fail
echo.
echo [X] Xatolik / oshibka. Internetni tekshirib, qayta urinib ko'ring / proverte internet i povtorite.
pause
exit /b 1
