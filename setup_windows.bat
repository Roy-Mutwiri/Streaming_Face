@echo off
rem One-time setup on Windows with an NVIDIA GPU. See README.md, "Installation".
cd /d "%~dp0"

where py >nul 2>&1
if errorlevel 1 (
    echo The Python launcher "py" was not found. Install Python 3.11 from python.org first.
    pause
    exit /b 1
)
py -3.11 --version >nul 2>&1
if errorlevel 1 (
    echo Python 3.11 is not installed. Install Python 3.11 ^(64-bit^) from python.org first.
    pause
    exit /b 1
)

if not exist venv\Scripts\python.exe (
    echo Creating the virtual environment in venv\ ...
    py -3.11 -m venv venv || goto :error
)

echo Installing packages ^(several GB on first install^) ...
venv\Scripts\python -m pip install --upgrade pip || goto :error
venv\Scripts\python -m pip install -r requirements.txt || goto :error

where ffmpeg >nul 2>&1
if errorlevel 1 (
    echo.
    echo NOTE: ffmpeg was not found. It is only needed for processing video files:
    echo       winget install Gyan.FFmpeg
)

echo.
echo Setup complete. Start the app with run-cuda.bat
pause
exit /b 0

:error
echo.
echo Setup failed. Scroll up for the error message.
pause
exit /b 1
