@echo off
rem Starts the app on the NVIDIA GPU (TensorRT + CUDA). Run setup_windows.bat once first.
cd /d "%~dp0"
if not exist venv\Scripts\python.exe (
    echo The virtual environment is missing. Run setup_windows.bat first.
    pause
    exit /b 1
)
venv\Scripts\python run.py --execution-provider cuda --execution-threads 5 %*
pause
