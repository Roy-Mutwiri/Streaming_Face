@echo off
setlocal EnableDelayedExpansion

:: 1. Setup your platform
echo Setting up your platform...
:: Python
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Python is not installed. Please install Python 3.10 or later.
    pause
    exit /b
)
:: Pip
where pip >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Pip is not installed. Please install Pip.
    pause
    exit /b
)
:: Git
where git >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Git is not installed. Installing Git...
    winget install --id Git.Git -e --source winget
)
:: FFMPEG
where ffmpeg >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo FFMPEG is not installed. Installing FFMPEG...
    winget install --id Gyan.FFmpeg -e --source winget
)
:: Visual Studio 2022 Runtimes
echo Installing Visual Studio 2022 Runtimes...
winget install --id Microsoft.VC++2015-2022Redist-x64 -e --source winget

:: 2. Clone Repository
@REM if exist iRoopDeepFaceCam (
@REM     echo iRoopDeepFaceCam directory already exists.
@REM     set /p overwrite="Do you want to overwrite? (Y/N): "
@REM     if /i "%overwrite%"=="Y" (
@REM         rmdir /s /q iRoopDeepFaceCam
@REM         git clone https://github.com/iVideoGameBoss/iRoopDeepFaceCam.git
@REM     ) else (
@REM         echo Skipping clone, using existing directory.
@REM     )
@REM ) else (
@REM     git clone https://github.com/iVideoGameBoss/iRoopDeepFaceCam.git
@REM )
@REM cd iRoopDeepFaceCam

:: 3. Download Models
echo Downloading models...
if not exist models mkdir models
curl -L -o models\GFPGANv1.4.pth https://huggingface.co/ivideogameboss/iroopdeepfacecam/resolve/main/GFPGANv1.4.pth
curl -L -o models\inswapper_128_fp16.onnx https://huggingface.co/ivideogameboss/iroopdeepfacecam/resolve/main/inswapper_128_fp16.onnx

:: 4. Install dependencies
echo Creating a virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

echo Installing required Python packages...
pip install --upgrade pip
pip install -r requirements.txt
echo Setup complete. You can now run the application.

:menu
:: GPU Acceleration Options
echo.
echo Choose the GPU Acceleration Option if applicable:
echo 1. CUDA (Nvidia)
echo 2. CoreML (Apple Silicon)
echo 3. CoreML (Apple Legacy)
echo 4. DirectML (Windows)
echo 5. OpenVINO (Intel)
echo 6. None
set /p choice="Enter your choice (1-6): "

set "exec_provider="
if "%choice%"=="1" goto cuda
if "%choice%"=="2" goto coreml_silicon
if "%choice%"=="3" goto coreml_legacy
if "%choice%"=="4" goto directml
if "%choice%"=="5" goto openvino
if "%choice%"=="6" goto none
echo Invalid choice. Please try again.
goto menu

:cuda
echo Installing CUDA dependencies...
pip uninstall -y onnxruntime onnxruntime-gpu
pip install onnxruntime-gpu==1.16.3
set "exec_provider=cuda"
goto end_choice

:coreml_silicon
echo Installing CoreML (Apple Silicon) dependencies...
pip uninstall -y onnxruntime onnxruntime-silicon
pip install onnxruntime-silicon==1.13.1
set "exec_provider=coreml"
goto end_choice

:coreml_legacy
echo Installing CoreML (Apple Legacy) dependencies...
pip uninstall -y onnxruntime onnxruntime-coreml
pip install onnxruntime-coreml==1.13.1
set "exec_provider=coreml"
goto end_choice

:directml
echo Installing DirectML dependencies...
pip uninstall -y onnxruntime onnxruntime-directml
pip install onnxruntime-directml==1.15.1
set "exec_provider=directml"
goto end_choice

:openvino
echo Installing OpenVINO dependencies...
pip uninstall -y onnxruntime onnxruntime-openvino
pip install onnxruntime-openvino==1.15.0
set "exec_provider=openvino"
goto end_choice

:none
echo Skipping GPU acceleration setup.
set "exec_provider=none"
goto end_choice

:end_choice
echo.
echo GPU Acceleration setup complete.
echo Selected provider: !exec_provider!
echo.

:: Run the application
if defined exec_provider (
    echo Running the application with !exec_provider! execution provider...
    python run.py --execution-provider !exec_provider!
) else (
    echo Running the application...
    python run.py
)

:: Deactivate the virtual environment
call venv\Scripts\deactivate.bat

echo.
echo Script execution completed.
pause