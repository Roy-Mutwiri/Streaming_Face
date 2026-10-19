@echo off
rem Starts tools\auto_push.py at every Windows logon (no window) and starts it now.
rem Remove again with:  schtasks /Delete /TN "StreamingFace AutoPush" /F
cd /d "%~dp0.."

set PYW=
for /f "delims=" %%P in ('py -3.11 -c "import sys,os;print(os.path.join(os.path.dirname(sys.executable),'pythonw.exe'))" 2^>nul') do set PYW=%%P
if "%PYW%"=="" (
    echo Python 3.11 was not found. Install it first.
    pause
    exit /b 1
)

schtasks /Create /F /TN "StreamingFace AutoPush" /SC ONLOGON /RL LIMITED /TR "\"%PYW%\" \"%CD%\tools\auto_push.py\"" || goto :error
start "" "%PYW%" "%CD%\tools\auto_push.py"
echo.
echo Auto-push is installed and running. Changes are pushed to GitHub 2 minutes after you stop editing.
echo Log: %CD%\tools\auto_push.log
pause
exit /b 0

:error
echo Could not create the scheduled task.
pause
exit /b 1
