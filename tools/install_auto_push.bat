@echo off
rem Starts tools\auto_push.py at every Windows logon (Startup-folder shortcut, no admin rights
rem needed, no window) and starts it now.
rem Remove: delete "StreamingFace AutoPush" from the Startup folder (Win+R, shell:startup)
rem         and run:  py -3.11 tools\auto_push.py --stop
cd /d "%~dp0.."

set PYW=
for /f "delims=" %%P in ('py -3.11 -c "import sys,os;print(os.path.join(os.path.dirname(sys.executable),'pythonw.exe'))" 2^>nul') do set PYW=%%P
if "%PYW%"=="" (
    echo Python 3.11 was not found. Install it first.
    pause
    exit /b 1
)

powershell -NoProfile -Command ^
  "$s = (New-Object -ComObject WScript.Shell).CreateShortcut([IO.Path]::Combine([Environment]::GetFolderPath('Startup'), 'StreamingFace AutoPush.lnk'));" ^
  "$s.TargetPath = '%PYW%'; $s.Arguments = '\"%CD%\tools\auto_push.py\"'; $s.WorkingDirectory = '%CD%'; $s.WindowStyle = 7; $s.Save()" || goto :error
start "" "%PYW%" "%CD%\tools\auto_push.py"
echo.
echo Auto-push is installed and running. Changes are pushed to GitHub 2 minutes after you stop editing.
echo Log: %CD%\tools\auto_push.log
pause
exit /b 0

:error
echo Could not create the Startup shortcut.
pause
exit /b 1
