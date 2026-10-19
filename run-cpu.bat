echo Creating a virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

echo Setup complete. You can now run the application.

echo Running the application with cuda execution provider...
python run.py --execution-provider cpu

call venv\Scripts\deactivate.bat

echo.
echo Script execution completed.
pause