@echo off
echo Starting SarcastiX Web Application...

echo.
echo Step 1: Activating Python virtual environment...
cd %~dp0
call venv\Scripts\activate

echo.
echo Step 2: Starting Flask Backend API...
start cmd /k "python app.py"

echo.
echo Step 3: Starting React Frontend...
cd frontend
start cmd /k "npm run dev"

echo.
echo SarcastiX application is starting!
echo Backend API: http://localhost:3001
echo Frontend: http://localhost:3000
echo.
echo Press Ctrl+C in respective terminal windows to stop the servers.
