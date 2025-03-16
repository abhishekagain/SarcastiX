@echo off
echo Setting up SarcastiX Web Application...

echo.
echo Step 1: Creating Python virtual environment...
python -m venv venv
call venv\Scripts\activate

echo.
echo Step 2: Installing Python dependencies...
pip install -r requirements.txt

echo.
echo Step 3: Installing frontend dependencies...
cd frontend
call npm install

echo.
echo Setup complete! Run run_app.bat to start the application.
