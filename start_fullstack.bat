@echo off
REM Full-stack startup script (Backend + Frontend)

echo ========================================
echo PersonalAI - Intelligent Hybrid Recommendation System
echo ========================================
echo.
echo Starting BACKEND and FRONTEND...
echo.

REM Check if venv exists
if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv
)

REM Activate venv
call .venv\Scripts\activate.bat

REM Install backend dependencies
echo Installing backend dependencies...
pip install -r requirements.txt -q

REM Check if processed data exists
if not exist data\processed\ratings.parquet (
    echo.
    echo Processed data not found. Running data pipeline...
    echo This may take several minutes on first run...
    echo.
    python scripts\download_data.py
    python scripts\preprocess.py
)

REM Initialize database
echo Initializing database...
python -c "from src.api.database import init_db; init_db(); print('Database ready')"

REM Check if frontend dependencies are installed
if not exist app\frontend\node_modules (
    echo.
    echo Installing frontend dependencies...
    cd app\frontend
    call npm install
    cd ..\..
)

echo.
echo ========================================
echo Starting services...
echo ========================================
echo Backend API: http://localhost:8000/docs
echo Frontend UI: http://localhost:3000
echo.
echo Press Ctrl+C in either window to stop
echo ========================================
echo.

REM Start backend in new window
start "PersonalAI Backend" cmd /k "call .venv\Scripts\activate.bat && uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload"

REM Wait 3 seconds for backend to start
timeout /t 3 /nobreak >nul

REM Start frontend in new window
start "PersonalAI Frontend" cmd /k "cd app\frontend && npm run dev"

REM Open browser after 5 seconds
timeout /t 5 /nobreak >nul
start http://localhost:3000

echo.
echo Both services started in separate windows!
echo Close those windows to stop the services.
echo.
pause
