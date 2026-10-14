@echo off
REM PersonalAI — Windows startup script
REM Run this to start the full stack locally

echo ========================================
echo PersonalAI - Intelligent Hybrid Recommendation System
echo ========================================
echo.

REM Check if venv exists
if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv
)

REM Activate venv
call .venv\Scripts\activate.bat

REM Install/update dependencies
echo Installing dependencies...
pip install -r requirements.txt -q

REM Check if processed data exists
if not exist data\processed\ratings.parquet (
    echo.
    echo Processed data not found. Running data pipeline...
    echo This may take several minutes on first run...
    echo.
    python scripts/download_data.py
    python scripts/preprocess.py
)

REM Initialize database
echo Initializing database...
python -c "from src.api.database import init_db; init_db(); print('Database ready')"

echo.
echo Starting PersonalAI API server...
echo Frontend will be available at: http://localhost:3000
echo API docs at: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop
echo.

REM Start the API server
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload