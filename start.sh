#!/bin/bash
# PersonalAI — Linux/macOS startup script
# Run this to start the full stack locally

set -e

echo "========================================"
echo "PersonalAI - Intelligent Hybrid Recommendation System"
echo "========================================"
echo

# Check if venv exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate venv
source .venv/bin/activate

# Install/update dependencies
echo "Installing dependencies..."
pip install -r requirements.txt -q

# Check if processed data exists
if [ ! -f "data/processed/ratings.parquet" ]; then
    echo
    echo "Processed data not found. Running data pipeline..."
    echo "This may take several minutes on first run..."
    echo
    python scripts/download_data.py
    python scripts/preprocess.py
fi

# Initialize database
echo "Initializing database..."
python -c "from src.api.database import init_db; init_db(); print('Database ready')"

echo
echo "Starting PersonalAI API server..."
echo "Frontend will be available at: http://localhost:3000"
echo "API docs at: http://localhost:8000/docs"
echo
echo "Press Ctrl+C to stop"
echo

# Start the API server
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload