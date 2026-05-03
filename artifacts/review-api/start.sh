#!/bin/bash
set -e

cd artifacts/review-api

if [ ! -d ".venv" ]; then
  echo "Creating Python virtual environment..."
  python3 -m venv .venv
fi

source .venv/bin/activate

echo "Installing Python dependencies..."
pip install -q -r requirements.txt

echo "Starting ML API server on port ${PORT:-8090}..."
python main.py
