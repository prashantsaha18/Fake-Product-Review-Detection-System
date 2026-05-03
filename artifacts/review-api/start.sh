#!/bin/bash
set -e

echo "Starting ML API server on port ${PORT:-8090}..."
cd artifacts/review-api
python main.py
