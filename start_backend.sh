#!/bin/bash
set -e
cd "$(dirname "$0")"
echo "Starting Kisaan AI backend..."
venv/bin/uvicorn backend.main:app --reload --port 8000
