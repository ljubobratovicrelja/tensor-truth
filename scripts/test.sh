#!/bin/bash
# Test runner - used by CI and locally to ensure consistency.

# Auto-activate venv if it exists
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

set -e

echo "Running unit tests..."
pytest tests/unit/ -v --cov=tensortruth --cov-report=xml --cov-report=term

echo "Running integration tests..."
pytest tests/integration/ -v --run-network -m "not slow"

echo "Tests complete."
