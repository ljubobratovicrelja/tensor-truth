#!/bin/bash
# Test runner - used by CI and locally to ensure consistency.

set -e

echo "Running unit tests..."
pytest tests/unit/ -v --cov=tensortruth --cov-report=xml --cov-report=term

echo "Running integration tests..."
pytest tests/integration/ -v --run-network -m "not slow"

echo "Tests complete."
