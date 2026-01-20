#!/bin/bash
# Linting checks - used by CI and locally to ensure consistency.

set -e

TARGETS="src/tensortruth scripts tests"

echo "=== Python Linting ==="

echo "Running black..."
black --check $TARGETS

echo "Running isort..."
isort --check-only $TARGETS

echo "Running flake8..."
flake8 $TARGETS || true

echo "Running mypy..."
mypy src/tensortruth --ignore-missing-imports || true

echo ""
echo "=== Frontend Linting ==="

if [ -d "frontend" ]; then
    cd frontend
    echo "Running ESLint..."
    npm run lint
    echo "Checking Prettier formatting..."
    npm run format:check
    cd ..
else
    echo "Frontend directory not found, skipping frontend linting"
fi

echo ""
echo "Linting complete."
