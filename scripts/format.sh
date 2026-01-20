#!/bin/bash
# Formatting script - formats both Python and TypeScript/JavaScript code.

set -e

TARGETS="src/tensortruth scripts tests"

echo "=== Python Formatting ==="

echo "Running isort..."
isort $TARGETS

echo "Running black..."
black $TARGETS

echo ""
echo "=== Frontend Formatting ==="

if [ -d "frontend" ]; then
    cd frontend
    echo "Running ESLint fix..."
    npm run lint:fix || true
    echo "Running Prettier..."
    npm run format
    cd ..
else
    echo "Frontend directory not found, skipping frontend formatting"
fi

echo ""
echo "Formatting complete."
