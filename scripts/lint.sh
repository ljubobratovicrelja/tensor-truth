#!/bin/bash
# Linting checks - used by CI and locally to ensure consistency.

set -e

TARGETS="src/tensortruth scripts tests"

echo "Running black..."
black --check $TARGETS

echo "Running isort..."
isort --check-only $TARGETS

echo "Running flake8..."
flake8 $TARGETS || true

echo "Running mypy..."
mypy src/tensortruth --ignore-missing-imports || true

echo "Linting complete."
