#!/bin/bash
set -e

echo "Uninstalling existing pre-commit hooks..."
pre-commit uninstall --hook-type pre-commit || true
pre-commit uninstall --hook-type pre-push || true

echo "Installing pre-commit hooks..."
pre-commit install
pre-commit install --hook-type pre-push

echo "Pre-commit hooks installed successfully!"
