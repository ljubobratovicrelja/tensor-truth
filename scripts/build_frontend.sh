#!/usr/bin/env bash
# Build the React frontend and copy into the Python package for bundling.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== Building Frontend ==="

cd "$REPO_ROOT/frontend"

if [ ! -d "node_modules" ]; then
    echo "Installing frontend dependencies..."
    npm install
fi

echo "Running npm build..."
npm run build

echo "Copying build output to src/tensortruth/static/..."
rm -rf "$REPO_ROOT/src/tensortruth/static/"
cp -r dist/ "$REPO_ROOT/src/tensortruth/static/"

echo "Frontend build complete."
