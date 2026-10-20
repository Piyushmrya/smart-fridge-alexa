#!/usr/bin/env bash
set -e

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$APP_DIR/../.." && pwd)"

# Detect Python interpreter (priority: amazon_project/venv > active virtualenv > python3)
if [ -f "/home/niare/Projects/amazon_project/venv/bin/python" ]; then
    PYTHON_EXEC="/home/niare/Projects/amazon_project/venv/bin/python"
elif [ -n "$VIRTUAL_ENV" ] && [ -f "$VIRTUAL_ENV/bin/python" ]; then
    PYTHON_EXEC="$VIRTUAL_ENV/bin/python"
else
    PYTHON_EXEC="python3"
fi

PORT="${PORT:-8000}"

echo "=================================================="
echo "  🧊 Smart Fridge Bounding Box Labeling Web App"
echo "=================================================="
echo "Python interpreter: $PYTHON_EXEC"
echo "Application directory: $APP_DIR"
echo "Server URL: http://localhost:$PORT"
echo "=================================================="
echo "Press Ctrl+C to stop the server."
echo ""

cd "$APP_DIR"
exec "$PYTHON_EXEC" -m uvicorn app:app --host 0.0.0.0 --port "$PORT" --reload
