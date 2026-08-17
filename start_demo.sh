#!/usr/bin/env bash
# PM2.5 Hyperlocal Demo - two-terminal startup (Linux/macOS)
# Terminal 1: FastAPI backend on http://127.0.0.1:8000  (docs at /docs)
# Terminal 2: Dashboard on http://localhost:5173
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Starting PM2.5 Hyperlocal Demo..."
echo "  Backend : http://127.0.0.1:8000  (Swagger /docs)"
echo "  Dashboard: http://localhost:5173"
echo "Press Ctrl+C to stop."

.venv/bin/python -m uvicorn api.main:app --reload &
BACKEND_PID=$!
trap "kill $BACKEND_PID 2>/dev/null || true" EXIT

cd dashboard
npm run dev
