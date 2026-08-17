@echo off
REM PM2.5 Hyperlocal Demo - two-terminal startup (Windows)
REM Terminal 1: FastAPI backend on http://127.0.0.1:8000  (docs at /docs)
REM Terminal 2: Dashboard on http://localhost:5173

cd /d "%~dp0"

echo.
echo Starting PM2.5 Hyperlocal Demo...
echo   Backend : http://127.0.0.1:8000  (Swagger /docs)
echo   Dashboard: http://localhost:5173
echo Close both windows to stop the demo.
echo.

start "PM25 Backend" cmd /k ".venv\Scripts\python.exe -m uvicorn api.main:app --reload"
start "PM25 Dashboard" cmd /k "cd dashboard && npm run dev"
