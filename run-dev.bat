@echo off
echo Starting both frontend and backend development servers...
start "Frontend" cmd /k "cd frontend && npm run dev"
start "Backend" cmd /k "call .venv\Scripts\python.exe -m uvicorn mm_yoga.backend.app:app --host 0.0.0.0 --port 8000 --reload"

timeout /t 3 /nobreak >nul

start "" http://localhost:5173