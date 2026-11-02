@echo off
title S.P.I.D.E.Y Launcher
cd /d "%~dp0"
echo ========================================================
echo         Starting S.P.I.D.E.Y / ORBIT007 Environment
echo ========================================================

:: 0. Activate Virtual Environment (venv) if present
if exist "venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment (venv)...
    call "venv\Scripts\activate.bat"
) else if exist ".venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment (.venv)...
    call ".venv\Scripts\activate.bat"
) else (
    echo [WARNING] No venv found! Running with system Python.
)

:: 1. Start Ollama if installed
where ollama >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo [1/3] Starting Ollama server in background...
    start "Ollama Engine" /min cmd /c "ollama serve"
    timeout /t 2 /nobreak >nul
    ollama list 2>nul | findstr /i "qwen3.5:4b" >nul
    if %ERRORLEVEL% NEQ 0 (
        echo [WARNING] Ollama model 'qwen3.5:4b' not found in ollama list!
        echo Run: ollama pull qwen3.5:4b
    )
) else (
    echo [WARNING] Ollama not found in PATH!
    echo Please install Ollama from https://ollama.com and run: ollama pull qwen3.5:4b
)

:: 2. Start FastAPI Backend (Database, Agent WebSocket, REST API on Port 8000)
echo [2/3] Starting Python Backend with activated venv (Port 8000)...
if exist "venv\Scripts\activate.bat" (
    start "Spidey Backend" cmd /k "call venv\Scripts\activate.bat && python app.py --server"
) else if exist ".venv\Scripts\activate.bat" (
    start "Spidey Backend" cmd /k "call .venv\Scripts\activate.bat && python app.py --server"
) else (
    start "Spidey Backend" cmd /k "python app.py --server"
)

:: 3. Start Vite Frontend (Port 5173)
echo [3/3] Starting Frontend Web UI (Port 5173)...
start "Spidey Frontend" cmd /k "npm --prefix frontend run dev"

echo.
echo ========================================================
echo S.P.I.D.E.Y is starting!
echo Web UI: http://localhost:5173
echo Backend API: http://127.0.0.1:8000
echo ========================================================
echo.
pause
