@echo off
setlocal EnableDelayedExpansion

echo ===========================================================================
echo  Asmeranda AI - Localhost Launcher
echo ===========================================================================
echo.

REM 1. Cek Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python tidak ditemukan. Silakan install Python 3.10+ terlebih dahulu.
    pause
    exit /b 1
)

REM 2. Cek Node.js & npm
node --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js tidak ditemukan. Silakan install Node.js 18+ terlebih dahulu.
    pause
    exit /b 1
)

REM 3. Virtual Environment Python
if exist ".venv\Scripts\activate.bat" (
    echo [1/4] Mengaktifkan virtual environment (.venv)...
    call .venv\Scripts\activate.bat
) else if exist "venv\Scripts\activate.bat" (
    echo [1/4] Mengaktifkan virtual environment (venv)...
    call venv\Scripts\activate.bat
) else (
    echo [1/4] Membuat Python virtual environment (.venv)...
    python -m venv .venv
    call .venv\Scripts\activate.bat
    echo Menginstal dependensi backend...
    pip install -q --upgrade pip
    pip install -r backend\requirements-backend.txt
)

REM 4. Cek Dependensi Frontend
if not exist "frontend\node_modules" (
    echo [2/4] Menginstal dependensi frontend (npm install)...
    cd frontend
    call npm install
    cd ..
)

REM 5. Jalankan Backend Server
echo [3/4] Menjalankan Backend FastAPI (Port 8000)...
start "Asmeranda AI - Backend Server" cmd /k "call .venv\Scripts\activate.bat && python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload"

REM 6. Jalankan Frontend Server
echo [4/4] Menjalankan Frontend Next.js (Port 3000)...
start "Asmeranda AI - Frontend UI" cmd /k "cd frontend && npm run dev"

echo.
echo ===========================================================================
echo  Asmeranda AI sedang berjalan di Localhost!
echo  ---------------------------------------------------------------------------
echo  Frontend UI  : http://localhost:3000
echo  Backend API  : http://localhost:8000
echo  Swagger Docs : http://localhost:8000/docs
echo ===========================================================================
echo.
echo Menutup jendela ini tidak akan mematikan server (server berjalan di jendela terpisah).
echo.
timeout /t 5
