# ===========================================================================
# Asmeranda AI - Localhost PowerShell Launcher
# ===========================================================================

Write-Host "===========================================================================" -ForegroundColor Cyan
Write-Host " Asmeranda AI - Localhost Launcher" -ForegroundColor Cyan
Write-Host "===========================================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Cek Python
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ Python terdeteksi: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ [ERROR] Python tidak ditemukan. Silakan instal Python 3.10+" -ForegroundColor Red
    exit 1
}

# 2. Cek Node.js
try {
    $nodeVersion = node --version 2>&1
    Write-Host "✓ Node.js terdeteksi: $nodeVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ [ERROR] Node.js tidak ditemukan. Silakan instal Node.js 18+" -ForegroundColor Red
    exit 1
}

# 3. Setup Virtual Environment
if (Test-Path ".venv\Scripts\Activate.ps1") {
    Write-Host "[1/4] Mengaktifkan virtual environment (.venv)..." -ForegroundColor Yellow
    . .venv\Scripts\Activate.ps1
} elseif (Test-Path "venv\Scripts\Activate.ps1") {
    Write-Host "[1/4] Mengaktifkan virtual environment (venv)..." -ForegroundColor Yellow
    . venv\Scripts\Activate.ps1
} else {
    Write-Host "[1/4] Membuat Python virtual environment (.venv)..." -ForegroundColor Yellow
    python -m venv .venv
    . .venv\Scripts\Activate.ps1
    Write-Host "Menginstal dependensi backend..." -ForegroundColor Yellow
    pip install -q --upgrade pip
    pip install -r backend\requirements-backend.txt
}

# 4. Cek Dependensi Frontend
if (-not (Test-Path "frontend\node_modules")) {
    Write-Host "[2/4] Menginstal dependensi frontend (npm install)..." -ForegroundColor Yellow
    Push-Location frontend
    npm install
    Pop-Location
}

# 5. Jalankan Backend & Frontend
Write-Host "[3/4] Menjalankan Backend FastAPI (Port 8000)..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "if (Test-Path .venv\Scripts\Activate.ps1) { . .venv\Scripts\Activate.ps1 } elseif (Test-Path venv\Scripts\Activate.ps1) { . venv\Scripts\Activate.ps1 }; python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload"

Write-Host "[4/4] Menjalankan Frontend Next.js (Port 3000)..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location frontend; npm run dev"

Write-Host ""
Write-Host "===========================================================================" -ForegroundColor Green
Write-Host " Asmeranda AI sedang berjalan di Localhost!" -ForegroundColor Green
Write-Host " ---------------------------------------------------------------------------" -ForegroundColor Green
Write-Host " Frontend UI  : http://localhost:3000" -ForegroundColor White
Write-Host " Backend API  : http://localhost:8000" -ForegroundColor White
Write-Host " Swagger Docs : http://localhost:8000/docs" -ForegroundColor White
Write-Host "===========================================================================" -ForegroundColor Green
Write-Host ""
