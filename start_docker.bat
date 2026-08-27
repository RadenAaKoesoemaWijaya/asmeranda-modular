@echo off
REM ===========================================================================
REM Asmeranda AI - Start Docker Containers
REM ===========================================================================
echo ===========================================================================
echo  Starting Asmeranda AI via Docker Desktop...
echo ===========================================================================
echo.

where docker >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker tidak ditemukan di PATH.
    echo Pastikan Docker Desktop sudah terinstal dan sedang berjalan.
    pause
    exit /b 1
)

echo [1/2] Building / Checking Docker images...
docker compose build

echo.
echo [2/2] Starting containers in background...
docker compose up -d

echo.
echo ===========================================================================
echo  Asmeranda AI berhasil dijalankan!
echo  Frontend UI:  http://localhost:3000
echo  Backend API:  http://localhost:8000
echo  Swagger Docs: http://localhost:8000/docs
echo ===========================================================================
echo.
echo Untuk menghentikan kontainer: jalankan stop_docker.bat
echo Untuk melihat log kontainer:   docker compose logs -f
echo.
pause
