@echo off
setlocal EnableDelayedExpansion

echo ===========================================================================
echo  Asmeranda AI - Windows Installer & Portable Package Builder
echo ===========================================================================
echo.

set OUTPUT_DIR=InstallerOutput
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

REM 1. Cek Inno Setup Compiler (ISCC.exe)
set ISCC_PATH=""
if exist "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" (
    set ISCC_PATH="C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
) else if exist "C:\Program Files\Inno Setup 6\ISCC.exe" (
    set ISCC_PATH="C:\Program Files\Inno Setup 6\ISCC.exe"
) else (
    where iscc >nul 2>&1
    if not errorlevel 1 set ISCC_PATH="iscc"
)

if not %ISCC_PATH%=="" (
    echo [1/2] Menyusun Inno Setup Installer (.exe)...
    %ISCC_PATH% asmeranda.iss
    if not errorlevel 1 (
        echo.
        echo ✓ Installer .exe berhasil dibuat di folder: %OUTPUT_DIR%\
    ) else (
        echo [PERINGATAN] Kompilasi Inno Setup gagal.
    )
) else (
    echo [INFO] Inno Setup Compiler (ISCC.exe) tidak ditemukan di sistem.
    echo Anda dapat mengunduh Inno Setup dari https://jrsoftware.org/isdl.php
)

REM 2. Buat Portable ZIP Bundle
echo.
echo [2/2] Membuat Portable ZIP Package...
set ZIP_NAME=AsmerandaAI-Portable-v2.0.0.zip
powershell -NoProfile -Command "Compress-Archive -Path 'backend', 'frontend', 'core', 'nginx', 'docker-compose.yml', 'Dockerfile', 'Dockerfile.backend', 'run_local.bat', 'run_local.ps1', 'start_docker.bat', 'stop_docker.bat', 'deploy-docker-desktop.ps1', 'deploy-local.sh', 'workflow_validator.py', 'README.md', 'ARSITEKTUR ASMERANDA.md', '.env.example' -DestinationPath '%OUTPUT_DIR%\%ZIP_NAME%' -Force"

if exist "%OUTPUT_DIR%\%ZIP_NAME%" (
    echo ✓ Portable ZIP berhasil dibuat: %OUTPUT_DIR%\%ZIP_NAME%
)

echo.
echo ===========================================================================
echo  PROSES BUILD SELESAI
echo  Hasil output tersimpan di folder: %OUTPUT_DIR%\
echo ===========================================================================
echo.
pause
