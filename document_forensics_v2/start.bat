@echo off
REM Starts the backend using the project's .venv (D:\task1\.venv), not
REM whatever "python"/uvicorn resolves to on PATH. Two different Python
REM installs on this machine can both satisfy `pip show fastapi`, but only
REM the .venv one has a working PyMuPDF build for this project's pinned
REM version -- running from the wrong interpreter fails at import time
REM (main.py imports fitz transitively via metadata_extractor.py, before
REM the FastAPI app object even exists), so the whole backend won't start.
REM
REM Before starting, kill anything already listening on port 8000 so stale
REM servers can't silently keep serving old code next to a new one. Two
REM kill passes are needed: taskkill /T for a live process tree, PLUS a
REM sweep of children by ParentProcessId -- uvicorn --reload's listener has
REM repeatedly died on this machine while its multiprocessing spawn child
REM kept the socket alive, leaving a "ghost" PID that netstat reports but
REM taskkill says doesn't exist (the child holds the port, not the parent).

cd /d "%~dp0"

echo Checking for existing listeners on port 8000...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='SilentlyContinue'; $pids = Get-NetTCPConnection -LocalPort 8000 -State Listen | Select-Object -ExpandProperty OwningProcess -Unique; foreach ($p in $pids) { Write-Host ('Port 8000 in use by PID ' + $p + ' - killing it and any children'); taskkill /F /T /PID $p *>$null; Get-CimInstance Win32_Process -Filter ('ParentProcessId=' + $p) | ForEach-Object { Write-Host ('Killing orphaned child PID ' + $_.ProcessId + ' of PID ' + $p); Stop-Process -Id $_.ProcessId -Force } }; Start-Sleep -Seconds 1; if (Get-NetTCPConnection -LocalPort 8000 -State Listen) { exit 1 } else { exit 0 }"
if errorlevel 1 (
    echo ERROR: port 8000 is still in use after the kill attempt - refusing
    echo to stack a second server on top of it. Inspect what is holding it:
    echo   netstat -ano ^| findstr :8000
    exit /b 1
)

"..\.venv\Scripts\python.exe" -m uvicorn main:app --reload --port 8000
