@echo off
REM Starts the backend using the project's .venv (D:\task1\.venv), not
REM whatever "python"/uvicorn resolves to on PATH. Two different Python
REM installs on this machine can both satisfy `pip show fastapi`, but only
REM the .venv one has a working PyMuPDF build for this project's pinned
REM version -- running from the wrong interpreter fails at import time
REM (main.py imports fitz transitively via metadata_extractor.py, before
REM the FastAPI app object even exists), so the whole backend won't start.
cd /d "%~dp0"
"..\.venv\Scripts\python.exe" -m uvicorn main:app --reload --port 8000
