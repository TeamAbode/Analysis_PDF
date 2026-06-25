@echo off
REM Jury Analyst Pipeline - one-click launcher (Windows)
REM
REM Double-click this file to start the server. A command window will open
REM showing logs. Close that window to stop the server.
REM
REM First-run setup (venv + dependencies) happens automatically.

cd /d "%~dp0"

echo ============================================================
echo   Jury Analyst Pipeline
echo ============================================================
echo.

REM --- Verify Python is available ---
where python >nul 2>&1
if errorlevel 1 (
  echo ERROR: Python was not found on this PC.
  echo.
  echo   Install Python 3 from https://www.python.org/downloads/
  echo   IMPORTANT: tick "Add python.exe to PATH" during install.
  echo   Then re-run this launcher.
  echo.
  pause
  exit /b 1
)

REM --- Create venv on first run ---
if not exist ".venv" (
  echo First-run setup: creating virtual environment...
  python -m venv .venv
  if errorlevel 1 (
    echo ERROR: failed to create the virtual environment.
    pause
    exit /b 1
  )
  echo.
  echo Installing dependencies (one-time, ~30-60 seconds)...
  call .venv\Scripts\activate.bat
  python -m pip install --quiet --upgrade pip
  python -m pip install --quiet -r requirements.txt
  if errorlevel 1 (
    echo ERROR: dependency install failed.
    echo Try manually:  .venv\Scripts\activate  then  pip install -r requirements.txt
    pause
    exit /b 1
  )
  echo Setup complete.
  echo.
) else (
  call .venv\Scripts\activate.bat
)

REM --- Preflight: WeasyPrint native libraries ---
REM On Windows, WeasyPrint needs the GTK runtime. If the import fails, point
REM the user to the installer rather than failing later with a cryptic error.
python -c "import weasyprint" >nul 2>&1
if errorlevel 1 (
  echo ------------------------------------------------------------
  echo   One more setup step is needed (PDF engine libraries).
  echo ------------------------------------------------------------
  echo.
  echo   WeasyPrint needs the GTK3 runtime on Windows.
  echo   Install it from:
  echo     https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer/releases
  echo   Then open a NEW command window and re-run this launcher.
  echo.
  pause
  exit /b 1
)

REM --- Verify .env exists ---
if not exist ".env" (
  echo NOTE: .env file not found. Creating one...
  if exist ".env.example" (
    copy /y ".env.example" ".env" >nul
  ) else (
    echo ANTHROPIC_API_KEY=> .env
  )
  echo.
  echo   ^>^>^> You need to add your Anthropic API key to .env ^<^<^<
  echo   Opening it now in Notepad. Paste your key after the = sign
  echo   (it looks like sk-ant-...), save, then re-run this launcher.
  echo   Get a key at https://console.anthropic.com/
  echo.
  notepad .env
  pause
  exit /b 0
)

REM --- Open browser shortly, then start the server ---
start "" /b cmd /c "timeout /t 3 >nul & start http://localhost:8765"

echo Starting server on http://localhost:8765
echo Your browser will open automatically in a moment.
echo.
echo To stop the server: close this window.
echo ============================================================
echo.

python -u app.py

echo.
echo Server stopped.
pause
