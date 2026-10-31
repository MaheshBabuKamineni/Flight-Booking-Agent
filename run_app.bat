@echo off
REM Double-click to install dependencies and launch the Streamlit UI.
cd /d "%~dp0"
set "PYTHON=.venv\Scripts\python.exe"

if not exist "%PYTHON%" (
    py -3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)"
    if errorlevel 1 (
        echo Python 3.10 or newer is required. Install it and try again.
        pause
        exit /b 1
    )
    py -3 -m venv .venv
    if errorlevel 1 (
        echo Could not create the virtual environment.
        pause
        exit /b 1
    )
)

"%PYTHON%" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)"
if errorlevel 1 (
    echo The project's .venv uses Python older than 3.10.
    echo Recreate .venv with Python 3.10 or newer, then run this file again.
    pause
    exit /b 1
)

"%PYTHON%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo Dependency installation failed.
    pause
    exit /b 1
)

"%PYTHON%" -m streamlit run app.py
pause
