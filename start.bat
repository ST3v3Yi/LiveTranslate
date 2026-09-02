@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" goto setup
if not exist ".venv\.livetranslate-ready" goto setup
goto launch

:setup
    echo First run: setting up environment. This downloads Python and dependencies and may take several minutes...
    powershell -ExecutionPolicy Bypass -File "%~dp0bootstrap.ps1"
    if errorlevel 1 (
        echo.
        echo [ERROR] Setup failed. See messages above.
        pause
        exit /b 1
    )

:launch
echo Starting LiveTranslate...
.venv\Scripts\python.exe main.py
if errorlevel 1 (
    echo.
    echo [ERROR] LiveTranslate exited with an error.
    pause
)
