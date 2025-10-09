@echo off
echo ===============================
echo 🚀 Starting AI Healthcare Chatbot
echo ===============================

:: Activate virtual environment
call venv\Scripts\activate

:: Upgrade pip and install dependencies
python -m pip install --upgrade pip >nul
pip install -r requirements.txt >nul

:: Start FastAPI backend in a new window
echo Starting FastAPI backend...
start cmd /k "uvicorn src.main:app --reload"
echo Waiting for FastAPI to start...

:: Wait until the API is reachable (up to 30 seconds)
setlocal enabledelayedexpansion
set "READY=0"
for /L %%i in (1,1,30) do (
    timeout /t 1 >nul
    curl http://127.0.0.1:8000/docs >nul 2>&1
    if not errorlevel 1 (
        set "READY=1"
        goto :ready
    )
)
:ready

if "!READY!"=="1" (
    echo ✅ FastAPI is up and running!
) else (
    echo ⚠️ FastAPI did not respond after 30 seconds.
    echo Streamlit will start anyway...
)

:: Start Streamlit frontend
echo Launching Streamlit App...
start cmd /k "streamlit run src/app.py"

echo ===============================
echo ✅ Both FastAPI and Streamlit are running!
echo ===============================
pause
