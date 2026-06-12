@echo off
REM =============================================================================
REM AI Healthcare Chatbot — Environment Rebuild Script
REM =============================================================================
REM This script destroys the contaminated venv and rebuilds it from scratch.
REM Run from the project root: rebuild_environment.bat
REM =============================================================================

setlocal EnableDelayedExpansion

echo.
echo ============================================================
echo  AI Healthcare Chatbot — Environment Rebuild
echo ============================================================
echo.

REM --- Step 1: Verify we are in the right directory ---
if not exist "requirements_chainlit.txt" (
    echo [ERROR] requirements_chainlit.txt not found.
    echo         Run this script from the project root directory.
    exit /b 1
)
if not exist "src\ui\app.py" (
    echo [ERROR] src\ui\app.py not found.
    echo         Run this script from the project root directory.
    exit /b 1
)
echo [OK] Project root confirmed.

REM --- Step 2: Kill any running Python processes using the venv ---
echo.
echo [STEP 2] Stopping any running venv processes...
taskkill /F /FI "IMAGENAME eq python.exe" /FI "WINDOWTITLE eq *disease_prediction*" >nul 2>&1
REM Brief pause to release file handles
timeout /t 2 /nobreak >nul 2>&1

REM --- Step 3: Delete existing venv ---
echo.
echo [STEP 3] Removing existing virtual environment...
if exist "venv" (
    rmdir /s /q venv
    if exist "venv" (
        echo [WARNING] Could not fully remove venv. Retrying...
        timeout /t 3 /nobreak >nul 2>&1
        rmdir /s /q venv
    )
    if exist "venv" (
        echo [ERROR] Failed to remove venv. Close all terminals/IDEs using it and retry.
        exit /b 1
    )
    echo [OK] Old venv removed.
) else (
    echo [OK] No existing venv found.
)

REM --- Step 4: Create fresh venv ---
echo.
echo [STEP 4] Creating fresh virtual environment...
python -m venv venv
if errorlevel 1 (
    echo [ERROR] Failed to create venv. Ensure Python 3.11+ is installed.
    exit /b 1
)
echo [OK] Virtual environment created.

REM --- Step 5: Upgrade pip ---
echo.
echo [STEP 5] Upgrading pip...
.\venv\Scripts\python.exe -m pip install --upgrade pip --quiet
if errorlevel 1 (
    echo [WARNING] pip upgrade failed, continuing with existing pip.
)
echo [OK] pip upgraded.

REM --- Step 6: Install requirements ---
echo.
echo [STEP 6] Installing dependencies from requirements_chainlit.txt...
.\venv\Scripts\python.exe -m pip install -r requirements_chainlit.txt
if errorlevel 1 (
    echo [ERROR] pip install failed. Check the output above for errors.
    exit /b 1
)
echo [OK] Dependencies installed.

REM --- Step 7: Run pip check ---
echo.
echo [STEP 7] Running dependency audit (pip check)...
.\venv\Scripts\python.exe -m pip check 2>&1
echo [INFO] Review any conflicts above. Minor opentelemetry warnings are safe to ignore.

REM --- Step 8: Validate critical imports ---
echo.
echo [STEP 8] Validating critical package imports...
.\venv\Scripts\python.exe -c "import importlib.util; pkgs=[('chainlit','chainlit'),('langchain','langchain'),('langchain_core','langchain-core'),('langchain_community','langchain-community'),('langchain_google_genai','langchain-google-genai'),('chromadb','chromadb'),('sentence_transformers','sentence-transformers'),('pydantic','pydantic'),('dotenv','python-dotenv')]; failed=[p for m,p in pkgs if importlib.util.find_spec(m) is None]; print('ALL OK' if not failed else f'MISSING: {failed}'); exit(1 if failed else 0)"
if errorlevel 1 (
    echo [ERROR] Critical imports failed. See output above.
    exit /b 1
)
echo [OK] All critical packages verified.

REM --- Step 9: Test Gemini import specifically ---
echo.
echo [STEP 9] Testing Gemini LLM import chain...
.\venv\Scripts\python.exe -c "from langchain_google_genai import ChatGoogleGenerativeAI; print('[OK] ChatGoogleGenerativeAI loaded successfully')"
if errorlevel 1 (
    echo [ERROR] Gemini import chain broken.
    exit /b 1
)

REM --- Step 10: Report ---
echo.
echo ============================================================
echo  REBUILD COMPLETE
echo ============================================================
echo.
echo  All packages installed and validated.
echo.
echo  To start the Chainlit app:
echo    .\venv\Scripts\python.exe run_app.py
echo.
echo  To start the FastAPI server:
echo    .\venv\Scripts\python.exe -m uvicorn src.api.main:app --port 8001
echo.
echo ============================================================

REM --- Step 11: Auto-start the application ---
echo.
echo [STEP 11] Starting application...
.\venv\Scripts\python.exe run_app.py
