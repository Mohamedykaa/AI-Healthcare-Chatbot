@echo off
REM AI Healthcare Chatbot - Windows quick start
REM Prerequisites: Python 3.10+, Ollama installed and running, ollama pull llama3:8b

echo Installing dependencies...
pip install -r requirements_chainlit.txt
if errorlevel 1 exit /b 1

echo.
echo Ingesting medical data (first time only; skip if chroma_db already exists)...
python scripts/ingest_data.py
if errorlevel 1 (
    echo Ingest failed. If chroma_db exists, you can continue.
    pause
)

echo.
echo Starting Chainlit chatbot at http://localhost:8000 ...
python run_app.py
