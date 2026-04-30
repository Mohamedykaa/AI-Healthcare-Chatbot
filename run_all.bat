@echo off
REM AI Healthcare Chatbot - Windows quick start
REM Prerequisites: Python 3.10+, Ollama installed and running, ollama pull llama3:8b

echo Installing dependencies...
pip install -r requirements_chainlit.txt
if errorlevel 1 exit /b 1

echo.
REM Gate on all 3 knowledge files — ingestion creates these, not the ChromaDB
if exist data\medical_knowledge_medquad.txt (
    if exist data\medical_knowledge_medmcqa.txt (
        if exist data\medical_knowledge_public_health.txt (
            echo Found all knowledge files. Skipping ingestion.
            goto start_app
        )
    )
)

echo Ingesting medical data for the first run...
python scripts/ingest_data.py
if errorlevel 1 (
    echo Ingest failed. Check the output above.
    pause
    exit /b 1
)

:start_app
echo.
echo Starting Chainlit chatbot at http://localhost:8000 ...
python run_app.py
