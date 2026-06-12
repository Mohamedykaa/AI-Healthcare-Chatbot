# Project File Structure

The following tree outlines the active, deployed architecture of the Healthcare AI Chatbot. 

```text
d:\disease_prediction_project/
├── .chainlit/                  # Chainlit configuration and localization settings
│   └── config.toml             # UI config (forced to en-US to bypass missing translations)
├── chroma_db_v2/               # Active local vector database directory (SQLite schema v2)
├── data/                       # Raw dataset storage
│   ├── medical_knowledge_medquad.txt
│   └── medical_knowledge_medmcqa.txt
├── docs/                       # Project Documentation
│   └── archive/                # Legacy architecture documentation (Celery/Redis) - DO NOT USE
├── scripts/                    # Utility and evaluation scripts
│   ├── ingest_data.py          # Script to chunk and load data into ChromaDB
│   ├── evaluate_rag.py         # Quantitative evaluation script for RAG hit rates
│   └── admin_dashboard.py      # Streamlit-based admin and testing dashboard
├── src/                        # Main Application Source Code
│   ├── api/                    
│   │   └── main.py             # FastAPI REST endpoints
│   ├── core/                   
│   │   ├── config.py           # Centralized configuration variables
│   │   ├── logic.py            # Primary triage logic, RAG orchestration, Prompt Injection guard
│   │   └── risk.py             # Deterministic Emergency Risk Engine
│   ├── services/               
│   │   ├── llm.py              # Ollama LLaMA 3 integration setup
│   │   └── vectorstore.py      # ChromaDB retriever and embedding configuration
│   └── ui/                     
│   │   └── app.py              # Chainlit chat interface logic
├── tests/                      # Unit test suite for core logic
├── Dockerfile                  # Containerization script
├── README_NEW.md               # Main project documentation
├── requirements_api.txt        # Dependencies for FastAPI headless deployment
├── requirements_chainlit.txt   # Dependencies for Chainlit UI deployment
├── run_api.py                  # Entry point for the FastAPI server
└── run_app.py                  # Entry point for the Chainlit UI
```

## Core Directory Explanations

### `src/core/`
The heart of the application. 
- `logic.py` manages the dialogue state, checks for prompt injections, queries the vector database, formats prompts with contextual strategy, and invokes the LLM. 
- `risk.py` is entirely deterministic (no LLM) and intercepts critical inputs to ensure safety.

### `src/services/`
Handles external component wrappers. 
- `llm.py` connects locally to Ollama. 
- `vectorstore.py` manages HuggingFace embeddings and the local ChromaDB instance.

### `src/api/` & `src/ui/`
These are the two entry points. Both essentially wrap the `process_chat_message` function found in `src/core/logic.py`. This ensures identical safety and triage behavior regardless of whether a user connects via the Chainlit Web UI or a Flutter mobile app via the REST API.
