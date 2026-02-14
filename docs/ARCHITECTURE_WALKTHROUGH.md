# AI Healthcare Chatbot - Architecture Walkthrough

## Project Overview

This is a **medical decision-support chatbot** with the following core capabilities:
- **RAG-based retrieval** over curated medical knowledge texts
- **Multi-turn conversation** with history-aware context
- **Emergency detection** with immediate safety routing
- **Source citation** for every response
- **Offline operation** using local LLM (Ollama / Llama 3) and CPU embeddings

---

## High-Level Architecture

```mermaid
flowchart TB
    subgraph Chainlit ["Chainlit UI (app.py)"]
        Start[on_chat_start]
        Msg[on_message]
    end

    subgraph RAG ["RAG Pipeline"]
        LLM[ChatOllama - Llama 3 8B]
        Embed[HuggingFace Embeddings - MiniLM]
        VS[ChromaDB vectorstore]
        Chain[History-Aware Retrieval Chain]
    end

    subgraph Safety ["Safety Layer"]
        Emerg[check_for_emergency]
    end

    subgraph Data ["Data Sources"]
        MK1[medical_knowledge_medmcqa.txt]
        MK2[medical_knowledge_medquad.txt]
        MK3[medical_knowledge_public_health.txt]
    end

    Start --> VS
    Start --> Chain
    Msg --> Emerg
    Emerg -->|Safe| VS
    VS -->|retrieve docs| LLM
    LLM -->|response + sources| Msg
    Embed --> VS
    Data -->|ingested via ingest_data.py| VS
```

---

## Data Flow (Per Request)

```
User Input → Chainlit on_message handler
                        ↓
              1. check_for_emergency(user_input)
                        ↓ (if emergency → immediate safety response)
                        ↓ (if safe)
              2. Retrieve conversation history from cl.user_session
                        ↓
              3. Manual retrieval via vectorstore.as_retriever()
                 - search_kwargs: k=6, score_threshold=0.3
                        ↓
              4. Build custom prompt:
                 - SYSTEM_PROMPT (personality / guardrails)
                 - USER_INSTRUCTIONS (analysis template)
                 - Retrieved context documents
                 - Conversation history
                        ↓
              5. llm.ainvoke(prompt) → LLM response
                        ↓
              6. format_sources(context_documents)
                        ↓
              7. Append to chat history in session
                        ↓
              Response → Chainlit UI → User
```

---

## Core Components

### 1. Chainlit Application ([app.py](file:///d:/disease_prediction_project/app.py))

The **single-file application** serving as the chat UI; all RAG and risk logic lives in [backend/core.py](file:///d:/disease_prediction_project/backend/core.py).

**Handlers:**
- `on_chat_start()` — Session init: initializes shared components (if needed) and sets empty chat history in `cl.user_session`
- `on_message()` — Per-message handler: delegates to `backend.core.process_chat_message()` (emergency check → manual retrieval → LLM call → source formatting → response)

**Key logic (in [backend/core.py](file:///d:/disease_prediction_project/backend/core.py)):**
- `get_embedding_function()` — CPU-forced HuggingFace embeddings (all-MiniLM-L6-v2)
- `load_or_create_vectorstore()` — Loads ChromaDB from `./chroma_db` or creates from `data/medical_knowledge_*.txt`
- `validate_chroma_db()` — Integrity test query on startup
- `normalize_text()` — Whitespace / newline cleaning
- `get_llm()` — Initialises `ChatOllama` with Llama 3 8B
- **Manual retrieval only** — No RAG chain is built at runtime; the retriever is invoked directly and a custom prompt is built. A LangChain history-aware retrieval chain is **reserved for future use** if moving to a larger/cloud model (see *RAG Chain vs. Manual Retrieval* below).
- Emergency detection and response — Implemented in [backend/risk.py](file:///d:/disease_prediction_project/backend/risk.py) (pure, dependency-free); `format_sources()` in core formats retrieved documents as citations.

### 2. Data Ingestion Script ([scripts/ingest_data.py](file:///d:/disease_prediction_project/scripts/ingest_data.py))

**Purpose:** Pre-process and ingest medical knowledge texts into ChromaDB.

- Reads `data/medical_knowledge_*.txt` files
- Splits text using `RecursiveCharacterTextSplitter`
- Creates/updates the persistent ChromaDB vectorstore in `./chroma_db`

### 3. Emergency Detection (in [app.py](file:///d:/disease_prediction_project/app.py))

**Purpose:** Pre-validate user input for safety before RAG processing.

**Checks:**
- Emergency keyword matching → immediate safety response with emergency contacts
- Bypasses RAG pipeline entirely when triggered

---

## User Interface

### Chainlit UI (built into [app.py](file:///d:/disease_prediction_project/app.py))

The UI is provided by the **Chainlit** framework — no separate frontend process is required.

**Features:**
- Real-time streaming chat interface
- Automatic session management via `cl.user_session`
- Source citations appended to every response
- Welcome message with usage guidance (configured in `chainlit.md`)

**Session State (per user):**
- `chat_history` — List of `HumanMessage` / `AIMessage` objects (vectorstore and LLM are process-wide singletons in `backend.core`, not stored per session)

---

## Data Files

| File | Purpose |
|------|---------|
| [medical_knowledge_medmcqa.txt](file:///d:/disease_prediction_project/data/medical_knowledge_medmcqa.txt) | Medical Q&A knowledge (MedMCQA corpus) |
| [medical_knowledge_medquad.txt](file:///d:/disease_prediction_project/data/medical_knowledge_medquad.txt) | Medical Q&A knowledge (MedQuAD corpus) |
| [medical_knowledge_public_health.txt](file:///d:/disease_prediction_project/data/medical_knowledge_public_health.txt) | Public health knowledge base |
| `chroma_db/` | Persistent ChromaDB vector store (created by `scripts/ingest_data.py` or `app.py` on first run) |

---

## Archived / Legacy Architecture

> [!NOTE]
> The components listed below are **no longer active**. They have been moved to
> the [`archive/`](file:///d:/disease_prediction_project/archive) directory and
> are preserved for historical reference only.

The project previously used a multi-process architecture:

| Legacy Component | Original Path | Description |
|------------------|---------------|-------------|
| FastAPI Backend | `backend/app.py` | Orchestration API with `/orchestrate` endpoint |
| Streamlit Frontend | `frontend/streamlit_app.py` | Multi-page UI (chat, skin lesion, history, settings) |
| Agent Layer | `src/chatbot_system/` | SymptomAgent, DiagnosisAgent, FollowUpManager, RecommendationAgent |
| Safety Router | `backend/safety_router.py` | Input pre-validation (red flags, injection, harmful content) |
| Clean Architecture | `app/` | DDD-style layers (application, core, domain, infrastructure) |
| BioMistral Provider | `app/infrastructure/ai/llm_provider.py` | GGUF model via ctransformers |
| Docker / Nginx | `Dockerfile`, `docker-compose.yml`, `nginx/` | Containerised deployment stack |
| ML Pipeline | `.joblib` models | TF-IDF + Logistic Regression scoring |

All of the above now reside under `archive/` and are **not used** by the active system.

---

## LLM Integration

### ChatOllama — Llama 3 8B (configured in [backend/core.py](file:///d:/disease_prediction_project/backend/core.py))

- Served locally via **Ollama** (`ChatOllama` LangChain wrapper)
- Model: `llama3:8b` (configurable via `LLM_MODEL` env var)
- Temperature: `0.3` (conservative for medical context)
- Embeddings: `all-MiniLM-L6-v2` via HuggingFace, forced to CPU

---

## Testing Structure

```
tests/
├── conftest.py                    # Pytest fixtures & helpers
├── test_conversation_logic.py     # Multi-turn conversation flow tests
├── test_emergency_detection.py    # Emergency keyword detection tests
├── test_safety_filter.py          # Safety / injection filter tests
├── test_source_formatting.py      # Source citation formatting tests
└── test_text_processing.py        # Text normalisation & cleaning tests
```

Run with: `pytest tests/`

---

## Key Configuration

| Setting | Location | Purpose |
|---------|----------|---------|
| `CHROMA_PERSIST_DIR` | backend/core.py / `.env` | ChromaDB storage path (default `./chroma_db`) |
| `LLM_MODEL` | backend/core.py / `.env` | LLM model name (default `llama3:8b`) |
| `LLM_TEMPERATURE` | backend/core.py (hardcoded) | LLM sampling temperature (0.3) |
| Retriever `k` / `score_threshold` | backend/core.py | Number of documents and minimum similarity (see code) |

---

## Note: RAG Chain vs. Manual Retrieval

The active codebase uses **manual retrieval only**: [backend/core.py](file:///d:/disease_prediction_project/backend/core.py) calls the vectorstore retriever directly and builds a custom prompt for each message. No LangChain `create_history_aware_retriever` or `create_retrieval_chain` is built or invoked at runtime.

| Aspect | History-Aware RAG Chain | Manual Retrieval (active) |
|--------|------------------------|---------------------------|
| **Purpose** | Future use (e.g. larger or cloud-hosted model) | Current implementation |
| **Status** | Not present in codebase | Used for every user message in `process_chat_message()` |
| **Reason** | — | Full control over prompt formatting, echo-guardrails, and turn-phase logic for LLaMA 3 8B on local hardware |

This is a **conscious design decision**: the 8B model on CPU benefits from tight prompt engineering and an explicit echo guardrail, which are easier with direct LLM calls. A LangChain retrieval chain can be added later (e.g. in `backend/core.py` or a separate module) when moving to a larger model or cloud inference.

---

## Running the Application

```bash
# 1. (First time only) Ingest medical knowledge into ChromaDB
python scripts/ingest_data.py

# 2. Start the chatbot
chainlit run app.py
```

> [!TIP]
> The vectorstore is also auto-created on first run of `app.py` if
> `./chroma_db` does not exist, but running `ingest_data.py` separately
> gives more control over the ingestion process.
