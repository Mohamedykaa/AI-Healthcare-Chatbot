# 🏥 AI Healthcare Chatbot

> **📌 Project Note — Architecture Evolution**
>
> This project has undergone an architectural evolution. The original
> microservices stack (FastAPI, Celery, Redis, Nginx WAF, Docker Compose)
> has been **archived** in the `/archive` directory for reference.
>
> The **active, production-ready implementation** is a self-contained
> **Chainlit-based RAG chatbot** powered by LLaMA 3 (via Ollama),
> HuggingFace embeddings, and ChromaDB. All instructions below apply
> to this active version.

---

## 🏗️ Active Architecture

| Layer | Technology |
|-------|------------|
| **Chat Interface** | Chainlit |
| **LLM** | LLaMA 3 8B (ChatOllama, local) |
| **Embeddings** | all-MiniLM-L6-v2 (HuggingFace, CPU) |
| **Vector Store** | ChromaDB (persistent, local) |
| **RAG Framework** | LangChain (retrieval + generation) |
| **Data Sources** | MedQuad, MedMCQA, Medical Meadow WikiDoc |

**Key features:**
- Retrieval-Augmented Generation for grounded medical Q&A
- Emergency symptom detection with immediate escalation
- Safety-filtered knowledge base (no dosage/prescription content)
- Multi-turn conversation with history-aware context
- Source citation on every response

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.ai/) installed and running
- LLaMA 3 model pulled: `ollama pull llama3:8b`

### Requirements files
- **`requirements_chainlit.txt`** — Use this for the **Chainlit chat UI** (recommended). Covers Chainlit, LangChain, ChromaDB, embeddings, and Ollama integration.
- **`requirements_api.txt`** — Use this if you only need the **FastAPI REST API** (e.g. headless or programmatic access). Install this instead of (or in addition to) the Chainlit requirements when running `api.py`.

### Install & Run
```bash
# 1. Install dependencies (Chainlit UI)
pip install -r requirements_chainlit.txt

# 2. Ingest medical data (first time only)
python scripts/ingest_data.py

# 3. Start the chatbot
chainlit run app.py
```

The chatbot will be available at `http://localhost:8000`.

**Windows:** You can run `run_all.bat` to perform steps 1–3 in one go (install, optional ingest, then start Chainlit).

### Running the API instead of (or alongside) the UI
To run the **FastAPI** server (e.g. for programmatic access), use a **different port** so it does not conflict with Chainlit:
```bash
pip install -r requirements_api.txt   # if not already installed
uvicorn api:app --host 0.0.0.0 --port 8001
```
The API will be at `http://localhost:8001`. Chainlit and the API share the same backend logic; do not run both on port 8000.

### 🐳 Docker Deployment
Build and run the chatbot inside a Docker container. Ollama must be running on the **host machine**.
```bash
# Build the image
docker build -t medical-chatbot .

# Run (Linux/macOS — Ollama on host)
docker run -p 8000:8000 -e OLLAMA_BASE_URL=http://host.docker.internal:11434 medical-chatbot

# Run (Windows/Docker Desktop — same command works)
docker run -p 8000:8000 -e OLLAMA_BASE_URL=http://host.docker.internal:11434 medical-chatbot
```

---

## 🛡️ Safety Features

- **Emergency detection** — Escalates chest pain, seizures, suicidal ideation, etc.
- **Content filtering** — Ingestion pipeline strips dosage, prescription, and treatment data
- **Echo guardrails** — Detects and handles LLM echo/empty responses
- **Medical disclaimers** — Appended to every response automatically

---

## 🧪 Testing

```bash
# Run unit tests (no LLM or ChromaDB required)
pytest tests/

# Manual RAG chain verification (requires Ollama)
python scripts/test_rag_chain.py
```

## 📊 RAG Evaluation

Quantitatively measure the quality of the RAG pipeline:

```bash
# Full evaluation (retrieval + triage + answer faithfulness)
python scripts/evaluate_rag.py

# Retrieval-only mode (no LLM needed)
python scripts/evaluate_rag.py --retrieval
```

Metrics measured:
- **Triage Accuracy** — Deterministic emergency/urgent/routine classification against gold labels
- **Retrieval Hit Rate** — Percentage of questions where relevant documents are successfully retrieved
- **Answer Faithfulness** — Whether LLM answers reference retrieved context (not hallucinated)

Results are saved to `evaluation_results.json`.

## 🖥️ Admin Dashboard

A Streamlit-based monitoring dashboard for system administration:

```bash
pip install streamlit
streamlit run scripts/admin_dashboard.py
```

**Features:**
- 🩺 **System Health** — Live status of Ollama, ChromaDB, and embeddings
- 📚 **Knowledge Base Inspector** — Browse and preview ingested medical data
- 🚨 **Emergency Detection Tester** — Interactive risk triage classifier
- 🔍 **RAG Query Runner** — Send test queries through the full pipeline

## Project Integrity Check

Run the full validation suite:

```bash
python scripts/check_project_integrity.py
```

For CI/release gates (strict dependency enforcement):

```bash
python scripts/check_project_integrity.py --strict-imports
```

This verifies:
- Python version compatibility
- Required files presence
- Core dependency imports
- Full unit test suite execution

## Clinical Calibration Notes

The deterministic triage layer is intentionally calibrated to reduce false-positive emergency escalation while preserving safety:

- **Hard-stop critical signals** (e.g., suicidal intent, unconsciousness, stroke, heart attack, heatstroke/sunstroke) always return `EMERGENCY`, independent of score.
- **Score thresholds** are set to:
  - `EMERGENCY >= 6`
  - `URGENT >= 3`
  - otherwise `ROUTINE`
- **Why `EMERGENCY >= 6`?** This avoids escalating a single severe symptom phrase (e.g., severe chest pain) to emergency without additional red flags or critical hard-stop terms.
- **Why `URGENT >= 3`?** A core symptom can still trigger same-day caution (`URGENT`) without forcing immediate emergency messaging.
- **Deterministic scope note:** The matcher uses boundary-aware phrase matching and phrase-level deduplication. It does not perform synonym/NLP concept normalization by design.

---

## 📂 Archived Architecture (Reference Only)

The original production-grade microservices design is preserved in `/archive`:

- **Gateway**: Nginx + ModSecurity WAF + OWASP CRS (SSL/TLS)
- **Frontend**: Streamlit (Polling Architecture)
- **Backend**: FastAPI (Async Task Queue)
- **Worker**: Celery + Redis (Offloaded Inference)
- **AI Core**: BioMistral (LLM) + TensorFlow (Vision)
- **Observability**: OpenTelemetry + Jaeger + Prometheus

Security features from the archived version:
- **WAF**: Blocks SQL Injection, XSS (ModSecurity)
- **Semantic Firewall**: Blocks Prompt Injection attacks
- **Rate Limiting**: 20 req/min throttling (SlowAPI)
- **Compliance**: PII Scrubbing & Encrypted DB Connections
- **Supply Chain**: Automated Trivy vulnerability scanning

```bash
# (Archived) Docker deployment
cd archive
docker-compose up --build -d
```

---

## 👨‍💻 Credits

Developed by Mohamed Yaser | 2026

⭐ If you found this project useful, consider giving it a star on GitHub!
