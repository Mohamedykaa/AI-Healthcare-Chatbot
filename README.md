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

### Install & Run
```bash
# 1. Install dependencies
pip install -r requirements_chainlit.txt

# 2. Ingest medical data (first time only)
python scripts/ingest_data.py

# 3. Start the chatbot
chainlit run app.py
```

The chatbot will be available at `http://localhost:8000`.

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


## Project Integrity Check

Run the full validation suite:

```bash
python scripts/check_project_integrity.py
```

This verifies:
- Python version compatibility
- Required files presence
- Core dependency imports
- Full unit test suite execution

---

## Architectural Evolution

Earlier microservices-based iterations of this project 
(FastAPI, Docker, etc.) are preserved locally for reference 
but are not included in this repository to keep it lightweight.

---

## 👨‍💻 Credits

Developed by Mohamed Yaser | 2026

⭐ If you found this project useful, consider giving it a star on GitHub!
