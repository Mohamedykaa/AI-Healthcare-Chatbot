# 🏥 AI Healthcare Chatbot

## 📌 Project Description
The AI Healthcare Chatbot is a localized, privacy-first, educational medical symptom checker. It leverages Retrieval-Augmented Generation (RAG) to provide grounded medical guidance, powered by a local Large Language Model (LLaMA 3). 

The system acts as a digital triage assistant, designed to help users understand their symptoms and know when to seek urgent medical care. It features robust safety layers, deterministic emergency detection, and native bilingual support (English and Arabic), all while operating completely offline to ensure maximum patient data privacy.

---

## 🚀 Features

- **Retrieval-Augmented Generation (RAG):** Context-aware responses grounded in a curated medical knowledge base (MedQuad, MedMCQA).
- **Deterministic Risk Assessment:** Instantly escalates critical symptoms (e.g., heart attack, stroke) to emergency status, bypassing the LLM.
- **Prompt Injection Protection:** Rejects malicious instructions and requests for drug prescriptions before they reach the language model.
- **Sufficiency-Based Triage:** Intelligently asks clarifying questions to gather missing information (onset, severity, red flags) before providing a differential assessment.
- **Bilingual Interface:** Seamlessly processes and responds in English and Arabic based on the user's input.
- **Local Execution:** 100% offline inference using Ollama, ensuring zero patient data leaves the host machine.

---

## 🏗️ Architecture & Tech Stack

| Component | Technology |
|-----------|------------|
| **Core Framework** | FastAPI |
| **Chat Interface** | Chainlit (Web) / Flutter (Mobile Client) |
| **LLM** | LLaMA 3 8B (via local Ollama) |
| **Embeddings** | all-MiniLM-L6-v2 (HuggingFace, CPU) |
| **Vector Store** | ChromaDB (persistent local storage) |
| **Orchestration** | LangChain |

---

## 💻 Installation

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.ai/) installed and running locally
- LLaMA 3 model pulled: `ollama pull llama3:8b`

### Setup Steps
```bash
# 1. Clone the repository
git clone <repository_url>
cd <repository_directory>

# 2. Install dependencies (Chainlit UI & API)
pip install -r requirements_chainlit.txt
pip install -r requirements_api.txt

# 3. Ingest medical data (first time only)
python scripts/ingest_data.py
```

---

## 🏃‍♂️ Running the Project

You can run the project using either the Chainlit Web UI or the FastAPI REST server.

### Option A: Chainlit Web UI (Port 8000)
```bash
python run_app.py
# OR
chainlit run src/ui/app.py
```

### Option B: FastAPI REST API (Port 8001)
```bash
python run_api.py
# OR
uvicorn src.api.main:app --host 0.0.0.0 --port 8001
```

*(Note: Do not run both on the same port. They share the identical underlying logic).*

---

## 📡 API Endpoints

### `POST /chat`
Processes a user message and returns an educational medical response.

**Request:**
```json
{
  "message": "I have a severe headache and vision changes",
  "history": []
}
```

**Response:**
```json
{
  "response": "⚠️ **URGENT ADVICE REQUIRED:** Your symptoms may need same-day medical evaluation...\n\nBased on your description of a severe headache accompanied by vision changes...",
  "risk_level": "URGENT",
  "sources": "\n\n---\n**📚 References:** medical_knowledge_medquad.txt"
}
```

---

## 🛡️ Safety Notes & Limitations

- **Educational Use Only:** This system is not a doctor. It provides triage-oriented education and cannot issue definitive clinical diagnoses.
- **No Treatment Guidance:** The chatbot is explicitly hardcoded to refuse requests for medication names, dosages, or therapeutic plans.
- **Corpus Dependent:** The quality and breadth of the responses are strictly limited by the contents of the ingested medical vector database.
- **Hardware Constraints:** Running an 8B parameter model locally requires adequate system RAM and may experience inference latency on CPU-only machines.

---

## 🔮 Future Improvements

- Optimization of context window management for faster time-to-first-token.
- Expanding the localized Arabic vector database for higher retrieval precision.
- Refinement of the deterministic risk engine scoring weights based on broader clinical guidelines.
