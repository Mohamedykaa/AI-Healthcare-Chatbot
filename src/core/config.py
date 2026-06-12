import os

# ============================================================
# CONFIGURATION
# ============================================================

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
CHROMA_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db")
MEDICAL_KNOWLEDGE_FILES = [
    "data/medical_knowledge_medquad.txt",
    "data/medical_knowledge_medmcqa.txt",
    "data/medical_knowledge_public_health.txt",
]
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama").lower()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
LLM_MODEL = os.environ.get("LLM_MODEL", "llama3:8b")
LLM_TIMEOUT = int(os.environ.get("LLM_TIMEOUT", "30"))
LLM_MAX_RETRIES = int(os.environ.get("LLM_MAX_RETRIES", "3"))
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "50"))
RETRIEVER_K = int(os.environ.get("RETRIEVER_K", "10"))
RETRIEVER_SCORE_THRESHOLD = float(os.environ.get("RETRIEVER_SCORE_THRESHOLD", "0.2"))
CONTEXT_CHAR_LIMIT_PER_DOC = int(os.environ.get("CONTEXT_CHAR_LIMIT_PER_DOC", "2000"))
