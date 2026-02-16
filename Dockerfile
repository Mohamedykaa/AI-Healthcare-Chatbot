# ============================================================
# AI Healthcare Chatbot — Dockerfile
# ============================================================
# Runs the Chainlit chat UI.  Ollama must run on the HOST;
# set OLLAMA_BASE_URL=http://host.docker.internal:11434 or
# the actual host IP so the container can reach it.
# ============================================================

FROM python:3.11-slim AS base

# Prevent .pyc files and enable unbuffered output for logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# ---- Dependencies ----
COPY requirements_chainlit.txt .
RUN pip install --no-cache-dir -r requirements_chainlit.txt

# ---- Application code ----
COPY backend/          backend/
COPY scripts/          scripts/
COPY data/             data/
COPY app.py            .
COPY api.py            .
COPY chainlit.md       .
COPY .chainlit/        .chainlit/
COPY pytest.ini        .

# Expose the default Chainlit port
EXPOSE 8000

# Health-check: Chainlit serves on /
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')" || exit 1

# Default command: start Chainlit
CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8000"]
