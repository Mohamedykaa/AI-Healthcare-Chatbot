"""
AI Healthcare Chatbot — Environment Verification Script
========================================================
Run: .\venv\Scripts\python.exe verify_environment.py

Validates that the runtime environment is correctly configured.
Exit code 0 = all checks pass. Non-zero = failure.
"""

import importlib.util
import importlib.metadata
import sys
import os


def check(label: str, condition: bool, detail: str = "") -> bool:
    status = "PASS" if condition else "FAIL"
    msg = f"  [{status}] {label}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return condition


def main() -> int:
    print("=" * 60)
    print(" AI Healthcare Chatbot — Environment Verification")
    print("=" * 60)
    print()

    failures = 0

    # --- 1. Python version ---
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if not check("Python version", sys.version_info >= (3, 11), py_version):
        failures += 1

    # --- 2. Critical package presence ---
    print()
    print("Critical Packages:")
    required = [
        "chainlit", "langchain", "langchain_core", "langchain_community",
        "langchain_google_genai", "chromadb", "sentence_transformers",
        "pydantic", "dotenv", "tiktoken", "uvicorn", "fastapi",
    ]
    for mod in required:
        found = importlib.util.find_spec(mod) is not None
        if not check(f"  {mod}", found):
            failures += 1

    # --- 3. Version checks ---
    print()
    print("Version Pinning:")
    version_checks = {
        "chainlit": "1.0.505",
        "langchain": "0.1.20",
        "langchain-core": "0.1.52",
        "langchain-google-genai": "1.0.3",
        "chromadb": "0.4.24",
        "pydantic": "2.9.2",
        "sentence-transformers": "2.5.1",
    }
    for pkg, expected in version_checks.items():
        try:
            actual = importlib.metadata.version(pkg)
            ok = actual == expected
            if not check(f"  {pkg}", ok, f"expected={expected}, actual={actual}"):
                failures += 1
        except importlib.metadata.PackageNotFoundError:
            check(f"  {pkg}", False, "NOT INSTALLED")
            failures += 1

    # --- 4. Import chain tests ---
    print()
    print("Import Chain Tests:")

    # Gemini
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        check("  Gemini import chain", True, "ChatGoogleGenerativeAI loaded")
    except Exception as e:
        check("  Gemini import chain", False, str(e))
        failures += 1

    # ChromaDB
    try:
        from langchain_community.vectorstores import Chroma
        check("  ChromaDB import chain", True, "Chroma loaded")
    except Exception as e:
        check("  ChromaDB import chain", False, str(e))
        failures += 1

    # Embeddings
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        check("  Embeddings import chain", True, "HuggingFaceEmbeddings loaded")
    except Exception as e:
        check("  Embeddings import chain", False, str(e))
        failures += 1

    # Pydantic + FastAPI
    try:
        from pydantic import BaseModel
        from fastapi import FastAPI
        check("  Pydantic + FastAPI", True, "BaseModel + FastAPI loaded")
    except Exception as e:
        check("  Pydantic + FastAPI", False, str(e))
        failures += 1

    # --- 5. Conflict check ---
    print()
    print("Conflict Packages (must NOT be present):")
    conflict_packages = [
        "langchain-openai",   # Requires langchain-core >= 1.1.0
        "langgraph",          # Requires langchain-core >= 0.2.38
        "langgraph-checkpoint",
        "langgraph-prebuilt",
        "mcp",                # Requires pydantic >= 2.11.0
        "traceloop-sdk",      # Pulls 30+ opentelemetry conflicts
    ]
    for pkg in conflict_packages:
        try:
            importlib.metadata.version(pkg)
            check(f"  {pkg}", False, "INSTALLED — will cause conflicts!")
            failures += 1
        except importlib.metadata.PackageNotFoundError:
            check(f"  {pkg}", True, "not installed (correct)")

    # --- 6. .env file ---
    print()
    print("Configuration:")
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not check("  .env file exists", os.path.exists(env_path)):
        failures += 1
    else:
        with open(env_path, "r") as f:
            env_content = f.read()
        has_provider = "LLM_PROVIDER=gemini" in env_content
        has_key = "GEMINI_API_KEY=" in env_content and len(
            [l for l in env_content.splitlines() if l.startswith("GEMINI_API_KEY=") and len(l) > 16]
        ) > 0
        check("  LLM_PROVIDER=gemini", has_provider)
        check("  GEMINI_API_KEY set", has_key)
        if not has_provider:
            failures += 1
        if not has_key:
            failures += 1

    # --- Summary ---
    print()
    print("=" * 60)
    if failures == 0:
        print(" RESULT: ALL CHECKS PASSED ✓")
        print(" The environment is clean and ready for production.")
    else:
        print(f" RESULT: {failures} CHECK(S) FAILED ✗")
        print(" Run rebuild_environment.bat to fix.")
    print("=" * 60)

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
