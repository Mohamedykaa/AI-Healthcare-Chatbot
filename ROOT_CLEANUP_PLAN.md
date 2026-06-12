# Root Directory Cleanup Plan

This document outlines a safe reorganization plan to clean up the project root directory prior to the graduation defense. **No files have been modified or deleted.**

## 1. File Classification

### 🟢 KEEP IN ROOT
Files and folders required for normal project operation.
- **`.chainlit/`** — Chainlit UI configuration and localization settings. (Low Risk)
- **`.env`** — Environment variables. (Low Risk)
- **`.dockerignore` / `.gitignore`** — Git and Docker exclusions. (Low Risk)
- **`.pytest_cache/` / `.vscode/` / `.sixth/` / `.git/`** — IDE and system hidden folders. (Low Risk)
- **`Dockerfile`** — Container build instructions. (Low Risk)
- **`README.md`** — Current main documentation. (Low Risk)
- **`README_NEW.md`** — Pending manual review/replacement. (Low Risk)
- **`chainlit.md`** — Required markdown file for Chainlit homepage rendering. (Low Risk)
- **`chroma_db/` / `chroma_db_v2/`** — Vector database storage directories. (Low Risk)
- **`data/`** — Raw knowledge base text files. (Low Risk)
- **`docs/`** — Dedicated documentation directory. (Low Risk)
- **`pytest.ini`** — Test configuration. (Low Risk)
- **`requirements_api.txt` / `requirements_chainlit.txt`** — Dependency management. (Low Risk)
- **`run_all.bat` / `run_api.py` / `run_app.py`** — Main entry points for the application. (Low Risk)
- **`scripts/`** — Utility and maintenance scripts. (Low Risk)
- **`src/`** — Core application source code. (Low Risk)
- **`tests/`** — Unit test suite. (Low Risk)
- **`venv/`** — Python virtual environment. (Low Risk)
- **`verify_environment.py`** — Environment validation script. (Low Risk)

### 🔵 MOVE TO docs/
Documentation, audit reports, and generated markdown files that clutter the root.
- **`ARCHITECTURE.md`** — (Move)
- **`ARCHITECTURE_CLEANUP_REPORT.md`** — (Move)
- **`DEFENSE_SUMMARY.md`** — (Move)
- **`DOCUMENTATION_AUDIT.md`** — (Move)
- **`FINAL_DEFENSE_CONSISTENCY_REPORT.md`** — (Move)
- **`PROJECT_HEALTH_REPORT.md`** — (Move)
- **`PROJECT_STRUCTURE.md`** — (Move)
- **`SAFE_CLEANUP_CANDIDATES.md`** — (Move)

### 🟡 MOVE TO scripts/
Utility, helper, and maintenance tools currently orphaned in the root.
- **`generate_report.py`** — Helper script. (Low Risk to move)
- **`qa_runner.py`** — Evaluation tool. (Low Risk to move)
- **`rebuild_environment.bat`** — Maintenance tool. (Low Risk to move)
- **`run_real_fallback_test.py`** — Maintenance tool. (Low Risk to move)
- **`update_prompts.py`** — Migration helper. (Low Risk to move)

### 🟣 MOVE TO tests/
Orphaned unit tests that belong in the main test suite.
- **`test_fallback.py`** — (Low Risk to move)
- **`test_prompt.py`** — (Low Risk to move)
- **`test_rag.py`** — (Low Risk to move)

### 🟠 ARCHIVE
Legacy or deprecated files, such as the patch notes used during the recent version updates. (Move to `docs/archive/`).
- **`LOCAL_VERSION_MINIMAL_PATCH.md`** 
- **`LOCAL_VERSION_PATCH_ONLY.md`** 
- **`LOCAL_VERSION_REQUIRED_CHANGES.md`** 

### 🔴 DELETE CANDIDATE
Files proven unused, redundant, or orphaned output logs.
- **`evaluate_rag.py`** — Duplicate of `scripts/evaluate_rag.py`. (Low Risk to delete)
- **`chroma_db_qa_test/`** — Leftover test database folder. (Low Risk to delete)
- **`baseline_results.json`** — Output log. (Low Risk to delete)
- **`evaluation_results.json`** — Output log. (Low Risk to delete)
- **`post_change_results.json`** — Output log. (Low Risk to delete)
- **`rag_eval_results.json`** — Output log. (Low Risk to delete)
- **`real_test_output.txt`** — Output log. (Low Risk to delete)

---

## 2. Ideal Root Structure

If the above plan is executed, the professional graduation-project structure will look like this:

```text
project/
├── .chainlit/
├── .env.example
├── .gitignore
├── Dockerfile
├── README.md
├── chainlit.md
├── chroma_db/
├── chroma_db_v2/
├── data/
├── docs/
├── pytest.ini
├── requirements_api.txt
├── requirements_chainlit.txt
├── run_all.bat
├── run_api.py
├── run_app.py
├── scripts/
├── src/
├── tests/
└── verify_environment.py
```
