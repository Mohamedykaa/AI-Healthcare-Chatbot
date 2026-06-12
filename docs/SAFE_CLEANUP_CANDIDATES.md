# Safe Cleanup Candidates

The following items have been identified as cleanup candidates based on the `ARCHITECTURE_CLEANUP_REPORT.md` and subsequent codebase verification via `grep_search`. 

**No source files have been modified.** This list is for post-graduation defense review.

---

## 1. Legacy Architecture Documentation
* **File name:** `docs/archive/` (Directory)
* **Exact code/function/import:** All files within this directory.
* **Why it is unused:** Contains outdated notes on the deprecated Celery, Redis, and BioMistral microservices architecture. The active project uses Chainlit and FastAPI.
* **Where verified:** Verified by reviewing the active architecture documentation and confirming none of the active code relies on or references these markdown files.
* **Risk level:** **SAFE**

---

## 2. Duplicate Evaluation Script
* **File name:** `evaluate_rag.py` (in root directory)
* **Exact code/function/import:** Entire file (`evaluate_rag.py`, 4.8 KB).
* **Why it is unused:** This is an orphaned or older duplicate of the official evaluation script located at `scripts/evaluate_rag.py` (14.0 KB) which is the one explicitly documented in the README.
* **Where verified:** `list_dir` confirmed the presence of the official script in the `scripts/` directory, and `grep_search` confirmed the root file is not imported by any core modules.
* **Risk level:** **SAFE**

---

## 3. Orphaned Debugging/Testing Scripts
* **File names:** `qa_runner.py`, `generate_report.py`, `test_fallback.py`, `run_real_fallback_test.py` (All in root directory)
* **Exact code/function/import:** Entire files.
* **Why it is unused:** These appear to be temporary, local debugging or legacy testing scripts. They are not part of the official `tests/` suite or the `scripts/` utility folder.
* **Where verified:** `grep_search` across the entire codebase yielded **0 results** for any imports of these modules. They are completely disconnected from the FastAPI and Chainlit entry points.
* **Risk level:** **SAFE**

---

## 4. Redundant Risk Assessment Wrapper
* **File name:** `src/core/risk.py`
* **Exact code/function/import:** `def check_for_emergency(user_input: str) -> bool:`
* **Why it is unused:** The primary application logic in `src/core/logic.py` directly calls `assess_risk_level()` (which returns string enums like "EMERGENCY" or "ROUTINE") and bypasses `check_for_emergency` entirely.
* **Where verified:** `grep_search` confirmed this function is never called by `logic.py`, `app.py`, or `main.py`. It is *only* imported and executed inside `tests/test_risk_scoring.py` and `tests/test_emergency_detection.py`.
* **Risk level:** **MEDIUM** (Safe to remove from the application core, but will cause unit tests to fail unless the tests are refactored to use `assess_risk_level()` instead).

---

## 5. Legacy SQLite Database Validity Check
* **File name:** `src/services/vectorstore.py`
* **Exact code/function/import:** `def _is_valid_chroma_sqlite(db_path: str) -> bool:`
* **Why it is unused:** This was a migration helper designed to check for corrupted v1 ChromaDB SQLite files. If the project strictly uses `chroma_db_v2`, this extensive fallback verification is obsolete.
* **Where verified:** `grep_search` confirmed it is only called internally within `vectorstore.py` on line 136. It is not exposed to or used by any external module.
* **Risk level:** **MEDIUM** (Requires confirming that ChromaDB persistence is stable and that no edge-case database corruption errors will crash the application if the check is removed).
