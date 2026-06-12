# Architecture Cleanup Report

## 1. Overview
This report analyzes the current codebase of the Healthcare AI Chatbot project to identify dead code, unused imports, obsolete migration logic, and legacy debugging remnants.

**Disclaimer:** No files have been automatically modified. This is a recommendation report for final cleanup prior to the graduation defense.

## 2. Cleanup Recommendations

### 2.1 Unused or Redundant Logic

| File | Issue | Reason | Safe to remove? | Risk Level |
|------|-------|--------|-----------------|------------|
| `src/core/risk.py` | Unused Function: `check_for_emergency(user_input)` | The main pipeline in `logic.py` uses `assess_risk_level()` directly which returns "EMERGENCY" as a string. `check_for_emergency` simply wraps this and is never called. | Yes | Low |
| `src/services/vectorstore.py` | Legacy Directory Checks | The code handles migration from `./chroma_db` to `./chroma_db_v2`. If the database is now strictly `v2`, the fallback checks for the v1 SQLite database can be deprecated. | Yes | Medium |
| `src/api/main.py` | Extraneous CORS Origins | Uses `allow_origins=["*"]`. While acceptable for local dev, it is a security flag for production/defense review. | No (Change to specific domains instead) | Medium |

### 2.2 Obsolete Scripts and Files

| File/Folder | Issue | Reason | Safe to remove? | Risk Level |
|-------------|-------|--------|-----------------|------------|
| `docs/archive/` | Obsolete Architecture Notes | Contains old Celery, Redis, and BioMistral documentation which causes confusion regarding the active project state. | Yes | Low |
| `run_api.py` vs `src/api/main.py` | Redundant Entry Points | `run_api.py` is a simple wrapper around `uvicorn`. It can be kept for convenience, but the architecture should standardise on either direct `uvicorn` commands or the wrapper. | Yes | Low |

### 2.3 Dependency & Configuration Issues

| File | Issue | Reason | Safe to remove? | Risk Level |
|------|-------|--------|-----------------|------------|
| `requirements_api.txt` / `requirements_chainlit.txt` | Split Dependencies | The project maintains two requirements files, but `run_app.py` and `run_api.py` rely on the same underlying `src/core` logic (ChromaDB, LangChain, etc.). This split is confusing. | No (Merge into a single `requirements.txt` instead) | Low |
| `.chainlit/config.toml` | UI Language Warnings | As noted in the patch file, the lack of Arabic translation files causes startup warnings unless explicitly forced to `en-US`. | No (Fix configuration rather than removing) | Low |

## 3. Summary
The codebase is exceptionally clean and well-structured (`src/core`, `src/api`, `src/services`, `src/ui`). The primary cleanup tasks involve removing the external `archive/` documentation and pruning a single unused wrapper function in the risk engine. The architecture is sound and ready for defense.
