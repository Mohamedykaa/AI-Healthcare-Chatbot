"""Project integrity checks for the active Chainlit-based chatbot architecture."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _print_header(title: str) -> None:
    print(f"\n=== {title} ===")


def check_python_environment() -> bool:
    """Validate that the interpreter version is compatible."""
    _print_header("Python Environment")
    version = sys.version_info
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")

    if version < (3, 10):
        print("FAIL: Python 3.10+ is required.")
        return False

    print("PASS: Python version is compatible.")
    return True


def check_required_files() -> bool:
    """Ensure critical files for the active architecture exist."""
    _print_header("Required Files")
    required = [
        "app.py",
        "requirements_chainlit.txt",
        "README.md",
        "pytest.ini",
        "scripts/ingest_data.py",
        "tests/test_conversation_logic.py",
        "data/medical_knowledge_medquad.txt",
        "data/medical_knowledge_medmcqa.txt",
        "data/medical_knowledge_public_health.txt",
    ]

    all_found = True
    for rel in required:
        path = PROJECT_ROOT / rel
        if path.exists():
            print(f"PASS: Found {rel}")
        else:
            all_found = False
            print(f"FAIL: Missing {rel}")

    return all_found


def check_imports() -> bool:
    """Verify that core runtime/test dependencies are importable.

    Missing imports are reported as warnings because this script may run in
    lightweight CI/dev environments where only test dependencies are installed.
    """
    _print_header("Dependency Imports")
    modules = [
        "chainlit",
        "langchain",
        "langchain_community",
        "chromadb",
        "numpy",
        "pytest",
    ]

    missing = []
    for module in modules:
        if importlib.util.find_spec(module) is not None:
            print(f"PASS: import {module}")
        else:
            missing.append(module)
            print(f"WARN: cannot import {module}")

    if missing:
        print("WARN: Some runtime packages are not installed in this environment.")
        print("      Install requirements_chainlit.txt for full runtime validation.")

    return True


def check_pytest_suite() -> bool:
    """Run the unit test suite as part of integrity checks."""
    _print_header("Test Suite")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode == 0:
        print("PASS: test suite completed successfully")
        print(result.stdout.strip())
        return True
    
    print("FAIL: test suite failed")
    print(result.stdout.strip())
    print(result.stderr.strip())
    return False


def main() -> None:
    checks = [
        check_python_environment(),
        check_required_files(),
        check_imports(),
        check_pytest_suite(),
    ]

    if not all(checks):
        _print_header("Summary")
        print(f"Integrity check completed with failures.")
        sys.exit(1)

    print("\n=== Summary ===")
    print("Integrity check passed. Project looks good.")


if __name__ == "__main__":
    main()
