"""
AI Healthcare Chatbot — Application Launcher
=============================================
Safe startup with dependency validation using importlib.util.find_spec.
No module-level imports are triggered during validation.
"""

import importlib.util
import subprocess
import sys
import os


# ---------------------------------------------------------------------------
# Critical packages that MUST be importable before launching Chainlit.
# Format: (Python import name, pip package name)
# ---------------------------------------------------------------------------
_REQUIRED_PACKAGES = [
    ("chainlit", "chainlit"),
    ("langchain", "langchain"),
    ("langchain_core", "langchain-core"),
    ("langchain_community", "langchain-community"),
    ("langchain_google_genai", "langchain-google-genai"),
    ("chromadb", "chromadb"),
    ("sentence_transformers", "sentence-transformers"),
    ("dotenv", "python-dotenv"),
    ("tiktoken", "tiktoken"),
    ("pydantic", "pydantic"),
]


def check_and_install_dependencies() -> None:
    """Validate that all required packages are installed.

    Uses importlib.util.find_spec() which checks the package metadata
    WITHOUT executing any module-level code. This prevents the Chainlit
    CodeSettings / Pydantic crash that __import__("chainlit") triggers.
    """
    missing = []
    for module_name, package_name in _REQUIRED_PACKAGES:
        if importlib.util.find_spec(module_name) is None:
            missing.append(package_name)

    if not missing:
        return

    print(f"[run_app] Missing packages detected: {', '.join(missing)}")
    requirements_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "requirements_chainlit.txt"
    )

    if not os.path.exists(requirements_file):
        print(f"[run_app] ERROR: {requirements_file} not found. Cannot auto-install.")
        sys.exit(1)

    print("[run_app] Installing from requirements_chainlit.txt ...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", requirements_file],
        capture_output=False,
    )
    if result.returncode != 0:
        print("[run_app] ERROR: pip install failed. See output above.")
        sys.exit(1)

    # Re-check after install
    still_missing = [
        pkg for mod, pkg in _REQUIRED_PACKAGES
        if importlib.util.find_spec(mod) is None
    ]
    if still_missing:
        print(f"[run_app] FATAL: Still missing after install: {', '.join(still_missing)}")
        sys.exit(1)

    print("[run_app] All dependencies installed successfully.")


def main() -> None:
    # Ensure project root is on sys.path
    project_root = os.path.abspath(os.path.dirname(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Validate dependencies (safe — no side-effects)
    check_and_install_dependencies()

    # Launch Chainlit
    cmd = [sys.executable, "-m", "chainlit", "run", "src/ui/app.py"]
    print(f"[run_app] Launching: {' '.join(cmd)}")
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
