# D:\disease_prediction_project\check_project_integrity.py
import os
import sys
import json # To check the knowledge base JSON file
import joblib
import importlib.util
from pathlib import Path
from colorama import Fore, Style, init
import subprocess
import pandas as pd

# Initialize colorama
init(autoreset=True)

# Project root directory (assuming this script is in the root)
PROJECT_ROOT = Path(__file__).resolve().parent

def check_env():
    """Check if a virtual environment seems to be active."""
    print(f"\n{Fore.CYAN}🔹 Checking Virtual Environment...")
    python_path = sys.executable
    # Simplified check for common venv patterns in the path
    venv_indicators = ["venv", ".venv", "env", ".env"]
    is_venv = any(indicator in python_path for indicator in venv_indicators)

    if is_venv:
        print(f"{Fore.GREEN}✅ Virtual environment seems active: {python_path}")
    else:
        print(f"{Fore.YELLOW}⚠️ Warning: Not using a detected virtual environment. Current Python: {python_path}")
        print(f"{Fore.YELLOW}   Using a virtual environment (venv) is highly recommended.")

def check_requirements():
    """Check for requirements.txt and run 'pip check'."""
    print(f"\n{Fore.CYAN}🔹 Checking Requirements...")
    req_file = PROJECT_ROOT / "requirements.txt"
    if not req_file.exists():
        print(f"{Fore.RED}❌ requirements.txt not found in project root '{PROJECT_ROOT}'.")
        return
    else:
        print(f"{Fore.GREEN}✅ requirements.txt found.")

    try:
        print(f"{Fore.CYAN}   Running 'pip check' to verify installed dependencies...")
        # Use the current Python executable to run pip
        result = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True, encoding='utf-8', errors='ignore')
        if result.returncode == 0:
            print(f"{Fore.GREEN}✅ All installed requirements seem compatible.")
        else:
            # pip check can sometimes show warnings even if things work, so make it yellow
            print(f"{Fore.YELLOW}⚠️ 'pip check' reported potential inconsistencies (or warnings):")
            # Print stderr first if it exists, otherwise stdout
            output = result.stderr if result.stderr else result.stdout
            print(output)
    except Exception as e:
        print(f"{Fore.RED}❌ Error running 'pip check':", e)

def check_data_files():
    """Check for essential source and generated data files."""
    print(f"\n{Fore.CYAN}🔹 Checking Data Files...")
    data_path = PROJECT_ROOT / "data"

    # Essential source files needed by utility scripts
    source_files = [
        "DiseaseAndSymptoms.csv",     # Needed by auto_followup_generator.py (wide format expected now)
        "Symptom-severity.csv",       # Needed by generator & merger
        "symptom_Description.csv",    # Needed by generator & merger
        "symptom_precaution.csv",     # Needed by data_merger.py
        "dataset.csv"                 # Needed by data_merger.py (Kaggle dataset)
    ]

    # Important generated files
    output_files = [
        "merged_comprehensive_data.csv", # Output of data_merger.py (for training)
        "english_knowledge_base.json"    # Output of auto_followup_generator.py (for API)
    ]

    print(f"{Fore.CYAN}   Checking source files in '{data_path}':")
    all_source_found = True
    for file in source_files:
        file_path = data_path / file
        if file_path.exists():
            print(f"{Fore.GREEN}    ✅ Found: {file}")
        else:
            print(f"{Fore.RED}    ❌ Missing: {file} (Required by data processing scripts)")
            all_source_found = False

    if not all_source_found:
         print(f"{Fore.YELLOW}   ⚠️ Some source data files are missing. Data processing scripts might fail.")


    print(f"\n{Fore.CYAN}   Checking important generated files in '{data_path}':")
    merged_data_ok = False
    kb_ok = False

    # Check the merged data file for training
    merged_path = data_path / "merged_comprehensive_data.csv"
    if merged_path.exists():
        try:
            # Quickly check header and essential columns without loading everything
            df = pd.read_csv(merged_path, nrows=5)
            expected_cols = {"disease", "training_text"}
            if expected_cols.issubset(df.columns):
                 print(f"{Fore.GREEN}    ✅ {merged_path.name}: Exists and has required columns ('disease', 'training_text').")
                 merged_data_ok = True
            else:
                 print(f"{Fore.RED}    ❌ {merged_path.name}: Exists but missing required columns. Found: {list(df.columns)}")
        except Exception as e:
            print(f"{Fore.RED}    ❌ Error reading {merged_path.name}: {e}")
    else:
        print(f"{Fore.RED}    ❌ {merged_path.name}: Not found. Run 'python src/utils/data_merger.py' to generate it.")


    # Check the knowledge base file for the API
    kb_path = data_path / "english_knowledge_base.json"
    if kb_path.exists():
        try:
            # Check if it's valid JSON and contains the 'rules' key
            with open(kb_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "rules" in data and isinstance(data["rules"], list):
                 print(f"{Fore.GREEN}    ✅ {kb_path.name}: Exists and seems structurally valid (contains 'rules' list).")
                 kb_ok = True
            else:
                 print(f"{Fore.RED}    ❌ {kb_path.name}: Exists but structure is invalid (missing 'rules' list?).")
        except json.JSONDecodeError:
            print(f"{Fore.RED}    ❌ {kb_path.name}: Exists but is not a valid JSON file.")
        except Exception as e:
            print(f"{Fore.RED}    ❌ Error reading {kb_path.name}: {e}")
    else:
         print(f"{Fore.RED}    ❌ {kb_path.name}: Not found. Run 'python src/utils/auto_followup_generator.py' to generate it.")

    # Final hint if generated files are missing
    if not merged_data_ok or not kb_ok:
        print(f"{Fore.YELLOW}   💡 Hint: You might need to run the data processing scripts in 'src/utils/' first.")


def check_models():
    """Check for the presence and basic validity of saved model files."""
    print(f"\n{Fore.CYAN}🔹 Checking Model Files...")
    model_path = PROJECT_ROOT / "models"
    files = ["optimized_nlp_pipeline.joblib", "nlp_label_encoder.joblib"]
    all_models_found = True
    for f in files:
        file_path = model_path / f
        # Check existence and if file size is reasonably large (e.g., > 1KB)
        if file_path.exists() and file_path.stat().st_size > 1024:
            print(f"{Fore.GREEN}✅ Found: {f} (Size: {file_path.stat().st_size / 1024:.1f} KB)")
            # Removed joblib.load here to avoid potential import/dependency issues during check
        elif file_path.exists():
             print(f"{Fore.YELLOW}⚠️ Warning: {f} exists but is very small (Size: {file_path.stat().st_size} bytes). It might be corrupted.")
             all_models_found = False
        else:
            print(f"{Fore.RED}❌ Missing: {f}")
            all_models_found = False

    if not all_models_found:
        print(f"{Fore.YELLOW}   💡 Hint: Run the training script 'python src/train_model.py' to generate these model files.")


def check_core_scripts():
    """Check if essential Python script files exist."""
    print(f"\n{Fore.CYAN}🔹 Checking Core Scripts...")
    # List essential scripts relative to the project root
    core_files = [
        Path("src") / "main.py",                # Main API server
        Path("src") / "app_streamlit.py",       # Streamlit UI
        Path("src") / "train_model.py",         # Model training script
        Path("src") / "utils" / "data_merger.py", # Data merging utility
        Path("src") / "utils" / "auto_followup_generator.py", # KB generation utility
        Path("src") / "utils" / "text_cleaner.py", # Text cleaning utility
        Path("chatbot_system") / "followup_manager.py", # Part of chatbot logic
        Path("chatbot_system") / "recommendation_agent.py" # Part of chatbot logic
    ]
    all_scripts_found = True
    for f_rel in core_files:
        f_abs = PROJECT_ROOT / f_rel
        if f_abs.exists():
            print(f"{Fore.GREEN}✅ Found: {f_rel}")
        else:
            print(f"{Fore.RED}❌ Missing essential script: {f_rel}")
            all_scripts_found = False
    return all_scripts_found

def check_python_imports():
    """Check if major required libraries can be imported."""
    print(f"\n{Fore.CYAN}🔹 Checking Python Library Imports...")
    # List major libraries used across the project
    # Make sure these align with requirements.txt
    modules = ["pandas", "joblib", "fastapi", "uvicorn", "streamlit", "sklearn", "nltk", "geopy", "requests", "colorama", "deep_translator"]
    all_imports_ok = True
    for m in modules:
        try:
            # Attempt to import the module
            importlib.import_module(m)
            print(f"{Fore.GREEN}✅ Can import: {m}")
        except ImportError:
            print(f"{Fore.RED}❌ Cannot import: {m} (Likely not installed)")
            all_imports_ok = False
        except Exception as e:
            # Catch other potential import issues
            print(f"{Fore.YELLOW}⚠️ Problem importing {m}: {e}")
            all_imports_ok = False

    if not all_imports_ok:
        print(f"{Fore.YELLOW}   💡 Hint: Ensure all packages from requirements.txt are installed:")
        print(f"{Fore.YELLOW}      pip install -r requirements.txt")
    return all_imports_ok

def main():
    """Main function to run all checks."""
    print(f"\n{Fore.MAGENTA}{'='*60}")
    print(f"{Fore.YELLOW}🩺 Project Integrity Check - AI Disease Prediction System 🩺")
    print(f"{Fore.MAGENTA}{'='*60}")

    # Refine project root detection slightly
    global PROJECT_ROOT
    script_path = Path(__file__).resolve()
    # Assume the script is run from the project root directory
    PROJECT_ROOT = Path.cwd()
    if script_path.parent != PROJECT_ROOT:
         # If the script is elsewhere (e.g., in a 'scripts' folder), adjust
         # For now, assume it's run from the root where requirements.txt is.
         print(f"{Fore.YELLOW}⚠️ Warning: Script seems to be run from outside the project root. Checks assume current directory '{PROJECT_ROOT}' is the root.")

    print(f"{Fore.BLUE}ℹ️ Project Root for checks: {PROJECT_ROOT}")


    # Run checks - can collect boolean results if needed later
    check_env()
    check_requirements()
    check_data_files()
    check_models()
    check_core_scripts()
    check_python_imports()

    print(f"\n{Fore.CYAN}{'='*60}")
    # You could add logic here based on the results if functions returned True/False
    print(f"{Fore.GREEN}✅ Project integrity check completed.")
    print(f"{Fore.CYAN}{'='*60}\n")

if __name__ == "__main__":
    # Add src and chatbot_system to sys.path temporarily FOR THIS SCRIPT ONLY
    # This might be needed if check_models tries to load joblib files that depend on custom classes
    # However, since we removed joblib.load from check_models, this is less critical now.
    # Leaving it commented out unless loading proves necessary again.
    # script_dir = Path(__file__).resolve().parent
    # src_path = script_dir / 'src'
    # chatbot_sys_path = script_dir / 'chatbot_system'
    # if src_path.is_dir() and str(src_path) not in sys.path:
    #     sys.path.insert(0, str(src_path))
    # if chatbot_sys_path.is_dir() and str(chatbot_sys_path) not in sys.path:
    #      sys.path.insert(0, str(chatbot_sys_path))

    main()