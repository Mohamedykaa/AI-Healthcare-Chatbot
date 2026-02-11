#!/usr/bin/env python3
"""
Medical Data Ingestion Pipeline
================================

Production-grade, safe, and reproducible medical data ingestion for RAG pipelines.
Downloads and processes medical datasets from Hugging Face with strict safety filtering.

Author: AI Healthcare Assistant Team
"""

import os
import re
import shutil
from typing import Optional

from datasets import load_dataset
from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

# Resolve project root (one level up from scripts/)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OUTPUT_FILES = [
    os.path.join(ROOT_DIR, "data", "medical_knowledge_medquad.txt"),
    os.path.join(ROOT_DIR, "data", "medical_knowledge_medmcqa.txt"),
    os.path.join(ROOT_DIR, "data", "medical_knowledge_public_health.txt"),
]

VECTOR_DB_DIRS = [
    os.path.join(ROOT_DIR, "chroma_db"),
    os.path.join(ROOT_DIR, "chroma_db_medical"),
]

# Forbidden keywords for safety filtering (case-insensitive)
FORBIDDEN_KEYWORDS = [
    "dosage", "dose", "mg", "ml",
    "tablet", "capsule", "injection",
    "prescribe", "prescription",
    "treatment plan", "therapy", "surgery",
    "emergency",
    "diagnose", "diagnosis",
    "medication", "drug",
]

# Maximum entries per dataset (to limit embedding time)
# Set to None for unlimited
MAX_ENTRIES_PER_DATASET = 2000


# ============================================================
# HARD CLEANUP & RESET
# ============================================================

def force_cleanup() -> None:
    """
    Perform a full hard reset to prevent data contamination.
    Deletes all existing output files and vector database directories.
    """
    print("=" * 60)
    print("🧹 PHASE 1: HARD CLEANUP & RESET")
    print("=" * 60)
    
    # Delete output text files
    print("\n📄 Checking for existing output files...")
    for filepath in OUTPUT_FILES:
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"   ✅ Deleted: {filepath}")
        else:
            print(f"   ⏭️  Not found (skip): {filepath}")
    
    # Delete vector database directories
    print("\n📁 Checking for existing vector database directories...")
    for dirpath in VECTOR_DB_DIRS:
        # Handle both relative and absolute paths
        abs_path = os.path.abspath(dirpath)
        if os.path.exists(abs_path):
            shutil.rmtree(abs_path)
            print(f"   ✅ Deleted (recursive): {abs_path}")
        else:
            print(f"   ⏭️  Not found (skip): {dirpath}")
    
    print("\n✅ Hard cleanup complete. Starting fresh build.\n")


# ============================================================
# SAFETY FILTER
# ============================================================

def contains_forbidden_content(text: Optional[str]) -> bool:
    """
    Check if text contains any forbidden keywords.
    Returns True if forbidden content is found.
    """
    if not text:
        return False
    
    text_lower = text.lower()
    for keyword in FORBIDDEN_KEYWORDS:
        if keyword in text_lower:
            return True
    return False


def is_safe_entry(*fields: Optional[str]) -> bool:
    """
    Check if all fields are safe (contain no forbidden content).
    Returns True if the entry is safe to include.
    """
    for field in fields:
        if contains_forbidden_content(field):
            return False
    return True


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text: Optional[str]) -> str:
    """
    Normalize text for clean output.
    - Strip extra whitespace
    - Normalize newlines
    - Remove HTML tags
    - Remove markdown artifacts
    - Ensure clean paragraphs
    """
    if not text:
        return ""
    
    # Convert to string if needed
    text = str(text)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove markdown artifacts (links, images, etc.)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)  # [text](url) -> text
    text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', '', text)    # ![alt](url) -> ""
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)        # **bold** -> bold
    text = re.sub(r'\*([^*]+)\*', r'\1', text)            # *italic* -> italic
    text = re.sub(r'`([^`]+)`', r'\1', text)              # `code` -> code
    text = re.sub(r'#{1,6}\s*', '', text)                 # # headers -> ""
    
    # Normalize whitespace
    text = re.sub(r'[ \t]+', ' ', text)           # Multiple spaces -> single space
    text = re.sub(r'\n\s*\n', '\n\n', text)       # Multiple newlines -> double newline
    text = re.sub(r'^\s+', '', text, flags=re.MULTILINE)  # Leading whitespace per line
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


# ============================================================
# DATASET PROCESSORS
# ============================================================

def process_medquad() -> int:
    """
    Process MedQuad dataset (General Medical Q&A).
    Returns the number of entries written.
    """
    print("=" * 60)
    print("📥 PHASE 2A: Processing MedQuad Dataset")
    print("=" * 60)
    print("   Dataset: keivalya/MedQuad-MedicalQnADataset")
    print("   Purpose: General medical education, patient-friendly explanations\n")
    
    try:
        dataset = load_dataset("keivalya/MedQuad-MedicalQnADataset", split="train")
    except Exception as e:
        print(f"   ❌ Failed to load MedQuad dataset: {e}")
        return 0
    
    output_file = os.path.join(ROOT_DIR, "data", "medical_knowledge_medquad.txt")
    count = 0
    filtered = 0
    
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in tqdm(dataset, desc="   Processing MedQuad"):
            question = entry.get("Question", "") or entry.get("question", "")
            answer = entry.get("Answer", "") or entry.get("answer", "")
            
            # Normalize text
            question = normalize_text(question)
            answer = normalize_text(answer)
            
            # Skip empty entries
            if not question or not answer:
                continue
            
            # Safety filter
            if not is_safe_entry(question, answer):
                filtered += 1
                continue
            
            # Write formatted entry
            f.write(f"Question: {question}\n")
            f.write(f"Answer: {answer}\n")
            f.write("\n---\n\n")
            count += 1
            
            # Check limit
            if MAX_ENTRIES_PER_DATASET and count >= MAX_ENTRIES_PER_DATASET:
                print(f"\n   ⚠️ Reached limit of {MAX_ENTRIES_PER_DATASET} entries")
                break
    
    print(f"\n   ✅ MedQuad processing complete")
    print(f"   📊 Entries written: {count}")
    print(f"   🚫 Entries filtered (safety): {filtered}")
    print(f"   📄 Output: {output_file}\n")
    
    return count


def process_medmcqa() -> int:
    """
    Process MedMCQA dataset (Medical Concepts).
    Converts multiple-choice format to educational explanations.
    Returns the number of entries written.
    """
    print("=" * 60)
    print("📥 PHASE 2B: Processing MedMCQA Dataset")
    print("=" * 60)
    print("   Dataset: openlifescienceai/medmcqa")
    print("   Purpose: Medical concepts, educational explanations\n")
    
    try:
        dataset = load_dataset("openlifescienceai/medmcqa", split="train")
    except Exception as e:
        print(f"   ❌ Failed to load MedMCQA dataset: {e}")
        return 0
    
    output_file = os.path.join(ROOT_DIR, "data", "medical_knowledge_medmcqa.txt")
    count = 0
    filtered = 0
    invalid = 0
    
    def get_correct_option(entry) -> Optional[str]:
        """
        Extract the correct option text based on cop field.
        Handles various formats: int (0-3 or 1-4), string ("a"-"d").
        """
        cop = entry.get("cop")
        if cop is None:
            return None
        
        # Map option keys
        option_keys = ["opa", "opb", "opc", "opd"]
        
        # Handle string format ("a", "b", "c", "d")
        if isinstance(cop, str):
            cop_lower = cop.lower().strip()
            if cop_lower in ["a", "0"]:
                return entry.get("opa")
            elif cop_lower in ["b", "1"]:
                return entry.get("opb")
            elif cop_lower in ["c", "2"]:
                return entry.get("opc")
            elif cop_lower in ["d", "3"]:
                return entry.get("opd")
            else:
                # Try to parse as int
                try:
                    cop = int(cop_lower)
                except ValueError:
                    return None
        
        # Handle integer format (0-3 or 1-4)
        if isinstance(cop, int):
            if 0 <= cop <= 3:
                return entry.get(option_keys[cop])
            elif 1 <= cop <= 4:
                return entry.get(option_keys[cop - 1])
        
        return None
    
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in tqdm(dataset, desc="   Processing MedMCQA"):
            question = entry.get("question", "")
            correct_option = get_correct_option(entry)
            
            # Skip if we couldn't determine correct option
            if not correct_option:
                invalid += 1
                continue
            
            # Normalize text
            question = normalize_text(question)
            explanation = normalize_text(correct_option)
            
            # Skip empty entries
            if not question or not explanation:
                continue
            
            # Safety filter (check question, all options, and explanation)
            all_options = [
                entry.get("opa", ""),
                entry.get("opb", ""),
                entry.get("opc", ""),
                entry.get("opd", ""),
            ]
            if not is_safe_entry(question, explanation, *all_options):
                filtered += 1
                continue
            
            # Write formatted entry
            f.write(f"Question: {question}\n")
            f.write(f"Educational Explanation: {explanation}\n")
            f.write("\n---\n\n")
            count += 1
            
            # Check limit
            if MAX_ENTRIES_PER_DATASET and count >= MAX_ENTRIES_PER_DATASET:
                print(f"\n   ⚠️ Reached limit of {MAX_ENTRIES_PER_DATASET} entries")
                break
    
    print(f"\n   ✅ MedMCQA processing complete")
    print(f"   📊 Entries written: {count}")
    print(f"   🚫 Entries filtered (safety): {filtered}")
    print(f"   ⚠️  Entries invalid (no correct option): {invalid}")
    print(f"   📄 Output: {output_file}\n")
    
    return count


def process_public_health() -> int:
    """
    Process Medical Meadow WikiDoc dataset (Public Health Educational Text).
    Returns the number of entries written.
    """
    print("=" * 60)
    print("📥 PHASE 2C: Processing Public Health Dataset")
    print("=" * 60)
    print("   Dataset: medalpaca/medical_meadow_wikidoc")
    print("   Purpose: General medical education, public health explanations\n")
    
    try:
        dataset = load_dataset("medalpaca/medical_meadow_wikidoc", split="train")
    except Exception as e:
        print(f"   ❌ Failed to load Public Health dataset: {e}")
        return 0
    
    output_file = os.path.join(ROOT_DIR, "data", "medical_knowledge_public_health.txt")
    count = 0
    filtered = 0
    
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in tqdm(dataset, desc="   Processing Public Health"):
            # Handle different field names
            topic = entry.get("input", "") or entry.get("instruction", "")
            description = entry.get("output", "") or entry.get("response", "")
            
            # Normalize text
            topic = normalize_text(topic)
            description = normalize_text(description)
            
            # Skip empty entries
            if not topic or not description:
                continue
            
            # Skip very short entries (likely not useful)
            if len(description) < 50:
                continue
            
            # Safety filter
            if not is_safe_entry(topic, description):
                filtered += 1
                continue
            
            # Write formatted entry
            f.write(f"Topic: {topic}\n")
            f.write(f"Description: {description}\n")
            f.write("\n---\n\n")
            count += 1
            
            # Check limit
            if MAX_ENTRIES_PER_DATASET and count >= MAX_ENTRIES_PER_DATASET:
                print(f"\n   ⚠️ Reached limit of {MAX_ENTRIES_PER_DATASET} entries")
                break
    
    print(f"\n   ✅ Public Health processing complete")
    print(f"   📊 Entries written: {count}")
    print(f"   🚫 Entries filtered (safety): {filtered}")
    print(f"   📄 Output: {output_file}\n")
    
    return count


# ============================================================
# SUMMARY & VALIDATION
# ============================================================

def print_summary(counts: dict) -> None:
    """
    Print final summary of the ingestion pipeline.
    """
    print("=" * 60)
    print("📊 PHASE 3: INGESTION SUMMARY")
    print("=" * 60)
    
    total = sum(counts.values())
    
    print(f"\n   MedQuad entries:        {counts.get('medquad', 0):,}")
    print(f"   MedMCQA entries:        {counts.get('medmcqa', 0):,}")
    print(f"   Public Health entries:  {counts.get('public_health', 0):,}")
    print(f"   ─────────────────────────────────")
    print(f"   TOTAL entries:          {total:,}")
    
    print("\n   📄 Output files generated:")
    for filepath in OUTPUT_FILES:
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            size_kb = size / 1024
            print(f"      ✅ {filepath} ({size_kb:.1f} KB)")
        else:
            print(f"      ❌ {filepath} (not created)")
    
    print("\n" + "=" * 60)
    print("✅ Medical data ingestion pipeline complete!")
    print("=" * 60)
    print("\n📌 IMPORTANT REMINDERS:")
    print("   • This data is for EDUCATIONAL purposes only.")
    print("   • It does NOT enable diagnosis, treatment, or medication guidance.")
    print("   • Always consult qualified healthcare professionals for medical advice.")
    print()


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def main() -> None:
    """
    Main entry point for the medical data ingestion pipeline.
    """
    print()
    print("╔════════════════════════════════════════════════════════════╗")
    print("║    MEDICAL DATA INGESTION PIPELINE                         ║")
    print("║    Production-Grade • Safe • Reproducible                  ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print()
    
    # Phase 1: Hard cleanup
    force_cleanup()
    
    # Phase 2: Process all datasets
    counts = {}
    
    counts["medquad"] = process_medquad()
    counts["medmcqa"] = process_medmcqa()
    counts["public_health"] = process_public_health()
    
    # Phase 3: Summary
    print_summary(counts)


if __name__ == "__main__":
    main()
