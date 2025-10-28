import re
import pandas as pd
from pathlib import Path
from typing import List, Set, Optional

# --- Import TextCleaner ---
try:
    from src.utils.text_cleaner import TextCleaner
except ImportError:
    print("⚠️ Warning: Could not import TextCleaner from src.utils. Falling back to basic cleaning.")
    class TextCleaner:
        def transform(self, text):
            text = str(text).lower().strip()
            text = re.sub(r"[^a-zA-Z\u0621-\u064A\s_]", "", text)
            text = re.sub(r"\s+", " ", text)
            return text


class SymptomAgent:
    """
    Extracts and manages symptoms mentioned by the user during a session,
    using a standardized symptom list and the project's TextCleaner.
    """

    def __init__(self, symptom_list_path: str = "data/Symptom-severity.csv", max_ngram: int = 3):
        self.text_cleaner = TextCleaner()
        self.symptom_set: Set[str] = self._load_and_clean_symptom_set(symptom_list_path)
        self.collected_symptoms_session: Set[str] = set()
        self.raw_user_input: str = ""
        self.max_ngram = max_ngram
        if not self.symptom_set:
            print(f"⚠️ SymptomAgent initialized with an EMPTY symptom list from {symptom_list_path}. Extraction will likely fail.")

    # -------------------------------------------------------------------------
    # ✅ Load and clean symptoms from CSV
    # -------------------------------------------------------------------------
    def _load_and_clean_symptom_set(self, path: str) -> Set[str]:
        try:
            script_dir = Path(__file__).resolve().parent
            project_root = script_dir.parents[1]
            full_path = project_root / path

            if not full_path.exists():
                print(f"❌ Error: Symptom list file not found at calculated path: {full_path}")
                return set()

            df = pd.read_csv(full_path)
            symptom_col = next((col for col in df.columns if col.strip().lower() == 'symptom'), None)
            if symptom_col is None:
                print(f"❌ Error: 'symptom' column not found in {full_path}.")
                return set()

            symptoms_series = df[symptom_col].dropna().astype(str)
            cleaned_symptoms = [self.text_cleaner.transform(s) for s in symptoms_series]
            return {s.strip() for s in cleaned_symptoms if s}

        except Exception as e:
            print(f"❌ Error loading/cleaning symptom list from {path}: {e}")
            return set()

    # -------------------------------------------------------------------------
    # ✅ Multi-language, multi-word extraction (Arabic + English)
    # -------------------------------------------------------------------------
    def _extract_symptoms(self, cleaned_text: str) -> List[str]:
        if not self.symptom_set:
            return []

        # Normalize Arabic and underscores
        cleaned_text = re.sub(r"[إأآا]", "ا", cleaned_text)
        cleaned_text = cleaned_text.replace("_", " ")

        extracted = set()

        # Regex-based extraction for both Arabic & English multi-word symptoms
        for symptom in self.symptom_set:
            normalized_symptom = symptom.replace("_", " ")
            pattern = rf"\b{re.escape(normalized_symptom)}\b"
            if re.search(pattern, cleaned_text):
                extracted.add(normalized_symptom)

        # Also check n-grams for flexible detection
        words = cleaned_text.split()
        num_words = len(words)
        for n in range(min(self.max_ngram, num_words), 0, -1):
            for i in range(num_words - n + 1):
                phrase = " ".join(words[i:i + n])
                if phrase in self.symptom_set or phrase.replace(" ", "_") in self.symptom_set:
                    extracted.add(phrase.replace("_", " "))

        return sorted(list(extracted))

    # -------------------------------------------------------------------------
    # ✅ Collect user symptoms
    # -------------------------------------------------------------------------
    def collect_symptoms(self, user_input: str) -> str:
        self.raw_user_input = str(user_input).strip()
        cleaned_text = self.text_cleaner.transform(self.raw_user_input)

        if not cleaned_text:
            return "لم أتمكن من فهم المدخلات. هل يمكنك إعادة الصياغة؟"

        new_symptoms_found = self._extract_symptoms(cleaned_text)
        added_now_count = 0

        for s in new_symptoms_found:
            if s not in self.collected_symptoms_session:
                self.collected_symptoms_session.add(s)
                added_now_count += 1

        if not new_symptoms_found:
            if self.collected_symptoms_session:
                return (
                    "لم أتعرف على أعراض **جديدة** في هذه الرسالة. الأعراض المسجلة حتى الآن: "
                    + ", ".join(sorted(list(self.collected_symptoms_session)))
                )
            else:
                return "لم أتعرف على أعراض محددة من هذا الوصف. هل يمكنك تقديم تفاصيل أكثر؟"
        else:
            feedback = f"الأعراض التي تم التعرف عليها الآن: {', '.join(new_symptoms_found)}."
            if added_now_count < len(new_symptoms_found):
                feedback += " (بعضها كان مسجلًا بالفعل)"
            return feedback

    # -------------------------------------------------------------------------
    # ✅ Get clean readable text for diagnosis
    # -------------------------------------------------------------------------
    def get_symptom_text(self) -> str:
        if not self.collected_symptoms_session:
            return ""
        # Return space-separated list (to pass test expectations)
        symptoms = [s.replace("_", " ") for s in sorted(self.collected_symptoms_session)]
        return " ".join(symptoms)

    # -------------------------------------------------------------------------
    def clear_session(self):
        self.collected_symptoms_session.clear()
        self.raw_user_input = ""
        print("ℹ️ SymptomAgent session cleared.")


# -------------------------------------------------------------------------
# ✅ Example Run
# -------------------------------------------------------------------------
if __name__ == '__main__':
    agent = SymptomAgent("data/Symptom-severity.csv")

    inputs = [
        "I have a terrible headache and back pain, maybe fever too",
        "also suffering from nausea",
        "no other problems, just headache",
        "عندي صداع شديد وألم ظهر قوي"
    ]
    for text in inputs:
        print(f"\nUser: {text}")
        print("Bot:", agent.collect_symptoms(text))
        print("Collected:", agent.get_symptom_text())

    agent.clear_session()
