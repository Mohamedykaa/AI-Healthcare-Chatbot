"""
Pure risk-assessment logic for emergency detection.
Dependency-free (stdlib only) so core and tests can share one source of truth.
"""

import re
from typing import List, Set

# ============================================================
# CONFIGURATION (single source of truth)
# ============================================================

CRITICAL_KEYWORDS = [
    "heart attack", "stroke", "severe bleeding", "loss of consciousness",
    "unconscious", "seizure", "severe head injury", "poisoning", "overdose",
    "suicidal", "suicide", "self harm", "severe allergic reaction",
    "anaphylaxis", "choking", "heatstroke", "sunstroke",
]

ASSOCIATED_RED_FLAGS = ["shortness of breath", "cold sweating"]
CORE_SYMPTOM_KEYWORDS = ["chest pain", "cannot breathe", "cant breathe", "difficulty breathing", "fainting"]
SEVERE_MODIFIERS = ["severe", "crushing", "worst", "sudden"]
LOW_RISK_MODIFIERS = ["mild", "localized", "only when pressing", "brief"]

EMERGENCY_SCORE_THRESHOLD = 6
URGENT_SCORE_THRESHOLD = 3

# ============================================================
# PURE FUNCTIONS (no I/O, no heavy deps)
# ============================================================

def normalize_input(text: str) -> str:
    lowered = text.lower().strip()
    lowered = re.sub(r"[^\w\s]", " ", lowered)
    return re.sub(r"\s+", " ", lowered)


def contains_phrase(text: str, phrase: str) -> bool:
    escaped = re.escape(phrase)
    return re.search(rf"\b{escaped}\b", text) is not None


def _match_category_phrases(normalized_input: str, phrases: List[str], used_phrases: Set[str]) -> Set[str]:
    matched = set()
    for phrase in phrases:
        if phrase in used_phrases:
            continue
        if contains_phrase(normalized_input, phrase):
            matched.add(phrase)
    return matched


def calculate_risk_score(user_input: str) -> int:
    normalized = normalize_input(user_input)
    used_phrases: Set[str] = set()
    score = 0

    flags = _match_category_phrases(normalized, ASSOCIATED_RED_FLAGS, used_phrases)
    used_phrases.update(flags)
    score += 3 * len(flags)

    core = _match_category_phrases(normalized, CORE_SYMPTOM_KEYWORDS, used_phrases)
    used_phrases.update(core)
    score += 3 * len(core)

    severe = _match_category_phrases(normalized, SEVERE_MODIFIERS, used_phrases)
    used_phrases.update(severe)
    score += 2 * len(severe)

    low = _match_category_phrases(normalized, LOW_RISK_MODIFIERS, used_phrases)
    used_phrases.update(low)
    score -= 2 * len(low)
    return score


def assess_risk_level(user_input: str) -> str:
    normalized = normalize_input(user_input)
    for keyword in CRITICAL_KEYWORDS:
        if contains_phrase(normalized, keyword):
            return "EMERGENCY"

    score = calculate_risk_score(user_input)
    if score >= EMERGENCY_SCORE_THRESHOLD:
        return "EMERGENCY"
    if score >= URGENT_SCORE_THRESHOLD:
        return "URGENT"
    return "ROUTINE"


def check_for_emergency(user_input: str) -> bool:
    return assess_risk_level(user_input) == "EMERGENCY"


def get_emergency_response() -> str:
    return """🚨 **EMERGENCY ALERT** 🚨

Based on your description, this may be a medical emergency requiring immediate attention.

**IMPORTANT:**
- I cannot provide a diagnosis for emergency symptoms.
- Please seek immediate medical help.

**Recommended Actions:**
1. **Call Emergency Services** (911, 999, or your local emergency number)
2. **Go to the nearest Emergency Room** immediately
3. **Do not delay** - time is critical in medical emergencies

If you're with someone experiencing these symptoms:
- Stay calm and keep them comfortable
- Do not give them food or water unless instructed by medical personnel
- Be ready to describe the symptoms to emergency responders

**Remember:** I am an AI and cannot replace emergency medical care. Your safety is the priority.

---
*This is an automated safety response. Please seek professional medical attention immediately.*
"""
