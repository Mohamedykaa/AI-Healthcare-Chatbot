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
    "heart attack",
    "stroke",
    "severe bleeding",
    "loss of consciousness",
    "unconscious",
    "seizure",
    "severe head injury",
    "poisoning",
    "overdose",
    "suicidal",
    "suicide",
    "self harm",
    "severe allergic reaction",
    "anaphylaxis",
    "choking",
    "heatstroke",
    "sunstroke",
    "نوبة قلبية",
    "سكتة دماغية",
    "نزيف شديد",
    "فقدان الوعي",
    "غيبوبة",
    "تشنج",
    "تشنجات",
    "إصابة شديدة في الرأس",
    "تسمم",
    "جرعة زائدة",
    "انتحار",
    "أفكار انتحارية",
    "افكار انتحارية",
    "بأفكار انتحارية",
    "بافكار انتحارية",
    "إيذاء النفس",
    "حساسية شديدة",
    "اختناق",
    "ضربة شمس",
]

ASSOCIATED_RED_FLAGS = [
    "shortness of breath",
    "cold sweating",
    "ضيق تنفس",
    "تعرق بارد",
]
CORE_SYMPTOM_KEYWORDS = [
    "chest pain",
    "cannot breathe",
    "cant breathe",
    "can t breathe",
    "difficulty breathing",
    "fainting",
    "ألم في الصدر",
    "لا أستطيع التنفس",
    "صعوبة في التنفس",
    "إغماء",
]
SEVERE_MODIFIERS = [
    "severe",
    "crushing",
    "worst",
    "sudden",
    "شديد",
    "حاد",
    "مفاجئ",
]
LOW_RISK_MODIFIERS = [
    "mild",
    "localized",
    "only when pressing",
    "brief",
    "خفيف",
    "موضعي",
    "عند الضغط فقط",
]

EMERGENCY_SCORE_THRESHOLD = 6
URGENT_SCORE_THRESHOLD = 3

# ============================================================
# PURE FUNCTIONS (no I/O, no heavy deps)
# ============================================================


def normalize_input(text: str) -> str:
    lowered = (text or "").lower().strip()
    lowered = re.sub(r"[^\w\s]", " ", lowered)
    return re.sub(r"\s+", " ", lowered)


def contains_phrase(text: str, phrase: str) -> bool:
    escaped = re.escape(phrase)
    return re.search(rf"(?<!\w){escaped}(?!\w)", text) is not None


# Negation patterns: common English and Arabic negation words/phrases.
# A symptom mention preceded by these should NOT count as a positive match.
_NEGATION_RE = re.compile(
    r"(?:"
    # English negation words (word boundary enforced by \b)
    # NOTE: "can t" (from "can't") is intentionally excluded — "can't breathe"
    # means the person IS having difficulty breathing, not a negation.
    r"\b(?:no|not|never|without|don t|haven t|hasn t|didn t|wasn t|weren t|isn t|aren t|nor)\s+"
    # Arabic negation (must be preceded by start-of-string or whitespace)
    r"|(?:^|\s)(?:لا|ما|بدون|ليس|مش|مفيش|ماعنديش)\s+"
    r")"
    # Consume up to 5 words after the negation
    r"(?:\w+[\s,]*){1,5}",
    re.IGNORECASE,
)


def _strip_negated_phrases(normalized_text: str) -> str:
    """Remove symptom phrases that are preceded by negation words.

    Scans for negation + up to 5 following words and removes the entire
    negated span. This is intentionally conservative: it removes the
    negated phrase so it can't match, but any non-negated mentions of
    the same symptom elsewhere in the text will still score.
    """
    return _NEGATION_RE.sub(" ", normalized_text).strip()


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
    # Strip negated phrases so "no fainting" / "لا إغماء" don't score
    scoring_text = _strip_negated_phrases(normalized)
    used_phrases: Set[str] = set()
    score = 0

    flags = _match_category_phrases(scoring_text, ASSOCIATED_RED_FLAGS, used_phrases)
    used_phrases.update(flags)
    score += 3 * len(flags)

    core = _match_category_phrases(scoring_text, CORE_SYMPTOM_KEYWORDS, used_phrases)
    used_phrases.update(core)
    score += 3 * len(core)

    severe = _match_category_phrases(scoring_text, SEVERE_MODIFIERS, used_phrases)
    used_phrases.update(severe)
    score += 2 * len(severe)

    low = _match_category_phrases(scoring_text, LOW_RISK_MODIFIERS, used_phrases)
    used_phrases.update(low)
    score -= 2 * len(low)
    return score


def assess_risk_level(user_input: str) -> str:
    normalized = normalize_input(user_input)
    # Strip negated phrases for critical keyword check too
    scoring_text = _strip_negated_phrases(normalized)
    for keyword in CRITICAL_KEYWORDS:
        if contains_phrase(scoring_text, keyword):
            return "EMERGENCY"

    score = calculate_risk_score(user_input)
    if score >= EMERGENCY_SCORE_THRESHOLD:
        return "EMERGENCY"
    if score >= URGENT_SCORE_THRESHOLD:
        return "URGENT"
    return "ROUTINE"


def check_for_emergency(user_input: str) -> bool:
    return assess_risk_level(user_input) == "EMERGENCY"


def _contains_arabic(text: str) -> bool:
    return bool(re.search(r"[\u0600-\u06FF]", text or ""))


def get_emergency_response(user_input: str = "") -> str:
    if _contains_arabic(user_input):
        return """🚨 **تنبيه طارئ** 🚨

بناءً على وصفك، قد تكون هذه حالة طبية طارئة تحتاج إلى عناية فورية.

**مهم:**
- لا أستطيع تقديم تشخيص في حالات الطوارئ.
- يرجى طلب المساعدة الطبية فورًا.

**الإجراءات الموصى بها:**
1. **اتصل بالإسعاف أو برقم الطوارئ المحلي** فورًا
2. **توجه إلى أقرب قسم طوارئ** حالًا
3. **لا تؤخر طلب المساعدة** لأن الوقت قد يكون مهمًا

إذا كنت مع شخص يعاني من هذه الأعراض:
- حافظ على هدوئك وأبقِه مرتاحًا
- لا تعطه طعامًا أو شرابًا إلا إذا طلب الطاقم الطبي ذلك
- كن مستعدًا لشرح الأعراض للمسعفين

**تذكير:** أنا نظام ذكاء اصطناعي ولا أستطيع أن أحل محل الرعاية الطبية الطارئة.

---
*هذه استجابة أمان تلقائية. يرجى طلب رعاية طبية فورية.*
"""

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
