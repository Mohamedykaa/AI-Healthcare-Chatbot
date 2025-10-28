from typing import List, Dict, Optional, Any
from collections import defaultdict
import json
import os
import glob
from datetime import datetime
import random

class FollowUpManager:
    """
    FollowUpManager (complete, robust version)

    Responsibilities:
    - Manage pending follow-up questions (with disease context)
    - Track user answers & timestamps
    - Compute and return disease-level boosts
    - Provide disease-scoped question retrieval and global prioritization
    - Import/export state (safe JSON), load follow-ups from KB files
    """

    def __init__(self, negative_boost_multiplier: float = -0.25):
        self.pending_questions: List[Dict[str, Any]] = []
        self.user_answers: Dict[str, Dict[str, Any]] = {}
        self.question_meta: Dict[str, Dict[str, Any]] = {}
        self.asked_question_ids = set()
        self.disease_boosts: Dict[str, float] = {}
        self._questions_by_disease: Dict[str, List[str]] = defaultdict(list)
        self._seq_counter = 0
        self.negative_boost_multiplier = float(negative_boost_multiplier)

    # ---------------- adding / queue management ----------------
    def add_questions(self, questions: List[Dict[str, Any]], reorder: bool = True, disease_scope: Optional[str] = None) -> int:
        """
        Add multiple questions to the queue (avoiding duplicates).
        Supports optional disease_scope parameter for backward compatibility.
        Returns number of questions actually added.
        """
        added = 0
        if not isinstance(questions, (list, tuple)):
            return 0

        scope_key = disease_scope.strip().lower() if isinstance(disease_scope, str) and disease_scope.strip() else None

        for q in questions:
            if not isinstance(q, dict):
                continue
            qid = q.get("id") or q.get("qid") or None
            if not qid:
                continue
            if qid in self.asked_question_ids:
                continue

            text = q.get("question") or q.get("text") or q.get("prompt") or ""
            try:
                severity = int(q.get("severity", 0) or 0)
            except Exception:
                severity = 0

            disease = scope_key or (q.get("disease") or q.get("disease_name") or "")
            disease_key = disease.strip().lower() if disease else None

            boosts = q.get("boosts", []) or []
            boost_total = 0.0
            for b in boosts:
                try:
                    boost_total += abs(float(b.get("value", 0.0)))
                except Exception:
                    pass

            created_at = q.get("created_at") or datetime.utcnow().isoformat()

            meta = {
                "text": text,
                "boosts": boosts,
                "severity": severity,
                "boost_total": boost_total,
                "disease": disease_key,
                "created_at": created_at
            }

            self.question_meta[qid] = meta

            item = {
                "id": qid,
                "text": text,
                "severity": severity,
                "boost_total": boost_total,
                "disease": disease_key,
                "seq": self._seq_counter,
                "created_at": created_at,
                "asked": False
            }
            self._seq_counter += 1

            self.pending_questions.append(item)
            self.asked_question_ids.add(qid)
            if disease_key:
                self._questions_by_disease[disease_key].append(qid)

            added += 1

        if reorder and added > 0:
            self._reorder_queue()

        return added

    def _reorder_queue(self):
        """
        Reorder pending_questions by (severity desc, boost_total desc, older seq first).
        """
        if not self.pending_questions:
            return
        items = list(self.pending_questions)
        items.sort(key=lambda x: (x.get("severity", 0), x.get("boost_total", 0.0), -x.get("seq", 0)), reverse=True)
        self.pending_questions = items

    # ---------------- peek/pop next question ----------------
    def peek_next_question(self) -> Optional[Dict[str, Any]]:
        """Return the highest-priority question object without removing it. None if empty."""
        if not self.pending_questions:
            return None
        self._reorder_queue()

        item = dict(self.pending_questions[0])
        if item and "question" not in item and "text" in item:
            item["question"] = item["text"]
        return item

    def get_next_question(self) -> Optional[Dict[str, Any]]:
        """
        Pop and return the next global pending question (dict with id,text,...).
        """
        if not self.pending_questions:
            return None
        self._reorder_queue()
        item = self.pending_questions.pop(0)
        item["asked"] = True

        if item and "question" not in item and "text" in item:
            item["question"] = item["text"]

        return item

    def has_pending_questions(self) -> bool:
        return len(self.pending_questions) > 0

    # ---------------- disease-scoped helpers ----------------
    def has_followup_questions(self, disease_name: str) -> bool:
        """Return True if there are pending questions mapped to this disease."""
        if not disease_name:
            return False
        key = disease_name.strip().lower()
        pending_ids = {q["id"] for q in self.pending_questions}
        return any(qid in pending_ids for qid in self._questions_by_disease.get(key, []))

    def get_next_question_for_disease(self, disease_name: str) -> Optional[Dict[str, Any]]:
        """
        Pop and return the next pending question object for the specified disease.
        """
        if not disease_name:
            return None
        key = disease_name.strip().lower()
        if not self.pending_questions:
            return None

        items = list(self.pending_questions)
        for idx, q in enumerate(items):
            if q.get("disease") == key and not q.get("asked"):
                item = items.pop(idx)
                self.pending_questions = items
                item["asked"] = True

                if item and "question" not in item and "text" in item:
                    item["question"] = item["text"]

                return item
        return None

    def get_next_question_for_active_disease(self, active_disease: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        If active_disease given, try disease-specific first, else global.
        """
        if active_disease:
            res = self.get_next_question_for_disease(active_disease)
            if res:
                return res
        return self.get_next_question()

    # ---------------- recording answers & boosts ----------------

    # ✅ =================== START OF FIX ===================
    # This function is updated to fix the "i don't know" bug
    # and remove dangerous substring-matching heuristics.
    def _normalize_answer(self, answer: Any) -> str:
        """Normalize various user responses into canonical categories."""
        if answer is None:
            return ""
        s = str(answer).strip().lower()

        # --- YES ---
        if s in ("y", "yes", "true", "1", "yep", "yeah", "نعم", "ايوه", "ايوا"):
            return "yes"

        # --- NO ---
        if s in ("n", "no", "false", "0", "لا", "لأ", "لاا"):
            return "no"

        # --- PARTIAL / UNKNOWN ---
        # ✅ FIX: Added "i don't know" and "مش عارف" to the main set
        if s in (
            "maybe", "not sure", "sometimes", "a bit", "partially", "partial",
            "i don't know", "لا اعرف", "مش عارف",
            "ربما", "ممكن", "قد"
        ):
            return "partial_yes"

        # ✅ FIX: Removed dangerous 'any(tok in s)' heuristics (like 'no' in 'know').
        # Any other unknown text is safer as "partial_yes" than a wrong "no".
        return "partial_yes"
    # ✅ =================== END OF FIX ===================

    def record_answer(self, question_id: str, answer: Any, timestamp: Optional[str] = None):
        """
        Record an answer (yes/no/partial_yes). Update disease_boosts accordingly.
        """
        if not question_id:
            return
        norm = self._normalize_answer(answer)
        ts = timestamp or datetime.utcnow().isoformat()

        self.user_answers[question_id] = {"answer": norm, "timestamp": ts}

        meta = self.question_meta.get(question_id)
        if not meta:
            return

        if norm == "yes":
            multiplier = 1.0
        elif norm == "partial_yes":
            multiplier = 0.5
        elif norm == "no":
            multiplier = self.negative_boost_multiplier
        else:
            # Default for safety (e.g., if _normalize_answer returned "")
            multiplier = 0.5

        for b in meta.get("boosts", []):
            dname = (b.get("name") or "").strip().lower()
            if not dname:
                continue
            try:
                val = float(b.get("value", 0.0)) * multiplier
            except Exception:
                val = 0.0
            self.disease_boosts[dname] = self.disease_boosts.get(dname, 0.0) + val

    def record_answer_by_question_id(self, question_id: str, answer: Any, timestamp: Optional[str] = None):
        """Alias for clarity/backwards compatibility."""
        return self.record_answer(question_id, answer, timestamp=timestamp)

    def get_all_answers(self) -> Dict[str, Any]:
        """Return mapping question_id -> {answer, timestamp}."""
        return dict(self.user_answers)

    def get_all_answers_simple(self) -> Dict[str, str]:
        """
        Return mapping question_id -> simple answer string.
        """
        simple = {}
        for qid, data in self.user_answers.items():
            if isinstance(data, dict):
                simple[qid] = data.get("answer", "")
            else:
                simple[qid] = str(data)
        return simple

    def get_disease_boosts(self) -> Dict[str, float]:
        """Return mapping disease_name_lower -> accumulated boost value from answers."""
        return dict(self.disease_boosts)

    def get_followup_score(self, disease_name: str, include_unasked: bool = False) -> float:
        """
        Compute a normalized follow-up confidence score for a disease.
        """
        if not disease_name:
            return 0.0
        key = disease_name.strip().lower()

        total_possible = 0.0
        obtained = 0.0

        if include_unasked:
            qids = list(self._questions_by_disease.get(key, []))
        else:
            qids = [qid for qid in self.user_answers.keys() if qid in self._questions_by_disease.get(key, [])]

        for qid in qids:
            meta = self.question_meta.get(qid)
            if not meta:
                continue
            boosts = meta.get("boosts", [])
            answer_record = self.user_answers.get(qid)
            answer = answer_record.get("answer") if isinstance(answer_record, dict) else None
            for b in boosts:
                if (b.get("name") or "").strip().lower() != key:
                    continue
                try:
                    boost_val = abs(float(b.get("value", 0.0)))
                except Exception:
                    boost_val = 0.0
                total_possible += boost_val
                if answer == "yes":
                    obtained += boost_val
                elif answer == "partial_yes":
                    obtained += boost_val * 0.5
                elif answer == "no":
                    obtained += boost_val * (self.negative_boost_multiplier if self.negative_boost_multiplier < 0 else 0.0)

        if total_possible <= 0:
            return 0.0
        score = obtained / total_possible
        return float(max(0.0, min(1.0, score)))

    # ---------------- persistence / import-export ----------------
    def export_state(self) -> Dict[str, Any]:
        """Export internal state as JSON-serializable dict (safe)."""
        return {
            "pending_questions": list(self.pending_questions),
            "user_answers": dict(self.user_answers),
            "question_meta": dict(self.question_meta),
            "asked_question_ids": list(self.asked_question_ids),
            "disease_boosts": dict(self.disease_boosts),
            "_questions_by_disease": {k: list(v) for k, v in self._questions_by_disease.items()},
            "_seq_counter": self._seq_counter,
            "negative_boost_multiplier": self.negative_boost_multiplier
        }

    def import_state(self, state: Dict[str, Any]):
        """
        Safe import: apply only known keys and perform validation.
        """
        try:
            if not isinstance(state, dict):
                return
            self.clear()

            pq = state.get("pending_questions", []) or []
            safe_pq = []
            for itm in pq:
                if not isinstance(itm, dict):
                    continue
                if "id" not in itm:
                    continue
                safe_pq.append(itm)
            self.pending_questions = safe_pq

            ua = state.get("user_answers", {}) or {}
            safe_ua = {}
            if isinstance(ua, dict):
                for qid, val in ua.items():
                    if isinstance(val, dict) and "answer" in val:
                        safe_ua[qid] = {"answer": str(val.get("answer")), "timestamp": val.get("timestamp")}
                    else:
                        safe_ua[qid] = {"answer": str(val), "timestamp": None}
            self.user_answers = safe_ua

            qm = state.get("question_meta", {}) or {}
            if isinstance(qm, dict):
                self.question_meta = {k: v for k, v in qm.items() if isinstance(v, dict)}

            asked = state.get("asked_question_ids", []) or []
            if isinstance(asked, (list, set)):
                self.asked_question_ids = set(x for x in asked if isinstance(x, str))

            db = state.get("disease_boosts", {}) or {}
            safe_db = {}
            if isinstance(db, dict):
                for k, v in db.items():
                    try:
                        safe_db[k] = float(v)
                    except Exception:
                        safe_db[k] = 0.0
            self.disease_boosts = safe_db

            qby = state.get("_questions_by_disease", {}) or {}
            if isinstance(qby, dict):
                self._questions_by_disease = defaultdict(list, {k: list(v) for k, v in qby.items()})

            seq = state.get("_seq_counter")
            if isinstance(seq, int):
                self._seq_counter = seq

            nbm = state.get("negative_boost_multiplier")
            if nbm is not None:
                try:
                    self.negative_boost_multiplier = float(nbm)
                except Exception:
                    pass

            self._reorder_queue()

        except Exception:
            self.clear()

    def to_json(self) -> str:
        """Return JSON string of exported state (safe)."""
        try:
            return json.dumps(self.export_state(), ensure_ascii=False)
        except Exception:
            return json.dumps({
                "pending_questions": [],
                "user_answers": {},
                "question_meta": {},
                "asked_question_ids": [],
                "disease_boosts": {},
                "_questions_by_disease": {},
                "_seq_counter": getattr(self, "_seq_counter", 0),
                "negative_boost_multiplier": getattr(self, "negative_boost_multiplier", -0.25)
            }, ensure_ascii=False)

    def from_json(self, json_str: Optional[str]):
        """Import state from JSON string safely."""
        if not json_str:
            return
        try:
            obj = json.loads(json_str)
            self.import_state(obj)
        except Exception:
            try:
                self.clear()
            except Exception:
                pass

    # ---------------- dynamic loading from knowledge base ----------------

    def generate_followup_questions_from_kb(self, knowledge_base: dict):
        """
        Auto-generate follow-up questions dynamically from the knowledge base.
        NOTE: knowledge_base is expected to be kb_lookup: {disease_name: [list_of_rules]}
        """
        question_templates = [
            "Are you experiencing {symptom}?",
            "Do you have {symptom}?",
            "Have you been suffering from {symptom} recently?",
            "Do you notice any {symptom}?",
            "Have you felt {symptom} in the past few days?"
        ]

        seen_symptoms = set()
        generated_questions = []

        for disease, rules_list in knowledge_base.items():

            all_disease_symptoms = set()
            for rule in rules_list:
                for symptom in rule.get("symptoms", []):
                    all_disease_symptoms.add(symptom)

            symptoms_list = list(all_disease_symptoms)
            random.shuffle(symptoms_list)

            symptoms_added_for_this_disease = 0
            for symptom in symptoms_list:
                if symptoms_added_for_this_disease >= 3:
                    break

                if symptom not in seen_symptoms:
                    seen_symptoms.add(symptom)

                    question_text = random.choice(question_templates).format(symptom=symptom)
                    q_id = f"{disease}_{symptom.replace(' ', '_')}"

                    generated_questions.append({
                        "id": q_id,
                        "disease": disease,
                        "question": question_text,
                        "text": question_text,
                        "symptom": symptom,
                        "severity": random.randint(1, 5)
                    })
                    symptoms_added_for_this_disease += 1

        if generated_questions:
            self.add_questions(generated_questions)
            print(f"✅ Auto-generated {len(generated_questions)} follow-up questions (filtered & varied).")
        else:
            print("⚠️ No follow-up questions generated (empty or invalid knowledge base).")


    def load_followups_from_kb(self, kb_path: str = "data/english_knowledge_base.json") -> Dict[str, Any]:
        """
        Load follow-up questions from a KB file or all JSON files in a directory.
        """
        files_to_load: List[str] = []
        if not kb_path:
            return {"status": "error", "message": "No kb_path provided."}

        if os.path.isdir(kb_path):
            files_to_load = glob.glob(os.path.join(kb_path, "*.json"))
            if not files_to_load:
                return {"status": "empty", "message": f"No JSON KB files found in directory: {kb_path}"}
        elif os.path.isfile(kb_path):
            files_to_load = [kb_path]
        else:
            return {"status": "error", "message": f"KB path not found: {kb_path}"}

        total_added = 0
        for fp in files_to_load:
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    kb = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                continue

            all_questions: List[Dict[str, Any]] = []
            for rule in kb.get("rules", []):
                disease_name = ""
                if "conditions" in rule and rule["conditions"]:
                    disease_name = rule["conditions"][0].get("name", "") or ""

                for q in rule.get("follow_ups", []):
                    question_text = q.get("question") or q.get("text") or ""
                    qid = q.get("id") or q.get("qid") or f"kb_{abs(hash(question_text))}"
                    severity = int(q.get("severity", 5) or 5)
                    q_obj = {
                        "id": qid,
                        "question": question_text,
                        "text": question_text,
                        "boosts": q.get("boosts", []),
                        "severity": severity,
                        "disease": disease_name or None,
                        "created_at": q.get("created_at")
                    }
                    all_questions.append(q_obj)

            if all_questions:
                added = self.add_questions(all_questions, reorder=False)
                total_added += added

        if total_added > 0:
            self._reorder_queue()
            return {"status": "ok", "count": total_added}
        return {"status": "empty", "message": "No follow-up questions found in provided KB(s)."}

    # ---------------- progress / diagnostics ----------------
    def summarize_progress(self) -> Dict[str, Any]:
        """
        Return a diagnostic summary.
        """
        total_pending = len(self.pending_questions)
        total_answered = len(self.user_answers)

        answers_by_type: Dict[str, int] = {}
        for v in self.user_answers.values():
            ans = v.get("answer") if isinstance(v, dict) else v
            answers_by_type[ans] = answers_by_type.get(ans, 0) + 1

        pending_per_disease: Dict[str, int] = {}
        for q in self.pending_questions:
            d = q.get("disease") or "unknown"
            pending_per_disease[d] = pending_per_disease.get(d, 0) + 1

        top = sorted(self.disease_boosts.items(), key=lambda x: x[1], reverse=True)[:10]
        top_diseases = [{"disease": d, "boost": b} for d, b in top]

        return {
            "total_pending": total_pending,
            "total_answered": total_answered,
            "answers_by_type": answers_by_type,
            "pending_per_disease": pending_per_disease,
            "top_diseases_by_boost": top_diseases
        }

    # ---------------- utilities ----------------
    def clear(self):
        """Reset manager to an empty safe state."""
        self.pending_questions.clear()
        self.user_answers.clear()
        self.question_meta.clear()
        self.asked_question_ids.clear()
        self.disease_boosts.clear()
        self._questions_by_disease.clear()
        self._seq_counter = 0
        self.negative_boost_multiplier = float(self.negative_boost_multiplier or -0.25)

    def summary(self) -> str:
        """Human-readable summary useful for debugging."""
        s = self.summarize_progress()
        return (
            f"Pending: {s['total_pending']}, Answered: {s['total_answered']}, "
            f"Top boosts: {s['top_diseases_by_boost']}"
        )