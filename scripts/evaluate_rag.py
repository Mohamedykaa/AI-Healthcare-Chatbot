#!/usr/bin/env python3
"""
RAG Evaluation Script
======================

Measures the quality of the RAG pipeline with two metrics:

1. **Retrieval Hit Rate** — Does the vector store return relevant documents
   for a known medical question?  (No LLM needed.)

2. **Answer Faithfulness** — Does the full RAG pipeline produce an answer
   that references the retrieved context rather than hallucinating?
   (Requires Ollama.)

Usage:
    python scripts/evaluate_rag.py              # full eval (retrieval + faithfulness)
    python scripts/evaluate_rag.py --retrieval   # retrieval only (no LLM needed)

Results are printed to stdout AND saved to ``evaluation_results.json``.
"""

import argparse
import asyncio
import json
import os
import sys
import time

# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.services.vectorstore import (
    get_embedding_function,
    load_or_create_vectorstore,
)
from src.services.llm import get_llm
from src.core.config import (
    RETRIEVER_K,
    RETRIEVER_SCORE_THRESHOLD,
    CONTEXT_CHAR_LIMIT_PER_DOC,
)
from src.core.risk import assess_risk_level

# ---------------------------------------------------------------------------
# GOLD TEST SET
# Each entry has:
#   question          — the user query
#   expected_keywords — keywords we expect to see in the *retrieved docs*
#   answer_keywords   — keywords we expect in a correct *LLM answer*
#   risk_level        — expected deterministic triage result
# ---------------------------------------------------------------------------

GOLD_QA_PAIRS = [
    {
        "question": "What are the symptoms of diabetes?",
        "expected_keywords": ["diabetes", "blood sugar", "insulin"],
        "answer_keywords": ["diabetes", "sugar", "insulin"],
        "risk_level": "ROUTINE",
    },
    {
        "question": "What is hypertension?",
        "expected_keywords": ["blood pressure", "hypertension"],
        "answer_keywords": ["blood pressure", "hypertension"],
        "risk_level": "ROUTINE",
    },
    {
        "question": "What causes asthma attacks?",
        "expected_keywords": ["asthma", "airway", "breathing"],
        "answer_keywords": ["asthma", "airway"],
        "risk_level": "ROUTINE",
    },
    {
        "question": "How does pneumonia spread?",
        "expected_keywords": ["pneumonia", "lung", "infection"],
        "answer_keywords": ["pneumonia", "infection"],
        "risk_level": "ROUTINE",
    },
    {
        "question": "What are the risk factors for heart disease?",
        "expected_keywords": ["heart", "cholesterol", "blood pressure"],
        "answer_keywords": ["heart", "risk"],
        "risk_level": "ROUTINE",
    },
    {
        "question": "What is anemia and what causes it?",
        "expected_keywords": ["anemia", "blood", "iron"],
        "answer_keywords": ["anemia", "iron"],
        "risk_level": "ROUTINE",
    },
    {
        "question": "How do you prevent the common cold?",
        "expected_keywords": ["cold", "virus", "hygiene"],
        "answer_keywords": ["cold", "wash", "virus"],
        "risk_level": "ROUTINE",
    },
    {
        "question": "What are the symptoms of a urinary tract infection?",
        "expected_keywords": ["urinary", "infection", "bladder"],
        "answer_keywords": ["urinary", "infection"],
        "risk_level": "ROUTINE",
    },
    {
        "question": "What is tuberculosis?",
        "expected_keywords": ["tuberculosis", "lung", "bacteria"],
        "answer_keywords": ["tuberculosis", "lung"],
        "risk_level": "ROUTINE",
    },
    {
        "question": "What are the early signs of kidney disease?",
        "expected_keywords": ["kidney", "renal"],
        "answer_keywords": ["kidney"],
        "risk_level": "ROUTINE",
    },
    {
        "question": "What causes migraine headaches?",
        "expected_keywords": ["migraine", "headache"],
        "answer_keywords": ["migraine", "headache"],
        "risk_level": "ROUTINE",
    },
    {
        "question": "How is malaria transmitted?",
        "expected_keywords": ["malaria", "mosquito"],
        "answer_keywords": ["malaria", "mosquito"],
        "risk_level": "ROUTINE",
    },
    # --- Triage-specific cases ---
    {
        "question": "I am having a heart attack right now",
        "expected_keywords": [],
        "answer_keywords": [],
        "risk_level": "EMERGENCY",
    },
    {
        "question": "I have severe chest pain and shortness of breath",
        "expected_keywords": [],
        "answer_keywords": [],
        "risk_level": "EMERGENCY",
    },
    {
        "question": "I have mild chest pain only when pressing on it",
        "expected_keywords": [],
        "answer_keywords": [],
        "risk_level": "ROUTINE",
    },
]


# ---------------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------------

def retrieval_hit_rate(vectorstore, embedding_function) -> dict:
    """
    For each gold pair, retrieve top-K documents and check whether
    *any* expected keyword appears in the combined retrieved text.
    """
    results = []
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": RETRIEVER_K, "score_threshold": RETRIEVER_SCORE_THRESHOLD},
    )

    # Only test pairs that have expected retrieval keywords
    test_pairs = [p for p in GOLD_QA_PAIRS if p["expected_keywords"]]

    for pair in test_pairs:
        docs = retriever.invoke(pair["question"])
        combined = " ".join(d.page_content.lower() for d in docs)
        hits = [kw for kw in pair["expected_keywords"] if kw.lower() in combined]
        hit = len(hits) > 0
        results.append({
            "question": pair["question"],
            "hit": hit,
            "matched_keywords": hits,
            "docs_returned": len(docs),
        })

    total = len(results)
    hits = sum(1 for r in results if r["hit"])
    rate = hits / total if total > 0 else 0.0

    return {"hit_rate": rate, "hits": hits, "total": total, "details": results}


def triage_accuracy() -> dict:
    """
    Verify that the deterministic risk triage matches expected labels.
    """
    results = []
    for pair in GOLD_QA_PAIRS:
        predicted = assess_risk_level(pair["question"])
        correct = predicted == pair["risk_level"]
        results.append({
            "question": pair["question"],
            "expected": pair["risk_level"],
            "predicted": predicted,
            "correct": correct,
        })

    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    accuracy = correct / total if total > 0 else 0.0

    return {"accuracy": accuracy, "correct": correct, "total": total, "details": results}


async def answer_faithfulness(vectorstore, llm) -> dict:
    """
    Run the full RAG pipeline for each gold pair and check:
      1. The answer contains at least one expected answer keyword.
      2. The answer is not empty / a fallback.
    """
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": RETRIEVER_K, "score_threshold": RETRIEVER_SCORE_THRESHOLD},
    )

    # Only test pairs that have answer keywords (skip emergency-only pairs)
    test_pairs = [p for p in GOLD_QA_PAIRS if p["answer_keywords"]]
    results = []

    for pair in test_pairs:
        docs = retriever.invoke(pair["question"])
        context_text = "\n".join(
            d.page_content[:CONTEXT_CHAR_LIMIT_PER_DOC] for d in docs[:RETRIEVER_K]
        )

        prompt = f"""You are an educational medical symptom checker.
You talk to users who describe how they feel.
You respond by explaining symptoms in simple medical terms.

Medical context: {context_text[:800]}

User: {pair['question']}"""

        try:
            response = await llm.ainvoke(prompt)
            answer = response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            answer = f"[ERROR: {e}]"

        answer_lower = answer.lower()
        keyword_matches = [
            kw for kw in pair["answer_keywords"] if kw.lower() in answer_lower
        ]
        faithful = len(keyword_matches) > 0 and len(answer.strip()) > 30

        results.append({
            "question": pair["question"],
            "faithful": faithful,
            "matched_keywords": keyword_matches,
            "answer_preview": answer[:200],
        })

    total = len(results)
    faithful_count = sum(1 for r in results if r["faithful"])
    rate = faithful_count / total if total > 0 else 0.0

    return {
        "faithfulness_rate": rate,
        "faithful": faithful_count,
        "total": total,
        "details": results,
    }


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def print_table(title: str, rows: list, columns: list):
    """Pretty-print a simple text table."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

    widths = {c: max(len(c), max(len(str(r.get(c, ""))) for r in rows)) for c in columns}
    header = " | ".join(c.ljust(widths[c]) for c in columns)
    print(f"  {header}")
    print(f"  {'-' * len(header)}")
    for row in rows:
        line = " | ".join(str(row.get(c, "")).ljust(widths[c]) for c in columns)
        print(f"  {line}")


async def main():
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline")
    parser.add_argument("--retrieval", action="store_true", help="Run retrieval eval only (no LLM needed)")
    args = parser.parse_args()

    print()
    print("+" + "=" * 68 + "+")
    print("|   RAG EVALUATION PIPELINE                                          |")
    print("|   Retrieval Hit Rate  +  Triage Accuracy  +  Answer Faithfulness    |")
    print("+" + "=" * 68 + "+")
    print()

    start = time.time()
    all_results = {}

    # ---- 1. Triage Accuracy (always — no deps) ----
    print("[1/3] Evaluating Triage Accuracy …")
    triage = triage_accuracy()
    all_results["triage_accuracy"] = triage
    print(f"      Accuracy: {triage['accuracy']:.0%}  ({triage['correct']}/{triage['total']})")

    triage_rows = [
        {"Question": r["question"][:50], "Expected": r["expected"], "Predicted": r["predicted"], "OK": "YES" if r["correct"] else "NO"}
        for r in triage["details"]
    ]
    print_table("Triage Accuracy", triage_rows, ["Question", "Expected", "Predicted", "OK"])

    # ---- 2. Retrieval Hit Rate ----
    print("\n[2/3] Evaluating Retrieval Hit Rate …")
    try:
        emb = get_embedding_function()
        vs = load_or_create_vectorstore(emb)
    except Exception as e:
        print(f"      SKIP — could not load vector store: {e}")
        vs = None

    if vs:
        retrieval = retrieval_hit_rate(vs, emb)
        all_results["retrieval_hit_rate"] = retrieval
        print(f"      Hit Rate: {retrieval['hit_rate']:.0%}  ({retrieval['hits']}/{retrieval['total']})")

        ret_rows = [
            {"Question": r["question"][:45], "Hit": "YES" if r["hit"] else "NO", "Docs": str(r["docs_returned"]), "Keywords": ", ".join(r["matched_keywords"][:3])}
            for r in retrieval["details"]
        ]
        print_table("Retrieval Hit Rate", ret_rows, ["Question", "Hit", "Docs", "Keywords"])

    # ---- 3. Answer Faithfulness (skip if --retrieval) ----
    if not args.retrieval and vs:
        print("\n[3/3] Evaluating Answer Faithfulness (requires Ollama) …")
        try:
            llm = get_llm()
            faith = await answer_faithfulness(vs, llm)
            all_results["answer_faithfulness"] = faith
            print(f"      Faithfulness: {faith['faithfulness_rate']:.0%}  ({faith['faithful']}/{faith['total']})")

            faith_rows = [
                {"Question": r["question"][:45], "Faithful": "YES" if r["faithful"] else "NO", "Keywords": ", ".join(r["matched_keywords"][:3]), "Preview": r["answer_preview"][:60]}
                for r in faith["details"]
            ]
            print_table("Answer Faithfulness", faith_rows, ["Question", "Faithful", "Keywords", "Preview"])
        except Exception as e:
            print(f"      SKIP — LLM not available: {e}")
    elif args.retrieval:
        print("\n[3/3] Skipped (--retrieval mode)")

    # ---- Summary ----
    elapsed = time.time() - start
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  Triage Accuracy:       {triage['accuracy']:.0%}")
    if "retrieval_hit_rate" in all_results:
        print(f"  Retrieval Hit Rate:    {all_results['retrieval_hit_rate']['hit_rate']:.0%}")
    if "answer_faithfulness" in all_results:
        print(f"  Answer Faithfulness:   {all_results['answer_faithfulness']['faithfulness_rate']:.0%}")
    print(f"  Elapsed:               {elapsed:.1f}s")
    print(f"{'='*70}")

    # ---- Save to JSON ----
    out_path = os.path.join(ROOT_DIR, "evaluation_results.json")
    # Strip "details" for the summary file to keep it small
    summary = {}
    for k, v in all_results.items():
        summary[k] = {kk: vv for kk, vv in v.items() if kk != "details"}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to: {out_path}\n")


if __name__ == "__main__":
    asyncio.run(main())
