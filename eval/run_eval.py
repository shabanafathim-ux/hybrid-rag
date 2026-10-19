"""
run_eval.py - Step 8: test the whole system on a fixed set of questions.

Usage (from the hybrid-rag folder):
    python eval/run_eval.py                 # search + reranker + "no answer" check (fast, no LLM)
    python eval/run_eval.py --answers       # also ask Ollama and check every answer (slower)

Reads   eval/questions.csv
Writes  eval/results.csv   (one row per question, every detail)
Prints  a summary table per question type.

What is measured
----------------
1. Retrieval: is a correct chunk ranked 1st (Hit@1) or in the top 3 (Hit@3)?
   Compared for five methods, to see what each part adds:
     Dense      - embeddings only
     BM25       - keywords only
     RRF        - plain fusion of both (no confidence check)
     Hybrid     - fusion with the BM25 confidence check (hybrid_search.py)
     +Rerank    - hybrid, then re-sorted by the reranker
2. "No answer" check: does the reranker score correctly decide when to stop
   before the LLM?  (best score < NO_ANSWER_SCORE in answer.py)
3. Answers (--answers): does the LLM answer contain the expected facts, or
   say it doesn't know when it should?
"""

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import answer as answer_mod                     # noqa: E402
from hybrid_search import HybridSearcher, rrf, CANDIDATES  # noqa: E402
from rerank import rerank                       # noqa: E402

CHUNKS = ROOT / "output" / "employee_handbook.chunks.json"
QUESTIONS = ROOT / "eval" / "questions.csv"
RESULTS = ROOT / "eval" / "results.csv"

METHODS = ["Dense", "BM25", "RRF", "Hybrid", "+Rerank"]
REFUSAL_WORDS = ["don't know", "do not know", "couldn't find", "could not find", "not mentioned",
                 "not specified", "does not say", "doesn't say", "does not mention",
                 "doesn't mention", "not provided", "no information", "not stated"]


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def full_id(short: str) -> str:
    return f"employee_handbook_{short}"


def hits(ranked: list[str], expected: set[str], k: int) -> bool:
    """True if any expected chunk is in the top k."""
    return bool(expected & set(ranked[:k]))


def contains_all(text: str, must_contain: str) -> tuple[bool, list[str]]:
    """must_contain: 'a;b|c' means: 'a' AND ('b' OR 'c'). Case-insensitive.
    Returns (passed, missing groups)."""
    text = text.lower()
    missing = [g for g in must_contain.split(";") if g and not any(alt in text for alt in g.split("|"))]
    return not missing, missing


def is_refusal(text: str) -> bool:
    t = text.lower().replace("’", "'")
    return text == answer_mod.NO_ANSWER_TEXT or any(w in t for w in REFUSAL_WORDS)


def pct(values: list[bool]) -> str:
    return f"{100 * sum(values) / len(values):3.0f}%" if values else "  - "


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def evaluate_question(searcher: HybridSearcher, q: dict, with_answers: bool) -> dict:
    expected = {full_id(s) for s in q["expected_chunks"].split(";") if s}
    r = searcher.search(q["question"])

    dense = [c for c, _ in r["dense"]]
    bm25 = [c for c, _ in r["bm25"]]
    plain = [c for c, _ in rrf([dense, bm25] if bm25 else [dense])]
    hybrid = [c for c, _ in rrf([dense, bm25] if r["bm25_used"] else [dense])]
    candidates = [(c, searcher.chunks[c]["text"]) for c in hybrid[:CANDIDATES]]
    reranked = rerank(q["question"], candidates)          # full order, not only top 3
    ranked = {"Dense": dense, "BM25": bm25, "RRF": plain, "Hybrid": hybrid,
              "+Rerank": [c for c, _ in reranked]}

    best = reranked[0][1] if reranked else 0.0
    stopped = best < answer_mod.NO_ANSWER_SCORE
    # The stop is correct for questions with no answer chunk at all. For "trap"
    # questions (right chunk exists but lacks the detail) the LLM must refuse.
    should_stop = q["expect_no_answer"] == "yes" and not expected

    row = {"id": q["id"], "type": q["type"], "question": q["question"],
           "expected": " ".join(sorted(expected)) or "-",
           "best_rerank_score": round(best, 4), "llm_skipped": stopped,
           "skip_correct": stopped == should_stop,
           "bm25_coverage": round(r["bm25_coverage"], 2), "bm25_used": r["bm25_used"]}
    for m in METHODS:
        row[f"{m} top3"] = " ".join(c[-3:] for c in ranked[m][:3])
        if expected:
            row[f"{m} hit1"] = hits(ranked[m], expected, 1)
            row[f"{m} hit3"] = hits(ranked[m], expected, 3)
            row[f"{m} found_all"] = expected <= set(ranked[m][:3])

    if with_answers:
        out = answer_mod.answer(searcher, q["question"])
        text = out["answer"]
        if q["expect_no_answer"] == "yes":
            ok, missing = is_refusal(text), ([] if is_refusal(text) else ["(should refuse)"])
        else:
            ok, missing = contains_all(text, q["must_contain"])
            if is_refusal(text):
                missing.append("(refused wrongly)")
                ok = False
        row.update({"answer_ok": ok, "answer_missing": "; ".join(missing),
                    "answer": text.replace("\n", " ")})
    return row


def print_summary(rows: list[dict], with_answers: bool) -> None:
    types = sorted({r["type"] for r in rows})
    answerable = [r for r in rows if "Dense hit1" in r]

    def table(title: str, key: str, subset_rows: list[dict]) -> None:
        print(f"\n{title}")
        print(f"  {'Question type':<17}{'n':>3}  " + "".join(f"{m:>9}" for m in METHODS))
        for t in types + ["ALL"]:
            sel = [r for r in subset_rows if t == "ALL" or r["type"] == t]
            if not sel:
                continue
            line = f"  {t:<17}{len(sel):>3}  " + "".join(f"{pct([r[f'{m} {key}'] for r in sel]):>9}" for m in METHODS)
            print(("  " + "-" * 64 + "\n" if t == "ALL" else "") + line)

    table("RETRIEVAL - correct chunk ranked 1st (Hit@1)", "hit1", answerable)
    table("RETRIEVAL - correct chunk in top 3 (Hit@3)", "hit3", answerable)
    multi = [r for r in answerable if len(r["expected"].split()) > 1]
    if multi:
        table("RETRIEVAL - ALL needed chunks in top 3 (questions needing 2+ chunks)", "found_all", multi)

    print("\n\"NO ANSWER\" CHECK (skip the LLM when the best reranker score < "
          f"{answer_mod.NO_ANSWER_SCORE})")
    # Two kinds of mistakes, not equally bad:
    #   SERIOUS : an answerable question was skipped -> the user is told "not found" wrongly.
    #   OK      : an unanswerable question went to the LLM -> the LLM must refuse (checked
    #             by --answers). Costs one LLM call, but nothing is lost.
    serious = [r for r in rows if r["llm_skipped"] and r["expect_no_answer"] != "yes"]
    to_llm = [r for r in rows if not r["llm_skipped"] and r["expect_no_answer"] == "yes"]
    print(f"  answerable questions wrongly skipped (serious): {len(serious)}")
    print(f"  unanswerable questions sent to the LLM (LLM must refuse): {len(to_llm)}")
    for r in sorted(rows, key=lambda r: r["best_rerank_score"]):
        if r in serious:
            flag = "  <-- WRONGLY SKIPPED"
        elif r in to_llm:
            flag = "  <-- no answer exists: LLM must refuse"
        else:
            flag = ""
        print(f"    {r['best_rerank_score']:.4f}  {'skipped' if r['llm_skipped'] else 'to LLM ':<8} "
              f"{r['id']}  {r['question'][:55]}{flag}")

    if with_answers:
        print("\nANSWERS (expected facts present, or a refusal when there is no answer)")
        for t in types + ["ALL"]:
            sel = [r for r in rows if t == "ALL" or r["type"] == t]
            print(("  " + "-" * 30 + "\n" if t == "ALL" else "") +
                  f"  {t:<17}{len(sel):>3}  {pct([r['answer_ok'] for r in sel]):>5}")
        for r in rows:
            if not r["answer_ok"]:
                print(f"    FAIL {r['id']}  missing: {r['answer_missing']}\n         {r['answer'][:150]}")


def main() -> None:
    with_answers = "--answers" in sys.argv
    questions = list(csv.DictReader(QUESTIONS.open(encoding="utf-8")))

    print("Loading models...")
    searcher = HybridSearcher(CHUNKS)

    rows = []
    for q in questions:
        print(f"  {q['id']} {q['question'][:70]}")
        row = evaluate_question(searcher, q, with_answers)
        row["expect_no_answer"] = q["expect_no_answer"]
        rows.append(row)

    fields = list(dict.fromkeys(k for r in rows for k in r))
    with RESULTS.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print_summary(rows, with_answers)
    print(f"\nEvery detail per question: {RESULTS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
