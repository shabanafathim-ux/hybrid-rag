"""
hybrid_search.py - Step 6: combine dense search and BM25 (and optionally rerank).

Usage:
    python src/hybrid_search.py output/employee_handbook.chunks.json "your question"
    python src/hybrid_search.py output/employee_handbook.chunks.json "your question" --rerank

How it works:
  1. Dense search -> top 10 chunks (by meaning).
  2. BM25 search  -> top 10 chunks (by words).
  3. Confidence check: if BM25's best chunk contains less than half of the
     question's words, BM25 is guessing, so its list is ignored.
     (Found by testing: for "I saw something shady at work", BM25 only
     matched "work" and would have pushed the wrong chunks to the top.)
  4. Reciprocal Rank Fusion (RRF): every chunk gets 1 / (k + rank) points
     from each list it appears in. Most points wins.
  5. Optional reranker (rerank.py): reads question + chunk together and
     re-sorts the candidates.

Needs: output/<name>.dense.npy (dense_index.py build)
       output/<name>.bm25.json (bm25_index.py build)
"""

import json
import re
import sys
from pathlib import Path

import dense_index
from bm25_index import BM25Index, tokenize

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

RRF_K = 60               # see README: bigger k = agreement matters more than rank
CANDIDATES = 10          # how many chunks each search contributes
MIN_BM25_COVERAGE = 0.5  # BM25 votes only if its best chunk has >= 50% of the question words
TOP_K = 5                # how many results to return
RERANK_TOP_N = 3         # how many chunks to keep after reranking


# ---------------------------------------------------------------------------
# The pieces
# ---------------------------------------------------------------------------

def rrf(ranked_lists: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion. Each list is chunk ids, best first."""
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, chunk_id in enumerate(ranked, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


def question_words(question: str) -> set[str]:
    """The words used to measure coverage: simple words only, so
    '1-800-555-1212' counts as its parts 1, 800, 555, 1212."""
    tokens = set(tokenize(question))
    simple = {t for t in tokens if not re.search(r"[@.\-']", t)}
    return simple or tokens


def bm25_coverage(bm25: BM25Index, question: str, top_chunk: str) -> float:
    """Share of the question's words found in BM25's best chunk (0.0 - 1.0)."""
    words = question_words(question)
    if not words:
        return 0.0
    matched = set(bm25.matched_words(question, top_chunk)) & words
    return len(matched) / len(words)


class HybridSearcher:
    """Loads both indexes once, then answers many questions."""

    def __init__(self, chunks_path: Path):
        self.chunks = {c["chunk_id"]: c for c in json.loads(chunks_path.read_text(encoding="utf-8"))}
        self.vectors, self.dense_ids = dense_index.load(chunks_path)
        self.bm25 = BM25Index(chunks_path)

    def search(self, question: str, top_k: int = TOP_K, use_rerank: bool = False) -> dict:
        """Returns a dict with every intermediate list, so results can be explained."""
        dense = dense_index.search(question, self.vectors, self.dense_ids, CANDIDATES)
        bm25 = self.bm25.search(question, CANDIDATES)

        coverage = bm25_coverage(self.bm25, question, bm25[0][0]) if bm25 else 0.0
        bm25_used = coverage >= MIN_BM25_COVERAGE

        lists = [[cid for cid, _ in dense]]
        if bm25_used:
            lists.append([cid for cid, _ in bm25])
        fused = rrf(lists)

        result = {"dense": dense, "bm25": bm25, "bm25_coverage": coverage,
                  "bm25_used": bm25_used, "fused": fused[:top_k], "reranked": None}

        if use_rerank:
            from rerank import rerank
            candidates = [(cid, self.chunks[cid]["text"]) for cid, _ in fused[:CANDIDATES]]
            result["reranked"] = rerank(question, candidates, RERANK_TOP_N)
        return result

    def heading(self, chunk_id: str) -> str:
        return self.chunks[chunk_id]["text"].splitlines()[0]


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--rerank"]
    use_rerank = "--rerank" in sys.argv
    if len(args) < 2:
        sys.exit('Usage: python src/hybrid_search.py <chunks.json> "your question" [--rerank]')

    searcher = HybridSearcher(Path(args[0]))
    question = " ".join(args[1:])
    r = searcher.search(question, use_rerank=use_rerank)
    h = searcher.heading

    print(f"\nQuestion: {question}\n")
    print("Dense search (meaning):")
    for i, (cid, s) in enumerate(r["dense"][:TOP_K], 1):
        print(f"  {i}. {s:.3f}  {h(cid)}")

    print("\nBM25 (words):")
    if not r["bm25"]:
        print("  no chunk contains any of the question words")
    for i, (cid, s) in enumerate(r["bm25"][:TOP_K], 1):
        print(f"  {i}. {s:5.2f}  {h(cid)}")
    verdict = "USED" if r["bm25_used"] else f"IGNORED (below {MIN_BM25_COVERAGE:.0%}, BM25 is guessing)"
    print(f"  -> best chunk covers {r['bm25_coverage']:.0%} of the question words: {verdict}")

    print(f"\nHybrid (RRF, k={RRF_K}):")
    for i, (cid, s) in enumerate(r["fused"], 1):
        print(f"  {i}. {s:.4f}  {h(cid)}")

    if r["reranked"] is not None:
        print("\nAfter reranking:")
        for i, (cid, s) in enumerate(r["reranked"], 1):
            print(f"  {i}. {s:.3f}  {h(cid)}")


if __name__ == "__main__":
    main()
