"""
rerank.py - Step 6b: re-sort the hybrid search results with a reranker.

Dense search and BM25 are fast because they never read the question and a
chunk TOGETHER. A reranker (a "cross-encoder") does: it reads each
(question, chunk) pair and gives one score for "how well does this chunk
answer this question?". That is slower but much more accurate, so it is used
only on the ~10 candidates the hybrid search already found.

Used by hybrid_search.py with the --rerank flag:
    python src/hybrid_search.py output/employee_handbook.chunks.json "question" --rerank

The first run downloads the model (~1.1 GB) from Hugging Face.
"""

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

# Same team (BAAI) as our embedding model. Free, runs on a laptop CPU,
# reads up to 512 tokens per (question + chunk) pair.
# Smaller, faster alternative (~90 MB): "cross-encoder/ms-marco-MiniLM-L-6-v2"
MODEL_NAME = "BAAI/bge-reranker-base"

MAX_LENGTH = 512  # tokens of question + chunk the model reads

_model = None


def get_model():
    """Load the model once and reuse it."""
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder
        _model = CrossEncoder(MODEL_NAME, max_length=MAX_LENGTH)
    return _model


def rerank(question: str, candidates: list[tuple[str, str]], top_n: int | None = None) -> list[tuple[str, float]]:
    """candidates: [(chunk_id, chunk_text), ...]
    Returns [(chunk_id, reranker_score), ...], best first.
    Higher score = better answer to the question."""
    if not candidates:
        return []
    pairs = [(question, text) for _, text in candidates]
    scores = get_model().predict(pairs)
    ranked = sorted(zip((cid for cid, _ in candidates), (float(s) for s in scores)),
                    key=lambda x: -x[1])
    return ranked[:top_n] if top_n else ranked
