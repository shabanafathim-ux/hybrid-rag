"""
dense_index.py - Step 4 of the pipeline: chunks -> vectors (dense search).

Build once:
    python src/dense_index.py build output/employee_handbook.chunks.json
    -> output/employee_handbook.dense.npy        (one vector per chunk)
    -> output/employee_handbook.dense_meta.json  (which chunk each row is)

Search:
    python src/dense_index.py search output/employee_handbook.chunks.json "Who do I tell about something shady?"

How it works:
  - An embedding model turns text into a list of 384 numbers (a vector).
    Texts with similar MEANING get similar vectors.
  - Building: every chunk is embedded once and saved.
  - Searching: the question is embedded, then compared with every chunk
    vector. The closest chunks (highest cosine similarity) win.

The first run downloads the model (~130 MB) from Hugging Face.
"""

import json
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

# bge-small: small, free, runs on a laptop CPU, reads up to 512 tokens.
# (all-MiniLM-L6-v2 is popular too, but it only reads the first 256 tokens,
#  which would cut off the end of our longest chunks.)
MODEL_NAME = "BAAI/bge-small-en-v1.5"

# bge models search better when the QUESTION (not the chunks) gets this prefix.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

TOP_K = 5

_model = None


def get_model():
    """Load the model once and reuse it."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer  # pip install sentence-transformers
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed(texts: list[str]) -> np.ndarray:
    """Texts -> matrix of vectors, one row per text, each of length 1.
    Length 1 ("normalized") means a simple dot product = cosine similarity."""
    return get_model().encode(texts, normalize_embeddings=True, convert_to_numpy=True)


def index_paths(chunks_path: Path) -> tuple[Path, Path]:
    stem = chunks_path.name.replace(".chunks.json", "")
    return chunks_path.parent / f"{stem}.dense.npy", chunks_path.parent / f"{stem}.dense_meta.json"


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def check_lengths(chunks: list[dict]) -> None:
    """Warn if a chunk is longer than the model can read (the rest is ignored)."""
    model = get_model()
    limit = model.max_seq_length
    for c in chunks:
        n_tokens = len(model.tokenizer.encode(c["text"]))
        if n_tokens > limit:
            print(f"  WARNING {c['chunk_id']}: {n_tokens} tokens, model reads only {limit}. "
                  f"Lower MAX_WORDS in chunk.py.")


def build(chunks_path: Path) -> None:
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    check_lengths(chunks)

    vectors = embed([c["text"] for c in chunks])

    vec_path, meta_path = index_paths(chunks_path)
    np.save(vec_path, vectors)
    meta = {"model": MODEL_NAME, "chunk_ids": [c["chunk_id"] for c in chunks]}
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Embedded {len(chunks)} chunks -> {vectors.shape[0]} vectors of {vectors.shape[1]} numbers")
    print(f"Saved {vec_path} and {meta_path}")


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def load(chunks_path: Path) -> tuple[np.ndarray, list[str]]:
    vec_path, meta_path = index_paths(chunks_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta["model"] != MODEL_NAME:
        sys.exit(f"Index was built with {meta['model']}, but MODEL_NAME is {MODEL_NAME}. Rebuild it.")
    return np.load(vec_path), meta["chunk_ids"]


def search(question: str, vectors: np.ndarray, chunk_ids: list[str], top_k: int = TOP_K) -> list[tuple[str, float]]:
    """Return [(chunk_id, similarity), ...], best first."""
    q = embed([QUERY_PREFIX + question])[0]
    scores = vectors @ q                      # one similarity per chunk
    best = np.argsort(-scores)[:top_k]        # positions of the highest scores
    return [(chunk_ids[i], float(scores[i])) for i in best]


def main() -> None:
    if len(sys.argv) < 3 or sys.argv[1] not in ("build", "search"):
        sys.exit('Usage:\n  python src/dense_index.py build <chunks.json>\n'
                 '  python src/dense_index.py search <chunks.json> "your question"')

    chunks_path = Path(sys.argv[2])
    if sys.argv[1] == "build":
        build(chunks_path)
        return

    question = " ".join(sys.argv[3:])
    vectors, chunk_ids = load(chunks_path)
    chunks = {c["chunk_id"]: c for c in json.loads(chunks_path.read_text(encoding="utf-8"))}

    print(f"Question: {question}\n")
    for rank, (cid, score) in enumerate(search(question, vectors, chunk_ids), start=1):
        heading = chunks[cid]["text"].splitlines()[0]
        print(f"  {rank}. {score:.3f}  {cid}  {heading}")


if __name__ == "__main__":
    main()
