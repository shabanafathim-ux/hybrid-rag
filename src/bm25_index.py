"""
bm25_index.py - Step 5 of the pipeline: chunks -> word table (keyword search).

Build once:
    python src/bm25_index.py build output/employee_handbook.chunks.json
    -> output/employee_handbook.bm25.json  (the words of every chunk)

Search:
    python src/bm25_index.py search output/employee_handbook.chunks.json "What is 1-800-555-1212?"

How it works:
  - Every chunk is split into words ("tokens").
  - A question is split the same way.
  - BM25 scores each chunk by the question words it contains:
      * rare words (whistleblower, PPE) count a lot,
      * common words (contoso, employees) count very little,
      * repeating a word helps, but less and less,
      * long chunks get a small penalty.
  - No model, no vectors: just counting words.
"""

import json
import re
import sys
from pathlib import Path

from rank_bm25 import BM25Okapi  # pip install rank_bm25

TOP_K = 5

# Very common words that say nothing about the topic. BM25 already gives
# them little weight; removing them keeps the scores easier to read.
STOPWORDS = set("""
a an and are as at be been but by can could did do does for from had has have
how i if in into is it its me my no not of on or our should so than that the
their them then there these they this to was we were what when where which who
whom why will with would you your
""".split())

# A token is letters/digits, optionally joined by @ . - ' so that
# compliance@contoso.com, 1-800-555-1212, cutting-edge, employee's stay whole.
TOKEN = re.compile(r"[a-z0-9]+(?:[@.\-'][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Text -> list of words for BM25.

    'Call 1-800-555-1212 or compliance@contoso.com'
      -> ['call', '1-800-555-1212', '1', '800', '555', '1212',
          'compliance@contoso.com', 'compliance', 'contoso', 'com']

    Joined tokens are kept whole AND split into parts, so a search for
    'compliance hotline' or '555-1212' still finds them.
    """
    tokens = []
    for tok in TOKEN.findall(text.lower()):
        if tok in STOPWORDS:
            continue
        tokens.append(tok)
        parts = re.split(r"[@.\-']", tok)
        if len(parts) > 1:
            tokens.extend(p for p in parts if p and p not in STOPWORDS)
    return tokens


def index_path(chunks_path: Path) -> Path:
    stem = chunks_path.name.replace(".chunks.json", "")
    return chunks_path.parent / f"{stem}.bm25.json"


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build(chunks_path: Path) -> None:
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    data = {
        "chunk_ids": [c["chunk_id"] for c in chunks],
        "tokens": [tokenize(c["text"]) for c in chunks],
    }
    index_path(chunks_path).write_text(json.dumps(data), encoding="utf-8")

    vocab = {t for toks in data["tokens"] for t in toks}
    print(f"Indexed {len(chunks)} chunks, {len(vocab)} different words -> {index_path(chunks_path)}")


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

class BM25Index:
    def __init__(self, chunks_path: Path):
        data = json.loads(index_path(chunks_path).read_text(encoding="utf-8"))
        self.chunk_ids = data["chunk_ids"]
        self.tokens = data["tokens"]
        self.bm25 = BM25Okapi(self.tokens)  # k1=1.5, b=0.75 by default

    def search(self, question: str, top_k: int = TOP_K) -> list[tuple[str, float]]:
        """Return [(chunk_id, score), ...], best first. Chunks with score 0
        (no question word at all) are left out: BM25 found nothing there."""
        scores = self.bm25.get_scores(tokenize(question))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        return [(self.chunk_ids[i], float(scores[i])) for i in order[:top_k] if scores[i] > 0]

    def matched_words(self, question: str, chunk_id: str) -> list[str]:
        """Which question words appear in this chunk (to explain a result)."""
        words = set(self.tokens[self.chunk_ids.index(chunk_id)])
        return sorted({t for t in tokenize(question) if t in words})


def main() -> None:
    if len(sys.argv) < 3 or sys.argv[1] not in ("build", "search"):
        sys.exit('Usage:\n  python src/bm25_index.py build <chunks.json>\n'
                 '  python src/bm25_index.py search <chunks.json> "your question"')

    chunks_path = Path(sys.argv[2])
    if sys.argv[1] == "build":
        build(chunks_path)
        return

    question = " ".join(sys.argv[3:])
    index = BM25Index(chunks_path)
    chunks = {c["chunk_id"]: c for c in json.loads(chunks_path.read_text(encoding="utf-8"))}

    print(f"Question: {question}")
    print(f"Words searched: {tokenize(question)}\n")
    results = index.search(question)
    if not results:
        print("  No chunk contains any of these words.")
    for rank, (cid, score) in enumerate(results, start=1):
        heading = chunks[cid]["text"].splitlines()[0]
        print(f"  {rank}. {score:5.2f}  {cid}  {heading}")
        print(f"            matched: {index.matched_words(question, cid)}")


if __name__ == "__main__":
    main()
