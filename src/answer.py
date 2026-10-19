"""
answer.py - Step 7: search + rerank + let a local LLM (Ollama) write the answer.

Usage:
    Chat mode (models load once, then ask as many questions as you like):
        python src/answer.py output/employee_handbook.chunks.json

    One question:
        python src/answer.py output/employee_handbook.chunks.json "I saw something shady at work, who do I tell?"

Needs:
  - both indexes built (dense_index.py build, bm25_index.py build)
  - Ollama running on this computer, with the model below downloaded:
        ollama pull llama3.2

How it works:
  1. Hybrid search + reranker find the best chunks (hybrid_search.py).
  2. "No answer" check: if the reranker's best score is below
     NO_ANSWER_SCORE, nothing in the documents answers the question.
     We say so and do NOT ask the LLM (so it cannot make something up).
  3. Relevance filter: keep only chunks scoring at least
     KEEP_RELATIVE x the best score (drops the noise below the answer),
     but only when the reranker is sure (best score >= UNSURE_SCORE).
     When it is unsure, all its top 3 chunks go to the LLM.
  4. The LLM gets the question + those chunks, with strict instructions:
     answer only from this text, cite the sources, say "I don't know" otherwise.

Nothing is sent to the internet: Ollama runs the model locally.
"""

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

from hybrid_search import HybridSearcher

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2"         # run `ollama list` to see your models; any chat model works

# Best reranker score below this -> "not in the documents", LLM not called.
# Set from the evaluation: unanswerable questions scored 0.0000, 0.0005, 0.0050,
# but one answerable casual question scored 0.0066, so 0.01 wrongly skipped it.
# Skipping a real answer is worse than one extra LLM call (the LLM refuses), so
# the cut-off is kept low.
NO_ANSWER_SCORE = 0.002
KEEP_RELATIVE = 0.3        # keep chunks scoring >= 30% of the best score...

# ...but only when the reranker is SURE. Below this best score it is unsure
# (casual wording), and its #1 can be wrong: for "How often will my boss check
# how I'm doing?" it ranked Job Roles 1st (0.0066) and Performance Reviews 2nd,
# and the 30% filter then removed the right chunk. When unsure, send its whole
# top 3 and let the LLM pick. In the evaluation, sure answers scored 0.65-0.99.
UNSURE_SCORE = 0.1

NO_ANSWER_TEXT = "I couldn't find this in the company documents. Please contact HR or your manager."

SYSTEM_PROMPT = """You are a helpful assistant for Contoso Electronics employees.
Answer the question using ONLY the documents provided.
Rules:
- If the documents do not contain the answer, say: "I don't know based on the company documents."
- Do not use outside knowledge and do not guess.
- Be short and clear. Use a short list if there are several options or steps.
- Copy exact details (emails, phone numbers, names) exactly as written.
- End with "Sources:" and the document numbers you used, for example [1]."""


# ---------------------------------------------------------------------------
# The pieces
# ---------------------------------------------------------------------------

def select_chunks(reranked: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """Choose which chunks the LLM gets. Returns [] when nothing answers the question.

    - best score < NO_ANSWER_SCORE : nothing relevant -> []
    - best score < UNSURE_SCORE    : reranker is unsure -> send all its top chunks
                                     and let the LLM pick (its #1 may be wrong)
    - otherwise                    : reranker is sure -> keep only chunks close to #1
    """
    if not reranked or reranked[0][1] < NO_ANSWER_SCORE:
        return []
    best = reranked[0][1]
    if best < UNSURE_SCORE:
        return reranked
    return [(cid, s) for cid, s in reranked if s >= KEEP_RELATIVE * best]


def build_prompt(question: str, chunks: list[dict]) -> str:
    """The text the LLM receives: numbered documents, then the question."""
    docs = []
    for i, c in enumerate(chunks, start=1):
        label = "page" if len(c["pages"]) == 1 else "pages"
        pages = ", ".join(str(p) for p in c["pages"])
        docs.append(f"[{i}] (source: {c['source']}, {label} {pages})\n{c['text']}")
    return "Documents:\n\n" + "\n\n---\n\n".join(docs) + f"\n\nQuestion: {question}"


def ask_ollama(prompt: str) -> str:
    """Send the prompt to the local Ollama server and return the reply text."""
    body = json.dumps({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": 0},  # 0 = no creativity, same answer every time
    }).encode("utf-8")
    request = urllib.request.Request(OLLAMA_URL, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.loads(response.read())["message"]["content"].strip()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            sys.exit(f"Ollama doesn't have the model '{MODEL}'. Run:  ollama pull {MODEL}\n"
                     f"(or change MODEL in answer.py to one listed by `ollama list`)")
        raise
    except urllib.error.URLError:
        sys.exit("Can't reach Ollama. Open the Ollama app (or run `ollama serve`) and try again.")


def answer(searcher: HybridSearcher, question: str) -> dict:
    result = searcher.search(question, use_rerank=True)
    selected = select_chunks(result["reranked"])
    if not selected:
        return {"answer": NO_ANSWER_TEXT, "chunks": [], "search": result}
    chunks = [searcher.chunks[cid] for cid, _ in selected]
    reply = ask_ollama(build_prompt(question, chunks))
    return {"answer": reply, "chunks": selected, "search": result}


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def show(searcher: HybridSearcher, question: str) -> None:
    """Answer one question and print the answer plus how it was found."""
    r = answer(searcher, question)

    print(f"\n{r['answer']}")
    print("\n" + "-" * 60)
    top = r["search"]["reranked"]
    print(f"Best reranker score: {top[0][1]:.3f}" if top else "No search results")
    if r["chunks"]:
        print("Chunks given to the LLM:")
        for i, (cid, s) in enumerate(r["chunks"], start=1):
            pages = searcher.chunks[cid]["pages"]
            label = "page" if len(pages) == 1 else "pages"
            print(f"  [{i}] {s:.3f}  {searcher.heading(cid)}  ({label} {', '.join(map(str, pages))})")
    else:
        print(f"Below {NO_ANSWER_SCORE}: LLM not called.")
    print("-" * 60)


def chat(searcher: HybridSearcher) -> None:
    """Ask many questions; the models stay loaded between questions."""
    print("\nChat mode. Type a question and press Enter. Type 'quit' (or press Ctrl+C) to stop.")
    while True:
        try:
            question = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in ("quit", "exit", "q"):
            break
        if question:
            show(searcher, question)


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit('Usage:\n'
                 '  python src/answer.py <chunks.json>                    (chat mode)\n'
                 '  python src/answer.py <chunks.json> "your question"    (one question)')

    print("Loading models...")
    searcher = HybridSearcher(Path(sys.argv[1]))

    if len(sys.argv) == 2:
        chat(searcher)
    else:
        question = " ".join(sys.argv[2:])
        print(f"\nQuestion: {question}")
        show(searcher, question)


if __name__ == "__main__":
    main()
