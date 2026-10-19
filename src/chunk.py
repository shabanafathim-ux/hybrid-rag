"""
chunk.py - Step 3 of the pipeline: clean blocks -> chunks.

Input : output/<name>.clean.json   (made by clean.py)
Output: output/<name>.chunks.json  (what the search indexes will use)
        output/<name>.chunks.md    (same chunks, easy for a human to check)

How chunks are made (section-based chunking):
  1. Group paragraphs by section and subsection.
  2. Keep a list together with the sentence that introduces it.
  3. Tiny subsections are merged with their neighbour (never deleted).
  4. A group that is too long is split, with a small overlap.
  5. Every chunk starts with its heading path, e.g.
     "Whistleblower Policy > Reporting Procedures".

Sizes are counted in words (1 word is roughly 1.3 tokens).

Usage:
    python src/chunk.py output/employee_handbook.clean.json
"""

import json
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

MAX_WORDS = 350      # ~450 tokens. Bigger groups are split.
MIN_WORDS = 80       # Smaller groups are merged with a neighbour.
OVERLAP_WORDS = 50   # When splitting, repeat about this much text.
JUNK_WORDS = 5       # Chunks with fewer real words than this are dropped.


def words(text: str) -> int:
    return len(text.split())


# ---------------------------------------------------------------------------
# 1 + 2: blocks -> groups (one per subsection) made of units
# ---------------------------------------------------------------------------

def block_text(b: dict) -> str:
    if b["type"] == "list_item" and not re.match(r"^\d+\.", b["text"]):
        return "- " + b["text"]
    return b["text"]


def make_units(blocks: list[dict]) -> list[str]:
    """A unit is a piece that must never be cut: one paragraph, or a whole
    list together with the sentence that introduces it ("...include:")."""
    units: list[str] = []
    for b in blocks:
        text = block_text(b)
        if b["type"] == "list_item" and units:
            prev = units[-1]
            if prev.endswith(":") or prev.split("\n")[-1].startswith(("- ", *[f"{n}." for n in range(1, 100)])):
                units[-1] = prev + "\n" + text
                continue
        units.append(text)
    return units


def make_groups(doc: dict) -> list[dict]:
    """One group per (section, subsection), in document order."""
    groups: list[dict] = []
    for b in doc["blocks"]:
        if b["type"] in ("heading", "subheading"):
            continue
        key = (b["section"], b["subsection"])
        if not groups or (groups[-1]["section"], groups[-1]["subsection"]) != key:
            groups.append({"section": b["section"], "subsection": b["subsection"],
                           "blocks": [], "pages": set()})
        groups[-1]["blocks"].append(b)
        groups[-1]["pages"].add(b["page"])
    for g in groups:
        g["units"] = make_units(g.pop("blocks"))
    return groups


# ---------------------------------------------------------------------------
# 3: merge tiny groups inside the same section
# ---------------------------------------------------------------------------

def group_words(g: dict) -> int:
    return sum(words(u) for u in g["units"])


def merge_small(groups: list[dict]) -> list[dict]:
    merged: list[dict] = []
    for g in groups:
        g = {**g, "subsections": [g["subsection"]] if g["subsection"] else [],
             "body": [(g["subsection"], g["units"])]}
        prev = merged[-1] if merged else None
        same_section = prev is not None and prev["section"] == g["section"]
        if same_section and group_words(prev) < MIN_WORDS and group_words(prev) + group_words(g) <= MAX_WORDS:
            _absorb(prev, g)
        else:
            merged.append(g)
    # A small LAST group in a section joins the group before it
    result: list[dict] = []
    for g in merged:
        prev = result[-1] if result else None
        if (prev and prev["section"] == g["section"] and group_words(g) < MIN_WORDS
                and group_words(prev) + group_words(g) <= MAX_WORDS):
            _absorb(prev, g)
        else:
            result.append(g)
    return result


def _absorb(target: dict, g: dict) -> None:
    target["units"] = target["units"] + g["units"]
    target["subsections"] += g["subsections"]
    target["body"] += g["body"]
    target["pages"] |= g["pages"]


# ---------------------------------------------------------------------------
# 4 + 5: build chunk text, split long groups with overlap
# ---------------------------------------------------------------------------

def heading_path(section: str, body: list[tuple]) -> str:
    """Section, plus the subsection the chunk STARTS with (if any)."""
    first_sub = body[0][0]
    return section + (f" > {first_sub}" if first_sub else "")


def render(body: list[tuple]) -> list[str]:
    """Turn (subsection, units) pairs into text pieces. The FIRST subsection
    goes in the heading path, later ones are written inline as small titles."""
    pieces = []
    for i, (sub, units) in enumerate(body):
        if i > 0 and sub:
            pieces.append(sub)
        pieces.extend(units)
    return pieces


def last_sentences(text: str, limit: int) -> str:
    """The end of a text, about `limit` words, cut at a sentence start."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    out: list[str] = []
    for s in reversed(sentences):
        if out and words(" ".join([s] + out)) > limit:
            break
        out.insert(0, s)
    return " ".join(out)


def split_pieces(pieces: list[str]) -> list[list[str]]:
    parts: list[list[str]] = [[]]
    for p in pieces:
        if parts[-1] and sum(map(words, parts[-1])) + words(p) > MAX_WORDS:
            overlap = last_sentences(parts[-1][-1], OVERLAP_WORDS)
            parts.append([overlap] if overlap else [])
        parts[-1].append(p)
    return parts


def make_chunks(doc: dict) -> list[dict]:
    chunks: list[dict] = []
    for g in merge_small(make_groups(doc)):
        path = heading_path(g["section"], g["body"])
        parts = split_pieces(render(g["body"]))
        for n, part in enumerate(parts, start=1):
            body = "\n\n".join(part)
            if words(body) < JUNK_WORDS:
                continue
            chunks.append({
                "chunk_id": f"{Path(doc['source']).stem}_{len(chunks) + 1:03d}",
                "text": f"{path}\n\n{body}",
                "section": g["section"],
                "subsections": g["subsections"],
                "part": f"{n} of {len(parts)}",
                "pages": sorted(g["pages"]),
                "source": doc["source"],
                "title": doc.get("title"),
                "last_updated": doc.get("last_updated"),
                "words": words(body),
            })
    return chunks


def to_markdown(chunks: list[dict]) -> str:
    out = []
    for c in chunks:
        out.append(f"---\n**{c['chunk_id']}**  |  pages {c['pages']}  |  "
                   f"part {c['part']}  |  {c['words']} words\n")
        out.append("```\n" + c["text"] + "\n```\n")
    return "\n".join(out)


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("Usage: python src/chunk.py output/<name>.clean.json")

    clean_path = Path(sys.argv[1])
    stem = clean_path.name.replace(".clean.json", "")
    doc = json.loads(clean_path.read_text(encoding="utf-8"))

    chunks = make_chunks(doc)

    out_dir = clean_path.parent
    (out_dir / f"{stem}.chunks.json").write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / f"{stem}.chunks.md").write_text(to_markdown(chunks), encoding="utf-8")

    sizes = [c["words"] for c in chunks]
    print(f"{len(chunks)} chunks  |  words per chunk: min {min(sizes)}, "
          f"avg {sum(sizes) // len(sizes)}, max {max(sizes)}")
    for c in chunks:
        print(f"  {c['chunk_id']}  {c['words']:>4} words  {c['text'].splitlines()[0]}")


if __name__ == "__main__":
    main()
