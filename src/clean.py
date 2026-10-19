"""
clean.py - Step 2 of the pipeline: raw lines -> clean, structured blocks.

Input : output/<name>.raw.json   (made by extract.py)
Output: output/<name>.clean.json (blocks with section/subsection, for chunk.py)
        output/<name>.clean.md   (same content, easy for a human to read)
        output/<name>.clean_log.txt (what every step changed)

Cleaning steps (numbers match the README):
  2. Normalize text        - unicode, quotes, dashes, whitespace
  3. Fix line breaks       - rebuild paragraphs, join hyphenated line ends
  4. Remove boilerplate    - repeated headers/footers, page numbers, disclaimers
  5. Deduplicate           - drop repeated paragraphs
  6. Handle versions       - pull "Last Updated" date into metadata
  7. Detect structure      - headings, subheadings, list items

Rule of thumb: change the FORM of the text, never its MEANING.

Usage:
    python src/clean.py output/employee_handbook.raw.json
"""

import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Settings you may want to tune for other documents
# ---------------------------------------------------------------------------

# Text that marks a block as boilerplate (case-insensitive regex).
DISCLAIMER_PATTERNS = [
    r"generated using a language model",
    r"^all rights reserved",
    r"^page \d+( of \d+)?$",
    r"^\d+$",  # a lone page number
]

# A line repeated on at least this share of pages is treated as header/footer.
REPEATED_LINE_SHARE = 0.5

# Lines whose font is this many points bigger than body text are headings.
HEADING_SIZE_GAP = 2.0

# Short single-line blocks with at most this many words can be subheadings.
SUBHEADING_MAX_WORDS = 8

# Metadata lines: (metadata key, regex with one group for the value)
METADATA_PATTERNS = [
    ("last_updated", r"^last updated:\s*(.+)$"),
    ("version", r"^version:?\s*(.+)$"),
]

LIST_ITEM = re.compile(r"^(•|▪|◦|-|\*|\d+\.)\s+")


# ---------------------------------------------------------------------------
# Step 2: normalize
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    """Make the same word always look the same (helps BM25 match)."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("­", "")  # soft hyphen
    text = re.sub(r"[‘’‚‛]", "'", text)
    text = re.sub(r"[“”„‟]", '"', text)
    text = re.sub(r"[–—]", "-", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Step 4a: repeated headers / footers (line level, before paragraphs exist)
# ---------------------------------------------------------------------------

def find_repeated_lines(lines: list[dict]) -> set[str]:
    """Lines that appear on many pages, like 'Confidential - Page header'."""
    pages = {l["page"] for l in lines}
    if len(pages) < 3:
        return set()
    pages_per_text = Counter()
    for text in {(l["page"], l["text"].lower()) for l in lines if l["text"]}:
        pages_per_text[text[1]] += 1
    limit = REPEATED_LINE_SHARE * len(pages)
    return {text for text, count in pages_per_text.items() if count >= limit}


# ---------------------------------------------------------------------------
# Step 3 + 7: rebuild paragraphs and label them
# ---------------------------------------------------------------------------

def join_lines(previous: str, nxt: str) -> str:
    """Join two lines of one paragraph.

    'cutting-' + 'edge'   -> 'cutting-edge'   (keep hyphen, no space)
    '... at 1-' + '800-..' -> '... at 1-800-..'
    otherwise             -> 'previous nxt'
    """
    if previous.endswith("-"):
        return previous + nxt
    return previous + " " + nxt


def build_blocks(lines: list[dict], body_size: float, title_size: float) -> list[dict]:
    """Group lines into blocks: title, heading, subheading, list_item, paragraph."""
    blocks: list[dict] = []
    current_key = None

    for line in lines:
        key = (line["page"], line["block"])
        text = line["text"]

        if not text:  # blank line = paragraph break
            current_key = None
            continue

        if line["size"] >= title_size:
            kind = "title"
        elif line["size"] >= body_size + HEADING_SIZE_GAP:
            kind = "heading"
        elif LIST_ITEM.match(text):
            kind = "list_item"
        else:
            kind = "paragraph"

        # Every "1." or "•" starts a new item, even without a blank line before it
        same_block = key == current_key and blocks and blocks[-1]["type"] == kind and kind != "list_item"
        # A list item continues on the next line ("3. Innovation: ..." / "business.")
        continues_item = key == current_key and blocks and blocks[-1]["type"] == "list_item" and kind == "paragraph"
        # A paragraph cut by a page break: next block starts in lowercase ("website.")
        continues_page = (
            kind == "paragraph"
            and blocks
            and blocks[-1]["type"] in ("paragraph", "list_item")
            and line["page"] != blocks[-1]["page"]
            and text[:1].islower()
        )

        if same_block or continues_item or continues_page:
            blocks[-1]["text"] = join_lines(blocks[-1]["text"], text)
            blocks[-1]["n_lines"] += 1
        else:
            blocks.append({"type": kind, "text": text, "page": line["page"], "n_lines": 1})
        current_key = key

    # Strip the bullet symbol, keep numbers ("1." carries meaning, "•" does not)
    for b in blocks:
        if b["type"] == "list_item":
            b["text"] = re.sub(r"^(•|▪|◦|-|\*)\s+", "", b["text"])
    return blocks


def is_subheading(block: dict) -> bool:
    """Short one-line paragraph that doesn't end a sentence: 'Purpose', 'Definition:'."""
    text = block["text"]
    return (
        block["type"] == "paragraph"
        and block["n_lines"] == 1
        and len(text.split()) <= SUBHEADING_MAX_WORDS
        and text[:1].isupper()
        and not text.endswith((".", ",", ";", "!", "?"))
    )


# ---------------------------------------------------------------------------
# Main cleaning function
# ---------------------------------------------------------------------------

def clean(lines: list[dict], source: str) -> tuple[dict, list[str]]:
    log: list[str] = []

    # Step 2: normalize every line
    for l in lines:
        l["text"] = normalize(l["text"])
    text_lines = [l for l in lines if l["text"]]  # blank lines stay in `lines` as breaks

    # Font sizes: body = most common size, title = biggest size
    body_size = Counter(l["size"] for l in text_lines).most_common(1)[0][0]
    title_size = max(l["size"] for l in text_lines)

    # Step 4a: repeated headers/footers
    repeated = find_repeated_lines(lines)
    if repeated:
        log.append(f"[boilerplate] removed repeated lines: {sorted(repeated)}")
        lines = [l for l in lines if l["text"].lower() not in repeated]

    # Steps 3 + 7: paragraphs, headings, list items
    blocks = build_blocks(lines, body_size, title_size)

    # Document title (biggest font) becomes metadata, not content
    title_parts = [b["text"] for b in blocks if b["type"] == "title"]
    title = " ".join(title_parts)
    blocks = [b for b in blocks if b["type"] != "title"]

    # Step 4b: disclaimers and page numbers
    kept = []
    for b in blocks:
        if any(re.search(p, b["text"], re.IGNORECASE) for p in DISCLAIMER_PATTERNS):
            log.append(f"[boilerplate] p{b['page']}: removed '{b['text'][:70]}...'")
        else:
            kept.append(b)
    blocks = kept

    # Step 6: metadata such as "Last Updated: 2023-03-05"
    metadata = {}
    kept = []
    for b in blocks:
        for key, pattern in METADATA_PATTERNS:
            m = re.match(pattern, b["text"], re.IGNORECASE)
            if m and b["type"] != "heading":
                metadata[key] = m.group(1).strip()
                log.append(f"[metadata] {key} = {metadata[key]}")
                break
        else:
            kept.append(b)
    blocks = kept

    # Step 7: subheadings
    for b in blocks:
        if is_subheading(b):
            b["type"] = "subheading"
            b["text"] = b["text"].rstrip(":")

    # Step 5: deduplicate paragraphs.
    # We keep the LAST copy: intros often repeat text that properly belongs
    # in a later section (here: the intro paragraph = the "Our Mission" text).
    def fingerprint(text: str) -> str:
        return hashlib.md5(re.sub(r"\W+", " ", text.lower()).strip().encode()).hexdigest()

    last_index = {}
    for i, b in enumerate(blocks):
        if b["type"] in ("paragraph", "list_item"):
            last_index[fingerprint(b["text"])] = i
    kept = []
    for i, b in enumerate(blocks):
        if b["type"] in ("paragraph", "list_item") and last_index[fingerprint(b["text"])] != i:
            log.append(f"[dedup] p{b['page']}: removed duplicate '{b['text'][:60]}...'")
            continue
        kept.append(b)
    blocks = kept

    # Drop headings left with no content (e.g. after dedup)
    kept = []
    for i, b in enumerate(blocks):
        if b["type"] == "heading":
            following = blocks[i + 1] if i + 1 < len(blocks) else None
            if following is None or following["type"] == "heading":
                log.append(f"[structure] removed empty section '{b['text']}'")
                continue
        kept.append(b)
    blocks = kept

    # Give every block its section / subsection (the "heading path" for chunking)
    section, subsection = None, None
    for b in blocks:
        if b["type"] == "heading":
            section, subsection = b["text"], None
        elif b["type"] == "subheading":
            subsection = b["text"]
        b["section"], b["subsection"] = section, subsection
        del b["n_lines"]

    doc = {"source": source, "title": title, **metadata, "blocks": blocks}
    return doc, log


def to_markdown(doc: dict) -> str:
    """Readable version of the cleaned document, for spot-checking."""
    out = [f"# {doc['title']}", ""]
    if doc.get("last_updated"):
        out += [f"_Last updated: {doc['last_updated']}_", ""]
    for b in doc["blocks"]:
        if b["type"] in ("heading", "subheading") and out[-1]:
            out.append("")
        if b["type"] == "heading":
            out += [f"## {b['text']}", ""]
        elif b["type"] == "subheading":
            out += [f"### {b['text']}", ""]
        elif b["type"] == "list_item":
            out.append(b["text"] if re.match(r"^\d+\.", b["text"]) else f"- {b['text']}")
        else:
            if out and out[-1] and not out[-1].startswith("#"):
                out.append("")
            out += [b["text"], ""]
    return "\n".join(out).replace("\n\n\n", "\n\n")


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("Usage: python src/clean.py output/<name>.raw.json")

    raw_path = Path(sys.argv[1])
    stem = raw_path.name.replace(".raw.json", "")
    lines = json.loads(raw_path.read_text(encoding="utf-8"))

    doc, log = clean(lines, source=f"{stem}.pdf")

    out_dir = raw_path.parent
    (out_dir / f"{stem}.clean.json").write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / f"{stem}.clean.md").write_text(to_markdown(doc), encoding="utf-8")
    (out_dir / f"{stem}.clean_log.txt").write_text("\n".join(log) + "\n", encoding="utf-8")

    sections = [b["text"] for b in doc["blocks"] if b["type"] == "heading"]
    print(f"Title: {doc['title']}  |  last updated: {doc.get('last_updated', '-')}")
    print(f"{len(doc['blocks'])} blocks in {len(sections)} sections: {', '.join(sections)}")
    print(f"{len(log)} changes logged -> {out_dir / (stem + '.clean_log.txt')}")


if __name__ == "__main__":
    main()
