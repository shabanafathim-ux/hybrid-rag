"""
extract.py - Step 1 of the pipeline: PDF -> raw lines.

Reads a PDF with PyMuPDF and returns every line of text together with
its page number, font size and bold flag. The font information is what
lets clean.py find headings later, so we keep it instead of throwing it
away like a plain "get all text" call would.

Usage:
    python src/extract.py data/employee_handbook.pdf
    -> writes output/employee_handbook.raw.json
"""

import json
import sys
from pathlib import Path

import pymupdf  # pip install pymupdf

BOLD_FLAG = 16  # PyMuPDF sets bit 4 of a span's "flags" when the font is bold


def extract_lines(pdf_path: str) -> list[dict]:
    """Return one dict per text line: {page, block, text, size, bold}.

    "block" is PyMuPDF's text block on that page. Blank lines are kept
    (with text "") because clean.py uses them as paragraph breaks.
    """
    lines = []
    with pymupdf.open(pdf_path) as doc:
        for page_number, page in enumerate(doc, start=1):
            for block_number, block in enumerate(page.get_text("dict")["blocks"]):
                for line in block.get("lines", []):  # image blocks have no lines
                    spans = line["spans"]
                    text = "".join(span["text"] for span in spans)
                    # Blank lines are kept on purpose: in many PDFs they are
                    # the only sign of where one paragraph ends.
                    first = spans[0]
                    lines.append(
                        {
                            "page": page_number,
                            "block": block_number,
                            "text": text,
                            "size": round(first["size"], 1),
                            "bold": bool(first["flags"] & BOLD_FLAG),
                        }
                    )
    return lines


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("Usage: python src/extract.py <file.pdf>")

    pdf_path = Path(sys.argv[1])
    lines = extract_lines(str(pdf_path))

    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{pdf_path.stem}.raw.json"
    out_path.write_text(json.dumps(lines, indent=2, ensure_ascii=False), encoding="utf-8")

    pages = max((l["page"] for l in lines), default=0)
    print(f"Extracted {len(lines)} lines from {pages} pages -> {out_path}")


if __name__ == "__main__":
    main()
