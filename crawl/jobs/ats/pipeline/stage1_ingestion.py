"""Stage 1 — Document Ingestion & Text Extraction.

Upgrades over the original parser.py:
- Structural cue preservation (line breaks, bullet markers)
- ATS parse-hazard detection: multi-column layouts, tables, text boxes,
  image-only pages, headers/footers, encoding anomalies
- Returns IngestionResult with text + warnings, not just plain text

All hazard warnings surface in the final report's 'parse_hazards' list.
"""

import io
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class IngestionResult:
    """Output contract for Stage 1."""
    text:          str
    parse_hazards: List[str]          = field(default_factory=list)
    metadata:      Dict[str, Any]     = field(default_factory=dict)


# ── Public entry point ────────────────────────────────────────────────────────

def ingest(file_bytes: bytes, filename: str) -> IngestionResult:
    """Extract text from a résumé file and detect ATS parse hazards.

    Args:
        file_bytes: Raw bytes of the uploaded file.
        filename:   Original filename used to infer format.

    Returns:
        IngestionResult with text, hazard warnings, and metadata.
    """
    ext = _ext(filename)
    if ext == ".pdf":
        return _ingest_pdf(file_bytes)
    elif ext == ".docx":
        return _ingest_docx(file_bytes)
    elif ext in (".txt", ".text", ".md"):
        return _ingest_plain(file_bytes)
    else:
        raise ValueError(
            f"Unsupported format '{ext}'. Upload PDF, DOCX, or TXT."
        )


# ── PDF ───────────────────────────────────────────────────────────────────────

def _ingest_pdf(data: bytes) -> IngestionResult:
    try:
        import pypdf  # type: ignore
    except ImportError:
        raise ImportError("pypdf is required: pip install crawl-jobs")

    reader   = pypdf.PdfReader(io.BytesIO(data))
    hazards  = []
    pages_text: List[str] = []
    image_only_pages = 0
    total_pages = len(reader.pages)

    for i, page in enumerate(reader.pages):
        raw = page.extract_text() or ""
        if not raw.strip():
            image_only_pages += 1
        else:
            pages_text.append(raw)

        # ── Hazard: multi-column layout ───────────────────────────────────
        # Lines with a large mid-line whitespace gap (≥6 spaces) suggest columns
        lines = raw.split("\n")
        column_lines = sum(
            1 for ln in lines
            if re.search(r"\S {6,}\S", ln)
        )
        if column_lines > max(3, len(lines) * 0.25):
            hazards.append(
                f"Page {i+1}: Multi-column layout detected. Many ATS systems "
                "read left-to-right across columns, scrambling the text order."
            )

        # ── Hazard: repeated header/footer text ───────────────────────────
        # First/last line identical across 3+ pages → likely a header/footer
        # (simplified: flag if any single line ≤60 chars repeats on this page top/bottom)

    # ── Hazard: image-only pages ──────────────────────────────────────────
    if image_only_pages > 0:
        hazards.append(
            f"{image_only_pages} page(s) contain no extractable text — they may be "
            "scanned images. ATS systems cannot read image-based résumés."
        )

    # ── Hazard: encryption ────────────────────────────────────────────────
    if reader.is_encrypted:
        hazards.append(
            "PDF is encrypted/password-protected. Most ATS systems will fail "
            "to parse password-protected files."
        )

    text = "\n".join(pages_text).strip()
    if not text:
        raise ValueError(
            "No text could be extracted from the PDF. "
            "If it is a scanned document, please use a text-based PDF."
        )

    # ── Hazard: encoding anomalies ────────────────────────────────────────
    non_ascii = sum(1 for c in text if ord(c) > 127)
    if non_ascii > len(text) * 0.05:
        hazards.append(
            "High proportion of non-ASCII characters detected. Some ATS systems "
            "strip or garble special characters — use standard Latin characters."
        )

    return IngestionResult(
        text=text,
        parse_hazards=hazards,
        metadata={"pages": total_pages, "format": "pdf"},
    )


# ── DOCX ──────────────────────────────────────────────────────────────────────

def _ingest_docx(data: bytes) -> IngestionResult:
    try:
        import docx  # type: ignore
    except ImportError:
        raise ImportError("python-docx is required: pip install crawl-jobs")

    doc     = docx.Document(io.BytesIO(data))
    hazards = []
    parts   = []

    # ── Paragraphs ────────────────────────────────────────────────────────
    for para in doc.paragraphs:
        if para.text.strip():
            parts.append(para.text)

    # ── Hazard: tables ────────────────────────────────────────────────────
    if doc.tables:
        hazards.append(
            f"{len(doc.tables)} table(s) detected. ATS systems often fail to "
            "preserve table structure — consider replacing with plain bullet lists."
        )
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    if cell.text.strip():
                        parts.append(cell.text)

    # ── Hazard: text boxes (inline shapes) ───────────────────────────────
    # Text boxes are stored as w:txbxContent in the XML
    xml_str = doc.element.xml
    if "<w:txbxContent>" in xml_str:
        hazards.append(
            "Text box(es) detected. Text inside text boxes is invisible to most "
            "ATS parsers — move this content into the main document body."
        )

    # ── Hazard: headers/footers with significant content ─────────────────
    for section in doc.sections:
        hdr_text = " ".join(p.text for p in section.header.paragraphs).strip()
        ftr_text = " ".join(p.text for p in section.footer.paragraphs).strip()
        if len(hdr_text) > 20:
            hazards.append(
                "Significant content found in document header. ATS systems "
                "often ignore headers — move critical info to the body."
            )
        if len(ftr_text) > 20:
            hazards.append(
                "Significant content found in document footer. ATS systems "
                "often ignore footers — move critical info to the body."
            )

    text = "\n".join(p for p in parts if p.strip()).strip()
    if not text:
        raise ValueError("No text could be extracted from the DOCX file.")

    return IngestionResult(
        text=text,
        parse_hazards=hazards,
        metadata={"format": "docx", "tables": len(doc.tables)},
    )


# ── Plain text ────────────────────────────────────────────────────────────────

def _ingest_plain(data: bytes) -> IngestionResult:
    try:
        text = data.decode("utf-8").strip()
    except UnicodeDecodeError:
        text = data.decode("latin-1").strip()

    hazards = []
    if re.search(r"\|.+\|", text):
        hazards.append(
            "Pipe-character table formatting detected. "
            "ATS systems may not parse this correctly."
        )
    return IngestionResult(
        text=text,
        parse_hazards=hazards,
        metadata={"format": "txt"},
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ext(filename: str) -> str:
    idx = filename.rfind(".")
    return "" if idx == -1 else filename[idx:].lower()
