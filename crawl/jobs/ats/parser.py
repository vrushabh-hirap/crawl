"""Resume text extractor.

Supports PDF (.pdf), DOCX (.docx), and plain text (.txt / .text) files.
Returns the raw extracted text as a single string.
"""

import io
from typing import Optional


def extract_text(file_bytes: bytes, filename: str) -> str:
    """Extract plain text from a résumé file.

    Args:
        file_bytes: Raw bytes of the uploaded file.
        filename:   Original filename (used to determine format).

    Returns:
        Extracted plain-text content of the résumé.

    Raises:
        ValueError: If the file format is unsupported or extraction fails.
    """
    ext = _get_extension(filename)

    if ext == ".pdf":
        return _extract_pdf(file_bytes)
    elif ext == ".docx":
        return _extract_docx(file_bytes)
    elif ext in (".txt", ".text", ".md"):
        return _extract_plain(file_bytes)
    else:
        raise ValueError(
            f"Unsupported file format '{ext}'. "
            "Please upload a PDF, DOCX, or TXT file."
        )


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_extension(filename: str) -> str:
    """Return the lowercase file extension including the leading dot."""
    idx = filename.rfind(".")
    if idx == -1:
        return ""
    return filename[idx:].lower()


def _extract_pdf(data: bytes) -> str:
    """Extract text from a PDF using pypdf."""
    try:
        import pypdf  # type: ignore
    except ImportError:
        raise ImportError(
            "pypdf is required for PDF parsing. "
            "Install via: pip install crawl-jobs"
        )

    reader = pypdf.PdfReader(io.BytesIO(data))
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)

    result = "\n".join(pages).strip()
    if not result:
        raise ValueError(
            "Could not extract any text from the PDF. "
            "The file may be image-only (scanned). Please use a text-based PDF."
        )
    return result


def _extract_docx(data: bytes) -> str:
    """Extract text from a DOCX file using python-docx."""
    try:
        import docx  # type: ignore
    except ImportError:
        raise ImportError(
            "python-docx is required for DOCX parsing. "
            "Install via: pip install crawl-jobs"
        )

    doc = docx.Document(io.BytesIO(data))
    paragraphs = [para.text for para in doc.paragraphs]
    # Also pull text from tables
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                paragraphs.append(cell.text)

    result = "\n".join(p for p in paragraphs if p.strip()).strip()
    if not result:
        raise ValueError("Could not extract any text from the DOCX file.")
    return result


def _extract_plain(data: bytes) -> str:
    """Decode plain-text bytes, trying UTF-8 then latin-1 as fallback."""
    try:
        return data.decode("utf-8").strip()
    except UnicodeDecodeError:
        return data.decode("latin-1").strip()
