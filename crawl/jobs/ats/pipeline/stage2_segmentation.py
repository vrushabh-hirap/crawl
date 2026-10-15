"""Stage 2 — Section Segmentation.

Detects résumé (and JD) sections using positional/whitespace heuristics
and fuzzy label matching — not just exact string lookup.

Key improvements over the original scorer.py section detection:
- Positional heuristics: short ALL-CAPS or Title-Case lines preceded/followed
  by blank lines → high-confidence headers regardless of exact wording
- Fuzzy matching via simple normalized edit distance so "Professional Background"
  → Experience, "Credentials" → Certifications
- Also segments the JD into Required/Preferred/About us for Stage 5 weighting
- Returns a SectionMap (name → SectionBlock) consumed by all later stages
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class SectionBlock:
    """A detected section with its raw text and line span."""
    name:        str
    raw_text:    str
    line_start:  int
    line_end:    int
    confidence:  float   # 0.0–1.0; >0.7 = high confidence


SectionMap = Dict[str, SectionBlock]


# ── Canonical section labels and their synonym clusters ──────────────────────
_SECTION_ALIASES: Dict[str, List[str]] = {
    "Summary": [
        "summary", "objective", "profile", "about me", "professional summary",
        "career objective", "personal statement", "overview", "introduction",
        "highlights", "professional profile", "bio", "executive summary",
    ],
    "Experience": [
        "experience", "work experience", "employment", "work history",
        "professional experience", "career history", "employment history",
        "professional background", "relevant experience", "work background",
        "career experience", "job history", "positions held",
    ],
    "Education": [
        "education", "academic", "qualifications", "degree", "university",
        "schooling", "academic background", "educational background",
        "academic qualifications", "academic history", "academic credentials",
        "training and education",
    ],
    "Skills": [
        "skills", "technical skills", "core competencies", "competencies",
        "expertise", "technologies", "tools", "technical expertise",
        "key skills", "core skills", "skill set", "skillset",
        "languages and technologies", "tools and technologies",
        "tech stack", "proficiencies", "areas of expertise",
    ],
    "Projects": [
        "projects", "personal projects", "side projects", "portfolio",
        "key projects", "notable projects", "project experience",
        "selected projects", "open source", "independent projects",
    ],
    "Certifications": [
        "certifications", "certificates", "licenses", "credentials",
        "courses", "training", "professional development", "professional certifications",
        "accreditations", "certifications and licenses",
    ],
    "Awards": [
        "awards", "honors", "achievements", "recognition", "accomplishments",
        "honors and awards", "awards and recognition",
    ],
    "Languages": [
        "languages", "language proficiency", "spoken languages",
        "foreign languages", "language skills",
    ],
    "Publications": [
        "publications", "research", "papers", "articles", "patents",
        "research publications",
    ],
    "Volunteer": [
        "volunteer", "volunteering", "community service", "extracurricular",
        "activities", "leadership activities",
    ],
    "References": [
        "references", "referees", "professional references",
    ],
}

# JD section labels for keyword weighting in Stage 5
_JD_SECTION_ALIASES: Dict[str, List[str]] = {
    "Required": [
        "requirements", "required", "must have", "must-have", "qualifications",
        "required qualifications", "minimum qualifications", "what you need",
        "what we require", "you will need", "mandatory", "essential",
        "basic qualifications",
    ],
    "Preferred": [
        "preferred", "nice to have", "nice-to-have", "bonus", "plus",
        "ideally", "desired", "would be great", "additional qualifications",
        "preferred qualifications", "good to have", "not required but",
    ],
    "Responsibilities": [
        "responsibilities", "duties", "what you will do", "what you'll do",
        "role description", "job description", "in this role", "day-to-day",
        "you will be responsible", "key responsibilities",
    ],
    "About": [
        "about us", "about the company", "about the role", "company overview",
        "who we are", "our mission", "the team", "company culture",
        "why join", "what we offer", "benefits", "perks",
    ],
}


# ── Public entry points ───────────────────────────────────────────────────────

def segment_resume(text: str) -> SectionMap:
    """Segment a résumé text into named section blocks."""
    return _segment(text, _SECTION_ALIASES)


def segment_jd(text: str) -> SectionMap:
    """Segment a job description text into named section blocks."""
    return _segment(text, _JD_SECTION_ALIASES)


# ── Core segmentation ─────────────────────────────────────────────────────────

def _segment(text: str, alias_map: Dict[str, List[str]]) -> SectionMap:
    lines = text.split("\n")
    # Find candidate header lines with their confidence
    headers: List[Tuple[int, str, float]] = []  # (line_idx, canonical_name, confidence)

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        label, confidence = _classify_header(stripped, alias_map)
        if label and confidence > 0.4:
            headers.append((i, label, confidence))

    # Build section blocks from consecutive headers
    section_map: SectionMap = {}
    for j, (line_idx, label, conf) in enumerate(headers):
        # Determine where this section ends (start of next header or end of doc)
        end_idx = headers[j + 1][0] if j + 1 < len(headers) else len(lines)
        block_lines = lines[line_idx + 1 : end_idx]
        block_text  = "\n".join(block_lines).strip()

        # If same section label appears twice, merge
        if label in section_map:
            existing = section_map[label]
            section_map[label] = SectionBlock(
                name=label,
                raw_text=existing.raw_text + "\n" + block_text,
                line_start=existing.line_start,
                line_end=end_idx,
                confidence=max(existing.confidence, conf),
            )
        else:
            section_map[label] = SectionBlock(
                name=label,
                raw_text=block_text,
                line_start=line_idx,
                line_end=end_idx,
                confidence=conf,
            )

    # Fallback: if no sections found at all, treat entire text as "Body"
    if not section_map:
        section_map["Body"] = SectionBlock(
            name="Body", raw_text=text, line_start=0,
            line_end=len(lines), confidence=0.0
        )

    return section_map


def _classify_header(line: str, alias_map: Dict[str, List[str]]) -> Tuple[Optional[str], float]:
    """Classify a single line as a section header.

    Returns (canonical_label, confidence) or (None, 0.0).
    Confidence composed of:
      - Exact alias match: 1.0
      - Fuzzy alias match: 0.7–0.9
      - Structural heuristic only (short + uppercase/titlecase): 0.5
    """
    # Strip punctuation from the line for matching
    clean = re.sub(r"[^a-zA-Z0-9 /&-]", "", line).strip().lower()
    if not clean or len(clean.split()) > 8:
        return None, 0.0

    # ── Exact match ──────────────────────────────────────────────────────
    for canonical, aliases in alias_map.items():
        if clean in aliases or clean == canonical.lower():
            return canonical, 1.0

    # ── Fuzzy match (normalized edit distance) ───────────────────────────
    best_label: Optional[str] = None
    best_score = 0.0
    for canonical, aliases in alias_map.items():
        for alias in aliases:
            score = _similarity(clean, alias)
            if score > best_score:
                best_score = score
                best_label = canonical

    if best_score >= 0.72:
        return best_label, best_score * 0.9  # slight penalty for fuzzy

    # ── Structural heuristic ─────────────────────────────────────────────
    # Short (≤5 words), ALL-CAPS or Title-Case → structural hint
    words = clean.split()
    is_short = len(words) <= 5
    is_caps  = line.strip().isupper()
    is_title = line.strip().istitle()

    if is_short and (is_caps or is_title):
        # Return best fuzzy even if below threshold for structural hits
        if best_label and best_score >= 0.50:
            return best_label, best_score * 0.7
        # Unknown section header — label as "Unknown_N"
        return None, 0.0

    return None, 0.0


def _similarity(a: str, b: str) -> float:
    """Normalized edit distance similarity between two strings (0.0–1.0).

    Uses simple dynamic-programming Levenshtein — no external dependency.
    """
    if a == b:
        return 1.0
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 0.0
    # Early exit if lengths too different
    if abs(la - lb) > max(la, lb) * 0.6:
        return 0.0

    # DP table
    dp = list(range(lb + 1))
    for i in range(1, la + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev[j - 1] + cost)

    edit_dist = dp[lb]
    return 1.0 - edit_dist / max(la, lb)


# ── Utility: list found/missing sections ─────────────────────────────────────

def found_and_missing(section_map: SectionMap) -> Dict[str, List[str]]:
    """Return found section names and which standard sections are absent."""
    core_sections = list(_SECTION_ALIASES.keys())
    found   = [s for s in core_sections if s in section_map]
    missing = [s for s in core_sections if s not in section_map]
    return {"found": found, "missing": missing}
