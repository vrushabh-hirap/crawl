"""ATS scoring engine.

Performs offline NLP analysis — no external API calls.
Produces a structured score breakdown from résumé text and a job description.
"""

import re
from typing import Dict, List, Set, Tuple, Any


# ── Stop-words to exclude from keyword extraction ────────────────────────────
_STOP_WORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "up", "about", "into", "through", "during",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might",
    "shall", "can", "need", "dare", "ought", "used", "not", "no", "nor",
    "so", "yet", "both", "either", "neither", "each", "few", "more", "most",
    "other", "some", "such", "than", "too", "very", "just", "because",
    "as", "until", "while", "that", "this", "these", "those", "it", "its",
    "they", "them", "their", "we", "our", "you", "your", "he", "she", "his",
    "her", "who", "what", "which", "when", "where", "why", "how", "all",
    "any", "both", "each", "here", "there", "then", "once", "also", "i",
    "me", "my", "myself", "us", "am",
}

# ── Standard résumé section headings ─────────────────────────────────────────
_SECTION_PATTERNS: Dict[str, List[str]] = {
    "Summary":         ["summary", "objective", "profile", "about me", "professional summary", "career objective"],
    "Experience":      ["experience", "work experience", "employment", "work history", "professional experience", "career history"],
    "Education":       ["education", "academic", "qualifications", "degree", "university", "schooling"],
    "Skills":          ["skills", "technical skills", "core competencies", "competencies", "expertise", "technologies"],
    "Projects":        ["projects", "personal projects", "side projects", "portfolio", "key projects"],
    "Certifications":  ["certifications", "certificates", "licenses", "credentials", "courses", "training"],
    "Awards":          ["awards", "honors", "achievements", "recognition", "accomplishments"],
    "Languages":       ["languages", "language proficiency"],
    "References":      ["references", "referees"],
}

# ── Contact info patterns ─────────────────────────────────────────────────────
_EMAIL_RE    = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
_PHONE_RE    = re.compile(r"(\+?\d[\d\s\-().]{7,}\d)")
_LINKEDIN_RE = re.compile(r"linkedin\.com/in/[a-zA-Z0-9_-]+", re.IGNORECASE)
_GITHUB_RE   = re.compile(r"github\.com/[a-zA-Z0-9_-]+", re.IGNORECASE)

# ── Quantified achievement pattern (numbers in bullet points) ─────────────────
_QUANTITY_RE = re.compile(
    r"(?:^|\n)\s*[-•*]\s*.*?\b(\d+[\d,.]*\s*(?:%|x|times|percent|million|billion|k|m|b|users|customers|revenue|growth|improvement|reduction|increase|decrease|saved|hours|days|months|years|people|team|projects|engineers|developers|applications|servers))",
    re.IGNORECASE | re.MULTILINE,
)

# ── Table / multi-column heuristic ───────────────────────────────────────────
_TABLE_RE    = re.compile(r"\|.+\|")
_MULTISPACE_RE = re.compile(r"[ \t]{4,}")


def score(resume_text: str, job_description: str) -> Dict[str, Any]:
    """Run the full ATS scoring pipeline.

    Args:
        resume_text:     Extracted plain text from the résumé.
        job_description: Raw job description text pasted by the user.

    Returns:
        A dictionary containing all scoring dimensions.
    """
    jd_keywords    = _extract_keywords(job_description)
    resume_lower   = resume_text.lower()

    matched, missing = _match_keywords(jd_keywords, resume_lower)
    sections         = _detect_sections(resume_lower)
    contact          = _detect_contact(resume_text)
    formatting       = _score_formatting(resume_text)
    achievements     = _score_achievements(resume_text)

    # ── Weighted composite score ──────────────────────────────────────────────
    kw_score  = _keyword_score(matched, jd_keywords)        # 0–40
    sec_score = _section_score(sections["found"])           # 0–25
    con_score = _contact_score(contact)                     # 0–15
    fmt_score = formatting["score"]                         # 0–10
    ach_score = achievements["score"]                       # 0–10

    total = round(kw_score + sec_score + con_score + fmt_score + ach_score)
    total = max(0, min(100, total))

    return {
        "ats_score":   total,
        "grade":       _grade(total),
        "breakdown": {
            "keyword_match":   round(kw_score, 1),
            "sections":        round(sec_score, 1),
            "contact_info":    round(con_score, 1),
            "formatting":      round(fmt_score, 1),
            "achievements":    round(ach_score, 1),
        },
        "keywords": {
            "matched": sorted(matched),
            "missing": sorted(missing),
            "total_jd_keywords": len(jd_keywords),
        },
        "sections":       sections,
        "contact_info":   contact,
        "formatting":     formatting,
        "achievements":   achievements,
    }


# ── Keyword extraction ────────────────────────────────────────────────────────

def _extract_keywords(text: str) -> Set[str]:
    """Tokenize JD text and return significant, normalised keywords."""
    # Extract multi-word technical phrases first (e.g. "machine learning", "ci/cd")
    text_lower = text.lower()
    tokens: Set[str] = set()

    # Phrase patterns: 2–4 word sequences of non-stop words
    words = re.findall(r"[a-z][a-z0-9+#./\-]*", text_lower)
    for w in words:
        if len(w) >= 2 and w not in _STOP_WORDS:
            tokens.add(w)

    # Also capture hyphenated and slash-joined terms
    phrases = re.findall(r"[a-z][a-z0-9]+(?:[+#./\-][a-z0-9]+)+", text_lower)
    tokens.update(phrases)

    return tokens


def _match_keywords(jd_keywords: Set[str], resume_lower: str) -> Tuple[List[str], List[str]]:
    """Split JD keywords into those found/missing in the résumé."""
    matched = []
    missing = []
    for kw in jd_keywords:
        # Use word-boundary matching for short tokens; substring for longer phrases
        if len(kw) <= 3:
            pattern = rf"\b{re.escape(kw)}\b"
        else:
            pattern = re.escape(kw)
        if re.search(pattern, resume_lower):
            matched.append(kw)
        else:
            missing.append(kw)
    return matched, missing


# ── Section detection ─────────────────────────────────────────────────────────

def _detect_sections(resume_lower: str) -> Dict[str, List[str]]:
    """Check which standard résumé sections are present or absent."""
    found   = []
    missing = []
    for section, patterns in _SECTION_PATTERNS.items():
        detected = any(p in resume_lower for p in patterns)
        if detected:
            found.append(section)
        else:
            missing.append(section)
    return {"found": found, "missing": missing}


# ── Contact info detection ────────────────────────────────────────────────────

def _detect_contact(resume_text: str) -> Dict[str, bool]:
    return {
        "email":    bool(_EMAIL_RE.search(resume_text)),
        "phone":    bool(_PHONE_RE.search(resume_text)),
        "linkedin": bool(_LINKEDIN_RE.search(resume_text)),
        "github":   bool(_GITHUB_RE.search(resume_text)),
    }


# ── Formatting score ──────────────────────────────────────────────────────────

def _score_formatting(resume_text: str) -> Dict[str, Any]:
    """Heuristically check for ATS-unfriendly formatting patterns."""
    issues = []
    score  = 10  # start full, deduct per issue

    if _TABLE_RE.search(resume_text):
        issues.append("Possible table detected — tables may not parse correctly in ATS systems.")
        score -= 4

    if len(_MULTISPACE_RE.findall(resume_text)) > 5:
        issues.append("Multiple large whitespace gaps detected — may indicate multi-column layout.")
        score -= 3

    # Check if résumé is very short (< 200 words)
    word_count = len(resume_text.split())
    if word_count < 200:
        issues.append(f"Résumé appears very short ({word_count} words). Consider adding more detail.")
        score -= 3
    elif word_count > 1500:
        issues.append(f"Résumé is quite long ({word_count} words). ATS systems favour 1–2 pages.")
        score -= 1

    return {
        "score":      max(0, score),
        "word_count": word_count,
        "issues":     issues,
    }


# ── Quantified achievements ───────────────────────────────────────────────────

def _score_achievements(resume_text: str) -> Dict[str, Any]:
    """Count bullet points that contain measurable, quantified results."""
    hits = _QUANTITY_RE.findall(resume_text)
    count = len(hits)
    # 2 pts per quantified bullet, capped at 10
    score = min(10, count * 2)
    return {
        "score":            score,
        "quantified_count": count,
    }


# ── Sub-scores ────────────────────────────────────────────────────────────────

def _keyword_score(matched: List[str], jd_keywords: Set[str]) -> float:
    if not jd_keywords:
        return 40.0
    ratio = len(matched) / len(jd_keywords)
    return round(ratio * 40, 2)


def _section_score(found: List[str]) -> float:
    # Core sections worth most; extras are bonus
    core = {"Experience", "Education", "Skills"}
    core_found = len(core & set(found))
    extra_found = max(0, len(found) - core_found)
    score = (core_found / len(core)) * 20 + min(extra_found, 2) * 2.5
    return round(score, 2)


def _contact_score(contact: Dict[str, bool]) -> float:
    weights = {"email": 6, "phone": 5, "linkedin": 3, "github": 1}
    total   = sum(w for k, w in weights.items() if contact.get(k))
    return min(15.0, float(total))


def _grade(score: int) -> str:
    if score >= 85:  return "A+"
    if score >= 75:  return "A"
    if score >= 65:  return "B+"
    if score >= 55:  return "B"
    if score >= 45:  return "C+"
    if score >= 35:  return "C"
    return "D"
