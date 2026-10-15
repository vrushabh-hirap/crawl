"""Stage 5 — Skill & Keyword Matching.

Major upgrades over the original exact-string matching in scorer.py:

1. Normalized token matching — uses Stage 3 lemmas, so "developing" matches
   "develop" in the JD. Fixes the bug where inflected forms were flagged as missing.

2. Synonym resolution — loads skills_taxonomy.json so "K8s" in the résumé
   counts as a match for "Kubernetes" in the JD, and vice versa.

3. Punctuation fix — strips punctuation before matching, so "apis." in résumé
   correctly matches "apis" in the JD.

4. JD section weighting — keywords from "Required" section weight 1.5x,
   "Preferred" 1.0x, "About"/"Responsibilities" 0.3x (noise reduction).

5. Hard vs soft skill separation — soft skills (communication, leadership, etc.)
   are scored separately since ATS systems treat them differently.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from crawl.jobs.ats.pipeline.stage2_segmentation import SectionMap
from crawl.jobs.ats.pipeline.stage3_normalizer import NormalizerOutput


# ── Data paths ────────────────────────────────────────────────────────────────
_DATA_DIR     = Path(__file__).parent.parent / "data"
_TAXONOMY_FILE = _DATA_DIR / "skills_taxonomy.json"
_SOFT_FILE    = _DATA_DIR / "soft_skills.txt"


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class MatchedSkill:
    jd_term:    str
    resume_term: str           # what was actually found in the résumé
    is_synonym:  bool = False  # True if matched via taxonomy, not verbatim


@dataclass
class WeightedSkill:
    term:   str
    weight: float              # importance from JD section (1.5 = required, 1.0 = preferred)


@dataclass
class SkillMatchResult:
    """Output contract for Stage 5."""
    hard_matched:          List[MatchedSkill]   = field(default_factory=list)
    hard_missing_required: List[WeightedSkill]  = field(default_factory=list)
    hard_missing_preferred: List[WeightedSkill] = field(default_factory=list)
    soft_matched:          List[str]            = field(default_factory=list)
    soft_missing:          List[str]            = field(default_factory=list)
    synonym_matches:       List[MatchedSkill]   = field(default_factory=list)
    required_match_rate:   float                = 0.0
    preferred_match_rate:  float                = 0.0
    overall_keyword_score: float                = 0.0   # 0–30 pts


# ── Singleton data caches ─────────────────────────────────────────────────────

_taxonomy: Optional[Dict[str, List[str]]] = None
_soft_skills: Optional[Set[str]] = None


def _load_taxonomy() -> Dict[str, List[str]]:
    global _taxonomy
    if _taxonomy is None:
        try:
            with open(_TAXONOMY_FILE, encoding="utf-8") as f:
                _taxonomy = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            _taxonomy = {}
    return _taxonomy


def _load_soft_skills() -> Set[str]:
    global _soft_skills
    if _soft_skills is None:
        try:
            with open(_SOFT_FILE, encoding="utf-8") as f:
                _soft_skills = {
                    line.strip().lower() for line in f
                    if line.strip() and not line.startswith("#")
                }
        except FileNotFoundError:
            _soft_skills = set()
    return _soft_skills


# ── Public entry point ────────────────────────────────────────────────────────

def match_skills(
    resume_norm:  NormalizerOutput,
    jd_norm:      NormalizerOutput,
    jd_sections:  SectionMap,
) -> SkillMatchResult:
    """Run the full skill & keyword matching pipeline.

    Args:
        resume_norm:  Stage 3 output for the résumé.
        jd_norm:      Stage 3 output for the job description.
        jd_sections:  Stage 2 output for the JD (for section weighting).

    Returns:
        SkillMatchResult with all match categories and scores.
    """
    taxonomy   = _load_taxonomy()
    soft_skills = _load_soft_skills()

    # Build reverse synonym map: any synonym → canonical term
    # e.g. "k8s" → "kubernetes", "nodejs" → "javascript"
    synonym_to_canonical: Dict[str, str] = {}
    for canonical, synonyms in taxonomy.items():
        for syn in synonyms:
            synonym_to_canonical[_norm_key(syn)] = _norm_key(canonical)
        synonym_to_canonical[_norm_key(canonical)] = _norm_key(canonical)

    # ── Build weighted JD keyword set ─────────────────────────────────────
    weighted_jd_keywords: Dict[str, float] = {}  # normalized_term → weight

    # Section-based weighting
    section_weights = {
        "Required": 1.5,
        "Responsibilities": 0.8,
        "Preferred": 1.0,
        "About": 0.3,
    }
    default_weight = 0.9

    for section_name, block in jd_sections.items():
        weight = section_weights.get(section_name, default_weight)
        section_tokens = _extract_meaningful_tokens(block.raw_text)
        for token in section_tokens:
            norm = _norm_key(token)
            canonical = synonym_to_canonical.get(norm, norm)
            # Take max weight if keyword appears in multiple sections
            if canonical not in weighted_jd_keywords or weight > weighted_jd_keywords[canonical]:
                weighted_jd_keywords[canonical] = weight

    # If JD has no section structure, fall back to all tokens at weight 1.0
    if not weighted_jd_keywords:
        for token in jd_norm.tokens:
            norm = _norm_key(token)
            canonical = synonym_to_canonical.get(norm, norm)
            weighted_jd_keywords[canonical] = 1.0

    # ── Build résumé token set (normalized + canonicalized) ───────────────
    resume_token_set: Set[str] = set()
    resume_raw_to_canonical: Dict[str, str] = {}

    for token in resume_norm.tokens:
        norm = _norm_key(token)
        canonical = synonym_to_canonical.get(norm, norm)
        resume_token_set.add(canonical)
        resume_raw_to_canonical[norm] = canonical

    # ── Classify each JD keyword ──────────────────────────────────────────
    result = SkillMatchResult()

    required_total   = 0
    required_matched = 0
    preferred_total  = 0
    preferred_matched = 0
    weighted_score   = 0.0
    max_possible     = 0.0

    for canonical_term, weight in weighted_jd_keywords.items():
        max_possible += weight
        is_soft = _is_soft_skill(canonical_term, soft_skills)

        # Check if matched in résumé
        matched = canonical_term in resume_token_set
        resume_surface = canonical_term  # what was found

        # Track synonym match details
        is_synonym_match = False
        if matched:
            # Was it matched via a synonym?
            for orig_token in resume_norm.tokens:
                orig_norm = _norm_key(orig_token)
                if synonym_to_canonical.get(orig_norm) == canonical_term and orig_norm != canonical_term:
                    is_synonym_match = True
                    resume_surface = orig_token
                    break

        if matched:
            weighted_score += weight
            ms = MatchedSkill(
                jd_term=canonical_term,
                resume_term=resume_surface,
                is_synonym=is_synonym_match,
            )
            if is_soft:
                result.soft_matched.append(canonical_term)
            else:
                result.hard_matched.append(ms)
                if is_synonym_match:
                    result.synonym_matches.append(ms)
        else:
            if is_soft:
                result.soft_missing.append(canonical_term)
            else:
                ws = WeightedSkill(term=canonical_term, weight=weight)
                if weight >= 1.3:   # required
                    result.hard_missing_required.append(ws)
                    required_total += 1
                else:               # preferred / other
                    result.hard_missing_preferred.append(ws)
                    preferred_total += 1

        # Count for match rates
        if weight >= 1.3:
            required_total += 0  # already counted above
        else:
            preferred_total += 0

    # ── Calculate match rates ─────────────────────────────────────────────
    req_keywords = [t for t, w in weighted_jd_keywords.items() if w >= 1.3]
    pref_keywords = [t for t, w in weighted_jd_keywords.items() if 0.8 <= w < 1.3]

    matched_set = {ms.jd_term for ms in result.hard_matched} | set(result.soft_matched)
    result.required_match_rate = (
        len([k for k in req_keywords if k in matched_set]) / len(req_keywords)
        if req_keywords else 1.0
    )
    result.preferred_match_rate = (
        len([k for k in pref_keywords if k in matched_set]) / len(pref_keywords)
        if pref_keywords else 1.0
    )

    # ── Weighted keyword score (0–30 pts) ─────────────────────────────────
    if max_possible > 0:
        result.overall_keyword_score = min(30.0, (weighted_score / max_possible) * 30)
    else:
        result.overall_keyword_score = 30.0

    # Sort missing by weight descending (most important first)
    result.hard_missing_required.sort(key=lambda x: -x.weight)
    result.hard_missing_preferred.sort(key=lambda x: -x.weight)

    return result


# ── Helpers ───────────────────────────────────────────────────────────────────

def _norm_key(s: str) -> str:
    """Normalize a skill string to a canonical lookup key.

    - Lowercase
    - Strip punctuation (fixes "apis." != "apis" bug)
    - Collapse whitespace
    """
    s = s.lower().strip()
    s = re.sub(r"[^\w\s/#+.-]", "", s)   # strip trailing/leading punctuation
    s = re.sub(r"\s+", " ", s)
    s = s.strip(".")
    return s


def _is_soft_skill(term: str, soft_set: Set[str]) -> bool:
    """Check whether a term is a soft skill."""
    term_lower = term.lower()
    return term_lower in soft_set or any(s in term_lower for s in soft_set)


def _extract_meaningful_tokens(text: str) -> List[str]:
    """Extract non-trivial tokens from raw text (simple, no spaCy dependency)."""
    stop = {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "we", "you", "they", "it", "this", "that", "as", "not",
    }
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9+#./\-]*", text.lower())
    return [t for t in tokens if len(t) >= 2 and t not in stop]
