"""Stage 7 — Scoring & Report Generation.

Computes the weighted composite ATS score from all prior stage outputs
and assembles the full structured report dictionary.

Score weights (total = 100 pts):
  - Keyword match (required + preferred):  30 pts  [was 40]
  - Section coverage:                      20 pts  [was 25]
  - Contact info:                          15 pts  [unchanged]
  - Formatting / parse safety:             10 pts  [unchanged]
  - Quantified achievements:               10 pts  [unchanged]
  - Semantic similarity (optional):        15 pts  [new; redistributed if unavailable]

When semantic scoring is unavailable, its 15 pts are redistributed:
  keyword_match → +7.5, sections → +7.5  (proportional to original weights)

Report shape is backwards-compatible with the v1.0 report EXCEPT:
  - 'recommendations' items are now {category, severity, text} objects
  - New fields: 'entities', 'semantic', 'formatting.parse_hazards'
  - 'keywords' now has 'hard_matched', 'hard_missing_required', etc.
    (legacy 'matched' / 'missing' keys are still included for UI compatibility)
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from crawl.jobs.ats.pipeline.stage1_ingestion  import IngestionResult
from crawl.jobs.ats.pipeline.stage2_segmentation import SectionMap, found_and_missing
from crawl.jobs.ats.pipeline.stage4_ner         import ResumeEntities, WorkEntry
from crawl.jobs.ats.pipeline.stage5_skills       import SkillMatchResult
from crawl.jobs.ats.pipeline.stage6_semantic     import SemanticResult


# ── Quantified achievement regex (same as original scorer.py) ────────────────
_QUANTITY_RE = re.compile(
    r"(?:^|\n)\s*[-•*]\s*.*?\b(\d+[\d,.]*\s*"
    r"(?:%|x|times|percent|million|billion|k|m|b|users|customers|revenue|"
    r"growth|improvement|reduction|increase|decrease|saved|hours|days|"
    r"months|years|people|team|projects|engineers|developers|applications|"
    r"servers|ms|latency))",
    re.IGNORECASE | re.MULTILINE,
)

# ── Contact regexes ───────────────────────────────────────────────────────────
_EMAIL_RE    = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
_PHONE_RE    = re.compile(r"(\+?\d[\d\s\-().]{7,}\d)")
_LINKEDIN_RE = re.compile(r"linkedin\.com/in/[a-zA-Z0-9_-]+", re.IGNORECASE)
_GITHUB_RE   = re.compile(r"github\.com/[a-zA-Z0-9_-]+", re.IGNORECASE)

# ── Formatting heuristics ─────────────────────────────────────────────────────
_TABLE_RE      = re.compile(r"\|.+\|")
_MULTISPACE_RE = re.compile(r"[ \t]{4,}")


# ── Public entry point ────────────────────────────────────────────────────────

def build_report(
    ingestion:    IngestionResult,
    resume_sections: SectionMap,
    jd_sections:  SectionMap,
    entities:     ResumeEntities,
    skills:       SkillMatchResult,
    semantic:     Optional[SemanticResult],
) -> Dict[str, Any]:
    """Assemble the complete ATS report from all pipeline stage outputs.

    Returns a JSON-serializable dictionary.
    """
    resume_text = ingestion.text

    # ── Sub-scores ────────────────────────────────────────────────────────
    kw_score  = skills.overall_keyword_score          # 0–30
    sec_score = _section_score(resume_sections)        # 0–20
    con_score = _contact_score(resume_text)            # 0–15 (returns dict+score)
    fmt_score = _format_score(resume_text, ingestion)  # 0–10 (returns dict+score)
    ach       = _achievement_score(resume_text)        # 0–10 (returns dict+score)
    sem_score = semantic.semantic_score if semantic else 0.0  # 0–15

    # Redistribute semantic pts if unavailable
    if semantic is None:
        kw_score  = min(37.5, kw_score  * (37.5 / 30))
        sec_score = min(27.5, sec_score * (27.5 / 20))
        sem_score = 0.0

    total = round(kw_score + sec_score + con_score["score"] + fmt_score["score"] + ach["score"] + sem_score)
    total = max(0, min(100, total))

    # ── Section map ───────────────────────────────────────────────────────
    sec_info = found_and_missing(resume_sections)

    # ── Backwards-compatible keyword dict ─────────────────────────────────
    matched_names  = [ms.jd_term for ms in skills.hard_matched] + skills.soft_matched
    missing_names  = (
        [ws.term for ws in skills.hard_missing_required] +
        [ws.term for ws in skills.hard_missing_preferred] +
        skills.soft_missing
    )

    # ── Recommendations ───────────────────────────────────────────────────
    recs = _generate_recommendations(
        total, skills, sec_info, con_score, fmt_score, ach,
        semantic, ingestion.parse_hazards, entities,
    )

    return {
        # ── Core score ────────────────────────────────────────────────────
        "ats_score": total,
        "grade":     _grade(total),

        # ── Breakdown ─────────────────────────────────────────────────────
        "breakdown": {
            "keyword_match":  round(kw_score, 1),
            "sections":       round(sec_score, 1),
            "contact_info":   round(con_score["score"], 1),
            "formatting":     round(fmt_score["score"], 1),
            "achievements":   round(ach["score"], 1),
            "semantic_match": round(sem_score, 1),
        },

        # ── Keywords (new + legacy keys for UI) ───────────────────────────
        "keywords": {
            # Legacy keys (UI already renders these)
            "matched": sorted(matched_names),
            "missing": sorted(missing_names),
            "total_jd_keywords": (
                len(skills.hard_matched) + len(skills.hard_missing_required) +
                len(skills.hard_missing_preferred)
            ),
            # New structured keys
            "hard_matched":           [
                {"jd_term": ms.jd_term, "resume_term": ms.resume_term, "synonym": ms.is_synonym}
                for ms in skills.hard_matched
            ],
            "hard_missing_required":  [
                {"term": ws.term, "weight": ws.weight}
                for ws in skills.hard_missing_required
            ],
            "hard_missing_preferred": [
                {"term": ws.term, "weight": ws.weight}
                for ws in skills.hard_missing_preferred
            ],
            "soft_matched":  skills.soft_matched,
            "soft_missing":  skills.soft_missing,
            "synonym_matches": [
                {"resume_term": ms.resume_term, "jd_term": ms.jd_term}
                for ms in skills.synonym_matches
            ],
            "required_match_rate":   round(skills.required_match_rate, 3),
            "preferred_match_rate":  round(skills.preferred_match_rate, 3),
        },

        # ── Sections ──────────────────────────────────────────────────────
        "sections": sec_info,

        # ── Contact info ──────────────────────────────────────────────────
        "contact_info": con_score["info"],

        # ── Formatting ────────────────────────────────────────────────────
        "formatting": {
            "score":      fmt_score["score"],
            "word_count": fmt_score["word_count"],
            "issues":     fmt_score["issues"],
            "parse_hazards": ingestion.parse_hazards,
        },

        # ── Achievements ──────────────────────────────────────────────────
        "achievements": {
            "score":            ach["score"],
            "quantified_count": ach["count"],
        },

        # ── Entities (new) ────────────────────────────────────────────────
        "entities": {
            "work_history": [
                {
                    "company":         e.company,
                    "title":           e.title,
                    "start_date":      e.start_date,
                    "end_date":        e.end_date,
                    "duration_months": e.duration_months,
                    "bullet_count":    len(e.bullets),
                }
                for e in entities.work_history
            ],
            "total_experience_months": entities.total_exp_months,
            "years_of_experience":     entities.years_of_experience,
            "education": [
                {
                    "degree":      e.degree,
                    "institution": e.institution,
                    "year":        e.year,
                    "gpa":         e.gpa,
                }
                for e in entities.education
            ],
            "certifications": entities.certifications,
            "locations":      entities.locations,
        },

        # ── Semantic (new) ────────────────────────────────────────────────
        "semantic": {
            "available":          semantic is not None,
            "overall_similarity": semantic.overall_similarity if semantic else None,
            "section_scores":     semantic.section_scores     if semantic else {},
            "semantic_score":     round(sem_score, 2),
        },

        # ── Structured recommendations (new format) ───────────────────────
        "recommendations": recs,
    }


# ── Recommendation generator ─────────────────────────────────────────────────

def _generate_recommendations(
    total_score, skills, sec_info, con_score, fmt_score, ach,
    semantic, parse_hazards, entities,
) -> List[Dict[str, str]]:
    recs: List[Dict[str, str]] = []

    def add(category: str, severity: str, text: str):
        recs.append({"category": category, "severity": severity, "text": text})

    # ── Parse hazards (highest priority — structural failures) ────────────
    for hazard in parse_hazards:
        add("parse_hazard", "high", hazard)

    # ── Contact info ──────────────────────────────────────────────────────
    info = con_score["info"]
    if not info.get("email"):
        add("contact", "high", "Add your email address — it is required by every ATS system.")
    if not info.get("phone"):
        add("contact", "medium", "Add a phone number so recruiters can reach you directly.")
    if not info.get("linkedin"):
        add("contact", "medium", "Include your LinkedIn profile URL (linkedin.com/in/…) to boost credibility.")

    # ── Missing priority sections ─────────────────────────────────────────
    priority = ["Summary", "Skills", "Experience", "Education"]
    missing_priority = [s for s in priority if s in sec_info.get("missing", [])]
    if missing_priority:
        add("sections", "high",
            f"Add the following missing résumé section(s): {', '.join(missing_priority)}.")

    # ── Required keywords missing ─────────────────────────────────────────
    top_required_missing = [ws.term for ws in skills.hard_missing_required[:10]]
    if top_required_missing:
        add("keyword", "high",
            f"Missing {len(skills.hard_missing_required)} required keyword(s). "
            f"Add these to your résumé: {', '.join(top_required_missing)}.")

    # ── Synonym matches (informational — good news) ───────────────────────
    if skills.synonym_matches:
        examples = [
            f"'{m.resume_term}' → '{m.jd_term}'"
            for m in skills.synonym_matches[:3]
        ]
        add("keyword", "info",
            f"Good news: {len(skills.synonym_matches)} synonym match(es) detected "
            f"({', '.join(examples)}). Consider also spelling out the full terms for maximum ATS compatibility.")

    # ── Preferred keywords missing ────────────────────────────────────────
    top_preferred_missing = [ws.term for ws in skills.hard_missing_preferred[:8]]
    if top_preferred_missing and skills.preferred_match_rate < 0.6:
        add("keyword", "medium",
            f"Consider adding these preferred keywords: {', '.join(top_preferred_missing)}.")

    # ── Match rate feedback ───────────────────────────────────────────────
    if skills.required_match_rate < 0.4:
        add("keyword", "high",
            "Your required-keyword match rate is very low. Mirror exact phrasing from the "
            "'Requirements' section of the JD where possible.")
    elif skills.required_match_rate < 0.65:
        add("keyword", "medium",
            f"Required keyword match rate: {skills.required_match_rate:.0%}. "
            "Tailoring your résumé more closely to this JD's requirements will improve your score.")

    # ── Formatting issues ─────────────────────────────────────────────────
    for issue in fmt_score.get("issues", []):
        add("formatting", "medium", f"Formatting: {issue}")

    # ── Quantified achievements ───────────────────────────────────────────
    if ach["count"] < 3:
        add("content", "medium",
            "Add more quantified achievements (e.g. 'Reduced API latency by 40%'). "
            "Metrics make your impact concrete and ATS ranking algorithms reward them.")

    # ── Work history ──────────────────────────────────────────────────────
    if entities.work_history and entities.years_of_experience > 0:
        entries_with_bullets = [e for e in entities.work_history if e.bullets]
        if len(entries_with_bullets) < len(entities.work_history):
            add("content", "medium",
                "Some work history entries have no bullet points. Add achievement-focused bullets "
                "to each role so ATS systems can extract your contributions.")

    # ── Semantic insight ──────────────────────────────────────────────────
    if semantic and semantic.available:
        sim = semantic.overall_similarity
        if sim < 0.45:
            add("semantic", "high",
                f"Overall semantic similarity to the JD is low ({sim:.0%}). "
                "The overall language and framing of your résumé may not align with this role — "
                "consider rewriting your Summary to mirror the JD's terminology.")
        elif sim < 0.60:
            add("semantic", "medium",
                f"Semantic similarity: {sim:.0%}. Moderate alignment — "
                "adjusting your Experience bullet points to echo JD language will help.")
        else:
            add("semantic", "info",
                f"Strong semantic alignment ({sim:.0%}) — your résumé language closely matches the JD.")

    # ── Overall score comment ─────────────────────────────────────────────
    if total_score < 40:
        add("overall", "high",
            "Overall, your résumé needs significant tailoring. Focus on parse hazards, "
            "missing required keywords, and missing sections first.")
    elif total_score < 65:
        add("overall", "medium",
            "Your résumé is a partial match. Address the required keyword gaps and add "
            "quantified results to meaningfully improve your ATS score.")
    elif total_score >= 85:
        add("overall", "info",
            "Excellent match! Your résumé is well-optimised for this role. "
            "Review the minor gaps above for final polish.")

    return recs


# ── Sub-score helpers ─────────────────────────────────────────────────────────

def _section_score(section_map: SectionMap) -> float:
    core = {"Experience", "Education", "Skills"}
    found = set(section_map.keys())
    core_found  = len(core & found)
    extra_found = max(0, len(found) - core_found)
    score = (core_found / len(core)) * 15 + min(extra_found, 2) * 2.5
    return round(min(20.0, score), 2)


def _contact_score(text: str) -> Dict[str, Any]:
    info = {
        "email":    bool(_EMAIL_RE.search(text)),
        "phone":    bool(_PHONE_RE.search(text)),
        "linkedin": bool(_LINKEDIN_RE.search(text)),
        "github":   bool(_GITHUB_RE.search(text)),
    }
    weights = {"email": 6, "phone": 5, "linkedin": 3, "github": 1}
    score   = min(15.0, sum(w for k, w in weights.items() if info.get(k)))
    return {"score": score, "info": info}


def _format_score(text: str, ingestion: IngestionResult) -> Dict[str, Any]:
    issues = []
    score  = 10

    if _TABLE_RE.search(text):
        issues.append("Pipe-character table formatting detected — may break ATS parsers.")
        score -= 2

    if len(_MULTISPACE_RE.findall(text)) > 5:
        issues.append("Multiple large whitespace gaps detected — may indicate multi-column layout.")
        score -= 2

    word_count = len(text.split())
    if word_count < 200:
        issues.append(f"Résumé appears very short ({word_count} words). Consider adding more detail.")
        score -= 4
    elif word_count > 1500:
        issues.append(f"Résumé is quite long ({word_count} words). ATS systems favour 1–2 pages.")
        score -= 1

    # Deduct for parse hazards found in Stage 1 (cap at -3 for non-content issues)
    hazard_penalty = min(3, len(ingestion.parse_hazards))
    score -= hazard_penalty

    return {"score": max(0, score), "word_count": word_count, "issues": issues}


def _achievement_score(text: str) -> Dict[str, Any]:
    hits  = _QUANTITY_RE.findall(text)
    count = len(hits)
    score = min(10, count * 2)
    return {"score": score, "count": count}


def _grade(score: int) -> str:
    if score >= 85: return "A+"
    if score >= 75: return "A"
    if score >= 65: return "B+"
    if score >= 55: return "B"
    if score >= 45: return "C+"
    if score >= 35: return "C"
    return "D"
