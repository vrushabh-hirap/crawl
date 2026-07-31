"""ATS report builder.

Assembles the raw score data from scorer.py into a user-friendly,
fully structured report dictionary with actionable recommendations.
"""

from typing import Any, Dict, List


def build_report(score_data: Dict[str, Any]) -> Dict[str, Any]:
    """Transform raw scoring output into the final ATS report.

    Args:
        score_data: Dict returned by ``scorer.score()``.

    Returns:
        Enriched report dict ready for JSON serialisation.
    """
    recommendations = _generate_recommendations(score_data)
    return {**score_data, "recommendations": recommendations}


# ── Recommendations generator ─────────────────────────────────────────────────

def _generate_recommendations(data: Dict[str, Any]) -> List[str]:
    recs: List[str] = []

    # ── Contact info ─────────────────────────────────────────────────────────
    contact = data.get("contact_info", {})
    if not contact.get("email"):
        recs.append("Add your email address to the résumé — it is required by every ATS.")
    if not contact.get("phone"):
        recs.append("Add a phone number so recruiters can reach you directly.")
    if not contact.get("linkedin"):
        recs.append("Include your LinkedIn profile URL (linkedin.com/in/…) to boost credibility.")

    # ── Missing sections ──────────────────────────────────────────────────────
    sections_missing = data.get("sections", {}).get("missing", [])
    priority = ["Summary", "Skills", "Experience", "Education"]
    missing_priority = [s for s in priority if s in sections_missing]
    if missing_priority:
        recs.append(
            f"Add the following missing résumé section(s): {', '.join(missing_priority)}."
        )

    # ── Missing keywords ──────────────────────────────────────────────────────
    missing_kw = data.get("keywords", {}).get("missing", [])
    # Only surface the top-15 most impactful missing keywords to avoid overwhelm
    top_missing = [k for k in missing_kw if len(k) >= 4][:15]
    if top_missing:
        recs.append(
            f"Incorporate these keywords from the job description into your résumé: "
            f"{', '.join(top_missing)}."
        )

    # ── Keyword match rate ────────────────────────────────────────────────────
    total_kw = data.get("keywords", {}).get("total_jd_keywords", 0)
    matched  = data.get("keywords", {}).get("matched", [])
    if total_kw > 0:
        ratio = len(matched) / total_kw
        if ratio < 0.3:
            recs.append(
                "Your keyword match rate is low. Mirror the exact phrasing from the job description "
                "where appropriate — ATS systems look for verbatim matches."
            )

    # ── Formatting ────────────────────────────────────────────────────────────
    fmt_issues = data.get("formatting", {}).get("issues", [])
    for issue in fmt_issues:
        recs.append(f"Formatting: {issue}")

    # ── Quantified achievements ───────────────────────────────────────────────
    ach_count = data.get("achievements", {}).get("quantified_count", 0)
    if ach_count < 3:
        recs.append(
            "Add more quantified achievements (e.g. 'Improved API response time by 40%'). "
            "Numbers make your impact concrete and memorable."
        )

    # ── Overall score advice ──────────────────────────────────────────────────
    score = data.get("ats_score", 0)
    if score < 40:
        recs.append(
            "Overall, your résumé needs significant tailoring for this role. "
            "Focus on the keyword gaps and missing sections first."
        )
    elif score < 65:
        recs.append(
            "Your résumé is a partial match. Address the keyword gaps and add quantified results "
            "to meaningfully improve your ATS score."
        )
    elif score >= 85:
        recs.append(
            "Excellent match! Your résumé is well-optimised for this role. "
            "Review the minor gaps above for final polish."
        )

    return recs
