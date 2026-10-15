"""Stage 4 — Named Entity Recognition (NER).

ADR-2 Decision: Hybrid approach — spaCy en_core_web_sm + custom regex gazetteer.
Rationale:
  - spaCy en_core_web_sm (~12 MB) handles: ORG (companies), GPE (locations),
    PERSON, CARDINAL out-of-the-box, fully offline.
  - Fine-tuned transformer NER (e.g. dslim/bert-base-NER) adds 400+ MB and
    conflicts with the "runs completely locally with small footprint" positioning.
  - Resume-specific entities (DEGREE, DATE_RANGE, SKILL, JOB_TITLE) are poorly
    handled by generic NER models — custom regex/gazetteers give higher precision
    for these structured, pattern-driven fields.
  - Combined: ~95% of the entity extraction value at ~3% of the model size cost.

Output: ResumeEntities dataclass consumed by Stage 5 and Stage 7.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from crawl.jobs.ats.pipeline.stage2_segmentation import SectionMap


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class WorkEntry:
    """A single work history entry extracted from the Experience section."""
    company:         str
    title:           str
    start_date:      Optional[str] = None
    end_date:        Optional[str] = None
    duration_months: Optional[int] = None
    bullets:         List[str]     = field(default_factory=list)


@dataclass
class EducationEntry:
    """A single education entry."""
    degree:      str
    institution: str
    year:        Optional[str] = None
    gpa:         Optional[str] = None


@dataclass
class ResumeEntities:
    """All structured entities extracted from the résumé."""
    work_history:         List[WorkEntry]     = field(default_factory=list)
    education:            List[EducationEntry] = field(default_factory=list)
    skills_raw:           List[str]            = field(default_factory=list)
    certifications:       List[str]            = field(default_factory=list)
    locations:            List[str]            = field(default_factory=list)
    total_exp_months:     int                  = 0
    years_of_experience:  float                = 0.0


# ── Regex patterns ────────────────────────────────────────────────────────────

# Date range: "Jan 2021 – Present", "2019–2021", "March 2018 - June 2020"
_DATE_RANGE_RE = re.compile(
    r"(?P<start>"
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\.?\s+\d{4}|\d{4}"
    r")\s*"
    r"(?:–|—|-|to)\s*"
    r"(?P<end>"
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\.?\s+\d{4}|\d{4}|[Pp]resent|[Cc]urrent|[Nn]ow"
    r")",
    re.IGNORECASE,
)

# Degree patterns: B.S., M.S., Ph.D., MBA, B.E., M.Tech, B.Tech, B.A., M.A.
_DEGREE_RE = re.compile(
    r"\b("
    r"Ph\.?D\.?|Doctor(?:ate)? of (?:Philosophy|Science)|"
    r"M\.?S\.?|M\.?Sc\.?|Master(?:s)? of (?:Science|Arts|Engineering|Technology|Business)|MBA|M\.?B\.?A\.?|"
    r"B\.?S\.?|B\.?Sc\.?|Bachelor(?:s)? of (?:Science|Arts|Engineering|Technology)|"
    r"B\.?E\.?|B\.?Tech\.?|M\.?Tech\.?|B\.?A\.?|M\.?A\.?|"
    r"Associate(?:s)? (?:of|in|degree)|A\.?S\.?|A\.?A\.?"
    r")\b",
    re.IGNORECASE,
)

# GPA pattern: 3.8/4.0, GPA: 3.9
_GPA_RE = re.compile(r"(?:GPA|gpa|G\.P\.A\.?)\s*:?\s*(\d\.\d+)", re.IGNORECASE)

# Certification patterns (common certifications)
_CERT_RE = re.compile(
    r"\b("
    r"AWS\s+(?:Certified|Solutions\s+Architect|Developer|SysOps|DevOps|Cloud\s+Practitioner)|"
    r"Google\s+Cloud\s+(?:Certified|Professional|Associate)|"
    r"Microsoft\s+(?:Azure\s+)?(?:Certified|MCSA|MCSE|MCP)|"
    r"CKA|CKAD|CKS|Kubernetes\s+(?:Administrator|Developer)|"
    r"PMP|PMI(?:-ACP|-RMP)?|PRINCE2|"
    r"CISSP|CISM|CISA|CEH|CompTIA\s+(?:Security\+|Network\+|A\+|Cloud\+)|"
    r"Scrum\s+Master|CSM|SAFe|ITIL|"
    r"TensorFlow\s+Developer|"
    r"Salesforce\s+(?:Certified|Administrator|Developer)|"
    r"Oracle\s+Certified"
    r")",
    re.IGNORECASE,
)

# Bullet point pattern
_BULLET_RE = re.compile(r"^[\s]*[-•*◦▪▸→>]\s+(.+)$", re.MULTILINE)

# Month name mapping for duration calculation
_MONTH_MAP = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "september": 9, "oct": 10, "october": 10,
    "nov": 11, "november": 11, "dec": 12, "december": 12,
}


# ── Public entry point ────────────────────────────────────────────────────────

def extract_entities(section_map: SectionMap, full_text: str) -> ResumeEntities:
    """Extract structured entities from the résumé section map.

    Args:
        section_map: Section map from Stage 2.
        full_text:   Full extracted text for fallback scanning.

    Returns:
        ResumeEntities with work history, education, skills, etc.
    """
    entities = ResumeEntities()

    # ── Organizations and locations via spaCy ─────────────────────────────
    org_list, loc_list = _extract_spacy_entities(full_text)
    entities.locations = loc_list

    # ── Work history ──────────────────────────────────────────────────────
    exp_text = ""
    for key in ("Experience", "Body"):
        if key in section_map:
            exp_text = section_map[key].raw_text
            break
    if exp_text:
        entities.work_history = _parse_work_history(exp_text, org_list)
        total_months = sum(
            e.duration_months for e in entities.work_history
            if e.duration_months is not None
        )
        entities.total_exp_months = total_months
        entities.years_of_experience = round(total_months / 12, 1)

    # ── Education ─────────────────────────────────────────────────────────
    edu_text = section_map.get("Education", None)
    entities.education = _parse_education(
        edu_text.raw_text if edu_text else "", full_text
    )

    # ── Skills (raw list from Skills section) ─────────────────────────────
    skills_section = section_map.get("Skills", None)
    if skills_section:
        entities.skills_raw = _parse_skills_section(skills_section.raw_text)

    # ── Certifications ────────────────────────────────────────────────────
    cert_section = section_map.get("Certifications", None)
    cert_text = cert_section.raw_text if cert_section else full_text
    entities.certifications = _parse_certifications(cert_text)

    return entities


# ── spaCy NER ─────────────────────────────────────────────────────────────────

def _extract_spacy_entities(text: str):
    """Use spaCy to extract ORG and GPE (location) entities."""
    try:
        import spacy  # type: ignore
        nlp = _get_nlp()
        if nlp is None:
            return [], []
        doc = nlp(text[:100_000])  # cap at 100k chars for speed
        orgs = list({ent.text.strip() for ent in doc.ents if ent.label_ == "ORG"})
        locs = list({ent.text.strip() for ent in doc.ents if ent.label_ in ("GPE", "LOC")})
        return orgs, locs
    except Exception:
        return [], []


_spacy_nlp = None

def _get_nlp():
    global _spacy_nlp
    if _spacy_nlp is None:
        try:
            import spacy  # type: ignore
            _spacy_nlp = spacy.load("en_core_web_sm")
        except Exception:
            _spacy_nlp = False
    return _spacy_nlp if _spacy_nlp is not False else None


# ── Work history parsing ──────────────────────────────────────────────────────

def _parse_work_history(exp_text: str, known_orgs: List[str]) -> List[WorkEntry]:
    """Parse work entries from the Experience section text."""
    entries: List[WorkEntry] = []
    lines = exp_text.split("\n")

    current_entry: Optional[Dict[str, Any]] = None

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Detect date ranges
        date_match = _DATE_RANGE_RE.search(stripped)

        # Check if this line looks like a job entry header
        # (contains a date range AND is not a pure bullet)
        is_bullet = bool(re.match(r"^[-•*◦▪▸→>]\s", stripped))

        if date_match and not is_bullet:
            # Save previous entry
            if current_entry:
                entries.append(_finalize_entry(current_entry))

            start_str = date_match.group("start").strip()
            end_str   = date_match.group("end").strip()
            duration  = _calc_duration(start_str, end_str)

            # Try to extract title from the same line (text before the date)
            title_part = stripped[:date_match.start()].strip()
            title_part = re.sub(r"[|–—-]+$", "", title_part).strip()

            # Company: look in known_orgs or take from line after title
            company = ""
            for org in known_orgs:
                if org.lower() in stripped.lower():
                    company = org
                    break

            current_entry = {
                "title":      title_part or "Unknown Title",
                "company":    company,
                "start_date": start_str,
                "end_date":   end_str,
                "duration_months": duration,
                "bullets":    [],
            }
        elif current_entry is not None and is_bullet:
            # Add bullet to current entry
            bullet_match = _BULLET_RE.match(line)
            if bullet_match:
                current_entry["bullets"].append(bullet_match.group(1).strip())
        elif current_entry is not None and not date_match:
            # Could be company name line (right after title/date line)
            if not current_entry["company"] and len(stripped.split()) <= 6:
                current_entry["company"] = stripped

    # Don't forget the last entry
    if current_entry:
        entries.append(_finalize_entry(current_entry))

    return entries


def _finalize_entry(d: Dict[str, Any]) -> WorkEntry:
    return WorkEntry(
        company=d.get("company", ""),
        title=d.get("title", ""),
        start_date=d.get("start_date"),
        end_date=d.get("end_date"),
        duration_months=d.get("duration_months"),
        bullets=d.get("bullets", []),
    )


def _calc_duration(start: str, end: str) -> Optional[int]:
    """Calculate duration in months between two date strings."""
    import datetime
    try:
        s_month, s_year = _parse_date(start)
        if end.lower() in ("present", "current", "now"):
            e_month = datetime.date.today().month
            e_year  = datetime.date.today().year
        else:
            e_month, e_year = _parse_date(end)

        if s_year is None or e_year is None:
            return None
        months = (e_year - s_year) * 12 + (e_month - s_month)
        return max(0, months)
    except Exception:
        return None


def _parse_date(s: str):
    """Parse a date string into (month, year) tuple."""
    s = s.strip().lower()
    # Year only: "2021"
    if re.match(r"^\d{4}$", s):
        return 1, int(s)
    # Month Year: "jan 2021", "january 2021"
    m = re.match(r"([a-z]+)\.?\s+(\d{4})", s)
    if m:
        month_str = m.group(1)[:3]
        return _MONTH_MAP.get(month_str, 1), int(m.group(2))
    return 1, None


# ── Education parsing ─────────────────────────────────────────────────────────

def _parse_education(edu_text: str, full_text: str) -> List[EducationEntry]:
    """Extract education entries from education section text."""
    text = edu_text or full_text
    entries: List[EducationEntry] = []

    for m in _DEGREE_RE.finditer(text):
        degree = m.group(0)
        # Grab surrounding context (±100 chars) for institution + year
        start = max(0, m.start() - 20)
        end   = min(len(text), m.end() + 120)
        ctx   = text[start:end]

        # Year
        year_m = re.search(r"\b(20\d\d|19\d\d)\b", ctx)
        year = year_m.group(0) if year_m else None

        # GPA
        gpa_m = _GPA_RE.search(ctx)
        gpa = gpa_m.group(1) if gpa_m else None

        # Institution heuristic: look for "University", "College", "Institute", "School"
        inst_m = re.search(
            r"([A-Z][a-zA-Z ]+(?:University|College|Institute|School|Academy)[a-zA-Z ]*)",
            ctx,
        )
        institution = inst_m.group(0).strip() if inst_m else ""

        entries.append(EducationEntry(
            degree=degree, institution=institution, year=year, gpa=gpa
        ))

    return entries


# ── Skills section parsing ────────────────────────────────────────────────────

def _parse_skills_section(skills_text: str) -> List[str]:
    """Extract individual skill tokens from the Skills section."""
    # Skills are often comma or pipe separated, or one per line
    skills = []
    for line in skills_text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        # Split by common delimiters: comma, pipe, bullet, semicolon
        parts = re.split(r"[,|•;·/]", stripped)
        for part in parts:
            clean = part.strip().strip("-").strip()
            if clean and 2 <= len(clean) <= 40:
                skills.append(clean)
    return skills


# ── Certification parsing ─────────────────────────────────────────────────────

def _parse_certifications(text: str) -> List[str]:
    """Extract certification names."""
    matches = _CERT_RE.findall(text)
    return list({m.strip() for m in matches if m.strip()})
