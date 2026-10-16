"""ATSAnalyzer — public facade for the production ATS pipeline.

This orchestrator runs a 7-stage NLP pipeline:

    Stage 1 — Document Ingestion & hazard detection
    Stage 2 — Section Segmentation (résumé + JD)
    Stage 3 — Text Normalization (lemmatization / stemming, acronym expansion)
    Stage 4 — Named Entity Recognition (spaCy + regex gazetteer)
    Stage 5 — Skill & Keyword Matching (taxonomy synonyms, weighted by JD section)
    Stage 6 — Semantic Similarity (sentence-transformers, optional)
    Stage 7 — Scoring & Report Generation

Public API is unchanged — server.py needs no modification:

    analyzer = ATSAnalyzer()
    report   = analyzer.analyze(file_bytes, filename, job_description)
"""

from typing import Any, Dict, Literal, Optional

from crawl.utils import setup_logger

# ── Stage imports ─────────────────────────────────────────────────────────────
from crawl.jobs.ats.pipeline.stage1_ingestion    import ingest
from crawl.jobs.ats.pipeline.stage2_segmentation import segment_resume, segment_jd
from crawl.jobs.ats.pipeline.stage3_normalizer   import normalize, NormalizerConfig
from crawl.jobs.ats.pipeline.stage4_ner          import extract_entities
from crawl.jobs.ats.pipeline.stage5_skills       import match_skills
from crawl.jobs.ats.pipeline.stage6_semantic     import compute_semantic_similarity
from crawl.jobs.ats.pipeline.stage7_reporter     import build_report

logger = setup_logger(__name__)

# Maximum accepted résumé file size (bytes)
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB


class ATSAnalyzer:
    """Orchestrates the 7-stage production ATS analysis pipeline.

    Args:
        normalizer: Text normalization method — "lemmatizer" (default) or "stemmer".
                    See ADR-1 in stage3_normalizer.py for the trade-off rationale.
    """

    def __init__(self, normalizer: Literal["lemmatizer", "stemmer", "wordnet"] = "lemmatizer"):
        self._norm_config = NormalizerConfig(method=normalizer)

    def analyze(
        self,
        file_bytes:      bytes,
        filename:        str,
        job_description: str,
    ) -> Dict[str, Any]:
        """Run the complete ATS analysis pipeline.

        Args:
            file_bytes:      Raw bytes of the uploaded résumé file.
            filename:        Original filename (used to infer format).
            job_description: Plain-text job description from the user.

        Returns:
            Structured ATS report dictionary, ready for JSON serialisation.
            Shape is backwards-compatible with v1.0 + new fields (entities,
            semantic, structured recommendations).

        Raises:
            ValueError: On file-size violation, unsupported format, or text
                        extraction failure.
        """
        # ── Guards ────────────────────────────────────────────────────────
        if len(file_bytes) > MAX_FILE_SIZE:
            raise ValueError(
                f"File too large ({len(file_bytes) // 1024} KB). "
                "Maximum allowed size is 5 MB."
            )
        if not job_description or not job_description.strip():
            raise ValueError("Job description cannot be empty.")

        logger.info(f"ATS analysis started for file: {filename!r}")

        # ── Stage 1: Ingestion ────────────────────────────────────────────
        ingestion = ingest(file_bytes, filename)
        logger.info(
            f"Stage 1 complete: {len(ingestion.text.split())} words extracted, "
            f"{len(ingestion.parse_hazards)} parse hazard(s) detected."
        )

        # ── Stage 2: Section segmentation ────────────────────────────────
        resume_sections = segment_resume(ingestion.text)
        jd_sections     = segment_jd(job_description)
        logger.info(
            f"Stage 2 complete: {len(resume_sections)} résumé section(s) detected — "
            f"{list(resume_sections.keys())}"
        )

        # ── Stage 3: Text normalization ───────────────────────────────────
        resume_norm = normalize(ingestion.text,   self._norm_config)
        jd_norm     = normalize(job_description,  self._norm_config)
        logger.info(
            f"Stage 3 complete: {len(resume_norm.tokens)} résumé tokens, "
            f"{len(jd_norm.tokens)} JD tokens (method={self._norm_config.method})."
        )

        # ── Stage 4: Named entity recognition ────────────────────────────
        entities = extract_entities(resume_sections, ingestion.text)
        logger.info(
            f"Stage 4 complete: {len(entities.work_history)} work entries, "
            f"{len(entities.education)} education entries, "
            f"{entities.years_of_experience} yrs experience."
        )

        # ── Stage 5: Skill & keyword matching ────────────────────────────
        skills = match_skills(resume_norm, jd_norm, jd_sections)
        logger.info(
            f"Stage 5 complete: {len(skills.hard_matched)} hard-skill matches, "
            f"{len(skills.hard_missing_required)} required missing, "
            f"required_rate={skills.required_match_rate:.0%}."
        )

        # ── Stage 6: Semantic similarity (optional) ───────────────────────
        semantic = compute_semantic_similarity(
            ingestion.text, job_description, resume_sections, jd_sections
        )
        if semantic:
            logger.info(
                f"Stage 6 complete: semantic similarity={semantic.overall_similarity:.3f}, "
                f"score={semantic.semantic_score:.1f} pts."
            )
        else:
            logger.info(
                "Stage 6 skipped: sentence-transformers not installed. "
                "Run 'pip install crawl-jobs[semantic]' to enable."
            )

        # ── Stage 7: Scoring & report ─────────────────────────────────────
        report = build_report(
            ingestion=ingestion,
            resume_sections=resume_sections,
            jd_sections=jd_sections,
            entities=entities,
            skills=skills,
            semantic=semantic,
        )

        logger.info(
            f"ATS analysis complete: score={report['ats_score']} "
            f"grade={report['grade']}"
        )
        return report
