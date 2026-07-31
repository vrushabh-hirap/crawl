"""ATSAnalyzer — public facade for the ATS pipeline.

Usage::

    from crawl.jobs.ats import ATSAnalyzer

    analyzer = ATSAnalyzer()
    with open("resume.pdf", "rb") as f:
        report = analyzer.analyze(f.read(), "resume.pdf", job_description_text)
    print(report)
"""

from typing import Any, Dict

from crawl.jobs.ats.parser   import extract_text
from crawl.jobs.ats.scorer   import score
from crawl.jobs.ats.reporter import build_report
from crawl.utils import setup_logger

logger = setup_logger(__name__)

# Maximum accepted résumé file size (bytes)
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB


class ATSAnalyzer:
    """Orchestrates résumé parsing → scoring → report generation."""

    def analyze(
        self,
        file_bytes: bytes,
        filename: str,
        job_description: str,
    ) -> Dict[str, Any]:
        """Run the complete ATS analysis pipeline.

        Args:
            file_bytes:      Raw bytes of the uploaded résumé file.
            filename:        Original filename (used to infer format).
            job_description: Plain-text job description from the user.

        Returns:
            Structured ATS report dictionary, ready for JSON serialisation.

        Raises:
            ValueError: On file-size violation or extraction failure.
        """
        # ── Guard: file size ──────────────────────────────────────────────────
        if len(file_bytes) > MAX_FILE_SIZE:
            raise ValueError(
                f"File too large ({len(file_bytes) // 1024} KB). "
                "Maximum allowed size is 5 MB."
            )

        # ── Guard: JD not empty ───────────────────────────────────────────────
        if not job_description or not job_description.strip():
            raise ValueError("Job description cannot be empty.")

        logger.info(f"ATS analysis started for file: {filename!r}")

        # ── Step 1: Parse résumé ──────────────────────────────────────────────
        resume_text = extract_text(file_bytes, filename)
        logger.info(f"Extracted {len(resume_text.split())} words from résumé.")

        # ── Step 2: Score ─────────────────────────────────────────────────────
        score_data = score(resume_text, job_description)
        logger.info(f"ATS score: {score_data['ats_score']} ({score_data['grade']})")

        # ── Step 3: Build report ──────────────────────────────────────────────
        report = build_report(score_data)

        return report
