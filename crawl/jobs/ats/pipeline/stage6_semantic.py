"""Stage 6 — Semantic Similarity Scoring (optional).

ADR-3 Decision: sentence-transformers with all-MiniLM-L6-v2.
Rationale:
  - Best offline sentence-level embeddings (~90 MB model, cached after first DL).
  - Cosine similarity of sentence embeddings catches semantic matches that
    keyword matching misses: "led a 12-person team" ↔ "team management".
  - Strictly optional: if sentence-transformers is not installed, this stage
    returns None and the final score simply redistributes its weight.
  - Does NOT call external APIs — model runs 100% locally after first download.
  - Flagged: requires ~90 MB download on first run (cached in ~/.cache/).

Install:
    pip install crawl-jobs[semantic]
    # or: pip install sentence-transformers

First run warning:
    Logs a warning about the initial model download so users aren't surprised.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_MODEL_NAME  = "all-MiniLM-L6-v2"
_model_cache = None         # module-level singleton


@dataclass
class SemanticResult:
    """Output contract for Stage 6."""
    overall_similarity: float              # 0.0 – 1.0 cosine similarity
    section_scores:     Dict[str, float]   = field(default_factory=dict)
    semantic_score:     float              = 0.0    # 0–15 pts contribution
    available:          bool               = True


# ── Public entry point ────────────────────────────────────────────────────────

def compute_semantic_similarity(
    resume_text: str,
    jd_text:     str,
    resume_sections: Optional[Dict] = None,
    jd_sections:     Optional[Dict] = None,
) -> Optional[SemanticResult]:
    """Compute semantic similarity between résumé and JD.

    Returns:
        SemanticResult if sentence-transformers is available, else None.
    """
    model = _load_model()
    if model is None:
        return None

    try:
        from sentence_transformers import util  # type: ignore

        # ── Whole-document similarity ─────────────────────────────────────
        # Cap text to 5000 chars each for speed (MiniLM max is 256 tokens)
        res_text_short = _truncate(resume_text, 5000)
        jd_text_short  = _truncate(jd_text,     5000)

        emb_resume = model.encode(res_text_short, convert_to_tensor=True)
        emb_jd     = model.encode(jd_text_short,  convert_to_tensor=True)
        overall    = float(util.cos_sim(emb_resume, emb_jd)[0][0])

        # ── Section-level similarity ──────────────────────────────────────
        section_scores: Dict[str, float] = {}
        if resume_sections and jd_sections:
            exp_block  = resume_sections.get("Experience")
            req_block  = jd_sections.get("Required") or jd_sections.get("Responsibilities")

            if exp_block and req_block:
                emb_exp = model.encode(
                    _truncate(exp_block.raw_text, 2000), convert_to_tensor=True
                )
                emb_req = model.encode(
                    _truncate(req_block.raw_text, 2000), convert_to_tensor=True
                )
                section_scores["experience_vs_requirements"] = float(
                    util.cos_sim(emb_exp, emb_req)[0][0]
                )

        # ── Scale to 0–15 pts ─────────────────────────────────────────────
        # Similarity of 0.5+ is meaningful; 0.85+ is excellent
        # Map [0.4, 0.9] → [0, 15]
        clamped = max(0.0, min(1.0, (overall - 0.4) / 0.5))
        semantic_score = round(clamped * 15, 2)

        return SemanticResult(
            overall_similarity=round(overall, 4),
            section_scores=section_scores,
            semantic_score=semantic_score,
            available=True,
        )

    except Exception as e:
        logger.warning(f"Semantic similarity computation failed: {e}")
        return None


# ── Model loader ──────────────────────────────────────────────────────────────

def _load_model():
    global _model_cache
    if _model_cache is not None:
        return _model_cache if _model_cache is not False else None

    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        logger.info(
            f"Loading semantic model '{_MODEL_NAME}'. "
            "First run may download ~90 MB — subsequent runs are instant."
        )
        _model_cache = SentenceTransformer(_MODEL_NAME)
        logger.info("Semantic model loaded successfully.")
        return _model_cache

    except ImportError:
        logger.info(
            "sentence-transformers not installed — semantic scoring disabled. "
            "Run 'pip install crawl-jobs[semantic]' to enable."
        )
        _model_cache = False
        return None
    except Exception as e:
        logger.warning(f"Failed to load semantic model: {e}")
        _model_cache = False
        return None


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."
