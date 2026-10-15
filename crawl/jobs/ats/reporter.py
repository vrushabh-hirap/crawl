"""ATS report builder.

Delegates to Stage 7 reporter for backward compatibility.
"""

from typing import Any, Dict
from crawl.jobs.ats.pipeline.stage7_reporter import build_report as stage7_build_report


def build_report(data: Dict[str, Any]) -> Dict[str, Any]:
    """Backward compatibility shim for report generation.

    If data is already a full report dictionary from stage7, returns it.
    """
    if "recommendations" in data and "breakdown" in data:
        return data
    return data
