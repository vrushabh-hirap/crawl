"""Data models representing search parameters and normalized job listings."""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any

@dataclass
class JobQuery:
    """Represents the structured search parameters entered by the user."""
    title: str = ""
    work_modes: List[str] = field(default_factory=list)
    location: str = ""
    skills: List[str] = field(default_factory=list)
    recency: str = ""  # e.g., "24h", "week", "month"

    def to_dict(self) -> Dict[str, Any]:
        """Converts query properties to a dictionary."""
        return asdict(self)


@dataclass
class JobResult:
    """Represents a normalized job search result."""
    title: str
    company: str
    location: str
    work_mode: str
    source: str
    posting_date: str
    link: str

    def to_dict(self) -> Dict[str, Any]:
        """Converts result properties to a dictionary."""
        return asdict(self)
