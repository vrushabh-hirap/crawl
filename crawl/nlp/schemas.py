"""Schemas and data models for Crawl NLP Lab."""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class ToolOption:
    """Represents a configurable parameter for an NLP tool."""
    name: str
    type: str  # "select" | "boolean" | "text" | "number"
    label: str
    description: str
    default: Any
    choices: Optional[List[Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NLPTool:
    """Definition and metadata for a registered NLP tool."""
    id: str
    name: str
    category: str
    description: str
    icon: str
    badge: str
    status: str = "available"  # "available" | "pending_implementation" | "active"
    options: List[ToolOption] = field(default_factory=list)
    sample_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["options"] = [opt.to_dict() if hasattr(opt, "to_dict") else opt for opt in self.options]
        return data


@dataclass
class NLPRequest:
    """Normalized payload for NLP execution requests."""
    tool_id: str
    text: str
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NLPResponse:
    """Standard response model for NLP tool execution."""
    success: bool
    tool_id: str
    tool_name: str
    status: str  # "completed" | "pending_implementation"
    message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    validation: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
