"""Service layer for NLP Lab: validation, tool registry, and execution dispatch."""

from typing import Any, Callable, Dict, Optional, Tuple
from crawl.nlp.registry import get_tool_by_id, get_tool_registry
from crawl.nlp.schemas import NLPResponse
from crawl.utils import setup_logger

logger = setup_logger(__name__)

# Registry for connecting tool implementation handlers as they are developed
_TOOL_HANDLERS: Dict[str, Callable[[str, Dict[str, Any]], Dict[str, Any]]] = {}


def register_tool_handler(tool_id: str, handler: Callable[[str, Dict[str, Any]], Dict[str, Any]]) -> None:
    """Registers an implementation handler for a specific NLP tool.

    Args:
        tool_id: The identifier of the tool (e.g. 'text_analysis', 'stem_lemma').
        handler: Callable accepting (text, options) and returning a result dictionary.
    """
    tool = get_tool_by_id(tool_id)
    if not tool:
        raise ValueError(f"Cannot register handler for unknown tool_id: {tool_id!r}")
    _TOOL_HANDLERS[tool_id] = handler
    logger.info(f"Registered NLP tool handler for {tool_id!r}")


def unregister_tool_handler(tool_id: str) -> None:
    """Removes a registered tool handler."""
    _TOOL_HANDLERS.pop(tool_id, None)


def is_tool_connected(tool_id: str) -> bool:
    """Checks whether an implementation handler is registered for a given tool."""
    return tool_id in _TOOL_HANDLERS


def validate_nlp_request(
    tool_id: str,
    text: str,
    options: Optional[Dict[str, Any]] = None
) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    """Validates the tool ID, input text, and tool-specific options.

    Args:
        tool_id: Unique string identifying the tool.
        text: Input string to process.
        options: Optional configuration dictionary.

    Returns:
        A tuple of (is_valid, error_message, validated_options).
    """
    tool = get_tool_by_id(tool_id)
    if not tool:
        available = [t.id for t in get_tool_registry()]
        return False, f"Unknown NLP tool: {tool_id!r}. Supported tools: {', '.join(available)}", {}

    if not isinstance(text, str) or not text.strip():
        return False, "Input text is required and cannot be empty.", {}

    raw_options = options if isinstance(options, dict) else {}
    validated_options: Dict[str, Any] = {}

    # Validate against tool option specifications
    for opt in tool.options:
        val = raw_options.get(opt.name, opt.default)

        if opt.type == "boolean":
            if isinstance(val, bool):
                validated_options[opt.name] = val
            elif isinstance(val, str):
                validated_options[opt.name] = val.lower() in ("true", "1", "yes")
            else:
                validated_options[opt.name] = bool(val)

        elif opt.type == "select":
            if opt.choices is not None:
                # Convert both for loose string matching (e.g. 1 vs "1")
                str_choices = [str(c) for c in opt.choices]
                if str(val) not in str_choices:
                    choices_str = ", ".join(f"{c!r}" for c in opt.choices)
                    return (
                        False,
                        f"Invalid value for option '{opt.name}': {val!r}. Allowed choices: [{choices_str}]",
                        {}
                    )
                # Preserve int type if choice was originally int
                if isinstance(opt.choices[0], int):
                    try:
                        val = int(val)
                    except (ValueError, TypeError):
                        pass
            validated_options[opt.name] = val

        elif opt.type == "number":
            try:
                validated_options[opt.name] = int(val)
            except (ValueError, TypeError):
                try:
                    validated_options[opt.name] = float(val)
                except (ValueError, TypeError):
                    return False, f"Option '{opt.name}' must be a valid number, got {val!r}", {}

        elif opt.type == "text":
            validated_options[opt.name] = str(val) if val is not None else ""

        else:
            validated_options[opt.name] = val

    return True, None, validated_options


def execute_nlp_tool(
    tool_id: str,
    text: str,
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Dispatches execution to the registered tool handler or returns structured validation status.

    Args:
        tool_id: Tool identifier.
        text: Input string.
        options: Tool options.

    Returns:
        Structured response dictionary.
    """
    is_valid, err_msg, validated_options = validate_nlp_request(tool_id, text, options)
    if not is_valid:
        raise ValueError(err_msg or "Invalid NLP request")

    tool = get_tool_by_id(tool_id)
    assert tool is not None

    if is_tool_connected(tool_id):
        handler = _TOOL_HANDLERS[tool_id]
        logger.info(f"Executing registered NLP tool handler for: {tool_id}")
        try:
            result_data = handler(text, validated_options)
            resp = NLPResponse(
                success=True,
                tool_id=tool.id,
                tool_name=tool.name,
                status="completed",
                result=result_data,
                validation={
                    "text_characters": len(text),
                    "text_words": len(text.split()),
                    "options": validated_options,
                }
            )
            return resp.to_dict()
        except Exception as e:
            logger.exception(f"Error during execution of tool '{tool_id}'")
            return {
                "success": False,
                "tool_id": tool.id,
                "tool_name": tool.name,
                "status": "error",
                "error": f"Tool execution failed: {str(e)}"
            }

    # Tool is defined and validated, but computational engine is scheduled for future task
    logger.info(f"Tool {tool_id!r} validated; implementation scheduled for upcoming task.")
    resp = NLPResponse(
        success=True,
        tool_id=tool.id,
        tool_name=tool.name,
        status="pending_implementation",
        message=(
            f"Tool '{tool.name}' route and input schema validation are active. "
            f"The computational implementation is queued to be connected in the following task."
        ),
        result=None,
        validation={
            "text_characters": len(text),
            "text_words": len(text.split()),
            "options": validated_options,
        }
    )
    return resp.to_dict()


# ── Register Built-in Handlers ────────────────────────────────────────────────
from crawl.nlp.tools.text_analysis import analyze_text
from crawl.nlp.tools.stem_lemma import stem_and_lemmatize
from crawl.nlp.tools.pos_embeddings_classification import run_pos_embeddings_classification
from crawl.nlp.tools.tfidf import compute_tfidf
from crawl.nlp.tools.ngram_language_model import run_ngram_tool
from crawl.nlp.tools.lsi import run_lsi_tool

register_tool_handler("text_analysis", lambda text, options: analyze_text(text, options))
register_tool_handler("stem_lemma", lambda text, options: stem_and_lemmatize(text, options))
register_tool_handler("pos_embeddings_classification", lambda text, options: run_pos_embeddings_classification(text, options))
register_tool_handler("tfidf", lambda text, options: compute_tfidf(text, options))
register_tool_handler("ngram_probability", lambda text, options: run_ngram_tool(text, options))
register_tool_handler("lsi_search", lambda text, options: run_lsi_tool(text, options))

