"""Export engine for the crawl-any-website library.

Provides helper functions to format and save extracted structured data
into standard CSV, JSON, or plain TXT formats, completely decoupled from
the DOM parsing modules.
"""

import csv
import json
import io
from typing import List, Dict, Any, Union
from crawl.exceptions import ExportError


def to_csv(data: List[Dict[str, Any]]) -> str:
    """Converts a list of dictionaries to a CSV-formatted string.

    Automatically handles sparse data row entries by compiling the union of all keys.

    Args:
        data: List of dictionaries where each dict represents a row.

    Returns:
        A CSV-formatted string with headers as the first line.

    Raises:
        ExportError: If conversion fails due to formatting or type mismatch.
    """
    if not data:
        return ""
    try:
        # Collect union of all keys present in any row to handle varying schemas gracefully
        headers_set = set()
        for row in data:
            headers_set.update(row.keys())
        headers = sorted(list(headers_set))

        # Output to memory stream
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=headers)
        writer.writeheader()
        writer.writerows(data)
        
        return output.getvalue()
    except Exception as e:
        raise ExportError(f"CSV conversion failed: {str(e)}") from e


def to_json(data: Any, indent: int = 4) -> str:
    """Converts data structure to a formatted, human-readable JSON string.

    Args:
        data: Any JSON-serializable Python data structure (dict, list, string, etc.).
        indent: Indentation spacing level for pretty printing.

    Returns:
        A JSON-formatted string.

    Raises:
        ExportError: If serialization fails (e.g. data contains non-serializable objects).
    """
    try:
        # Ensure ascii is disabled so characters like accented text or emojis aren't escaped
        return json.dumps(data, indent=indent, ensure_ascii=False)
    except (TypeError, ValueError) as e:
        raise ExportError(f"JSON serialization failed: {str(e)}") from e


def export_csv(data: List[Dict[str, Any]], filepath: str) -> None:
    """Saves a list of dictionaries to a CSV file on the local file system.

    Creates parent directories automatically if needed.

    Args:
        data: List of dictionaries representing rows.
        filepath: Absolute or relative target destination file path.

    Raises:
        ExportError: If writing to the file fails due to system permissions or layout errors.
    """
    try:
        # 1. Convert to CSV payload in memory
        csv_content = to_csv(data)
        # 2. Write content to the file with UTF-8 encoding
        with open(filepath, "w", encoding="utf-8", newline="") as f:
            f.write(csv_content)
    except Exception as e:
        if isinstance(e, ExportError):
            raise
        raise ExportError(f"Failed to export CSV to {filepath}: {str(e)}") from e


def export_json(data: Any, filepath: str, indent: int = 4) -> None:
    """Saves any serializable data structure to a JSON file on the local file system.

    Args:
        data: Any JSON-serializable Python data structure.
        filepath: Absolute or relative target destination file path.
        indent: Pretty print indentation index.

    Raises:
        ExportError: If writing to the file fails due to system permissions or layout errors.
    """
    try:
        # 1. Convert payload to pretty JSON string
        json_content = to_json(data, indent=indent)
        # 2. Write JSON string to destination filepath
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(json_content)
    except Exception as e:
        if isinstance(e, ExportError):
            raise
        raise ExportError(f"Failed to export JSON to {filepath}: {str(e)}") from e


def export_txt(text: Union[str, List[str]], filepath: str) -> None:
    """Saves plain text or a list of strings to a TXT file.

    If a list of strings is supplied, each item is written as a separate line.

    Args:
        text: String content or list of lines to write.
        filepath: Absolute or relative target destination file path.

    Raises:
        ExportError: If writing to the file fails due to system permissions or layout errors.
    """
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            if isinstance(text, list):
                # Write each element as a separate line
                for line in text:
                    f.write(str(line) + "\n")
            else:
                f.write(str(text))
    except Exception as e:
        raise ExportError(f"Failed to export TXT to {filepath}: {str(e)}") from e
