"""Export engine for the crawl library.

Provides helper functions to format and save extracted structured data
into CSV, JSON, or TXT formats, completely independent of the DOM parser.
"""

import csv
import json
import io
from typing import List, Dict, Any, Union
from crawl.exceptions import ExportError


def to_csv(data: List[Dict[str, Any]]) -> str:
    """Converts a list of dictionaries to a CSV string.

    Args:
        data: List of dictionaries representing rows of data.

    Returns:
        A CSV-formatted string.

    Raises:
        ExportError: If conversion fails.
    """
    if not data:
        return ""
    try:
        # Collect union of all keys to handle sparse data
        headers_set = set()
        for row in data:
            headers_set.update(row.keys())
        headers = sorted(list(headers_set))

        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=headers)
        writer.writeheader()
        writer.writerows(data)
        return output.getvalue()
    except Exception as e:
        raise ExportError(f"CSV conversion failed: {str(e)}") from e


def to_json(data: Any, indent: int = 4) -> str:
    """Converts data structure to a formatted JSON string.

    Args:
        data: Any JSON-serializable Python data structure.
        indent: Indentation level for pretty printing.

    Returns:
        JSON-formatted string.

    Raises:
        ExportError: If serialization fails.
    """
    try:
        return json.dumps(data, indent=indent, ensure_ascii=False)
    except (TypeError, ValueError) as e:
        raise ExportError(f"JSON serialization failed: {str(e)}") from e


def export_csv(data: List[Dict[str, Any]], filepath: str) -> None:
    """Saves a list of dictionaries to a CSV file.

    Args:
        data: List of dictionaries representing rows.
        filepath: Target destination file path.

    Raises:
        ExportError: If writing to the file fails.
    """
    try:
        csv_content = to_csv(data)
        with open(filepath, "w", encoding="utf-8", newline="") as f:
            f.write(csv_content)
    except Exception as e:
        if isinstance(e, ExportError):
            raise
        raise ExportError(f"Failed to export CSV to {filepath}: {str(e)}") from e


def export_json(data: Any, filepath: str, indent: int = 4) -> None:
    """Saves a data structure to a JSON file.

    Args:
        data: Any JSON-serializable Python data structure.
        filepath: Target destination file path.
        indent: Pretty print indentation index.

    Raises:
        ExportError: If writing to the file fails.
    """
    try:
        json_content = to_json(data, indent=indent)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(json_content)
    except Exception as e:
        if isinstance(e, ExportError):
            raise
        raise ExportError(f"Failed to export JSON to {filepath}: {str(e)}") from e


def export_txt(text: Union[str, List[str]], filepath: str) -> None:
    """Saves plain text or a list of strings to a TXT file.

    Args:
        text: String content or list of lines to write.
        filepath: Target destination file path.

    Raises:
        ExportError: If writing to the file fails.
    """
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            if isinstance(text, list):
                # Write each element as a line
                for line in text:
                    f.write(str(line) + "\n")
            else:
                f.write(str(text))
    except Exception as e:
        raise ExportError(f"Failed to export TXT to {filepath}: {str(e)}") from e
