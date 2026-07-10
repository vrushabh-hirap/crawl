"""CSS Selector engine for the crawl library.

Parses CSS selector strings and runs them against a DOM Node.
Supports:
- Tag names (e.g. 'div', '*')
- IDs (e.g. '#main')
- Classes (e.g. '.product', '.product.featured')
- Descendant combinators (space)
- Direct child combinators ('>')
- Comma-separated alternative selectors (e.g. 'h1, h2')
Exposes matched nodes in document order without duplicates.
"""

import re
from typing import List, Optional, Tuple, Set
from crawl.dom import Node
from crawl.exceptions import InvalidSelector


def parse_selector_part(part: str) -> Tuple[Optional[str], Optional[str], List[str]]:
    """Parses a single selector component (e.g., 'div#main.class1.class2').

    Args:
        part: The selector component string.

    Returns:
        A tuple of (tag, id, classes).
    """
    # Extract tag name from start (alphanumeric, dash, underscore, or '*')
    tag_match = re.match(r"^([a-zA-Z0-9_-]+|\*)", part)
    tag = tag_match.group(1) if tag_match else None

    # Extract ID (starts with #)
    id_match = re.search(r"#([a-zA-Z0-9_-]+)", part)
    id_ = id_match.group(1) if id_match else None

    # Extract all classes (starts with .)
    classes = re.findall(r"\.([a-zA-Z0-9_-]+)", part)

    return tag, id_, classes


def matches_part(node: Node, tag: Optional[str], id_: Optional[str], classes: List[str]) -> bool:
    """Checks if a DOM Node matches a parsed selector component.

    Args:
        node: The Node to inspect.
        tag: Optional tag name constraint.
        id_: Optional ID constraint.
        classes: List of class name constraints.

    Returns:
        True if the Node matches all constraints, False otherwise.
    """
    # CSS selectors do not match text, comment, or doctype nodes directly
    if node.tag is None or node.tag.startswith("#"):
        return False

    # Tag constraint matching
    if tag is not None and tag != "*":
        if node.tag != tag.lower():
            return False

    # ID constraint matching
    if id_ is not None:
        if node.get_attribute("id") != id_:
            return False

    # Class constraint matching (all classes must be present)
    if classes:
        class_attr = node.get_attribute("class") or ""
        node_classes = set(class_attr.split())
        for cls in classes:
            if cls not in node_classes:
                return False

    return True


def get_descendants(node: Node) -> List[Node]:
    """Recursively retrieves all element descendants of a node in pre-order traversal."""
    results = []

    def recurse(curr: Node) -> None:
        for child in curr.children:
            # Ignore non-element nodes (text, comments, doctypes)
            if child.tag is not None and not child.tag.startswith("#"):
                results.append(child)
                recurse(child)

    recurse(node)
    return results


def parse_selector_path(selector_str: str) -> List[Tuple[str, str]]:
    """Parses a single selector path into a list of steps: (combinator, part).

    Args:
        selector_str: CSS selector path (e.g. 'div > span.highlight').

    Returns:
        List of tuples containing (combinator, part).
    """
    # Standardize spaces around direct child combinators
    s = re.sub(r"\s*>\s*", " > ", selector_str)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)

    if not s:
        raise InvalidSelector("Empty selector path")

    parts = s.split(" ")
    steps = []

    # The first item cannot be a combinator
    if parts[0] == ">":
        raise InvalidSelector("CSS Selector path cannot begin with a '>' combinator")

    steps.append(("", parts[0]))

    i = 1
    while i < len(parts):
        part = parts[i]
        if part == ">":
            if i + 1 >= len(parts):
                raise InvalidSelector("CSS Selector path cannot end with a '>' combinator")
            next_part = parts[i + 1]
            if next_part == ">":
                raise InvalidSelector("Consecutive '>' combinators are invalid")
            steps.append((">", next_part))
            i += 2
        else:
            # Default space represents descendant combinator
            steps.append((" ", part))
            i += 1

    return steps


def select_single_path(start_node: Node, selector_path: str) -> List[Node]:
    """Evaluates a single CSS selector path starting from a given node.

    This breaks the selector path (e.g. 'div > span.highlight') into sequential steps,
    processes each step one-by-one, and filters the list of candidate nodes accordingly.

    Args:
        start_node: The reference DOM Node.
        selector_path: Selector path string.

    Returns:
        A list of matching DOM Nodes.
    """
    steps = parse_selector_path(selector_path)
    if not steps:
        return []

    # Initial candidate set for step 0 is all matching descendants of the starting node
    first_comb, first_part = steps[0]
    first_tag, first_id, first_classes = parse_selector_part(first_part)

    candidates = [
        node for node in get_descendants(start_node)
        if matches_part(node, first_tag, first_id, first_classes)
    ]

    # Evaluate remaining combinator steps left-to-right
    for i in range(1, len(steps)):
        comb, part = steps[i]
        tag, id_, classes = parse_selector_part(part)
        new_candidates = []

        if comb == ">":
            # Direct child combinator: candidates must be direct children of the current matching nodes
            for cand in candidates:
                for child in cand.children:
                    if matches_part(child, tag, id_, classes):
                        if child not in new_candidates:
                            new_candidates.append(child)
        else:
            # Descendant combinator (space): candidates can be any sub-level descendant of current matching nodes
            for cand in candidates:
                for desc in get_descendants(cand):
                    if matches_part(desc, tag, id_, classes):
                        if desc not in new_candidates:
                            new_candidates.append(desc)

        # Update candidate set for the next selector step
        candidates = new_candidates

    return candidates


def select(node: Node, selector_str: str) -> List[Node]:
    """Finds all element descendants matching the CSS selector.

    Handles comma-separated alternate paths (alternative selector lists)
    and returns unique matches sorted in pre-order document sequence.

    Args:
        node: The reference DOM Node.
        selector_str: Comma-separated CSS selector string.

    Returns:
        Deduplicated list of matching DOM Nodes in document order.
    """
    if not selector_str.strip():
        raise InvalidSelector("Empty selector query")

    # Split selector list by commas
    raw_paths = selector_str.split(",")
    paths = [p.strip() for p in raw_paths if p.strip()]

    # If the count of trimmed paths does not match the raw count, it means the user
    # entered double commas or trailing commas (e.g. "div, , span" or "div,"), which is invalid CSS.
    if len(paths) != len(raw_paths) or not paths:
        raise InvalidSelector("CSS Selector query has invalid comma usage or is empty")

    all_results = []
    seen = set()

    # Collect matches from all comma-separated selectors
    for path in paths:
        path_matches = select_single_path(node, path)
        for match in path_matches:
            if match not in seen:
                seen.add(match)
                all_results.append(match)

    # Sort results to guarantee exact pre-order document sequence.
    # This prevents results from being returned out-of-order when multiple queries merge.
    descendants = get_descendants(node)
    results_set = set(all_results)
    return [d for d in descendants if d in results_set]
