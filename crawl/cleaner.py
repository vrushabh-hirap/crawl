"""Cleaning engine for the crawl library.

Provides utilities to clean and normalize a parsed DOM tree:
- Normalizing whitespace in text nodes.
- Decoding HTML entities.
- Recursively pruning empty elements (ignoring void tags).
- Deduplicating structurally identical sibling subtrees.
"""

import html
import re
from typing import Set
from crawl.dom import Node


def normalize_whitespace(text: str) -> str:
    """Collapses duplicate spaces/tabs/newlines into a single space and strips.

    Args:
        text: The raw input string.

    Returns:
        The normalized string.
    """
    if not text:
        return ""
    # Replace any sequence of whitespace with a single space
    return re.sub(r"\s+", " ", text).strip()


def decode_entities(text: str) -> str:
    """Decodes HTML entities (standard and numerical) in a string.

    Args:
        text: Input string potentially containing HTML entities.

    Returns:
        String with entities decoded.
    """
    if not text:
        return ""
    return html.unescape(text)


def clean_text_nodes(node: Node) -> None:
    """Recursively normalizes whitespace and decodes HTML entities in all text nodes.

    Args:
        node: The reference DOM Node to clean in-place.
    """
    if node.tag is None:  # Text node
        if node.text is not None:
            node.text = normalize_whitespace(decode_entities(node.text))
    elif node.tag == "#comment" or node.tag == "#doctype":
        # Decode entities in comment content but don't force whitespace normalization
        if node.text is not None:
            node.text = decode_entities(node.text)
    else:
        for child in node.children:
            clean_text_nodes(child)


def is_identical_subtree(n1: Node, n2: Node) -> bool:
    """Checks if two DOM subtrees are structurally and contextually identical.

    Args:
        n1: First Node.
        n2: Second Node.

    Returns:
        True if subtrees match exactly, False otherwise.
    """
    if n1.tag != n2.tag:
        return False
    if n1.attributes != n2.attributes:
        return False
    if n1.text != n2.text:
        return False
    if len(n1.children) != len(n2.children):
        return False
    for c1, c2 in zip(n1.children, n2.children):
        if not is_identical_subtree(c1, c2):
            return False
    return True


def clean_duplicate_nodes(node: Node) -> None:
    """Prunes duplicate sibling elements that are structurally identical in the DOM tree.

    Args:
        node: The parent Node to clean.
    """
    # Recurse down children first (bottom-up)
    for child in node.children:
        clean_duplicate_nodes(child)

    unique_children = []
    for child in node.children:
        # Keep all text/comment/doctype nodes as-is to preserve text flows,
        # but deduplicate element nodes.
        if child.tag is None or child.tag.startswith("#"):
            unique_children.append(child)
            continue

        duplicate = False
        for uc in unique_children:
            if uc.tag == child.tag and is_identical_subtree(child, uc):
                duplicate = True
                break

        if not duplicate:
            unique_children.append(child)

    node.children = unique_children


def clean_empty_tags(node: Node, keep_with_attrs: bool = True) -> None:
    """Recursively removes empty element nodes.

    An element node is empty if it is not a void tag, has no children, and has
    no non-whitespace text.

    Args:
        node: The reference DOM Node.
        keep_with_attrs: If True, empty elements that have attributes (like id/class/src)
            are preserved.
    """
    # Bottom-up recursion
    for child in list(node.children):
        clean_empty_tags(child, keep_with_attrs)

    # Process immediate children
    for child in list(node.children):
        if child.tag is None or child.tag.startswith("#"):
            continue

        # Void tags must not be pruned since they never have children/text (e.g. <img>, <br>)
        if child.tag in Node.VOID_TAGS:
            continue

        # Check if the element is empty
        has_children = len(child.children) > 0
        has_text = bool(child.text and child.text.strip())

        if not has_children and not has_text:
            if keep_with_attrs and child.attributes:
                # Keep because it has styling classes, IDs, or other configurations
                continue
            node.remove_child(child)


def clean_dom(node: Node) -> None:
    """Runs the full sequence of DOM cleaning operations in-place.

    Args:
        node: Root DOM Node of the tree.
    """
    clean_text_nodes(node)
    clean_empty_tags(node)
    clean_duplicate_nodes(node)
