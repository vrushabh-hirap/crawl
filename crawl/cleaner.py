"""Cleaning engine for the crawl-any-website library.

Provides utilities to clean and normalize a parsed DOM tree in place:
1. Whitespace Normalization: Collapses duplicate space characters.
2. Entity Decoding: Unescapes XML/HTML characters (e.g. '&amp;' -> '&').
3. Empty Tag Pruning: Deletes element nodes that do not contain text/children.
4. Sibling Deduplication: Removes duplicate identical subtrees to clean pages.
"""

import html
import re
from typing import Set
from crawl.dom import Node


def normalize_whitespace(text: str) -> str:
    """Collapses duplicate spaces, tabs, and newlines into a single space and strips.

    This prevents layout spacing from bloating extracted text data.

    Args:
        text: The raw input string containing excessive spacing or newlines.

    Returns:
        The normalized and stripped string.
    """
    if not text:
        return ""
    # Replaces any sequence of whitespace characters (\s+) with a single space and trims ends
    return re.sub(r"\s+", " ", text).strip()


def decode_entities(text: str) -> str:
    """Decodes HTML entities (standard, named, and numerical ones) in a string.

    For example, decodes '&quot;' to '"', '&amp;' to '&', and '&#x27;' to "'".

    Args:
        text: Input string potentially containing HTML entities.

    Returns:
        The decoded plain text representation.
    """
    if not text:
        return ""
    # Decodes standard entities using standard library html module unescape
    return html.unescape(text)


def clean_text_nodes(node: Node) -> None:
    """Recursively normalizes whitespace and decodes HTML entities in all text nodes.

    Walks down the DOM tree. For text nodes (node.tag is None), it runs cleaning.
    For comments/doctypes, it decodes entities but preserves spacing.

    Args:
        node: The reference DOM Node to clean in-place.
    """
    # If the tag name is None, this is a plain text node holding page text
    if node.tag is None:
        if node.text is not None:
            # Decode HTML entities first, then collapse consecutive spaces
            node.text = normalize_whitespace(decode_entities(node.text))
    # If this is a comment or a DOCTYPE declaration
    elif node.tag == "#comment" or node.tag == "#doctype":
        # Decode entities to be readable, but do not normalize whitespace (preserve formats)
        if node.text is not None:
            node.text = decode_entities(node.text)
    else:
        # Otherwise, this is a structural element. Clean its children recursively.
        for child in node.children:
            clean_text_nodes(child)


def is_identical_subtree(n1: Node, n2: Node) -> bool:
    """Checks if two DOM subtrees are structurally and contextually identical.

    Compares tag names, attributes, text, child counts, and recursively compares children.

    Args:
        n1: First DOM Node to compare.
        n2: Second DOM Node to compare.

    Returns:
        True if subtrees match exactly in tags, structure, and text; False otherwise.
    """
    # 1. Compare Tag names
    if n1.tag != n2.tag:
        return False
    # 2. Compare Element Attributes dictionary
    if n1.attributes != n2.attributes:
        return False
    # 3. Compare Text content
    if n1.text != n2.text:
        return False
    # 4. Compare child count
    if len(n1.children) != len(n2.children):
        return False
    # 5. Recursively verify each child subtree in sequence
    for c1, c2 in zip(n1.children, n2.children):
        if not is_identical_subtree(c1, c2):
            return False
    return True


def clean_duplicate_nodes(node: Node) -> None:
    """Prunes duplicate sibling elements that are structurally identical in the DOM tree.

    Keeps the first occurrence of a sibling subtree and drops subsequent copies.
    Preserves text, comments, and doctypes to prevent losing page text flows.

    Args:
        node: The parent Node whose children will be deduplicated.
    """
    # Recurse down children first (bottom-up tree cleaning)
    for child in node.children:
        clean_duplicate_nodes(child)

    unique_children = []
    for child in node.children:
        # Preserve text flows, comments, and doctype declarations as-is.
        # Deduplication only applies to structural HTML element tags (e.g. redundant tables/divs).
        if child.tag is None or child.tag.startswith("#"):
            unique_children.append(child)
            continue

        duplicate = False
        # Check against previously kept sibling nodes
        for uc in unique_children:
            if uc.tag == child.tag and is_identical_subtree(child, uc):
                duplicate = True
                break

        # Append to our active list only if it's unique
        if not duplicate:
            unique_children.append(child)

    # Overwrite the parent's children with the deduplicated elements list
    node.children = unique_children


def clean_empty_tags(node: Node, keep_with_attrs: bool = True) -> None:
    """Recursively removes empty element nodes from the DOM.

    An element node is empty if it is not a void tag, has no child nodes, and has
    no non-whitespace text.

    Args:
        node: The reference DOM Node to prune.
        keep_with_attrs: If True, empty elements containing attributes (like id/class/src)
            are preserved to avoid breaking layout elements.
    """
    # Bottom-up recursion: clean deeper children first
    for child in list(node.children):
        clean_empty_tags(child, keep_with_attrs)

    # Process immediate children of this node
    for child in list(node.children):
        # Ignore plain text, comments, or doctype declarations
        if child.tag is None or child.tag.startswith("#"):
            continue

        # Void tags (like <img>, <br>, <hr>) must never be pruned since they do not
        # contain children or text by definition under the HTML spec.
        if child.tag in Node.VOID_TAGS:
            continue

        # Check if the element contains children or text
        has_children = len(child.children) > 0
        has_text = bool(child.text and child.text.strip())

        if not has_children and not has_text:
            if keep_with_attrs and child.attributes:
                # Keep the element because it has styling classes, IDs, or other configurations
                continue
            # Remove the empty element from the parent's children list
            node.remove_child(child)


def clean_dom(node: Node) -> None:
    """Runs the full sequence of DOM cleaning operations in-place.

    Performs text cleaning (entity + whitespace), prunes empty element tags, and
    removes duplicate sibling subtrees in a single pipeline call.

    Args:
        node: Root DOM Node of the tree.
    """
    # 1. Clean and normalize all text nodes
    clean_text_nodes(node)
    # 2. Prune redundant empty container elements
    clean_empty_tags(node)
    # 3. Deduplicate structural sibling nodes
    clean_duplicate_nodes(node)
