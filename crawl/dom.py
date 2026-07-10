"""DOM (Document Object Model) node implementation.

This module provides the Node class, representing HTML elements, text nodes,
comments, and doctypes. It includes methods for manipulating the tree,
retrieving attributes, and serializing HTML.
"""

from typing import Dict, List, Optional, Any


class Node:
    """Represents a node in the custom DOM tree."""

    # Set of standard HTML void tags that do not require closing tags
    VOID_TAGS = {
        "area", "base", "br", "col", "embed", "hr", "img", "input",
        "link", "meta", "param", "source", "track", "wbr"
    }

    def __init__(
        self,
        tag: Optional[str] = None,
        attributes: Optional[Dict[str, str]] = None,
        children: Optional[List["Node"]] = None,
        parent: Optional["Node"] = None,
        text: Optional[str] = None
    ) -> None:
        """Initializes a DOM Node.

        Args:
            tag: The HTML tag name (e.g. 'div', 'p'). None for text/comment/doctype nodes.
            attributes: A dictionary of attributes (name -> value).
            children: A list of child Nodes.
            parent: The parent Node.
            text: The text content if this is a text, comment, or doctype node.
        """
        self.tag = tag.lower() if tag else None
        self.attributes = {k.lower(): v for k, v in (attributes or {}).items()}
        self.children = children or []
        self.parent = parent
        self.text = text

        # Set parent for initial children
        for child in self.children:
            child.parent = self

    def append_child(self, child: "Node") -> None:
        """Appends a child node to this node.

        If the child already has a parent, it is removed from that parent first.
        """
        if child.parent is not None:
            child.parent.remove_child(child)
        child.parent = self
        self.children.append(child)

    def remove_child(self, child: "Node") -> None:
        """Removes a child node from this node."""
        if child in self.children:
            child.parent = None
            self.children.remove(child)

    def get_attribute(self, name: str, default: Optional[str] = None) -> Optional[str]:
        """Retrieves the value of an attribute by name (case-insensitive)."""
        return self.attributes.get(name.lower(), default)

    def set_attribute(self, name: str, value: str) -> None:
        """Sets or updates an attribute value."""
        self.attributes[name.lower()] = value

    def find(self, selector: str) -> Optional["Node"]:
        """Finds the first descendant matching the given CSS selector."""
        from crawl.selector import select
        results = select(self, selector)
        return results[0] if results else None

    def find_all(self, selector: str) -> List["Node"]:
        """Finds all descendants matching the given CSS selector."""
        from crawl.selector import select
        return select(self, selector)

    def get_text(self) -> str:
        """Extracts and returns the text content of this node and all its descendants."""
        if self.tag is None:  # Text node, comment, or doctype
            # Only return text for actual text nodes, not comments or doctypes
            # (which we identify by tag = None and custom labels if we want, or by checking self.text)
            # We'll tag text nodes with tag=None or tag="#text".
            # If tag is None, check if it's text. Let's make text node tag "#text" or None.
            # If tag is None, we treat as a text node, except if it is a comment or doctype.
            # Let's check comment/doctype signatures.
            return self.text if self.text is not None else ""

        # For element nodes, concatenate child texts recursively
        return "".join(child.get_text() for child in self.children)

    def inner_html(self) -> str:
        """Returns the serialized HTML content of the node's children."""
        return "".join(child.outer_html() for child in self.children)

    def outer_html(self) -> str:
        """Returns the serialized HTML representation of this node and its descendants."""
        # Case 1: Text node
        if self.tag is None:
            return self.text if self.text is not None else ""

        # Case 2: Special Nodes (Comments & Doctypes)
        if self.tag == "#comment":
            return f"<!--{self.text}-->"
        if self.tag == "#doctype":
            return f"<!DOCTYPE {self.text}>"

        # Case 3: Standard Element Node
        # Build attribute string
        attr_parts = []
        for k, v in self.attributes.items():
            if v == "":
                # Boolean attribute (e.g. disabled)
                attr_parts.append(f" {k}")
            else:
                attr_parts.append(f' {k}="{v}"')
        attrs_str = "".join(attr_parts)

        # Check for void tag
        if self.tag in self.VOID_TAGS:
            return f"<{self.tag}{attrs_str}>"

        # Return tag with children
        return f"<{self.tag}{attrs_str}>{self.inner_html()}</{self.tag}>"

    def __repr__(self) -> str:
        if self.tag is None:
            return f"TextNode(text={repr(self.text)})"
        if self.tag == "#comment":
            return f"CommentNode(text={repr(self.text)})"
        if self.tag == "#doctype":
            return f"DoctypeNode(text={repr(self.text)})"
        attrs = f", attrs={self.attributes}" if self.attributes else ""
        children = f", children={len(self.children)}" if self.children else ""
        return f"ElementNode(tag='{self.tag}'{attrs}{children})"
