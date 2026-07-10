"""DOM (Document Object Model) node implementation.

This module provides the Node class, representing HTML elements, text nodes,
comments, and doctypes. It includes methods for manipulating the tree,
retrieving attributes, and serializing HTML.
"""

from typing import Dict, List, Optional, Any


class Node:
    """Represents a node in the custom DOM tree."""

    # Set of standard HTML void tags that do not require closing tags under the HTML spec.
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
        # Save tag in lower case to make selector queries case-insensitive
        self.tag = tag.lower() if tag else None
        # Convert all attribute names to lowercase for standard compliance (e.g. Class -> class)
        self.attributes = {k.lower(): v for k, v in (attributes or {}).items()}
        self.children = children or []
        self.parent = parent
        self.text = text

        # Establish parent-child reference pointers for initial children list
        for child in self.children:
            child.parent = self

    def append_child(self, child: "Node") -> None:
        """Appends a child node to this node.

        If the child already has a parent, it is removed from that parent first.

        Args:
            child: The Node to append.
        """
        if child.parent is not None:
            child.parent.remove_child(child)
        # Point the child's parent pointer to this node
        child.parent = self
        self.children.append(child)

    def remove_child(self, child: "Node") -> None:
        """Removes a child node from this node.

        Clears the parent pointer on the child node.

        Args:
            child: The Node to remove.
        """
        if child in self.children:
            child.parent = None
            self.children.remove(child)

    def get_attribute(self, name: str, default: Optional[str] = None) -> Optional[str]:
        """Retrieves the value of an attribute by name (case-insensitive).

        Args:
            name: The attribute name to fetch.
            default: Return fallback value if attribute does not exist.

        Returns:
            The string attribute value, or default value.
        """
        return self.attributes.get(name.lower(), default)

    def set_attribute(self, name: str, value: str) -> None:
        """Sets or updates an attribute value.

        Converts the attribute name to lowercase.

        Args:
            name: Attribute name.
            value: Attribute value.
        """
        self.attributes[name.lower()] = value

    def find(self, selector: str) -> Optional["Node"]:
        """Finds the first descendant matching the given CSS selector.

        Args:
            selector: CSS selector query string.

        Returns:
            The first matching Node, or None if none found.
        """
        from crawl.selector import select
        results = select(self, selector)
        return results[0] if results else None

    def find_all(self, selector: str) -> List["Node"]:
        """Finds all descendants matching the given CSS selector.

        Args:
            selector: CSS selector query string.

        Returns:
            A list of all matching Nodes in document pre-order.
        """
        from crawl.selector import select
        return select(self, selector)

    def get_text(self) -> str:
        """Extracts and returns the text content of this node and all its descendants.

        Concatenates plain text nodes, ignoring comments and doctype contents.

        Returns:
            Plain text representation.
        """
        # If this is a text node (or a comment/doctype wrapper)
        if self.tag is None:
            # Return string content. We only print value if text is set.
            return self.text if self.text is not None else ""

        # For structural element tags, concatenate the text values of children recursively
        return "".join(child.get_text() for child in self.children)

    def inner_html(self) -> str:
        """Returns the serialized HTML content of the node's children.

        Returns:
            Serialized HTML string.
        """
        # Concatenates outer_html of all children nodes
        return "".join(child.outer_html() for child in self.children)

    def outer_html(self) -> str:
        """Returns the serialized HTML representation of this node and its descendants.

        Translates tags, attributes, comments, doctypes, and nested children back into markup.

        Returns:
            Full serialized HTML code string.
        """
        # Case 1: Plain Text Node
        if self.tag is None:
            return self.text if self.text is not None else ""

        # Case 2: Special Comment Tag
        if self.tag == "#comment":
            return f"<!--{self.text}-->"
            
        # Case 3: Special Doctype Tag
        if self.tag == "#doctype":
            return f"<!DOCTYPE {self.text}>"

        # Case 4: Standard HTML Element Tag
        # Compile attributes list (e.g. class="container" id="main")
        attr_parts = []
        for k, v in self.attributes.items():
            if v == "":
                # Boolean attribute represented without values (e.g., "disabled", "required")
                attr_parts.append(f" {k}")
            else:
                attr_parts.append(f' {k}="{v}"')
        attrs_str = "".join(attr_parts)

        # Handle void tags which must not have closing tags (e.g. <img src="logo.png">)
        if self.tag in self.VOID_TAGS:
            return f"<{self.tag}{attrs_str}>"

        # Standard element: return opening tag, inner children HTML, and matching closing tag
        return f"<{self.tag}{attrs_str}>{self.inner_html()}</{self.tag}>"

    def __repr__(self) -> str:
        """Prints a human-readable node description for debug inspections."""
        if self.tag is None:
            return f"TextNode(text={repr(self.text)})"
        if self.tag == "#comment":
            return f"CommentNode(text={repr(self.text)})"
        if self.tag == "#doctype":
            return f"DoctypeNode(text={repr(self.text)})"
        attrs = f", attrs={self.attributes}" if self.attributes else ""
        children = f", children={len(self.children)}" if self.children else ""
        return f"ElementNode(tag='{self.tag}'{attrs}{children})"
