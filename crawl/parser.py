"""HTML Parser for the crawl library.

Consumes lexical tokens from the HTMLTokenizer and builds a DOM tree structure.
Implements stack-based tree construction and recovers gracefully from unclosed tags.
"""

from typing import List
from crawl.tokenizer import Token, TokenType, HTMLTokenizer
from crawl.dom import Node
from crawl.exceptions import ParserError


class HTMLParser:
    """Consumes tokens and constructs a DOM tree."""

    def __init__(self, html: str) -> None:
        """Initializes the parser with target HTML.

        Args:
            html: HTML string to parse.
        """
        self.tokenizer = HTMLTokenizer(html)
        # Create a document root node to act as the parent for all top-level elements
        self.root = Node(tag="root")
        self.stack: List[Node] = [self.root]

    def parse(self) -> Node:
        """Parses the HTML and returns the document root node.

        Returns:
            The root Node of the constructed DOM tree.
        """
        try:
            for token in self.tokenizer.tokenize():
                self._handle_token(token)
        except Exception as e:
            raise ParserError(f"Unrecoverable error during HTML parsing: {str(e)}") from e

        return self.root

    def _handle_token(self, token: Token) -> None:
        """Processes a single token and updates the DOM tree/stack."""
        if token.type == TokenType.DOCTYPE:
            doctype_node = Node(tag="#doctype", text=token.value)
            self.stack[-1].append_child(doctype_node)

        elif token.type == TokenType.COMMENT:
            comment_node = Node(tag="#comment", text=token.value)
            self.stack[-1].append_child(comment_node)

        elif token.type == TokenType.TEXT:
            text_node = Node(tag=None, text=token.value)
            self.stack[-1].append_child(text_node)

        elif token.type == TokenType.START_TAG:
            el = Node(tag=token.value, attributes=token.attributes)
            self.stack[-1].append_child(el)

            # Check if this tag should be pushed onto the stack
            is_void = el.tag in Node.VOID_TAGS
            if not token.self_closing and not is_void:
                self.stack.append(el)

        elif token.type == TokenType.END_TAG:
            tag_name = token.value.lower()

            # Find matching tag in the stack from top to bottom (excluding dummy root)
            match_idx = -1
            for i in range(len(self.stack) - 1, 0, -1):
                if self.stack[i].tag == tag_name:
                    match_idx = i
                    break

            if match_idx != -1:
                # Close all tags nested inside the matching tag by popping them
                self.stack = self.stack[:match_idx]


def parse_html(html: str) -> Node:
    """Helper function to parse an HTML string and return the DOM tree root.

    Args:
        html: Raw HTML code.

    Returns:
        Root Node of the DOM.
    """
    parser = HTMLParser(html)
    return parser.parse()
