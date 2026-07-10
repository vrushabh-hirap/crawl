"""HTML Tokenizer for the crawl library.

This module scans HTML strings character by character and produces lexical tokens
representing the structure of the document (tags, text, comments, doctypes).
"""

from enum import Enum, auto
from typing import Dict, Generator, Optional, Tuple, Any


class TokenType(Enum):
    """Enumeration of token types produced by the tokenizer."""
    START_TAG = auto()
    END_TAG = auto()
    TEXT = auto()
    COMMENT = auto()
    DOCTYPE = auto()


class Token:
    """A lexical token representing an HTML construct.

    Attributes:
        type: The TokenType of this token.
        value: The main text content of the token (e.g. tag name, text, comment contents).
        attributes: Dictionary of attributes (only for START_TAG).
        self_closing: True if tag ends in '/>' (only for START_TAG).
    """

    def __init__(
        self,
        type_: TokenType,
        value: str = "",
        attributes: Optional[Dict[str, str]] = None,
        self_closing: bool = False
    ):
        self.type = type_
        self.value = value
        self.attributes = attributes or {}
        self.self_closing = self_closing

    def __repr__(self) -> str:
        attrs = f", attributes={self.attributes}" if self.attributes else ""
        sc = ", self_closing=True" if self.self_closing else ""
        return f"Token({self.type.name}, {repr(self.value)}{attrs}{sc})"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Token):
            return False
        return (
            self.type == other.type
            and self.value == other.value
            and self.attributes == other.attributes
            and self.self_closing == other.self_closing
        )


class HTMLTokenizer:
    """Character-by-character scanner for tokenizing HTML code.

    Uses a state machine to scan the HTML string and yield tokens.
    """

    def __init__(self, html: str) -> None:
        """Initializes the tokenizer with the target HTML.

        Args:
            html: Raw HTML string to tokenize.
        """
        self.html = html
        self.index = 0
        self.length = len(html)

    def _peek(self, offset: int = 0) -> str:
        """Peeks at the character offset positions ahead of the current cursor.

        Returns:
            The character at the peek position, or an empty string if EOF is reached.
        """
        pos = self.index + offset
        if pos < self.length:
            return self.html[pos]
        return ""

    def _consume(self) -> str:
        """Consumes the character at the current index and advances the cursor.

        Returns:
            The consumed character, or an empty string if EOF.
        """
        if self.index < self.length:
            char = self.html[self.index]
            self.index += 1
            return char
        return ""

    def _match_str(self, prefix: str, case_insensitive: bool = False) -> bool:
        """Checks if the subsequent characters match a given prefix.

        If matched, consumes the prefix characters and returns True.
        """
        end_pos = self.index + len(prefix)
        if end_pos > self.length:
            return False

        slice_str = self.html[self.index:end_pos]
        if case_insensitive:
            matched = slice_str.lower() == prefix.lower()
        else:
            matched = slice_str == prefix

        if matched:
            self.index = end_pos
            return True
        return False

    def _skip_whitespace(self) -> None:
        """Skips any leading whitespace characters from the current index."""
        while self.index < self.length and self.html[self.index].isspace():
            self.index += 1

    def tokenize(self) -> Generator[Token, None, None]:
        """Scans the HTML document and yields a sequence of Token objects.

        Yields:
            Token objects sequentially.
        """
        text_buffer = []

        while self.index < self.length:
            char = self._peek()

            # Check if a tag, comment, or doctype starts
            if char == "<":
                # Check for comment
                if self._peek(1) == "!" and self._peek(2) == "-" and self._peek(3) == "-":
                    if text_buffer:
                        yield Token(TokenType.TEXT, "".join(text_buffer))
                        text_buffer = []
                    yield self._parse_comment()
                    continue

                # Check for DOCTYPE
                if self._peek(1) == "!" and self.html[self.index + 1 : self.index + 9].lower() == "!doctype":
                    if text_buffer:
                        yield Token(TokenType.TEXT, "".join(text_buffer))
                        text_buffer = []
                    yield self._parse_doctype()
                    continue

                # Check for standard end/start tags
                next_char = self._peek(1)
                is_end_tag = next_char == "/"
                # A start tag needs to start with an ASCII letter, '!', '?', or an underscore to be parsed as a tag.
                # Otherwise we treat the '<' as normal text (e.g. in "a < b" or "< container").
                is_valid_tag_start = (
                    is_end_tag
                    or next_char.isalpha()
                    or next_char in ("!", "?", "_")
                )

                if is_valid_tag_start:
                    if text_buffer:
                        yield Token(TokenType.TEXT, "".join(text_buffer))
                        text_buffer = []

                    if is_end_tag:
                        yield self._parse_end_tag()
                    else:
                        yield self._parse_start_tag()
                    continue

            # Accumulate text content
            text_buffer.append(self._consume())

        # Yield any remaining text
        if text_buffer:
            yield Token(TokenType.TEXT, "".join(text_buffer))

    def _parse_comment(self) -> Token:
        """Parses an HTML comment of the form <!-- ... -->."""
        # Consume the "<!--" prefix
        self._match_str("<!--")
        comment_buffer = []

        while self.index < self.length:
            if self._match_str("-->"):
                break
            comment_buffer.append(self._consume())

        return Token(TokenType.COMMENT, "".join(comment_buffer))

    def _parse_doctype(self) -> Token:
        """Parses a <!DOCTYPE ...> tag."""
        # Consume "<!"
        self._consume() # <
        self._consume() # !
        # Consume "DOCTYPE" (case-insensitive)
        self._match_str("doctype", case_insensitive=True)
        self._skip_whitespace()

        doctype_buffer = []
        while self.index < self.length:
            char = self._peek()
            if char == ">":
                self._consume()
                break
            doctype_buffer.append(self._consume())

        return Token(TokenType.DOCTYPE, "".join(doctype_buffer).strip())

    def _parse_end_tag(self) -> Token:
        """Parses an end tag like </div >."""
        # Consume "</"
        self._consume() # <
        self._consume() # /

        name_buffer = []
        # Consume tag name
        while self.index < self.length:
            char = self._peek()
            if char == ">" or char.isspace() or char == "/":
                break
            name_buffer.append(self._consume())

        tag_name = "".join(name_buffer)

        # Skip any attributes or garbage inside end tags until matching '>'
        while self.index < self.length:
            char = self._consume()
            if char == ">":
                break

        return Token(TokenType.END_TAG, tag_name)

    def _parse_start_tag(self) -> Token:
        """Parses a start tag like <div id="main" class="container"> or <img />."""
        # Consume "<"
        self._consume()

        name_buffer = []
        while self.index < self.length:
            char = self._peek()
            if char == ">" or char == "/" or char.isspace():
                break
            name_buffer.append(self._consume())

        tag_name = "".join(name_buffer)
        attributes = {}
        self_closing = False

        # Parse attributes
        while self.index < self.length:
            self._skip_whitespace()
            char = self._peek()

            if char == ">":
                self._consume()
                break

            if char == "/":
                self._consume()
                if self._peek() == ">":
                    self._consume()
                    self_closing = True
                break

            # Handle unexpected EOF or broken structures
            if char == "":
                break

            # Parse attribute key-value pair
            attr_name, attr_val = self._parse_attribute()
            if attr_name:
                attributes[attr_name] = attr_val

        return Token(TokenType.START_TAG, tag_name, attributes, self_closing)

    def _parse_attribute(self) -> Tuple[str, str]:
        """Parses a single HTML attribute name/value pair."""
        name_buffer = []

        # Consume attribute name
        while self.index < self.length:
            char = self._peek()
            if char == "=" or char == ">" or char == "/" or char.isspace():
                break
            name_buffer.append(self._consume())

        attr_name = "".join(name_buffer)
        if not attr_name:
            # If name is empty, consume one char to avoid infinite loop
            self._consume()
            return "", ""

        self._skip_whitespace()

        # Check for value
        if self._peek() == "=":
            self._consume()  # Consume "="
            self._skip_whitespace()

            val_char = self._peek()
            val_buffer = []

            if val_char == '"':
                self._consume()  # Consume open double quote
                while self.index < self.length:
                    c = self._consume()
                    if c == '"':
                        break
                    val_buffer.append(c)
            elif val_char == "'":
                self._consume()  # Consume open single quote
                while self.index < self.length:
                    c = self._consume()
                    if c == "'":
                        break
                    val_buffer.append(c)
            else:
                # Unquoted value
                while self.index < self.length:
                    c = self._peek()
                    if c == ">" or c == "/" or c.isspace():
                        break
                    val_buffer.append(self._consume())

            attr_val = "".join(val_buffer)
        else:
            # Boolean attribute like "disabled" or "required"
            attr_val = ""

        return attr_name, attr_val
