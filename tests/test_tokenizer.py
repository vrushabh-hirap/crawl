"""Unit tests for the HTML Tokenizer."""

import unittest
from crawl.tokenizer import HTMLTokenizer, TokenType, Token


class TestHTMLTokenizer(unittest.TestCase):
    """Verifies that the tokenizer correctly emits tokens for standard/malformed HTML."""

    def test_plain_text(self) -> None:
        """Tests that text without HTML structures is emitted as a single TEXT token."""
        tok = HTMLTokenizer("Hello, World!")
        tokens = list(tok.tokenize())
        self.assertEqual(len(tokens), 1)
        self.assertEqual(tokens[0], Token(TokenType.TEXT, "Hello, World!"))

    def test_doctype(self) -> None:
        """Tests <!DOCTYPE html> parsing."""
        tok = HTMLTokenizer("<!DOCTYPE html>")
        tokens = list(tok.tokenize())
        self.assertEqual(len(tokens), 1)
        self.assertEqual(tokens[0], Token(TokenType.DOCTYPE, "html"))

    def test_comment(self) -> None:
        """Tests comment parsing."""
        tok = HTMLTokenizer("<!-- this is a comment -->")
        tokens = list(tok.tokenize())
        self.assertEqual(len(tokens), 1)
        self.assertEqual(tokens[0], Token(TokenType.COMMENT, " this is a comment "))

    def test_simple_tags(self) -> None:
        """Tests open and close tags without attributes."""
        tok = HTMLTokenizer("<div>Hello</div>")
        tokens = list(tok.tokenize())
        self.assertEqual(len(tokens), 3)
        self.assertEqual(tokens[0], Token(TokenType.START_TAG, "div"))
        self.assertEqual(tokens[1], Token(TokenType.TEXT, "Hello"))
        self.assertEqual(tokens[2], Token(TokenType.END_TAG, "div"))

    def test_attributes(self) -> None:
        """Tests tag attribute extraction (double quotes, single quotes, unquoted, boolean)."""
        html = '<div id="main" class=\'container\' data-value=123 disabled>'
        tok = HTMLTokenizer(html)
        tokens = list(tok.tokenize())
        self.assertEqual(len(tokens), 1)

        expected_attrs = {
            "id": "main",
            "class": "container",
            "data-value": "123",
            "disabled": ""
        }
        self.assertEqual(tokens[0], Token(TokenType.START_TAG, "div", expected_attrs))

    def test_self_closing_tag(self) -> None:
        """Tests self-closing tags (both space-separated and direct slash)."""
        tok1 = HTMLTokenizer('<img src="pic.jpg"/>')
        tok2 = HTMLTokenizer('<img src="pic.jpg" />')

        t1 = list(tok1.tokenize())[0]
        t2 = list(tok2.tokenize())[0]

        self.assertTrue(t1.self_closing)
        self.assertEqual(t1.value, "img")
        self.assertEqual(t1.attributes, {"src": "pic.jpg"})

        self.assertTrue(t2.self_closing)
        self.assertEqual(t2.value, "img")
        self.assertEqual(t2.attributes, {"src": "pic.jpg"})

    def test_malformed_bracket(self) -> None:
        """Tests text comparison signs '<' that are not part of tag structures."""
        # '<' with trailing whitespace or numbers should treat it as text
        tok = HTMLTokenizer("a < b and 3 < 5")
        tokens = list(tok.tokenize())
        self.assertEqual(len(tokens), 1)
        self.assertEqual(tokens[0].type, TokenType.TEXT)
        self.assertEqual(tokens[0].value, "a < b and 3 < 5")


if __name__ == "__main__":
    unittest.main()
