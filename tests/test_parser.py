"""Unit tests for the HTML Parser."""

import unittest
from crawl.parser import parse_html, HTMLParser
from crawl.exceptions import ParserError


class TestHTMLParser(unittest.TestCase):
    """Verifies that HTMLParser constructs correct DOM structures."""

    def test_basic_tree(self) -> None:
        """Tests standard well-formed HTML parses to correct tree structure."""
        html = "<html><body><div><p>Hello</p></div></body></html>"
        root = parse_html(html)

        # root is the root dummy node
        self.assertEqual(len(root.children), 1)
        html_node = root.children[0]
        self.assertEqual(html_node.tag, "html")

        body_node = html_node.children[0]
        self.assertEqual(body_node.tag, "body")

        div_node = body_node.children[0]
        self.assertEqual(div_node.tag, "div")

        p_node = div_node.children[0]
        self.assertEqual(p_node.tag, "p")

        text_node = p_node.children[0]
        self.assertIsNone(text_node.tag)
        self.assertEqual(text_node.text, "Hello")

    def test_void_tags_non_nesting(self) -> None:
        """Tests that void tags (like img, br) do not wrap subsequent elements as children."""
        html = "<div><img src='pic.jpg'><p>Text</p></div>"
        root = parse_html(html)

        div_node = root.children[0]
        self.assertEqual(len(div_node.children), 2)

        img_node = div_node.children[0]
        self.assertEqual(img_node.tag, "img")
        self.assertEqual(len(img_node.children), 0)

        p_node = div_node.children[1]
        self.assertEqual(p_node.tag, "p")

    def test_unclosed_tags_recovery(self) -> None:
        """Tests that unclosed nested tags are auto-closed when a parent tag is closed."""
        html = "<div><p>Hello</div>"
        root = parse_html(html)

        div_node = root.children[0]
        self.assertEqual(div_node.tag, "div")
        self.assertEqual(len(div_node.children), 1)

        p_node = div_node.children[0]
        self.assertEqual(p_node.tag, "p")
        self.assertEqual(p_node.get_text(), "Hello")

    def test_doctype_and_comments(self) -> None:
        """Tests that doctypes and comments are parsed as special node types."""
        html = "<!DOCTYPE html><!-- comment --><div>text</div>"
        root = parse_html(html)

        self.assertEqual(len(root.children), 3)

        doctype = root.children[0]
        self.assertEqual(doctype.tag, "#doctype")
        self.assertEqual(doctype.text, "html")

        comment = root.children[1]
        self.assertEqual(comment.tag, "#comment")
        self.assertEqual(comment.text, " comment ")

        div = root.children[2]
        self.assertEqual(div.tag, "div")


if __name__ == "__main__":
    unittest.main()
