"""Unit tests for the CSS Selector engine."""

import unittest
from crawl.parser import parse_html
from crawl.selector import select
from crawl.exceptions import InvalidSelector


class TestSelector(unittest.TestCase):
    """Verifies all selector types, combinators, and error states."""

    def setUp(self) -> None:
        self.html = """
        <div id="main" class="container active">
            <span class="highlight">Hello</span>
            <p class="text">
                <span class="highlight bold">Nested Span</span>
            </p>
            <div class="footer">
                <span class="links">Link</span>
            </div>
        </div>
        <p class="text">Outside Paragraph</p>
        """
        self.root = parse_html(self.html)

    def test_tag_selector(self) -> None:
        """Tests selecting elements by tag name."""
        spans = select(self.root, "span")
        self.assertEqual(len(spans), 3)
        self.assertEqual(spans[0].get_text(), "Hello")

    def test_id_selector(self) -> None:
        """Tests selecting by ID."""
        main = select(self.root, "#main")
        self.assertEqual(len(main), 1)
        self.assertEqual(main[0].get_attribute("id"), "main")

    def test_class_selectors(self) -> None:
        """Tests selecting by single and multiple classes."""
        # Single class
        highlights = select(self.root, ".highlight")
        self.assertEqual(len(highlights), 2)

        # Multiple classes (logical AND)
        bold_highlight = select(self.root, ".highlight.bold")
        self.assertEqual(len(bold_highlight), 1)
        self.assertEqual(bold_highlight[0].get_text(), "Nested Span")

    def test_descendant_selector(self) -> None:
        """Tests descendant spaces combinators (e.g. div span)."""
        div_spans = select(self.root, "div span")
        # Should match: span.highlight, span.highlight.bold, span.links
        self.assertEqual(len(div_spans), 3)

        p_spans = select(self.root, "p span")
        # Should match span.highlight.bold inside p
        self.assertEqual(len(p_spans), 1)
        self.assertEqual(p_spans[0].get_text(), "Nested Span")

    def test_direct_child_selector(self) -> None:
        """Tests direct child combinator (e.g. div > span)."""
        direct_spans = select(self.root, "div > span")
        # span.highlight is direct child of div#main
        # span.links is direct child of div.footer
        # span.highlight.bold is NOT a direct child of div (it is inside p)
        self.assertEqual(len(direct_spans), 2)
        self.assertEqual(direct_spans[0].get_text(), "Hello")
        self.assertEqual(direct_spans[1].get_text(), "Link")

    def test_comma_separated_selectors(self) -> None:
        """Tests comma-separated alternative selector queries (e.g. h1, h2)."""
        matches = select(self.root, ".footer, .links")
        # Should return div.footer and span.links in document order
        self.assertEqual(len(matches), 2)
        self.assertEqual(matches[0].tag, "div")
        self.assertEqual(matches[1].tag, "span")

    def test_invalid_selectors(self) -> None:
        """Tests that malformed CSS selector strings raise InvalidSelector exception."""
        invalid_queries = [
            "",
            "div >",
            "> span",
            "div > > span",
            ", div",
        ]
        for query in invalid_queries:
            with self.subTest(query=query):
                with self.assertRaises(InvalidSelector):
                    select(self.root, query)


if __name__ == "__main__":
    unittest.main()
