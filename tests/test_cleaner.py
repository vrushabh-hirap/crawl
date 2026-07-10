"""Unit tests for the Cleaning engine."""

import unittest
from crawl.parser import parse_html
from crawl.dom import Node
from crawl.cleaner import (
    normalize_whitespace,
    decode_entities,
    clean_dom,
    is_identical_subtree
)


class TestCleaner(unittest.TestCase):
    """Verifies entity resolution, whitespace formatting, and node pruning."""

    def test_normalize_whitespace(self) -> None:
        """Tests that multiple spaces/newlines/tabs are collapsed to one space."""
        self.assertEqual(normalize_whitespace("  hello   world  \n  "), "hello world")
        self.assertEqual(normalize_whitespace("\tinfo\tdata"), "info data")
        self.assertEqual(normalize_whitespace(""), "")

    def test_decode_entities(self) -> None:
        """Tests standard HTML entities decoding."""
        self.assertEqual(decode_entities("&amp;&lt;&gt;&quot;&apos;&#x27;"), '&<>"\'\'')
        # Let's check a concrete decoded string
        self.assertEqual(decode_entities("A &amp; B"), "A & B")
        self.assertEqual(decode_entities("3 &lt; 5"), "3 < 5")

    def test_is_identical_subtree(self) -> None:
        """Tests matching check for identical subtrees."""
        n1 = parse_html("<div><p class='text'>Hello</p></div>").children[0]
        n2 = parse_html("<div><p class='text'>Hello</p></div>").children[0]
        n3 = parse_html("<div><p class='other'>Hello</p></div>").children[0]

        self.assertTrue(is_identical_subtree(n1, n2))
        self.assertFalse(is_identical_subtree(n1, n3))

    def test_clean_dom_operations(self) -> None:
        """Tests complete cleaning operations (entities, whitespace, empty tags, duplicates)."""
        html = """
        <div id="main">
            <!-- Comment with entity &amp; -->
            <p>Hello  &amp;   Welcome  </p>
            <div></div> <!-- empty, should be pruned -->
            <div class="styled"></div> <!-- empty but has attrs, should NOT be pruned by default -->
            <img src="pic.jpg"> <!-- void element, should NOT be pruned -->
            
            <ul>
                <li>Item 1</li>
                <li>Item 1</li> <!-- duplicate element, should be pruned -->
                <li>Item 2</li>
            </ul>
        </div>
        """
        root = parse_html(html)
        clean_dom(root)

        div_main = root.find("#main")
        self.assertIsNotNone(div_main)

        # 1. Text node entity decode & whitespace check
        p = div_main.find("p")
        self.assertIsNotNone(p)
        self.assertEqual(p.get_text(), "Hello & Welcome")

        comments = [c for c in div_main.children if c.tag == "#comment"]
        self.assertTrue(len(comments) >= 1)
        comment = comments[0]
        self.assertEqual(comment.text, " Comment with entity & ")

        # 3. Empty tag checks
        # The plain empty div should be gone.
        # The styled empty div and the void img should remain.
        all_divs = div_main.find_all("div")
        self.assertEqual(len(all_divs), 1)  # Only the .styled one remains
        self.assertEqual(all_divs[0].get_attribute("class"), "styled")

        img = div_main.find("img")
        self.assertIsNotNone(img)

        # 4. Duplicate tag check
        # Sibling duplicate `<li>Item 1</li>` should be pruned.
        ul = div_main.find("ul")
        self.assertIsNotNone(ul)
        ul_elements = [c for c in ul.children if c.tag == "li"]
        self.assertEqual(len(ul_elements), 2)  # One 'Item 1' and one 'Item 2'
        self.assertEqual(ul_elements[0].get_text(), "Item 1")
        self.assertEqual(ul_elements[1].get_text(), "Item 2")


if __name__ == "__main__":
    unittest.main()
