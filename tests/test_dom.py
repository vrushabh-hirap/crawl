"""Unit tests for the DOM Node class."""

import unittest
from crawl.dom import Node


class TestDOMNode(unittest.TestCase):
    """Verifies DOM tree operations and serialization."""

    def test_append_and_remove_child(self) -> None:
        """Tests adding and removing child nodes, verifying parent reassignments."""
        parent = Node(tag="div")
        child = Node(tag="span")

        parent.append_child(child)
        self.assertEqual(len(parent.children), 1)
        self.assertEqual(child.parent, parent)

        # Move to a new parent
        parent2 = Node(tag="p")
        parent2.append_child(child)
        self.assertEqual(len(parent.children), 0)
        self.assertEqual(len(parent2.children), 1)
        self.assertEqual(child.parent, parent2)

        # Remove child
        parent2.remove_child(child)
        self.assertEqual(len(parent2.children), 0)
        self.assertIsNone(child.parent)

    def test_attributes(self) -> None:
        """Tests setting and getting attributes, and case insensitivity."""
        node = Node(tag="div", attributes={"ID": "header", "class": "main-nav"})

        self.assertEqual(node.get_attribute("id"), "header")
        self.assertEqual(node.get_attribute("ID"), "header")
        self.assertEqual(node.get_attribute("class"), "main-nav")
        self.assertIsNone(node.get_attribute("href"))

        node.set_attribute("Href", "/home")
        self.assertEqual(node.get_attribute("href"), "/home")

    def test_serialization(self) -> None:
        """Tests inner_html and outer_html serialization for elements, void tags, and texts."""
        parent = Node(tag="div", attributes={"class": "container"})
        span = Node(tag="span")
        text = Node(tag=None, text="Hello")

        span.append_child(text)
        parent.append_child(span)

        # Non-void elements
        self.assertEqual(span.outer_html(), "<span>Hello</span>")
        self.assertEqual(span.inner_html(), "Hello")

        # Parent containing span
        self.assertEqual(parent.inner_html(), "<span>Hello</span>")
        self.assertEqual(parent.outer_html(), '<div class="container"><span>Hello</span></div>')

        # Void element serialization (should not have a closing tag)
        img = Node(tag="img", attributes={"src": "logo.png"})
        self.assertEqual(img.outer_html(), '<img src="logo.png">')
        self.assertEqual(img.inner_html(), "")

    def test_get_text_recursive(self) -> None:
        """Verifies recursive text extraction gets all text nodes in pre-order."""
        parent = Node(tag="div")
        t1 = Node(tag=None, text="Hello ")
        span = Node(tag="span")
        t2 = Node(tag=None, text="World")

        span.append_child(t2)
        parent.append_child(t1)
        parent.append_child(span)

        self.assertEqual(parent.get_text(), "Hello World")


if __name__ == "__main__":
    unittest.main()
