"""Unit tests for the Data and Link Extractor engine."""

import unittest
from crawl.parser import parse_html
from crawl.crawler import Page
from crawl.downloader import Response
import crawl.extractor as extractor


class TestExtractor(unittest.TestCase):
    """Verifies all extraction endpoints extract data and resolve URLs properly."""

    def setUp(self) -> None:
        self.html = """
        <html>
        <head>
            <title>Test Page Title</title>
            <meta name="description" content="A test page description">
            <meta property="og:title" content="OpenGraph Title">
        </head>
        <body>
            <h1>Main Title</h1>
            <h2>Sub Title</h2>
            <p>First paragraph with <a href="/about" download="info">about link</a>.</p>
            <p>Second paragraph.</p>
            
            <img src="img/pic.png" alt="My Pic" title="Picture">
            <img src="http://external.com/logo.jpg">
            
            <table>
                <tr><th>Header 1</th><th>Header 2</th></tr>
                <tr><td>Val 1</td><td>Val 2</td></tr>
            </table>

            <form action="/submit" method="post">
                <input type="text" name="username" value="vrushabh">
                <select name="role">
                    <option value="admin">Administrator</option>
                    <option>User</option>
                </select>
                <textarea name="bio">My bio</textarea>
                <button type="submit">Submit Form</button>
            </form>
            
            <ul>
                <li>Item A</li>
                <li>Item B</li>
            </ul>
            
            <input type="button" name="cancel" value="Cancel Action">
        </body>
        </html>
        """
        self.root = parse_html(self.html)
        # Wrap in Page for testing Page extraction shortcuts too
        dummy_response = Response(url="https://example.com/index.html", status_code=200, headers={}, content=self.html.encode("utf-8"))
        self.page = Page(url="https://example.com/index.html", response=dummy_response, root_node=self.root)

    def test_extract_title(self) -> None:
        """Tests page title extraction."""
        self.assertEqual(extractor.extract_title(self.root), "Test Page Title")

        # Fallback to H1 if no title tag
        no_title_html = "<body><h1>First Heading</h1></body>"
        no_title_root = parse_html(no_title_html)
        self.assertEqual(extractor.extract_title(no_title_root), "First Heading")

    def test_extract_paragraphs(self) -> None:
        """Tests paragraphs extraction."""
        paragraphs = extractor.extract_paragraphs(self.root)
        self.assertEqual(len(paragraphs), 2)
        self.assertEqual(paragraphs[0], "First paragraph with about link.")
        self.assertEqual(paragraphs[1], "Second paragraph.")

    def test_extract_images(self) -> None:
        """Tests images extraction and absolute URL resolution."""
        images = extractor.extract_images(self.root, base_url=self.page.url)
        self.assertEqual(len(images), 2)
        # Relative URL resolved
        self.assertEqual(images[0]["src"], "https://example.com/img/pic.png")
        self.assertEqual(images[0]["alt"], "My Pic")
        self.assertEqual(images[0]["title"], "Picture")
        # Absolute URL preserved
        self.assertEqual(images[1]["src"], "http://external.com/logo.jpg")

    def test_extract_tables(self) -> None:
        """Tests HTML tables extraction."""
        tables = extractor.extract_tables(self.root)
        self.assertEqual(len(tables), 1)
        table = tables[0]
        self.assertEqual(len(table), 2)  # Two rows
        self.assertEqual(table[0], ["Header 1", "Header 2"])
        self.assertEqual(table[1], ["Val 1", "Val 2"])

    def test_extract_forms(self) -> None:
        """Tests form structure and inputs extraction."""
        forms = extractor.extract_forms(self.root, base_url=self.page.url)
        self.assertEqual(len(forms), 1)
        form = forms[0]
        self.assertEqual(form["action"], "https://example.com/submit")
        self.assertEqual(form["method"], "POST")
        
        inputs = form["inputs"]
        self.assertEqual(len(inputs), 3)
        self.assertEqual(inputs[0], {"tag": "input", "name": "username", "type": "text", "value": "vrushabh"})
        self.assertEqual(inputs[1], {"tag": "select", "name": "role", "options": ["admin", "User"]})
        self.assertEqual(inputs[2], {"tag": "textarea", "name": "bio", "value": "My bio"})

    def test_extract_lists(self) -> None:
        """Tests ordered/unordered lists extraction."""
        lists = extractor.extract_lists(self.root)
        self.assertEqual(len(lists), 1)
        self.assertEqual(lists[0], ["Item A", "Item B"])

    def test_extract_links(self) -> None:
        """Tests anchors href extraction and attribute tracking."""
        links = extractor.extract_links(self.root, base_url=self.page.url)
        self.assertEqual(len(links), 1)
        link = links[0]
        self.assertEqual(link["text"], "about link")
        self.assertEqual(link["href"], "https://example.com/about")
        self.assertEqual(link["download"], "info")

    def test_extract_all_links_attributes(self) -> None:
        """Tests extracting all occurrences of href/src/action/download attributes."""
        all_links = extractor.extract_all_links_attributes(self.root, base_url=self.page.url)
        # Should find meta name description (no link attribute),
        # a href, img src, img src, form action, a download...
        # Let's verify we found specific ones
        hrefs = [l for l in all_links if l["attribute"] == "href"]
        srcs = [l for l in all_links if l["attribute"] == "src"]
        actions = [l for l in all_links if l["attribute"] == "action"]
        downloads = [l for l in all_links if l["attribute"] == "download"]

        self.assertEqual(len(hrefs), 1)
        self.assertEqual(hrefs[0]["resolved_url"], "https://example.com/about")

        self.assertEqual(len(srcs), 2)
        self.assertEqual(srcs[0]["resolved_url"], "https://example.com/img/pic.png")

        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0]["resolved_url"], "https://example.com/submit")

        self.assertEqual(len(downloads), 1) # a has download attribute, and we check all tags for download
        # The anchor <a> has href AND download attributes, so it adds to list twice (once for href, once for download)
        # img does not have download.
        self.assertEqual(downloads[0]["tag"], "a")

    def test_extract_buttons(self) -> None:
        """Tests button elements and button-like inputs extraction."""
        buttons = extractor.extract_buttons(self.root)
        self.assertEqual(len(buttons), 2)
        self.assertEqual(buttons[0]["text"], "Submit Form")
        self.assertEqual(buttons[0]["type"], "submit")
        
        self.assertEqual(buttons[1]["text"], "Cancel Action")
        self.assertEqual(buttons[1]["type"], "button")

    def test_extract_meta_tags(self) -> None:
        """Tests page metadata extraction."""
        meta = extractor.extract_meta_tags(self.root)
        self.assertEqual(len(meta), 2)
        self.assertEqual(meta[0]["name"], "description")
        self.assertEqual(meta[0]["content"], "A test page description")
        self.assertEqual(meta[1]["name"], "og:title")
        self.assertEqual(meta[1]["content"], "OpenGraph Title")

    def test_extract_headers(self) -> None:
        """Tests h1-h6 headings extraction."""
        headers = extractor.extract_headers(self.root)
        self.assertEqual(len(headers), 2)
        self.assertEqual(headers[0], {"level": "h1", "text": "Main Title"})
        self.assertEqual(headers[1], {"level": "h2", "text": "Sub Title"})

    def test_page_shortcut_apis(self) -> None:
        """Verifies high-level shortcuts exposed on the Page wrapper class."""
        self.assertEqual(self.page.title(), "Test Page Title")
        self.assertEqual(len(self.page.links()), 1)
        self.assertEqual(self.page.links()[0], "https://example.com/about")
        self.assertEqual(len(self.page.images()), 2)
        self.assertEqual(self.page.images()[0], "https://example.com/img/pic.png")


if __name__ == "__main__":
    unittest.main()
