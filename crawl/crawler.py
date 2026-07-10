"""Main crawling interface for the crawl library.

Provides the user-facing Crawl class for fetching web pages, and the Page
wrapper class containing extraction, traversal, and cleanup methods.
"""

from typing import Dict, List, Optional, Any
from crawl.downloader import Downloader, Response
from crawl.parser import parse_html
from crawl.dom import Node
from crawl.cleaner import clean_dom
import crawl.extractor as extractor
import crawl.exporter as exporter
from crawl.utils import get_default_headers, setup_logger

logger = setup_logger(__name__)


class Page:
    """Represents a downloaded and parsed web page.

    Wraps the parsed DOM tree and exposes high-level APIs for extraction,
    selection, DOM manipulation, cleaning, and exporting.
    """

    def __init__(self, url: str, response: Response, root_node: Node) -> None:
        """Initializes the Page.

        Args:
            url: The final absolute URL of the page.
            response: The raw HTTP Response.
            root_node: The root DOM Node of the parsed tree.
        """
        self.url = url
        self.response = response
        self.node = root_node

    def find(self, selector: str) -> Optional[Node]:
        """Finds the first element matching the CSS selector.

        Args:
            selector: CSS selector string.

        Returns:
            The first matching DOM Node, or None.
        """
        return self.node.find(selector)

    def find_all(self, selector: str) -> List[Node]:
        """Finds all elements matching the CSS selector.

        Args:
            selector: CSS selector string.

        Returns:
            List of matching DOM Nodes.
        """
        return self.node.find_all(selector)

    def get_text(self) -> str:
        """Extracts all text content from the document."""
        return self.node.get_text()

    def inner_html(self) -> str:
        """Returns the serialized inner HTML of the root document."""
        return self.node.inner_html()

    def outer_html(self) -> str:
        """Returns the serialized outer HTML of the root document."""
        return self.node.outer_html()

    # Data extraction APIs

    def title(self) -> str:
        """Returns the page title (falls back to first H1)."""
        return extractor.extract_title(self.node)

    def paragraphs(self) -> List[str]:
        """Returns the text contents of all paragraph (<p>) tags."""
        return extractor.extract_paragraphs(self.node)

    def images(self) -> List[str]:
        """Returns the absolute URLs of all images (<img> src attributes)."""
        return [img["src"] for img in extractor.extract_images(self.node, self.url) if img.get("src")]

    def tables(self) -> List[List[List[str]]]:
        """Returns structured row/cell data for all tables in the page."""
        return extractor.extract_tables(self.node)

    def forms(self) -> List[Dict[str, Any]]:
        """Returns structured metadata for all form elements in the page."""
        return extractor.extract_forms(self.node, self.url)

    def lists(self) -> List[List[str]]:
        """Returns items of all ordered and unordered lists on the page."""
        return extractor.extract_lists(self.node)

    def links(self) -> List[str]:
        """Returns the absolute target URLs (href attributes) of all hyperlink anchors."""
        return [lnk["href"] for lnk in extractor.extract_links(self.node, self.url) if lnk.get("href")]

    def buttons(self) -> List[Dict[str, str]]:
        """Returns structured metadata for all button and clickable input elements."""
        return extractor.extract_buttons(self.node)

    def meta_tags(self) -> List[Dict[str, str]]:
        """Returns metadata from all meta tags in the page header."""
        return extractor.extract_meta_tags(self.node)

    def headers(self) -> List[Dict[str, str]]:
        """Returns all headings (H1 through H6) in order of appearance."""
        return extractor.extract_headers(self.node)

    # Cleaning and Normalization

    def clean(self) -> None:
        """Cleans the parsed DOM in-place (normalizes whitespace, decodes entities, trims empty tags)."""
        clean_dom(self.node)

    # Exporting helper shortcuts

    def export_json(self, filepath: str) -> None:
        """Helper to export all primary extracted data to a JSON file.

        Args:
            filepath: Target file path.
        """
        data = {
            "title": self.title(),
            "headers": self.headers(),
            "paragraphs": self.paragraphs(),
            "links": self.links(),
            "images": self.images(),
            "meta": self.meta_tags()
        }
        exporter.export_json(data, filepath)

    def __repr__(self) -> str:
        return f"<Page URL: {self.url} Title: {repr(self.title()[:30])}>"


class Crawl:
    """The main interface of the crawl engine.

    Coordinates downloader, parser, and exporter steps.
    Designed to allow extension with JS rendering, robots.txt checking,
    rate limiting, concurrent downloads, and plugin hooks.
    """

    def __init__(
        self,
        default_headers: Optional[Dict[str, str]] = None,
        timeout: float = 10.0,
        verify_ssl: bool = True,
        rate_limit_delay: float = 0.0,
        obey_robots: bool = False,
        plugins: Optional[List[Any]] = None
    ) -> None:
        """Initializes the Crawl instance.

        Args:
            default_headers: Headers to send with every request.
            timeout: Connection timeout.
            verify_ssl: Verify SSL/TLS certificates.
            rate_limit_delay: Delay (in seconds) between sequential requests.
            obey_robots: Future expansion placeholder to respect robots.txt rules.
            plugins: Future expansion placeholder for engine middleware/plugins.
        """
        headers = get_default_headers()
        if default_headers:
            headers.update(default_headers)

        self.downloader = Downloader(default_headers=headers, timeout=timeout, verify_ssl=verify_ssl)
        self.rate_limit_delay = rate_limit_delay
        self.obey_robots = obey_robots
        self.plugins = plugins or []

        # Future extensions could initialize robots.txt parser, rate limiter, or JS rendering context here.
        if self.obey_robots:
            logger.info("Robots.txt parsing is currently not implemented, but configured to obey.")
        if self.plugins:
            logger.info(f"Loaded {len(self.plugins)} plugins.")

    def fetch(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None
    ) -> Page:
        """Downloads and parses a web page.

        Args:
            url: Absolute HTTP/HTTPS URL.
            headers: Request-specific headers.
            timeout: Request-specific timeout.

        Returns:
            A Page object representing the parsed resource.
        """
        # Future extension: Rate limiting sleep here
        # Future extension: Robots.txt rules checking here

        # 1. Download
        response = self.downloader.download(url, headers=headers, timeout=timeout)

        # 2. Parse HTML text to DOM tree
        root_node = parse_html(response.text)

        # 3. Construct and return Page wrapper
        return Page(url=response.url, response=response, root_node=root_node)
