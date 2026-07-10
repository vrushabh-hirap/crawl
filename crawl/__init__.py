"""Crawl: A production-quality web crawling and scraping library written from scratch.

This package provides:
- A custom HTML tokenizer and parser
- A manual DOM implementation with selectors
- A structured data extractor and cleaning engine
- Multi-format exporters (CSV, JSON, TXT)
"""

from crawl.version import __version__
from crawl.exceptions import (
    CrawlError,
    DownloadError,
    ParserError,
    InvalidSelector,
    NodeNotFound,
    ExportError
)
from crawl.downloader import Downloader, Response
from crawl.dom import Node
from crawl.crawler import Crawl

__all__ = [
    "__version__",
    "CrawlError",
    "DownloadError",
    "ParserError",
    "InvalidSelector",
    "NodeNotFound",
    "ExportError",
    "Downloader",
    "Response",
    "Node",
    "Crawl"
]
