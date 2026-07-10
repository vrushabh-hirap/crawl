"""Custom exceptions for the crawl library.

This module defines all the structured errors that can be raised by the
downloader, parser, selector engine, or export tools.
"""

class CrawlError(Exception):
    """Base exception class for all errors raised by the crawl package."""
    pass


class DownloadError(CrawlError):
    """Raised when the Downloader fails to fetch a URL (e.g. network failure, timeout, non-200 status)."""
    pass


class ParserError(CrawlError):
    """Raised when the HTML Parser encounters an unrecoverable parsing error."""
    pass


class InvalidSelector(CrawlError):
    """Raised when a CSS selector string cannot be parsed or is malformed."""
    pass


class NodeNotFound(CrawlError):
    """Raised when a DOM query requires finding a node but none matches."""
    pass


class ExportError(CrawlError):
    """Raised when exporting data to CSV, JSON, or TXT fails."""
    pass
