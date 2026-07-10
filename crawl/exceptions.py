"""Custom exceptions for the crawl-any-website library.

This module defines the entire exception hierarchy used across all modules.
By separating exceptions, users of the library can selectively catch failures
raised during downloading, parsing, query selector execution, data extraction,
or format exporting.
"""

class CrawlError(Exception):
    """Base exception class for all errors raised by the crawl-any-website package.
    
    All other custom exceptions in this package inherit from this class. This allows
    users to catch any library-specific error with a single 'except CrawlError' block.
    """
    pass


class DownloadError(CrawlError):
    """Raised when the Downloader fails to fetch a URL.
    
    This includes scenarios such as:
    - Network connection failures (e.g. host not found, connection reset)
    - Request timeouts (when server exceeds designated wait times)
    - SSL handshake failures (e.g. invalid certificates)
    - HTTP protocol errors
    """
    pass


class ParserError(CrawlError):
    """Raised when the HTML Parser encounters an unrecoverable parsing error.
    
    This exception indicates that the lexical scanner or the stack-based tree construction
    module encountered malformed data or internal state errors that prevent building the DOM.
    """
    pass


class InvalidSelector(CrawlError):
    """Raised when a CSS selector string cannot be parsed or is malformed.
    
    Examples of invalid selectors that trigger this error:
    - Double commas, trailing commas, or leading commas (e.g. 'div,,span', 'div,')
    - Empty selector strings
    - Invalid combinator chains (e.g. consecutive direct child operators 'div > > span')
    """
    pass


class NodeNotFound(CrawlError):
    """Raised when a DOM query requires finding a specific node but none is found.
    
    Used when query operations expect at least one matching node but find an empty set.
    """
    pass


class ExportError(CrawlError):
    """Raised when exporting data to CSV, JSON, or TXT formats fails.
    
    This covers issues like:
    - Failures during serialization (e.g. trying to serialize non-serializable objects to JSON)
    - File system write errors (e.g. permission denied, disk full, invalid filepath)
    """
    pass
