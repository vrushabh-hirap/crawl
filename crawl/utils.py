"""Utility functions for the crawl-any-website library.

Provides auxiliary helper functions used across various modules for tasks like:
- Configuring standardized logging formats.
- Validating HTTP/HTTPS URLs.
- Compiling default headers to bypass simple scraper filters.
"""

import logging
import urllib.parse
from typing import Dict


def setup_logger(name: str = "crawl", level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a package-level logger.

    Sets up a default stream handler printing to stdout with date, level, and name
    formatting. Prevents duplicate handler registration if the logger is re-initialized.

    Args:
        name: Name of the logger channel. Defaults to 'crawl'.
        level: Logging severity threshold (e.g. logging.INFO, logging.DEBUG).

    Returns:
        A configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    
    # Check if handlers are already configured to avoid duplicate logs printing out
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler()
        
        # Format style: "[2026-07-10 20:00:00] INFO [crawl]: message"
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    return logger


def is_valid_url(url: str) -> bool:
    """Checks if a URL has a valid HTTP or HTTPS scheme and network location (domain).

    Ensures that the URL string is structurally valid and can be safely downloaded.

    Args:
        url: The string URL to check.

    Returns:
        True if the URL contains a valid scheme (http/https) and netloc, False otherwise.
    """
    try:
        # Parse the URL into components
        parsed = urllib.parse.urlparse(url)
        # Scheme must be either HTTP or HTTPS, and the domain (netloc) cannot be empty
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except Exception:
        # Any parsing exceptions return False indicating an invalid target URL
        return False


def get_default_headers() -> Dict[str, str]:
    """Returns a dictionary of default HTTP headers for web scraping.

    These headers mimic standard browser configurations to avoid immediate blocks by
    simple site filters.

    Returns:
        A dictionary containing standard browser headers (Accept, Accept-Language, Connection).
    """
    return {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Connection": "keep-alive"
    }
