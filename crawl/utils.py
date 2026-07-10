"""Utility functions for the crawl library.

Provides helper functions for URL validation, default configurations,
and custom logging setup.
"""

import logging
import urllib.parse
from typing import Dict


def setup_logger(name: str = "crawl", level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a package-level logger.

    Args:
        name: Logger name.
        level: Logging level (e.g. logging.INFO).

    Returns:
        A configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def is_valid_url(url: str) -> bool:
    """Checks if a URL has a valid HTTP/HTTPS scheme and network location.

    Args:
        url: The string URL to check.

    Returns:
        True if valid, False otherwise.
    """
    try:
        parsed = urllib.parse.urlparse(url)
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except Exception:
        return False


def get_default_headers() -> Dict[str, str]:
    """Returns a dictionary of default HTTP headers for web scraping.

    Includes standard Accept and connection headers.
    """
    return {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Connection": "keep-alive"
    }
