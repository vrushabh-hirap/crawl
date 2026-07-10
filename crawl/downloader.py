"""HTTP/HTTPS Downloader for the crawl-any-website library.

This module is responsible only for opening network connections,
fetching HTML content, handling redirects, headers, timeouts, and
constructing HTTP Response objects. It contains no HTML parsing logic.
Supports automatic content decoding of compressed transfer payloads (gzip/deflate).
"""

import urllib.request
import urllib.error
import urllib.parse
import ssl
import gzip
import zlib
from typing import Dict, List, Optional, Any
from crawl.exceptions import DownloadError


def decompress_content(content: bytes, headers: Dict[str, str]) -> bytes:
    """Decompresses transfer body content if encoded in gzip or deflate formats.

    Args:
        content: Raw binary response bytes.
        headers: Unfiltered response headers dictionary.

    Returns:
        Decompressed content bytes. If decoding fails or is not required, returns raw content.
    """
    if not content:
        return content
        
    # Search response headers case-insensitively for the 'content-encoding' indicator
    content_encoding = ""
    for k, v in headers.items():
        if k.lower() == "content-encoding":
            content_encoding = v.lower()
            break

    # Handle Gzip compression
    if "gzip" in content_encoding:
        try:
            return gzip.decompress(content)
        except Exception:
            pass  # Fallback to returning raw bytes if decompression fails
            
    # Handle Deflate compression
    elif "deflate" in content_encoding:
        try:
            # Try decompressing with raw deflate format (sans zlib headers, using max negative window bits)
            return zlib.decompress(content, -zlib.MAX_WBITS)
        except Exception:
            try:
                # Fallback to standard zlib decompression
                return zlib.decompress(content)
            except Exception:
                pass
                
    return content


class Response:
    """Represents an HTTP response returned by the Downloader.

    Attributes:
        url: The final absolute URL after following all redirection hops.
        status_code: The HTTP status code of the response (e.g. 200, 404).
        headers: A dictionary of response headers (normalized to lowercase keys).
        content: The raw binary content of the response.
        redirect_history: A list of dicts detailing each redirection step.
    """

    def __init__(
        self,
        url: str,
        status_code: int,
        headers: Dict[str, str],
        content: bytes,
        redirect_history: Optional[List[Dict[str, Any]]] = None
    ):
        self.url = url
        self.status_code = status_code
        # Standardize headers keys to lowercase for standard lookup
        self.headers = {k.lower(): v for k, v in headers.items()}
        self.content = content
        self.redirect_history = redirect_history or []

    @property
    def text(self) -> str:
        """Decodes the raw binary content into a Python string.

        Tries to extract the charset from the 'Content-Type' header.
        Defaults to UTF-8 if no charset is specified.
        """
        content_type = self.headers.get("content-type", "")
        encoding = "utf-8"
        
        # Look for text charset declaration (e.g., "text/html; charset=ISO-8859-1")
        if "charset=" in content_type:
            parts = content_type.split("charset=")
            if len(parts) > 1:
                # Extract clean encoding tag name
                encoding = parts[1].split(";")[0].strip().strip('"\'')

        try:
            return self.content.decode(encoding)
        except (LookupError, UnicodeDecodeError):
            # Fallback with replacement characters if the parsed encoding is unsupported
            return self.content.decode("utf-8", errors="replace")

    def __repr__(self) -> str:
        return f"<Response [{self.status_code}] URL: {self.url}>"


class TrackingRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Custom redirect handler that tracks the redirect chain history.

    Inherits from urllib's HTTPRedirectHandler and intercepts redirects to log each
    hop's details (URL, status code, headers).
    """

    def __init__(self) -> None:
        super().__init__()
        # List of tracked redirection hops
        self.history: List[Dict[str, Any]] = []

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str
    ) -> urllib.request.Request:
        # Record redirect hop before forwarding the request
        self.history.append({
            "status": code,
            "url": req.full_url,
            "headers": dict(headers.items()) if hasattr(headers, "items") else {}
        })
        # Proceed with the standard redirect execution
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class Downloader:
    """HTTP/HTTPS Downloader for fetching web resources.

    Ensures proper connection management, custom header injection, timeout handling,
    User-Agent configuration, and tracking of HTTP redirects.
    """

    DEFAULT_USER_AGENT = "CrawlEngine/0.1.0 (+https://github.com/vrushabhhirap/Crawl)"

    def __init__(
        self,
        default_headers: Optional[Dict[str, str]] = None,
        timeout: float = 10.0,
        verify_ssl: bool = True
    ):
        """Initializes the Downloader.

        Args:
            default_headers: Headers to include in every request.
            timeout: Default request timeout in seconds.
            verify_ssl: If False, SSL certificates won't be verified.
        """
        self.default_headers = default_headers or {}
        # Ensure a user agent is set by default if not specified
        has_ua = any(k.lower() == "user-agent" for k in self.default_headers)
        if not has_ua:
            self.default_headers["User-Agent"] = self.DEFAULT_USER_AGENT
        self.timeout = timeout
        self.verify_ssl = verify_ssl

    def download(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None
    ) -> Response:
        """Downloads the content of the specified URL.

        Args:
            url: The HTTP/HTTPS URL to download.
            headers: Optional request-specific headers to merge or override defaults.
            timeout: Optional request-specific timeout override.

        Returns:
            A Response object containing status, headers, and body.

        Raises:
            DownloadError: If a connection error, SSL failure, or timeout occurs.
        """
        # 1. Merge default and request-specific headers
        req_headers = self.default_headers.copy()
        if headers:
            req_headers.update(headers)

        # 2. Build urllib request target
        req = urllib.request.Request(url, headers=req_headers)
        req_timeout = timeout if timeout is not None else self.timeout

        # 3. Configure SSL validation contexts
        if not self.verify_ssl:
            # Disable certificate checking (useful for self-signed certificates or local testing issues)
            ssl_context = ssl._create_unverified_context()
        else:
            # Enable standard secure certificate verification
            ssl_context = ssl.create_default_context()

        # 4. Construct opener with Cookie support, Redirect history tracking, and SSL context
        redirect_handler = TrackingRedirectHandler()
        cookie_processor = urllib.request.HTTPCookieProcessor()
        https_handler = urllib.request.HTTPSHandler(context=ssl_context)
        
        opener = urllib.request.build_opener(
            redirect_handler,
            cookie_processor,
            https_handler
        )

        try:
            # Open connection
            with opener.open(req, timeout=req_timeout) as res:
                # Read raw bytes payload
                content = res.read()
                response_headers = dict(res.info())
                # Handle decompression if the response is gzip or deflate compressed
                content = decompress_content(content, response_headers)
                status_code = res.getcode()
                final_url = res.geturl()

                return Response(
                    url=final_url,
                    status_code=status_code,
                    headers=response_headers,
                    content=content,
                    redirect_history=redirect_handler.history
                )

        except urllib.error.HTTPError as e:
            # Handle HTTP errors gracefully (e.g. 404 Not Found, 500 Server Error)
            try:
                content = e.read()
                # Decompress error body content if compressed
                content = decompress_content(content, dict(e.headers))
            except Exception:
                content = b""
            return Response(
                url=e.geturl(),
                status_code=e.code,
                headers=dict(e.headers),
                content=content,
                redirect_history=redirect_handler.history
            )
        except urllib.error.URLError as e:
            # Handle network level failures like unreachable hosts, bad domains, or timeouts
            raise DownloadError(f"Network error downloading {url}: {e.reason}") from e
        except TimeoutError as e:
            # Handle socket level timeouts
            raise DownloadError(f"Timeout occurred while downloading {url}") from e
        except Exception as e:
            # Catch all unexpected connection failures
            raise DownloadError(f"Unexpected error downloading {url}: {str(e)}") from e
