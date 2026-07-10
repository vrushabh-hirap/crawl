"""HTTP/HTTPS Downloader for the crawl library.

This module is responsible only for opening network connections,
fetching HTML content, handling redirects, headers, timeouts, and
constructing HTTP Response objects. It contains no parsing logic.
"""

import urllib.request
import urllib.error
import urllib.parse
import ssl
from typing import Dict, List, Optional, Any
from crawl.exceptions import DownloadError


class Response:
    """Represents an HTTP response returned by the Downloader.

    Attributes:
        url: The final URL after any redirects.
        status_code: The HTTP status code of the response.
        headers: A dictionary of response headers (case-insensitive keys).
        content: The raw binary content of the response.
        redirect_history: A list of dicts detailing redirect hops.
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
        self.headers = {k.lower(): v for k, v in headers.items()}
        self.content = content
        self.redirect_history = redirect_history or []

    @property
    def text(self) -> str:
        """Decodes the raw binary content into a string.

        Tries to detect charset from Content-Type, defaulting to utf-8.
        """
        content_type = self.headers.get("content-type", "")
        encoding = "utf-8"
        if "charset=" in content_type:
            parts = content_type.split("charset=")
            if len(parts) > 1:
                encoding = parts[1].split(";")[0].strip().strip('"\'')

        try:
            return self.content.decode(encoding)
        except (LookupError, UnicodeDecodeError):
            return self.content.decode("utf-8", errors="replace")

    def __repr__(self) -> str:
        return f"<Response [{self.status_code}] URL: {self.url}>"


class TrackingRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Custom redirect handler that tracks the redirect chain history."""

    def __init__(self) -> None:
        super().__init__()
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
        # Record the redirection hop before forwarding
        self.history.append({
            "status": code,
            "url": req.full_url,
            "headers": dict(headers.items()) if hasattr(headers, "items") else {}
        })
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
        # Ensure a user agent is set
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
        req_headers = self.default_headers.copy()
        if headers:
            req_headers.update(headers)

        req = urllib.request.Request(url, headers=req_headers)
        req_timeout = timeout if timeout is not None else self.timeout

        # Configure SSL context
        if not self.verify_ssl:
            ssl_context = ssl._create_unverified_context()
        else:
            ssl_context = ssl.create_default_context()

        # Build custom opener to track redirects
        redirect_handler = TrackingRedirectHandler()
        cookie_processor = urllib.request.HTTPCookieProcessor()
        opener = urllib.request.build_opener(redirect_handler, cookie_processor)

        try:
            # We must pass the context to urlopen directly or install the opener.
            # Building opener and using opener.open is standard.
            # But opener.open does not directly take context. Instead, we can pass
            # an HTTPSHandler with our context.
            https_handler = urllib.request.HTTPSHandler(context=ssl_context)
            opener = urllib.request.build_opener(
                redirect_handler,
                cookie_processor,
                https_handler
            )

            with opener.open(req, timeout=req_timeout) as res:
                content = res.read()
                # res.getcode() and res.info() retrieve the final response attributes
                status_code = res.getcode()
                response_headers = dict(res.info())
                final_url = res.geturl()

                return Response(
                    url=final_url,
                    status_code=status_code,
                    headers=response_headers,
                    content=content,
                    redirect_history=redirect_handler.history
                )

        except urllib.error.HTTPError as e:
            # Server responded with an error (e.g. 404, 500)
            try:
                content = e.read()
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
            # Network failures, host not found, timeouts
            raise DownloadError(f"Network error downloading {url}: {e.reason}") from e
        except TimeoutError as e:
            raise DownloadError(f"Timeout occurred while downloading {url}") from e
        except Exception as e:
            raise DownloadError(f"Unexpected error downloading {url}: {str(e)}") from e
