"""Unit tests for the crawl Downloader."""

import unittest
from unittest.mock import patch, MagicMock
import urllib.error
from crawl.downloader import Downloader, Response, TrackingRedirectHandler
from crawl.exceptions import DownloadError


class TestDownloader(unittest.TestCase):
    """Verifies downloader configurations, mock server calls, and exception raising."""

    def test_response_text_decoding(self) -> None:
        """Tests that binary content is correctly decoded based on charset."""
        headers = {"content-type": "text/html; charset=utf-8"}
        res = Response(url="http://test.com", status_code=200, headers=headers, content=b"Hello World")
        self.assertEqual(res.text, "Hello World")

        # Guess encoding when not specified
        res_no_charset = Response(url="http://test.com", status_code=200, headers={}, content=b"Default UTF-8")
        self.assertEqual(res_no_charset.text, "Default UTF-8")

        # ISO-8859-1 override
        headers_iso = {"content-type": "text/html; charset=iso-8859-1"}
        res_iso = Response(url="http://test.com", status_code=200, headers=headers_iso, content=b"\xe9") # e with acute accent
        self.assertEqual(res_iso.text, "é")

    @patch("urllib.request.build_opener")
    def test_download_success(self, mock_build_opener: MagicMock) -> None:
        """Verifies downloder makes requests and configures correct values on Response."""
        mock_response = MagicMock()
        mock_response.read.return_value = b"<html>Test</html>"
        mock_response.getcode.return_value = 200
        mock_response.info.return_value = {"Content-Type": "text/html"}
        mock_response.geturl.return_value = "http://test.com"
        mock_response.__enter__.return_value = mock_response

        mock_opener = MagicMock()
        mock_opener.open.return_value = mock_response
        mock_build_opener.return_value = mock_opener

        dl = Downloader(default_headers={"X-Custom": "Value"}, timeout=5.0)
        res = dl.download("http://test.com")

        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.text, "<html>Test</html>")
        self.assertEqual(res.url, "http://test.com")
        self.assertEqual(res.headers["content-type"], "text/html")

    @patch("urllib.request.build_opener")
    def test_download_http_error(self, mock_build_opener: MagicMock) -> None:
        """HTTPError should return a Response object containing status code and content."""
        # HTTPError requires (url, code, msg, hdrs, fp)
        fp = MagicMock()
        fp.read.return_value = b"Not Found"
        error = urllib.error.HTTPError("http://test.com/404", 404, "Not Found", {}, fp)

        mock_opener = MagicMock()
        mock_opener.open.side_effect = error
        mock_build_opener.return_value = mock_opener

        dl = Downloader()
        res = dl.download("http://test.com/404")

        self.assertEqual(res.status_code, 404)
        self.assertEqual(res.text, "Not Found")

    @patch("urllib.request.build_opener")
    def test_download_url_error(self, mock_build_opener: MagicMock) -> None:
        """URLError should trigger DownloadError."""
        error = urllib.error.URLError("Connection refused")

        mock_opener = MagicMock()
        mock_opener.open.side_effect = error
        mock_build_opener.return_value = mock_opener

        dl = Downloader()
        with self.assertRaises(DownloadError):
            dl.download("http://invalid-domain.com")


if __name__ == "__main__":
    unittest.main()
