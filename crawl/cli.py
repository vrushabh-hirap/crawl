"""Command Line Interface for the crawl-any-website library.

Provides a CLI command to query the installed package version or test the crawler
and parser modules against target web URLs directly from the terminal.
"""

import sys
import argparse
from crawl.version import __version__
from crawl.crawler import Crawl
from crawl.exceptions import CrawlError


def main() -> None:
    """Entry point for the crawl command-line application.

    Parses CLI flags. If 'webpage' is specified, launches the local job search UI.
    Otherwise, treats arguments as a target URL to download, parse, and verify.
    """
    import sys

    # 1. Route to webpage server if requested
    if len(sys.argv) > 1 and sys.argv[1] == "webpage":
        parser = argparse.ArgumentParser(
            description="crawl webpage: Start the local job-search Web UI server."
        )
        parser.add_argument(
            "--host",
            default="127.0.0.1",
            help="Host address to bind the local web server to (default: 127.0.0.1)."
        )
        parser.add_argument(
            "-p", "--port",
            type=int,
            default=8080,
            help="Port to run the local web server on (default: 8080)."
        )
        parser.add_argument(
            "--no-browser",
            action="store_true",
            help="Do not automatically open the web browser when starting the server."
        )
        
        args = parser.parse_args(sys.argv[2:])
        
        try:
            from crawl.jobs.web.server import run_server
            run_server(host=args.host, port=args.port, no_browser=args.no_browser)
        except ImportError as e:
            print(f"\n[Error] {e}", file=sys.stderr)
            sys.exit(1)
        return

    # 2. Legacy fallback behavior for URL parsing
    parser = argparse.ArgumentParser(
        description="crawl-any-website: A scratch-built HTML parser and web crawling engine."
    )
    
    # Define version option (-v / --version)
    parser.add_argument(
        "-v", "--version",
        action="version",
        version=f"crawl-any-website {__version__}",
        help="Show the package version and exit."
    )
    
    # Define positional URL argument
    parser.add_argument(
        "url",
        nargs="?",
        help="Optional HTTP/HTTPS URL to download, parse, and verify."
    )

    args = parser.parse_args()

    # If user provided a URL to fetch
    if args.url:
        try:
            print(f"Crawling: {args.url} ...")
            # Instantiate crawler (bypassing SSL verification for command-line convenience)
            crawler = Crawl(verify_ssl=False)
            page = crawler.fetch(args.url)
            page.clean()  # Normalize whitespace and HTML entities
            
            # Print page statistics report to console
            print("\n" + "=" * 40)
            print("Crawl Success!")
            print("=" * 40)
            print(f"Title:       {page.title()}")
            print(f"Status Code: {page.response.status_code}")
            print(f"Links count: {len(page.links())}")
            print(f"Image count: {len(page.images())}")
            print("=" * 40)
        except CrawlError as e:
            # Catch library exceptions and report to stderr
            print(f"\n[Error] Failed to crawl URL: {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            # Catch unexpected standard library errors
            print(f"\n[Error] Unexpected exception: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        # If no arguments or URLs are passed, show the CLI help menu
        parser.print_help()


if __name__ == "__main__":
    main()
