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
    """Entry point for the crawl-any-website command-line application.

    Parses CLI flags. If a target URL is supplied, crawls, cleans, and prints
    metadata summaries directly to stdout.
    """
    parser = argparse.ArgumentParser(
        description="crawl-any-website: A scratch-built HTML parser and web crawling engine."
    )
    
    # 1. Define version option (-v / --version)
    parser.add_argument(
        "-v", "--version",
        action="version",
        version=f"crawl-any-website {__version__}",
        help="Show the package version and exit."
    )
    
    # 2. Define positional URL argument
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
