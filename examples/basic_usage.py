"""Example demonstrating how to use the crawl library.

This script parses a mock HTML page, performs DOM selection, cleans the document,
extracts structured details, and exports them to CSV, JSON, and TXT files.
It also shows how to fetch a real webpage.
"""

import os
from crawl import Crawl
from crawl.parser import parse_html
from crawl.cleaner import clean_dom
from crawl.extractor import (
    extract_title,
    extract_paragraphs,
    extract_links,
    extract_images,
    extract_tables
)
from crawl.exporter import export_csv, export_json, export_txt


def main() -> None:
    # -------------------------------------------------------------------------
    # Scenario 1: Parsing and Scraping Mock HTML String
    # -------------------------------------------------------------------------
    print("=== Scenario 1: Parsing and Scraping Mock HTML ===")
    mock_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Shop   &amp;   Compare Products</title>
    </head>
    <body>
        <div id="wrapper">
            <h1 class="header main">Welcome to the Store</h1>
            <p>Check out our   <b>latest</b>   deals below!</p>
            
            <div class="product-card" id="p1">
                <span class="product-title">Item A</span>
                <span class="price">$19.99</span>
                <a href="/products/item-a" class="details-link">Details</a>
            </div>
            
            <div class="product-card" id="p2">
                <span class="product-title">Item B</span>
                <span class="price">$29.99</span>
                <a href="/products/item-b" class="details-link">Details</a>
            </div>

            <!-- Empty element to clean -->
            <div class="empty-promo"></div>
            
            <table>
                <tr><th>Product</th><th>In Stock</th></tr>
                <tr><td>Item A</td><td>Yes</td></tr>
                <tr><td>Item B</td><td>No</td></tr>
            </table>
        </div>
    </body>
    </html>
    """

    # 1. Parse HTML to build the DOM Tree
    root = parse_html(mock_html)
    print("Parsed DOM Root:")
    print(repr(root))

    # 2. Clean the DOM (whitespace normalization, entity decoding, empty tag removal)
    print("\nCleaning DOM...")
    clean_dom(root)

    # 3. Query elements using CSS Selector Engine
    print("\nSelecting elements via CSS queries:")
    # Tag Selector
    h1 = root.find("h1")
    print(f"H1 Title: {h1.get_text() if h1 else 'None'}")

    # Class Selector
    prices = root.find_all(".price")
    for price in prices:
        print(f"Found price: {price.get_text()}")

    # Nested / Child Selectors
    product_links = root.find_all("div.product-card > a")
    for link in product_links:
        print(f"Product Link target: {link.get_attribute('href')}")

    # 4. Extract structured data
    print("\nExtracting data:")
    title = extract_title(root)
    paragraphs = extract_paragraphs(root)
    links = extract_links(root, base_url="https://example-store.com")
    tables = extract_tables(root)

    print(f"Page Title: {title}")
    print(f"Paragraphs: {paragraphs}")
    print(f"Links (resolved): {links}")
    print(f"Tables: {tables}")

    # 5. Export extracted data
    print("\nExporting extracted data...")
    export_dir = "./export_demo"
    os.makedirs(export_dir, exist_ok=True)

    csv_path = os.path.join(export_dir, "products.csv")
    json_path = os.path.join(export_dir, "metadata.json")
    txt_path = os.path.join(export_dir, "page_text.txt")

    # Export links as CSV
    export_csv(links, csv_path)
    # Export full structured details as JSON
    export_json({"title": title, "paragraphs": paragraphs, "tables": tables}, json_path)
    # Export page plain text as TXT
    export_txt(root.get_text(), txt_path)

    print(f"Data exported to:\n- {csv_path}\n- {json_path}\n- {txt_path}")

    # -------------------------------------------------------------------------
    # Scenario 2: Crawling a Real URL using public crawler API
    # -------------------------------------------------------------------------
    print("\n=== Scenario 2: Fetching Real Web Resource ===")
    crawler = Crawl(timeout=5.0)
    target_url = "https://httpbin.org/html"

    print(f"Fetching URL: {target_url} ...")
    try:
        page = crawler.fetch(target_url)
        print(f"Status Code: {page.response.status_code}")
        print(f"Page Title: {page.title()}")
        print(f"First Paragraph: {page.paragraphs()[0] if page.paragraphs() else 'None'}")
        print(f"Total Page Links: {len(page.links())}")
    except Exception as e:
        print(f"Failed to fetch {target_url}: {e}")


if __name__ == "__main__":
    main()
