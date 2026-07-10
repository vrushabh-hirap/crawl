# Crawl Package Documentation

Welcome to the `crawl` library documentation.

`crawl` is a Python-only, dependency-free web crawling and scraping library containing its own HTML parsing, DOM, and selector engines built from scratch.

## Modules Reference

- **Downloader (`crawl.downloader`)**: Manages HTTP connection handlers, headers, timeouts, and redirect history.
- **Tokenizer (`crawl.tokenizer`)**: Scans HTML character-by-character into token streams.
- **Parser (`crawl.parser`)**: Construct DOM trees from token streams with auto-closing tag error recovery.
- **DOM Node (`crawl.dom`)**: Represents elements, text nodes, doctypes, and comments. Supports HTML serialization.
- **Selector Engine (`crawl.selector`)**: Executes CSS queries (Tags, IDs, Classes, Descendants, Child combinators, Comma lists).
- **Extractor (`crawl.extractor`)**: Structured endpoints to extract tags, forms, tables, links, and buttons.
- **Cleaner (`crawl.cleaner`)**: Whitespace merging, entity unescaping, empty tag pruning, sibling subtree deduplication.
- **Exporter (`crawl.exporter`)**: Independent file export utility for CSV, JSON, and TXT.
