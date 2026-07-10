"""Data extraction engine for the crawl-any-website library.

Provides helper functions to extract structured components from a DOM Node tree:
- Page Title & Headings (H1 to H6)
- Text paragraphs (<p> tags)
- Image metadata (src, alt, title attributes)
- Table grids (rows, cells data)
- Form metadata (action, method, and fields like inputs, selects, textareas)
- List items (ul, ol list items)
- Anchor hyperlinks (href, text, title)
- Interactive buttons and metadata (meta tags)
- Link-related elements (resolving relative URLs against the target base URL)
"""

import urllib.parse
from typing import Dict, List, Optional, Any
from crawl.dom import Node


def resolve_url(url: str, base_url: Optional[str]) -> str:
    """Helper to resolve a relative URL to an absolute one given a base URL.

    Args:
        url: The raw URL string extracted from an HTML element.
        base_url: The absolute base URL of the source page.

    Returns:
        The resolved absolute URL, or the original if base_url is not set.
    """
    if not url:
        return ""
    if not base_url:
        return url
    # Use standard library urljoin to handle relative paths, queries, and anchors
    return urllib.parse.urljoin(base_url, url.strip())


def extract_title(node: Node) -> str:
    """Extracts the page title from the DOM.

    Checks the <title> tag first. If not present, falls back to the first <h1> tag.

    Args:
        node: The reference DOM Node (usually the root or <head> element).

    Returns:
        The title string, or an empty string if none is found.
    """
    # 1. Search for <title> tag anywhere in the DOM
    title_nodes = node.find_all("title")
    if title_nodes:
        return title_nodes[0].get_text().strip()

    # 2. Fall back to the first <h1> tag if title is missing
    h1_nodes = node.find_all("h1")
    if h1_nodes:
        return h1_nodes[0].get_text().strip()

    return ""


def extract_paragraphs(node: Node) -> List[str]:
    """Extracts text contents of all paragraph (<p>) tags.

    Args:
        node: The reference DOM Node.

    Returns:
        List of paragraph text strings, with empty paragraphs ignored.
    """
    # Loops through all <p> tags and retrieves their plain text representation
    return [p.get_text().strip() for p in node.find_all("p") if p.get_text().strip()]


def extract_images(node: Node, base_url: Optional[str] = None) -> List[Dict[str, str]]:
    """Extracts metadata for all image (<img>) tags.

    Resolves relative paths in 'src' against the base URL.

    Args:
        node: The reference DOM Node.
        base_url: Optional base URL to resolve absolute image src paths.

    Returns:
        A list of dictionaries containing image src, alt, and title details.
    """
    images = []
    for img in node.find_all("img"):
        src = img.get_attribute("src", "")
        if src:
            images.append({
                # Resolve relative URL to absolute URL
                "src": resolve_url(src, base_url),
                "alt": img.get_attribute("alt", ""),
                "title": img.get_attribute("title", "")
            })
    return images


def extract_tables(node: Node) -> List[List[List[str]]]:
    """Extracts data grids from HTML <table> elements.

    Args:
        node: The reference DOM Node.

    Returns:
        A list of tables. Each table is represented as a list of rows,
        and each row is a list of cell text strings (td/th contents).
    """
    tables = []
    for table in node.find_all("table"):
        table_data = []
        # Find all table rows
        for tr in table.find_all("tr"):
            row = []
            # Extract cell data from direct children (cells must be <td> or <th>)
            for cell in tr.children:
                if cell.tag in ("td", "th"):
                    row.append(cell.get_text().strip())
            if row:
                table_data.append(row)
        if table_data:
            tables.append(table_data)
    return tables


def extract_forms(node: Node, base_url: Optional[str] = None) -> List[Dict[str, Any]]:
    """Extracts structured form metadata from the DOM.

    Collects inputs, select dropdowns, and textareas, resolving target URLs.

    Args:
        node: The reference DOM Node.
        base_url: Optional base URL to resolve form action paths.

    Returns:
        A list of dictionaries describing form action, method, and input fields.
    """
    forms = []
    for form in node.find_all("form"):
        inputs = []
        
        # 1. Find all standard <input> fields
        for inp in form.find_all("input"):
            name = inp.get_attribute("name")
            if name:
                inputs.append({
                    "tag": "input",
                    "name": name,
                    "type": inp.get_attribute("type", "text"),
                    "value": inp.get_attribute("value", "")
                })

        # 2. Find all <select> dropdown fields and their options
        for sel in form.find_all("select"):
            name = sel.get_attribute("name")
            if name:
                options = []
                for opt in sel.find_all("option"):
                    val = opt.get_attribute("value")
                    options.append(val if val is not None else opt.get_text().strip())
                inputs.append({
                    "tag": "select",
                    "name": name,
                    "options": options
                })

        # 3. Find all <textarea> multi-line text input fields
        for ta in form.find_all("textarea"):
            name = ta.get_attribute("name")
            if name:
                inputs.append({
                    "tag": "textarea",
                    "name": name,
                    "value": ta.get_text()
                })

        # Resolve relative form submission action to absolute URL
        action = form.get_attribute("action", "")
        forms.append({
            "action": resolve_url(action, base_url) if action else "",
            "method": form.get_attribute("method", "get").upper(),
            "inputs": inputs
        })
    return forms


def extract_lists(node: Node) -> List[List[str]]:
    """Extracts list items from unordered (<ul>) and ordered (<ol>) lists.

    Args:
        node: The reference DOM Node.

    Returns:
        List of lists containing text strings of <li> list items.
    """
    lists = []
    # Search all lists on the page
    for lst in node.find_all("ul") + node.find_all("ol"):
        items = []
        for li in lst.find_all("li"):
            items.append(li.get_text().strip())
        if items:
            lists.append(items)
    return lists


def extract_links(node: Node, base_url: Optional[str] = None) -> List[Dict[str, str]]:
    """Extracts all hyperlink anchor (<a>) tags.

    Resolves relative paths in 'href' against the base URL.

    Args:
        node: The reference DOM Node.
        base_url: Optional base URL to resolve links.

    Returns:
        A list of dicts with link text, resolved href, title, and download attribute if present.
    """
    links = []
    for a in node.find_all("a"):
        href = a.get_attribute("href", "")
        if href:
            link_data = {
                "text": a.get_text().strip(),
                "href": resolve_url(href, base_url),
                "title": a.get_attribute("title", "")
            }
            # Add download attribute if present (indicates a file download)
            download = a.get_attribute("download")
            if download is not None:
                link_data["download"] = download
            links.append(link_data)
    return links


def extract_all_links_attributes(node: Node, base_url: Optional[str] = None) -> List[Dict[str, str]]:
    """Extracts URLs from tags containing link-related attributes: href, src, action, download.

    Walks down all DOM nodes and returns a unified list of resource endpoints.

    Args:
        node: The reference DOM Node.
        base_url: Optional base URL to resolve URLs.

    Returns:
        List of dicts representing extracted links (tag, attribute, raw_value, resolved_url).
    """
    results = []

    # Get descendants and check node itself
    from crawl.selector import get_descendants
    candidates = [node] + get_descendants(node)

    for cand in candidates:
        # Skip text, comments, and doctypes
        if cand.tag is None or cand.tag.startswith("#"):
            continue

        # Check 'href' attributes (hyperlinks, stylesheets)
        href = cand.get_attribute("href")
        if href:
            results.append({
                "tag": cand.tag,
                "attribute": "href",
                "raw_value": href,
                "resolved_url": resolve_url(href, base_url)
            })

        # Check 'src' attributes (images, scripts, iframes)
        src = cand.get_attribute("src")
        if src:
            results.append({
                "tag": cand.tag,
                "attribute": "src",
                "raw_value": src,
                "resolved_url": resolve_url(src, base_url)
            })

        # Check 'action' attributes (form submissions)
        action = cand.get_attribute("action")
        if action:
            results.append({
                "tag": cand.tag,
                "attribute": "action",
                "raw_value": action,
                "resolved_url": resolve_url(action, base_url)
            })

        # Check 'download' attributes (file downloads)
        download = cand.get_attribute("download")
        if download:
            results.append({
                "tag": cand.tag,
                "attribute": "download",
                "raw_value": download,
                "resolved_url": resolve_url(download, base_url)
            })

    return results


def extract_buttons(node: Node) -> List[Dict[str, str]]:
    """Extracts button elements (both <button> and input types: submit, button, reset).

    Args:
        node: The reference DOM Node.

    Returns:
        List of dictionaries with button details (text, type, name).
    """
    buttons = []
    
    # 1. Parse standard <button> tags
    for btn in node.find_all("button"):
        buttons.append({
            "text": btn.get_text().strip(),
            "type": btn.get_attribute("type", "button"),
            "name": btn.get_attribute("name", "")
        })
        
    # 2. Parse <input> tags that act as interactive buttons
    for inp in node.find_all("input"):
        type_ = inp.get_attribute("type", "").lower()
        if type_ in ("button", "submit", "reset"):
            buttons.append({
                "text": inp.get_attribute("value", "").strip() or type_,
                "type": type_,
                "name": inp.get_attribute("name", "")
            })
    return buttons


def extract_meta_tags(node: Node) -> List[Dict[str, str]]:
    """Extracts metadata from all <meta> tags in the DOM.

    Useful for fetching SEO tags, viewport details, and OpenGraph/Twitter descriptions.

    Args:
        node: The reference DOM Node.

    Returns:
        A list of dictionaries with meta tag name/property, content, and http-equiv.
    """
    meta_tags = []
    for meta in node.find_all("meta"):
        meta_tags.append({
            "name": meta.get_attribute("name") or meta.get_attribute("property") or "",
            "content": meta.get_attribute("content") or "",
            "http_equiv": meta.get_attribute("http-equiv") or ""
        })
    return meta_tags


def extract_headers(node: Node) -> List[Dict[str, str]]:
    """Extracts all structural headings (<h1> through <h6>).

    Returned list is ordered in the sequence of their appearance in the DOM.

    Args:
        node: The reference DOM Node.

    Returns:
        List of dictionaries detailing heading tag level and text.
    """
    headers = []
    from crawl.selector import get_descendants
    candidates = get_descendants(node)
    
    # Check each candidate node in document order
    for cand in candidates:
        if cand.tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            headers.append({
                "level": cand.tag,
                "text": cand.get_text().strip()
            })
    return headers
