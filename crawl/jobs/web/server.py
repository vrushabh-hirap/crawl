"""Local Web Server for the crawl job search frontend."""

import os
import sys
import webbrowser
from threading import Timer
from typing import Dict, Any

# Ensure flask is imported only if installed, to support optional dependencies
try:
    from flask import Flask, request, jsonify, send_from_directory
except ImportError:
    # This will be raised and handled in the CLI module
    raise ImportError("Flask is required for the web server interface. Install via: pip install crawl-any-website[web]")

from crawl.jobs.models import JobQuery
from crawl.jobs.engine import JobSearchEngine
from crawl.utils import setup_logger

logger = setup_logger(__name__)

# Initialize Flask app serving static folder from package resources
current_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(current_dir, "static")

app = Flask(__name__, static_folder=static_dir, static_url_path="")

# Instance of concurrent job search orchestrator
search_engine = JobSearchEngine()

@app.route("/")
def index() -> Any:
    """Serves the main frontend index.html."""
    return send_from_directory(app.static_folder, "index.html")

@app.route("/api/search", methods=["POST"])
def api_search() -> Any:
    """Endpoint executing concurrent job search."""
    try:
        data = request.json or {}
        
        # Build JobQuery schema
        query = JobQuery(
            title=data.get("title", "").strip(),
            work_modes=data.get("work_modes", []),
            location=data.get("location", "").strip(),
            skills=data.get("skills", []),
            recency=data.get("recency", "all")
        )

        logger.info(f"Received job search request: {query.to_dict()}")
        
        # Call search engine
        results, warnings = search_engine.search(query)
        
        # Convert dataclasses to dicts for JSON serialization
        results_serialized = [r.to_dict() for r in results]
        
        return jsonify({
            "success": True,
            "results": results_serialized,
            "warnings": warnings
        })
        
    except Exception as e:
        logger.exception("Error during job search API execution")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


def run_server(host: str, port: int, no_browser: bool) -> None:
    """Helper to start Flask and trigger browser auto-open.

    Args:
        host: Host to bind Flask.
        port: Port to listen.
        no_browser: If True, do not open browser.
    """
    logger.info(f"Starting job-search web server at http://{host}:{port}")

    # Launch browser in a separate thread after server starts
    if not no_browser:
        def open_browser() -> None:
            url = f"http://{host if host != '0.0.0.0' else '127.0.0.1'}:{port}"
            logger.info(f"Opening browser to {url} ...")
            webbrowser.open(url)
        Timer(1.2, open_browser).start()

    # Disable flask output logging details for custom clean formatting unless debug requested
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.WARNING)

    try:
        app.run(host=host, port=port, debug=False, threaded=True)
    except Exception as e:
        logger.error(f"Failed to start web server: {e}")
        sys.exit(1)
