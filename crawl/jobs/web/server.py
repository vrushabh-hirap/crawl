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
    raise ImportError("Flask is required for the web server interface. Install via: pip install 'crawl-jobs[web]'")

from crawl.jobs.models import JobQuery
from crawl.jobs.engine import JobSearchEngine
from crawl.jobs.ats.analyzer import ATSAnalyzer
from crawl.nlp import get_tool_registry, validate_nlp_request, execute_nlp_tool
from crawl.utils import setup_logger

logger = setup_logger(__name__)

# Initialize Flask app serving static folder from package resources
current_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(current_dir, "static")

app = Flask(__name__, static_folder=static_dir, static_url_path="")

# Instance of concurrent job search orchestrator
search_engine = JobSearchEngine()

# Instance of ATS résumé analyzer
ats_analyzer = ATSAnalyzer()

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


@app.route("/api/ats/analyze", methods=["POST"])
def api_ats_analyze() -> Any:
    """ATS résumé analysis endpoint.

    Accepts multipart/form-data with:
      - ``resume``          (file)   — PDF, DOCX, or TXT résumé
      - ``job_description`` (string) — raw job description text
    """
    try:
        # ── Validate file upload ──────────────────────────────────────────
        if "resume" not in request.files:
            return jsonify({"success": False, "error": "No résumé file provided."}), 400

        resume_file = request.files["resume"]
        if not resume_file.filename:
            return jsonify({"success": False, "error": "Résumé file has no filename."}), 400

        job_description = (request.form.get("job_description") or "").strip()
        if not job_description:
            return jsonify({"success": False, "error": "Job description cannot be empty."}), 400

        file_bytes = resume_file.read()
        filename   = resume_file.filename

        logger.info(f"ATS analysis request: file={filename!r}, jd_len={len(job_description)}")

        # ── Run analysis ──────────────────────────────────────────────────
        report = ats_analyzer.analyze(file_bytes, filename, job_description)

        return jsonify({"success": True, "report": report})

    except ValueError as e:
        logger.warning(f"ATS validation error: {e}")
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        logger.exception("Unexpected error during ATS analysis")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/nlp/tools", methods=["GET"])
def api_nlp_tools() -> Any:
    """Returns the list of available NLP Lab tools, schemas, and default options."""
    try:
        tools = get_tool_registry()
        return jsonify({
            "success": True,
            "tools": [t.to_dict() for t in tools]
        })
    except Exception as e:
        logger.exception("Error retrieving NLP tool registry")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/nlp/run", methods=["POST"])
@app.route("/api/nlp/execute", methods=["POST"])
def api_nlp_run() -> Any:
    """Executes or validates an NLP Lab tool request."""
    try:
        if not request.is_json:
            return jsonify({
                "success": False,
                "error": "Request body must be JSON."
            }), 400

        data = request.get_json() or {}
        tool_id = (data.get("tool_id") or "").strip()
        text = data.get("text", "")
        options = data.get("options", {})

        if not tool_id:
            return jsonify({
                "success": False,
                "error": "Field 'tool_id' is required."
            }), 400

        if not isinstance(text, str) or not text.strip():
            return jsonify({
                "success": False,
                "error": "Input text is required and cannot be empty."
            }), 400

        if not isinstance(options, dict):
            return jsonify({
                "success": False,
                "error": "Field 'options' must be a dictionary/object."
            }), 400

        # Validate request parameters against tool schema
        is_valid, validation_err, validated_options = validate_nlp_request(tool_id, text, options)
        if not is_valid:
            return jsonify({
                "success": False,
                "error": validation_err
            }), 400

        response = execute_nlp_tool(tool_id, text, validated_options)
        return jsonify(response)

    except ValueError as e:
        logger.warning(f"NLP validation error: {e}")
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        logger.exception("Unexpected error executing NLP tool")
        return jsonify({"success": False, "error": str(e)}), 500



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
