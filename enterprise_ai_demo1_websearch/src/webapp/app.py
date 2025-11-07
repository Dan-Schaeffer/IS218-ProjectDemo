# pyright: reportMissingImports=false
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from flask import Flask, render_template, request, redirect, url_for, flash
from dotenv import load_dotenv

# Reuse your existing modules
from src.search_service import SearchService
from src.models import SearchOptions
from src.parser import ResponseParser
from src.code_explainer_service import explain_code
from src.models import ExplainRequest

# Load environment (.env) for OPENAI_API_KEY etc.
load_dotenv()


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).with_name("templates")),
        static_folder=str(Path(__file__).with_name("static")),
    )

    # For flash() messages; in real apps keep this secret via env
    app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-not-secret")

    # ----------------------------- Home (GET) -----------------------------
    @app.get("/")
    def index():
        return render_template("index.html")

    # ----------------------------- Web search (POST) ----------------------
    # Matches templates that use: action="{{ url_for('do_search') }}"
    @app.post("/search", endpoint="do_search")
    def do_search():
        query = (request.form.get("query") or "").strip()
        model = (request.form.get("model") or "gpt-4o-mini").strip()
        domains_raw = (request.form.get("domains") or "").strip()

        if not query:
            flash("Please enter a search query.", "error")
            return redirect(url_for("index"))

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            flash("OPENAI_API_KEY is not set.", "error")
            return redirect(url_for("index"))

        options = SearchOptions(model=model)
        if domains_raw:
            options.allowed_domains = [d.strip() for d in domains_raw.split(",") if d.strip()]

        try:
            service = SearchService(api_key=api_key)
            result = service.search(query, options)
            parsed = ResponseParser().format_for_display(result)
        except Exception as e:  # Keep page usable on failure
            flash(f"Search failed: {e}", "error")
            return render_template(
                "index.html",
                search_query=query,
                search_output=None,
            )

        return render_template(
            "index.html",
            search_query=query,
            search_output=parsed,
        )

    # ----------------------------- Code explainer (POST) ------------------
    # Matches templates that use: action="{{ url_for('do_explain') }}"
    @app.post("/explain", endpoint="do_explain")
    def do_explain():
        language = (request.form.get("language") or "").strip() or None
        context = (request.form.get("context") or "").strip() or None
        model = (request.form.get("explain_model") or "").strip() or None
        max_tokens_raw = (request.form.get("explain_max_tokens") or "").strip() or "8000"

        try:
            max_tokens = int(max_tokens_raw)
        except ValueError:
            max_tokens = 8000

        # Textarea content
        code_text = (request.form.get("code") or "").strip()

        # Optional file upload (takes precedence if provided)
        file = request.files.get("file")
        if file and file.filename:
            try:
                code_text = file.read().decode("utf-8", errors="replace")
            except Exception as e:
                flash(f"Could not read uploaded file: {e}", "error")
                return redirect(url_for("index"))

        if not code_text:
            flash("Please paste code or upload a file.", "error")
            return redirect(url_for("index"))

        req = ExplainRequest(
            code=code_text,
            language=language,
            extra_context=context,
            max_tokens=max_tokens,
        )

        try:
            result = explain_code(req, model=model)
            # Map pitfalls to "errors" so existing templates referencing
            # explain_result.errors continue to work.
            try:
                setattr(result, "errors", list(result.pitfalls or []))
            except Exception:
                # If setattr fails (e.g., non-dataclass), wrap it.
                result = SimpleNamespace(
                    summary=getattr(result, "summary", ""),
                    steps=getattr(result, "steps", []),
                    pitfalls=getattr(result, "pitfalls", []),
                    detected_language=getattr(result, "detected_language", None),
                    errors=list(getattr(result, "pitfalls", [])),
                )
        except Exception as e:
            flash(f"Explain failed: {e}", "error")
            return render_template(
                "index.html",
                explain_result=None,
                pasted_code=code_text,
                language=language or "auto",
            )

        return render_template(
            "index.html",
            explain_result=result,
            pasted_code=code_text,
            language=language or "auto",
        )

    return app


# For `python -m src.webapp.app`
app = create_app()

if __name__ == "__main__":
    # Run dev server: python -m src.webapp.app
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", "5000")), debug=True)
