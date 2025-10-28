import sys
import json
import types
import pytest
#test output
def test_cli_explain_from_code_happy_path(monkeypatch, capsys):
    # Import after monkeypatching to avoid caching issues
    import src.main as app

    # Fake explain_code to avoid network and control output
    def fake_explain_code(req, model=None):
        from src.models import Explanation
        return Explanation(
            summary="Prints 123.",
            steps=["Call print with 123."],
            pitfalls=[],
            detected_language="python",
        )

    monkeypatch.setattr(app, "explain_code", fake_explain_code)

    # Simulate argv: program explain --code ...
    monkeypatch.setattr(sys, "argv", ["prog", "explain", "--code", "print(123)", "--language", "python"])
    rc = app.main()
    captured = capsys.readouterr().out

    assert rc == 0
    assert "=== Code Explanation ===" in captured
    assert "Detected language: python" in captured
    assert "Summary:" in captured and "Prints 123." in captured
    assert "How it works:" in captured and "Call print with 123." in captured
    assert "Errors:" in captured

def test_cli_explain_from_file_happy_path(tmp_path, monkeypatch, capsys):
    codefile = tmp_path / "snippet.py"
    codefile.write_text("def add(a,b): return a+b", encoding="utf-8")

    import src.main as app

    def fake_explain_code(req, model=None):
        from src.models import Explanation
        return Explanation(
            summary="Adds two numbers.",
            steps=["Define function", "Return a+b"],
            pitfalls=["Type mismatch if not numbers"],
            detected_language="python",
        )

    monkeypatch.setattr(app, "explain_code", fake_explain_code)
    monkeypatch.setattr(sys, "argv", ["prog", "explain", "--file", str(codefile), "--context", "unit test"])
    rc = app.main()
    captured = capsys.readouterr().out

    assert rc == 0
    assert "Adds two numbers." in captured
    assert "Type mismatch" in captured

def test_cli_explain_prints_empty_branches(monkeypatch, capsys):
    """
    Cover the '(no steps)' and '(none)' print branches in run_explain_command.
    """
    import sys
    import src.main as app

    # Return an Explanation with empty steps/pitfalls and no detected_language
    def fake_explain_code(req, model=None):
        from src.models import Explanation
        return Explanation(
            summary="Minimal explanation.",
            steps=[],
            pitfalls=[],
            detected_language=None,
        )

    monkeypatch.setattr(app, "explain_code", fake_explain_code)
    monkeypatch.setattr(sys, "argv", ["prog", ".", "--code", "x=1"])

    rc = app.main()
    out = capsys.readouterr().out

    assert rc == 0
    assert "Minimal explanation." in out
    assert "(no steps)" in out           # covers one missing line
    assert "(none)" in out               # covers the other missing line

def test_cli_explain_requires_code_or_file(capsys):
    # Call run_explain_command directly with args that have neither code nor file
    from types import SimpleNamespace
    from src.main import run_explain_command

    args = SimpleNamespace(
        code=None,
        file=None,
        language=None,
        context=None,
        explain_max_tokens=8000,
        explain_model=None,
    )

    rc = run_explain_command(args)
    out = capsys.readouterr().out

    assert rc == 2
    assert "Error: --code or --file is required for explanation." in out

