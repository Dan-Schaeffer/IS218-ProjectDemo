# Lab 05 – Chapter 5: CLI Interface

**Estimated time:** 45 minutes  
**Artifacts:** UX tweak with tests, demo rehearsal notes

---

## Objectives
- Understand how `src/main.py` delivers a polished CLI experience.
- Enhance user feedback while keeping error handling professional.
- Practice system-level testing with `pytest` capture fixtures.

---

## Before You Start
- Run `pytest tests/test_main.py -v`.
- Review how `argparse` is configured and how verbose mode behaves.
- Note where logging integrates with CLI output.

---

## Guided Exploration
1. Walk through `main()` and mark each log statement's purpose.
2. Inspect `display_results` formatting and identify improvement opportunities.
3. Check how tests simulate CLI runs with `capsys` and patched services.

---

## Core Exercise (TDD Loop)
### RED
Add a test ensuring the CLI warns users when no citations are returned.

Example snippet in `tests/test_main.py`:
```python
def test_main_warns_when_no_citations(mock_service, capsys, monkeypatch):
    mock_service.search.return_value = sample_result_without_citations
    monkeypatch.setattr("src.main.SearchService", lambda **_: mock_service)
    exit_code = main()
    captured = capsys.readouterr()
    assert "⚠️" in captured.out
    assert exit_code == 0
```
Expect failure due to missing warning.

### GREEN
Update `display_results` to append a line such as `"⚠️ This answer has no citations"` when appropriate.

Run `pytest tests/test_main.py -v` until the new test passes.

### REFACTOR
- Ensure formatting remains user-friendly (avoid duplicate newlines).
- Consider extracting display logic into helper functions for readability.
- Confirm manual CLI run still looks clean.

---

## Demo Rehearsal Drill
Script a 2-minute walkthrough: parse args → execute service → show output. Practice saying aloud what happens if the API key is missing and how the CLI responds. Jot notes for later demo prep.

---

## Stretch Challenges (Optional)
- Add a `--json` flag that prints raw JSON for automation workflows; cover with tests.
- Provide summary statistics (number of citations, sources) at the top of the output.
- Wire verbose mode to include timing information from `LogContext`.

---

## Reflection Prompt
Document:
- What UX detail did you improve?
- How do logs and printed output work together to aid support teams?
- Which argument parsing feature would you add next, and why?
- What is your comfort level with system tests after this lab?

---

## Exit Checklist
- [ ] Warning for no citations implemented and tested
- [ ] Demo rehearsal notes captured
- [ ] Reflection written
- [ ] Ready for `LAB_06_LOGGING`
