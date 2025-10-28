# Lab 06 – Chapter 6: Logging & Observability

**Estimated time:** 50 minutes  
**Artifacts:** Logging enhancement, captured log snippet, reflection

---

## Objectives
- Master the logging configuration in `src/logging_config.py`.
- Add structured context to an existing log statement.
- Ensure logs remain readable in both console and JSON formats.

---

## Before You Start
- Tail the current logs while running the CLI:
  ```bash
  python -m src.main "test query" && tail -n 20 logs/app.log
  ```
- Read the section on `JSONFormatter` and `LogContext`.
- Note how handlers differ between console and file output.

---

## Guided Exploration
1. Diagram the logging pipeline: logger → handlers → formatters → files/console.
2. Identify where performance metrics (duration_ms) are captured.
3. Check where `LogContext` is used in `main.py` and how it enriches log entries.

---

## Core Exercise (TDD Loop)
### RED
Add a test in `tests/test_search_service.py` or create a new logging-focused test to ensure that when `SearchService.search` succeeds, the log includes the query string in JSON output.

Example concept:
```python
@patch("src.logging_config.logging.Logger.info")
def test_log_context_includes_query(mock_logger_info, ...):
    ...
    service.search(sample_query)
    log_args, log_kwargs = mock_logger_info.call_args
    assert "query" in log_kwargs.get("extra", {})
```
Expect failure if the context is missing or not properly passed.

### GREEN
Enhance the `LogContext` usage (likely in `main.py`) to pass the query via `extra` args, or adjust the context manager to inject it automatically.

Re-run the targeted test and full suite if time allows.

### REFACTOR
- Confirm console logs remain clean (no JSON blobs unless requested).
- Update documentation in `logging_config.py` describing the new context field.
- Rotate logs to verify the new data appears in both `app.log` and `error.log` when appropriate.

---

## Observability Drill
Trigger an intentional error (e.g., remove API key temporarily) and inspect logs to understand how errors surface. Capture a snippet showing contextual data and store it in your notes.

---

## Stretch Challenges (Optional)
- Add an environment toggle to switch JSON logging on/off via `.env`.
- Capture a unique `request_id` per CLI invocation and propagate it through logs.
- Integrate logging with an external tool (e.g., send JSON logs to `jq` or Logstash) and document the steps.

---

## Reflection Prompt
Answer briefly:
- Which log statement did you enhance, and what new insight does it provide?
- How would you explain JSON vs. plain-text logging trade-offs to a stakeholder?
- What additional metrics or context would you add for production?
- How confident are you debugging using logs after this exercise?

---

## Exit Checklist
- [ ] Logging test added/passing
- [ ] Logs enriched with new context
- [ ] Error scenario observed and documented
- [ ] Reflection saved
- [ ] Narrative labs complete – move to practice loops
