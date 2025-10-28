# Lab 03 – Chapter 3: Response Parser

**Estimated time:** 50 minutes  
**Artifacts:** Updated tests covering malformed responses, reflection

---

## Objectives
- Decode how `ResponseParser` turns raw OpenAI payloads into `SearchResult` objects.
- Harden the parser against malformed or missing data.
- Practice writing tests that simulate messy API responses.

---

## Before You Start
- Run `pytest tests/test_parser.py -v`.
- Open `tests/fixtures/sample_responses.json` to see realistic payloads.
- Review `_extract_citations` and `_extract_sources` helpers.

---

## Guided Exploration
1. Draw a data-flow diagram: response → `output` list → `SearchResult` components.
2. Identify assumptions the parser currently makes (e.g., `output` exists, `web_search_call` present).
3. Note where defensive checks already exist and where they could be tightened.

---

## Core Exercise (TDD Loop)
### RED
Add a test to `tests/test_parser.py` that feeds a response missing the `web_search_call` item and asserts the parser still returns a `SearchResult` with empty `sources`.

Example skeleton:
```python
def test_parse_handles_missing_web_search_call(parser, sample_message_only_response):
    result = parser.parse(sample_message_only_response, "ai news")
    assert result.sources == []
```
Use a fixture or construct the dictionary inline. Expect failure or unexpected behavior.

### GREEN
Modify `parse` so it safely handles the absence of `web_search_call`, defaulting `sources` and `search_id` appropriately. Ensure existing functionality remains.

Re-run `pytest tests/test_parser.py -v`.

### REFACTOR
- Consolidate repeated `item.get("type")` checks via list comprehensions where sensible.
- Clarify docstrings to document the new fallback behavior.

---

## Data Quality Drill
Inject an annotation with missing `title` and confirm `_extract_citations` still returns a usable `Citation` (possibly with placeholder values). Add tests if desired.

---

## Stretch Challenges (Optional)
- Add support for multi-part messages by concatenating multiple `output_text` fragments; test the merge order.
- Record the parser timestamp as UTC and expose a helper that formats it relative to now ("2 minutes ago").
- Emit a warning log when annotations are dropped due to missing fields (coordinate with logging lab).

---

## Reflection Prompt
Log responses to:
- How does defensive parsing protect the user experience?
- What trade-offs exist between strict validation and graceful degradation?
- Which test gave you the best signal that the parser is resilient?
- What future API change would require the least code modifications, and why?

---

## Exit Checklist
- [ ] New parser test in place and passing
- [ ] Parser handles missing `web_search_call`
- [ ] Reflection recorded
- [ ] Ready for `LAB_04_SERVICE`
