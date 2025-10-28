# Lab 04 – Chapter 4: Search Service

**Estimated time:** 55 minutes  
**Artifacts:** Updated tests validating new business rule, service diagram sketch

---

## Objectives
- Study how `SearchService` coordinates validation, client calls, and parsing.
- Add a business rule while keeping responsibilities separated.
- Observe how integration-style tests provide safety nets.

---

## Before You Start
- Run `pytest tests/test_search_service.py -v`.
- Skim the `SearchService` docstrings and note the validation flow.
- Review how `apply_domain_filters` returns a new `SearchOptions` instance.
- Reference the system diagram in `docs/architecture.md` to see how the service coordinates collaborators.

---

## Guided Exploration
1. Trace `service.search` and list inputs/outputs for each collaborating component.
2. Identify which errors bubble up as `ValueError` vs `SearchError`.
3. Discuss with a peer or AI assistant: what other policy checks might the service own?

---

## Core Exercise (TDD Loop)
### RED
Add a test ensuring the service rejects queries containing banned phrases (e.g., "DROP TABLE").

Example:
```python
def test_search_rejects_banned_phrases(self, mock_client_class, test_api_key):
    service = SearchService(api_key=test_api_key)
    with pytest.raises(ValueError, match="contains restricted language"):
        service.search("DROP TABLE users")
```
Expect failure because the check is absent.

### GREEN
Implement a `BANNED_PHRASES` list in `SearchService` and extend `validate_query` to screen queries (case-insensitive). Raise `ValueError` with the message used in the test.

Re-run service tests until passing.

### REFACTOR
- Extract validation logic into focused helper methods (`_is_banned_phrase`).
- Document the new safeguard in the service docstring.
- Ensure `apply_domain_filters` tests and integrations still pass.

---

## Architecture Sketch
Create a quick diagram (paper or digital) showing the flow: CLI → Service → Client → Parser → Models. Highlight where validation lives. Snap a photo or save the file and reference it in your notes.

---

## Stretch Challenges (Optional)
- Add rate limiting at the service layer (e.g., simple in-memory counter) and test using time-mocking.
- Allow custom banned phrase lists via `SearchOptions`; ensure defaults still work.
- Introduce a metrics hook that counts successful/failed searches; plan how logging would capture it.

---

## Reflection Prompt
Answer briefly:
- Why does the service—not the CLI—own validation rules?
- How do integration tests differ from the unit tests earlier in the file?
- What new business rule would you add next, and where would it live?
- How might this service evolve if multiple clients (CLI, web, API) called it?

---

## Exit Checklist
- [ ] Banned phrase protection implemented and tested
- [ ] Architecture sketch stored
- [ ] Reflection noted
- [ ] Ready for `LAB_05_MAIN`
