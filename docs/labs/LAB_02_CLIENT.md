# Lab 02 – Chapter 2: OpenAI Client

**Estimated time:** 60 minutes  
**Artifacts:** Passing test demonstrating new validation, AI pairing log (optional)

---

## Objectives
- Understand how `src/client.py` packages requests, handles secrets, and translates errors.
- Practice mocking the OpenAI SDK in tests.
- Strengthen failure-mode thinking by adding defensive checks.

---

## Before You Start
- Run `pytest tests/test_client.py -v` to ensure a clean baseline.
- Skim `docs/AI_COLLABORATION.md`—consider pairing with an AI assistant for brainstorming edge cases.
- Review the error taxonomy in `SearchError` (Chapter 1).
- Keep `docs/web_search_openai.md` open as an API reference for payloads, sources, and filters.

---

## Guided Exploration
1. Trace the flow of `WebSearchClient.search` and list every place an exception could be raised.
2. Inspect `_construct_payload` and sketch the JSON payload the API receives.
3. Note how `_response_to_dict` shields the rest of the system from SDK object internals.

---

## Core Exercise (TDD Loop)
### RED
Add a test to `tests/test_client.py` that asserts a friendly error when the query contains only punctuation.

Example structure:
```python
def test_search_rejects_punctuation_only_query(web_client):
    with pytest.raises(ValueError, match="Query cannot contain only punctuation"):
        web_client.search("!!!")
```
Expect failure because the check does not exist yet.

### GREEN
Implement a guard in `WebSearchClient.search` that rejects punctuation-only strings (hint: use `str.isalnum` or a regex). Raise `ValueError` with the message asserted above.

Re-run `pytest tests/test_client.py -v` until green.

### REFACTOR
- Extract validation logic into a helper method for readability.
- Add a docstring snippet showing how validation protects API costs.
- Ensure no other tests regressed.

---

## Mocking Deep Dive
Try swapping the official SDK client with a simple fake:
```python
client.client = MagicMock()
client.client.responses.create.return_value = DummyResponse(...)
```
Observe how `_response_to_dict` keeps the rest of the code agnostic to the fake.

---

## Stretch Challenges (Optional)
- Simulate a `RateLimitError` with exponential backoff logic; add tests to ensure retries stop after N attempts.
- Add optional timeout support to `_construct_payload` and expose it through `SearchOptions`.
- Validate that `allowed_domains` are lowercased before being sent; test the transformation.

---

## Reflection Prompt
Capture short answers:
- Which failure mode surprised you the most?
- How does the client isolate the rest of the app from OpenAI’s SDK changes?
- What did mocking teach you about dependency management?
- Where might you add structured logging in the client, and why?

---

## Exit Checklist
- [ ] New validation test passing
- [ ] Client code updated responsibly
- [ ] Reflection logged
- [ ] Ready for `LAB_03_PARSER`
