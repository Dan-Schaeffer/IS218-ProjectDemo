# Lab 01 – Chapter 1: Data Models

**Estimated time:** 45 minutes  
**Artifacts:** Updated code (optional), lab reflection, failing/passing test screenshots (optional)

---

## Objectives
- Read `src/models.py` and understand how dataclasses, properties, and custom exceptions shape the domain.
- Practice the RED → GREEN → REFACTOR loop on a safe change.
- Observe how `tests/test_models.py` guards behavior.

---

## Before You Start
- Activate your virtual environment and ensure `pytest` passes.
- Open `src/models.py` side-by-side with `tests/test_models.py`.
- Skim comments labeled **📚 CONCEPT** and **📝 DESIGN DECISION**.

---

## Guided Exploration
1. Run `pytest tests/test_models.py -v` to baseline current behavior.
2. In a Python REPL, copy the interactive examples from the chapter and verify outputs.
3. Identify one design choice that surprised you and note why.

---

## Core Exercise (TDD Loop)
### RED
Add a new failing test in `tests/test_models.py`:
```python
def test_search_options_rejects_invalid_reasoning_effort():
    options = SearchOptions(reasoning_effort="ultra")
    assert options.reasoning_effort == "low"
```
Run the test and confirm it fails (the code does not yet enforce this rule).

### GREEN
Implement the minimal code in `SearchOptions` to clamp unrecognized `reasoning_effort` values to `"low"`.

Suggested approach:
```python
def __post_init__(self):
    allowed = {"low", "medium", "high"}
    if self.reasoning_effort not in allowed:
        self.reasoning_effort = "low"
```
Run `pytest tests/test_models.py -v` until it passes.

### REFACTOR
- Extract the allowed set to a module-level constant for clarity.
- Update the docstring block to mention the new validation.
- Ensure tests still pass.

---

## Stretch Challenges (Optional)
- Add a `@property` on `SearchResult` that returns a friendly timestamp string (e.g., `result.display_time`). Test it.
- Extend `SearchError.__str__` to include a hint when details contain a `"hint"` key; adjust tests.
- Explore immutability: experiment with `frozen=True` dataclasses and note trade-offs.

---

## Reflection Prompt
Write 4–5 sentences capturing:
- What new insight did you gain about dataclasses or properties?
- Which test failed first, and what did it teach you?
- How would you explain the difference between `Citation` and `Source` to a teammate?
- What is one question you still have about the data layer?

Record the reflection in your learning journal or cohort log.

---

## Exit Checklist
- [ ] New test added and passing
- [ ] Implementation change committed (or reverted if you chose not to keep it)
- [ ] Reflection captured
- [ ] Ready to proceed to `LAB_02_CLIENT`
