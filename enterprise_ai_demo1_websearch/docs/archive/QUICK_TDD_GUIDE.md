# ⚡ Quick TDD Guide

**⏱️ 15 min read | Everything you need to know about Test-Driven Development**

---

## 🎯 What Is TDD?

**TDD = Test-Driven Development**

Write tests BEFORE writing code. Sounds backwards, but it's genius.

### The 3-Step Process (RED-GREEN-REFACTOR)

```
🔴 RED:   Write a test → Run it → It fails (no code yet)
🟢 GREEN: Write minimal code → Run test → It passes
🔵 REFACTOR: Clean up code → Run test → Still passes
```

**Then repeat for next feature.**

---

## 🤔 Why Your Professor Wants This

### Without TDD:
```python
# You write a bunch of code
def search(query):
    # 50 lines of code
    pass

# Does it work? 🤷
# You test manually by running the program
# Maybe it works, maybe not
# You change something
# Did you break it? 🤷
# More manual testing
# Push to GitHub
# Demo day: Crashes
```

**Result:** Stress, bugs, bad grade

### With TDD:
```python
# Write test FIRST
def test_search_returns_results():
    results = search("Python")
    assert len(results) > 0  # ❌ FAILS

# Write code
def search(query):
    return api.search(query)  # ✅ PASSES

# Change something
def search(query, limit=10):
    return api.search(query)[:limit]  # ✅ STILL PASSES

# Every change → Run tests → Know immediately if broken
```

**Result:** Confidence, no bugs, A grade

---

## 🚀 TDD In Action (Real Example)

Let's build a simple search function.

### Step 1: RED (Write failing test)

```python
# tests/test_search.py
def test_search_returns_list():
    """Search should return a list of results"""
    results = search("Python tutorials")
    assert isinstance(results, list)
    assert len(results) > 0
```

**Run it:**
```bash
pytest tests/test_search.py
# ❌ FAILS - search() doesn't exist yet
```

---

### Step 2: GREEN (Make it pass)

```python
# src/search.py
def search(query):
    """Search for query and return results"""
    return [{"title": "Result 1"}]  # Minimal implementation
```

**Run it:**
```bash
pytest tests/test_search.py
# ✅ PASSES - test is green!
```

---

### Step 3: REFACTOR (Improve code)

```python
# src/search.py
from openai import OpenAI

def search(query):
    """Search using OpenAI web search"""
    if not query:
        raise ValueError("Query cannot be empty")
    
    client = OpenAI()
    response = client.search(query)
    return parse_results(response)

def parse_results(response):
    """Parse API response into list"""
    return [{"title": r.title, "url": r.url} for r in response.results]
```

**Run it:**
```bash
pytest tests/test_search.py
# ✅ STILL PASSES - refactored but not broken!
```

---

### Step 4: Add More Tests (Repeat RED-GREEN-REFACTOR)

```python
# tests/test_search.py
def test_search_empty_query_raises_error():
    """Empty query should raise error"""
    with pytest.raises(ValueError):
        search("")

def test_search_results_have_required_fields():
    """Each result should have title and url"""
    results = search("Python")
    assert all("title" in r for r in results)
    assert all("url" in r for r in results)

def test_search_filters_by_domain():
    """Should filter results by domain"""
    results = search("Python", domains=["python.org"])
    assert all("python.org" in r["url"] for r in results)
```

**Each test → RED → GREEN → REFACTOR → Next test**

---

## 📝 TDD Pattern Cheat Sheet

### 1. Data Models
```python
# RED: Write test
def test_create_search_result():
    result = SearchResult(title="Test", url="http://test.com")
    assert result.title == "Test"
    assert result.url == "http://test.com"

# GREEN: Write code
@dataclass
class SearchResult:
    title: str
    url: str

# ✅ Test passes
```

### 2. Functions
```python
# RED: Write test
def test_parse_response():
    raw = {"results": [{"title": "Test"}]}
    parsed = parse_response(raw)
    assert parsed[0].title == "Test"

# GREEN: Write code
def parse_response(raw_data):
    return [SearchResult(**r) for r in raw_data["results"]]

# ✅ Test passes
```

### 3. Error Handling
```python
# RED: Write test
def test_invalid_input_raises_error():
    with pytest.raises(ValueError, match="Invalid query"):
        search("")

# GREEN: Write code
def search(query):
    if not query:
        raise ValueError("Invalid query")
    # ... rest of code

# ✅ Test passes
```

### 4. API Calls (Use Mocks!)
```python
# RED: Write test
def test_search_calls_api(mock_openai):
    """Don't call real API in tests!"""
    mock_openai.search.return_value = {"results": []}
    search("test")
    mock_openai.search.assert_called_once_with("test")

# GREEN: Write code  
def search(query):
    client = OpenAI()
    return client.search(query)

# ✅ Test passes (without calling real API!)
```

---

## 🎯 Your First TDD Session

Follow this checklist:

### ☐ Setup
```bash
# Install pytest
pip install pytest pytest-cov pytest-mock

# Create test file
touch tests/test_my_feature.py
```

### ☐ Write First Test (RED)
```python
# tests/test_my_feature.py
def test_basic_functionality():
    result = my_function("input")
    assert result == "expected output"
```

### ☐ Run Test (Should Fail)
```bash
pytest tests/test_my_feature.py -v
# ❌ Should see red failure
```

### ☐ Write Code (GREEN)
```python
# src/my_feature.py
def my_function(input):
    return "expected output"
```

### ☐ Run Test Again (Should Pass)
```bash
pytest tests/test_my_feature.py -v
# ✅ Should see green pass
```

### ☐ Refactor
```python
# Improve code quality
# Add error handling
# Add docstrings
# But keep tests passing!
```

### ☐ Repeat!
Add next test → RED → GREEN → REFACTOR → Repeat

---

## 🎨 TDD With AI (How We Work Together)

### ❌ Wrong Way
```
You: "Write all my code"
AI: [Writes code without tests]
You: "Does it work?" 🤷
```

### ✅ Right Way
```
You: "I need to build [feature]. What should I test?"

AI: "Here are 5 tests you should write:
     1. Test basic case
     2. Test edge case
     3. Test error handling
     ..."

You: "Write the first test"
AI: [Writes test]

You: "Run it" → ❌ RED

You: "Now help me write code to pass this test"
AI: [Writes minimal code]

You: "Run it" → ✅ GREEN

You: "Good! Next test."
```

**Pattern:** Let AI suggest tests → You verify they make sense → Together implement

---

## 📊 How To Get 90% Test Coverage

Coverage = % of your code that tests actually run.

### Check Your Coverage
```bash
pytest --cov=src tests/
# Shows coverage report
```

### Example Output
```
Name           Stmts   Miss  Cover
----------------------------------
src/search.py     45      5    89%
src/parse.py      30      2    93%
----------------------------------
TOTAL            75      7    91%
```

### Tips To Hit 90%+

1. **Test every function**
   ```python
   def my_function():
       pass
   
   def test_my_function():  # ✅ Has test
       assert my_function() is None
   ```

2. **Test error paths**
   ```python
   def search(query):
       if not query:
           raise ValueError  # ← Test this!
       return results
   
   def test_empty_query_raises_error():
       with pytest.raises(ValueError):
           search("")
   ```

3. **Test edge cases**
   ```python
   def test_search_with_empty_results():
       # What if API returns nothing?
   
   def test_search_with_special_characters():
       # What if query has emojis?
   
   def test_search_with_very_long_query():
       # What if query is 1000 chars?
   ```

4. **Use fixtures for common setup**
   ```python
   # tests/conftest.py
   @pytest.fixture
   def sample_response():
       return {"results": [{"title": "Test"}]}
   
   # Now use in any test
   def test_parse(sample_response):
       parsed = parse(sample_response)
       assert len(parsed) == 1
   ```

---

## 🆘 Common TDD Problems

### "My test is failing but I don't know why"
```bash
# Run with verbose output
pytest -v

# Run specific test
pytest tests/test_file.py::test_name -v

# See print statements
pytest -s
```

### "Tests are too slow"
```python
# Use mocks instead of real API calls
@pytest.fixture
def mock_api(mocker):
    return mocker.patch('src.client.OpenAI')

def test_with_mock(mock_api):
    mock_api.search.return_value = [...]
    # Fast! No real API call
```

### "I don't know what to test"
Ask yourself:
1. What should this function do? (basic test)
2. What if input is empty? (edge case test)
3. What if input is wrong type? (error test)
4. What's the expected output format? (format test)

### "Coverage is at 85%, need 90%"
```bash
# See what's not covered
pytest --cov=src --cov-report=html tests/
open htmlcov/index.html

# Red lines = not covered
# Write tests for those lines
```

---

## 📋 TDD Checklist For Your Project

### ☐ Setup Phase
- [ ] Install pytest, pytest-cov, pytest-mock
- [ ] Create `pytest.ini` config file
- [ ] Create `tests/conftest.py` for fixtures
- [ ] Create folder structure: `tests/` mirrors `src/`

### ☐ For Each Component
- [ ] Write tests FIRST (RED)
- [ ] Run tests (confirm they fail)
- [ ] Write minimal code (GREEN)
- [ ] Run tests (confirm they pass)
- [ ] Refactor code (keep tests green)
- [ ] Check coverage → aim for 90%+

### ☐ Before Pushing Code
- [ ] All tests passing: `pytest -v`
- [ ] Coverage above 90%: `pytest --cov=src`
- [ ] No warnings
- [ ] Code is clean and documented

---

## ⏭️ Next Steps

1. ✅ You understand TDD now
2. ➡️ Look at `tests/` folder in this repo (real examples!)
3. ➡️ Pick a project from [PROJECT_IDEAS.md](PROJECT_IDEAS.md)
4. ➡️ Write your first test!

---

## 💡 The TDD Mindset Shift

**Old way:** Code → Hope it works → Manual test → Debug → Repeat

**TDD way:** Test → Code → Auto verify → Confidence → Move forward

**The magic:** You're never wondering "does this work?" - Tests tell you instantly.

---

**Ready to try TDD? Start with ONE simple function. Write a test. Watch it fail. Make it pass. Feel the power.** ⚡

---

*Need help writing tests? Ask Claude: "Help me write tests for [feature]. What cases should I cover?"*
