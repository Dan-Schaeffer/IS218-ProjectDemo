# 🚦 TDD Workflow - RED, GREEN, REFACTOR

**⏱️ 15 min read | Master Test-Driven Development**

---

## 🎯 What is TDD?

**Test-Driven Development = Write tests BEFORE code**

### The Cycle:

```
🔴 RED → Write a failing test
    ↓
🟢 GREEN → Write code to pass it
    ↓
🔵 REFACTOR → Clean up the code
    ↓
   Repeat!
```

---

## 🤔 Why TDD?

### Without TDD:
1. Write code
2. "Hope" it works
3. Find bugs in production
4. Scared to change anything
5. Tech debt accumulates

### With TDD:
1. Define what you want (test)
2. Build exactly that (code)
3. Know immediately if it works
4. Refactor confidently
5. Living documentation

**TDD = Confidence + Speed + Quality**

---

## 🔴 RED Phase - Write the Test First

### Step 1: Think About What You Want

**Don't think:** "I'll write a parser"
**Do think:** "I need to parse a user's name from a JSON response"

### Step 2: Write the Test

```python
# tests/test_parser.py
from src.parser import parse_user

def test_parse_user_with_name():
    """Should extract name from response"""
    # Arrange - Set up test data
    response = {"user": {"name": "Alice"}}
    
    # Act - Do the thing
    result = parse_user(response)
    
    # Assert - Check it worked
    assert result == "Alice"
```

### Step 3: Run the Test (Should FAIL)

```bash
pytest tests/test_parser.py
```

**Expected output:**
```
ImportError: cannot import name 'parse_user'
```

**This is GOOD! 🎉**

---

## 🟢 GREEN Phase - Make It Pass

### Write the SIMPLEST code to pass

```python
# src/parser.py
def parse_user(response):
    """Extract name from response"""
    return response["user"]["name"]
```

### Run the Test (Should PASS)

```bash
pytest tests/test_parser.py
```

**Expected output:**
```
tests/test_parser.py::test_parse_user_with_name PASSED
```

**Don't add extra features! Just pass the test!**

---

## 🔵 REFACTOR Phase - Clean It Up

### Now Make It Better

```python
# src/parser.py
def parse_user(response: dict) -> str:
    """Extract name from user response.
    
    Args:
        response: Dict with 'user' key containing 'name'
        
    Returns:
        str: The user's name
    """
    return response["user"]["name"]
```

### Run Tests Again (Should Still PASS)

```bash
pytest tests/test_parser.py
```

If tests pass → refactor is good!
If tests fail → you broke something, undo!

---

## 🔁 The Complete Flow (With Git)

### Cycle 1: First Test

```bash
# 1. RED - Write failing test
# Edit: tests/test_parser.py (write test_parse_user_with_name)
pytest tests/test_parser.py  # Fails ✓

# 2. GREEN - Make it pass
# Edit: src/parser.py (write parse_user function)
pytest tests/test_parser.py  # Passes ✓

# 3. REFACTOR - Clean it up
# Edit: src/parser.py (add types and docstring)
pytest tests/test_parser.py  # Still passes ✓

# 4. COMMIT
git add tests/test_parser.py src/parser.py
git commit -m "feat: Add user name parsing"
```

### Cycle 2: Handle Edge Case

```bash
# 1. RED - Test for missing name
# Edit: tests/test_parser.py (write test_parse_user_missing_name)
pytest tests/test_parser.py  # New test fails ✓

# 2. GREEN - Handle the error
# Edit: src/parser.py (add error handling)
pytest tests/test_parser.py  # All pass ✓

# 3. REFACTOR - Maybe extract validation
# Edit: src/parser.py (improve structure)
pytest tests/test_parser.py  # Still pass ✓

# 4. COMMIT
git add tests/test_parser.py src/parser.py
git commit -m "fix: Handle missing user name"
```

---

## 📋 Real Example: Building a Feature

### Feature: "Validate email addresses"

**Step 1: Write Test (RED)**

```python
# tests/test_validators.py
from src.validators import is_valid_email

def test_valid_email_returns_true():
    """Should return True for valid email"""
    assert is_valid_email("alice@example.com") is True

def test_invalid_email_returns_false():
    """Should return False for email without @"""
    assert is_valid_email("notanemail") is False
```

**Run it:**
```bash
pytest tests/test_validators.py -v
# ImportError: cannot import name 'is_valid_email'
```

✅ RED phase complete!

---

**Step 2: Write Code (GREEN)**

```python
# src/validators.py
def is_valid_email(email: str) -> bool:
    """Check if email is valid format"""
    return "@" in email and "." in email
```

**Run it:**
```bash
pytest tests/test_validators.py -v
# 2 passed
```

✅ GREEN phase complete!

---

**Step 3: Add More Tests (RED Again)**

```python
# tests/test_validators.py
def test_email_with_no_domain_returns_false():
    """Should return False for email without domain"""
    assert is_valid_email("alice@") is False
```

**Run it:**
```bash
pytest tests/test_validators.py -v
# FAILED - Expected False but got True
```

✅ Back to RED (found a bug in our logic!)

---

**Step 4: Fix (GREEN Again)**

```python
# src/validators.py
def is_valid_email(email: str) -> bool:
    """Check if email is valid format"""
    if "@" not in email:
        return False
    
    local, domain = email.split("@", 1)
    return len(local) > 0 and len(domain) > 0 and "." in domain
```

**Run it:**
```bash
pytest tests/test_validators.py -v
# 3 passed
```

✅ GREEN phase complete!

---

**Step 5: Refactor (REFACTOR)**

```python
# src/validators.py
import re

def is_valid_email(email: str) -> bool:
    """Check if email is valid format.
    
    Args:
        email: String to validate
        
    Returns:
        bool: True if valid email format, False otherwise
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))
```

**Run it:**
```bash
pytest tests/test_validators.py -v
# 3 passed
```

✅ REFACTOR complete - cleaner and more robust!

---

**Step 6: Commit**

```bash
git add tests/test_validators.py src/validators.py
git commit -m "feat: Add email validation with regex"
```

---

## 🎯 TDD Best Practices

### DO:
- ✅ Write test first (before code)
- ✅ Write ONE test at a time
- ✅ Run tests after every change
- ✅ Keep tests simple and focused
- ✅ Test behavior, not implementation
- ✅ Commit after each GREEN → REFACTOR cycle

### DON'T:
- ❌ Write code before test
- ❌ Write multiple tests at once
- ❌ Skip running tests
- ❌ Write complex tests
- ❌ Test internal details
- ❌ Commit failing tests

---

## 🧪 AAA Pattern (Arrange-Act-Assert)

### Every test has 3 parts:

```python
def test_example():
    # ARRANGE - Set up the test data
    user = User(name="Alice", age=30)
    
    # ACT - Do the thing you're testing
    result = user.is_adult()
    
    # ASSERT - Check it worked
    assert result is True
```

**Why AAA?**
- Clear structure
- Easy to read
- Easy to debug

---

## 📊 Test Coverage

### Check what's tested:

```bash
pytest --cov=src --cov-report=html
open htmlcov/index.html
```

**What you'll see:**
- Green lines = covered by tests ✅
- Red lines = not covered ❌

**Goal: 90%+ coverage**

### What to test:
- ✅ All public functions
- ✅ All classes
- ✅ Error handling
- ✅ Edge cases

### What NOT to test:
- ❌ Third-party libraries
- ❌ Python built-ins
- ❌ Simple property getters

---

## 🚨 Common TDD Mistakes

### Mistake #1: Writing Code First

**Wrong:**
```python
# Write function
def calculate_total(items):
    return sum(items)

# Then write test
def test_calculate_total():
    assert calculate_total([1, 2, 3]) == 6
```

**Right:**
```python
# Write test FIRST
def test_calculate_total():
    assert calculate_total([1, 2, 3]) == 6

# See it fail, THEN write code
def calculate_total(items):
    return sum(items)
```

---

### Mistake #2: Testing Too Much at Once

**Wrong:**
```python
def test_user_system():
    """Test user creation, validation, and saving"""
    user = User("Alice", "alice@test.com")
    assert user.is_valid()
    user.save()
    assert User.find("Alice").email == "alice@test.com"
```

**Right:**
```python
def test_user_creation():
    """Test user can be created"""
    user = User("Alice", "alice@test.com")
    assert user.name == "Alice"
    assert user.email == "alice@test.com"

def test_user_validation():
    """Test user validation works"""
    user = User("Alice", "alice@test.com")
    assert user.is_valid() is True

def test_user_saving():
    """Test user can be saved and retrieved"""
    user = User("Alice", "alice@test.com")
    user.save()
    found = User.find("Alice")
    assert found.email == "alice@test.com"
```

---

### Mistake #3: Not Running Tests Often

**Wrong:**
```
Write test 1 → Write code → Write test 2 → Write code → Run tests
```

**Right:**
```
Write test 1 → Run (RED) → Write code → Run (GREEN) → 
Write test 2 → Run (RED) → Write code → Run (GREEN)
```

---

## 🤖 TDD with Claude

### Effective Prompts:

**Step 1: Get the test**
```
"I need to validate user input.
Should check: name is not empty, email is valid.
Write tests for a validate_user function.
Return dictionary with 'valid': bool and 'errors': list."
```

**Step 2: See it fail**
```bash
pytest tests/test_validators.py  # RED
```

**Step 3: Get implementation**
```
"Here's my failing test: [paste test]
Write the validate_user function to pass it."
```

**Step 4: See it pass**
```bash
pytest tests/test_validators.py  # GREEN
```

**Step 5: Ask for improvements**
```
"The code works but looks messy.
Help me refactor while keeping tests passing."
```

---

## 📚 Real-World TDD Example

### Building an OpenAI Client

**RED - Test 1: Can create client**
```python
def test_create_client():
    client = OpenAIClient(api_key="test-key")
    assert client.api_key == "test-key"
```

**GREEN - Implement**
```python
class OpenAIClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
```

---

**RED - Test 2: Can make request**
```python
@patch('openai.ChatCompletion.create')
def test_chat_completion(mock_create):
    mock_create.return_value = {"choices": [{"message": {"content": "Hi"}}]}
    
    client = OpenAIClient(api_key="test-key")
    response = client.chat("Hello")
    
    assert response == "Hi"
```

**GREEN - Implement**
```python
import openai

class OpenAIClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        openai.api_key = api_key
    
    def chat(self, prompt: str) -> str:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response["choices"][0]["message"]["content"]
```

---

**RED - Test 3: Handle errors**
```python
@patch('openai.ChatCompletion.create')
def test_chat_handles_api_error(mock_create):
    mock_create.side_effect = openai.error.APIError("API down")
    
    client = OpenAIClient(api_key="test-key")
    
    with pytest.raises(openai.error.APIError):
        client.chat("Hello")
```

**GREEN - Already passes!** (No change needed)

---

**REFACTOR - Clean it up**
```python
import openai
from typing import Optional

class OpenAIClient:
    """Client for OpenAI Chat API"""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        """Initialize client with API key.
        
        Args:
            api_key: OpenAI API key
            model: Model to use for chat
        """
        self.api_key = api_key
        self.model = model
        openai.api_key = api_key
    
    def chat(self, prompt: str, temperature: float = 0.7) -> str:
        """Send chat message and get response.
        
        Args:
            prompt: Message to send
            temperature: Randomness (0-1)
            
        Returns:
            str: Response text
            
        Raises:
            openai.error.APIError: If API request fails
        """
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        return response["choices"][0]["message"]["content"]
```

**Run tests - still pass!**

---

## ✅ TDD Checklist

Before moving to next feature:

- [ ] Test written first
- [ ] Test failed (RED)
- [ ] Code written to pass test
- [ ] Test passed (GREEN)
- [ ] Code refactored
- [ ] Test still passes
- [ ] Committed with good message
- [ ] Coverage maintained/improved

---

## 🎯 Quick Reference

### TDD Commands:

```bash
# Run all tests
pytest

# Run specific file
pytest tests/test_parser.py

# Run specific test
pytest tests/test_parser.py::test_parse_user

# Run with coverage
pytest --cov=src

# Run with detailed output
pytest -v

# Run and open coverage report
pytest --cov=src --cov-report=html && open htmlcov/index.html
```

---

## 💡 Pro Tips

1. **Start simple** - Test happy path first, edge cases later
2. **One assertion per test** - Makes failures obvious
3. **Name tests well** - `test_parse_user_with_missing_name` not `test_1`
4. **Keep tests fast** - Use mocks for external services
5. **Don't test libraries** - Trust pytest, openai, etc.
6. **Red, green, commit** - Commit after each cycle
7. **Use coverage to find gaps** - But don't obsess over 100%

---

## 🚀 Level Up

### Beginner TDD:
- Write test
- Write code
- Move on

### Intermediate TDD:
- Write test
- See it fail
- Write minimal code
- See it pass
- Commit

### Advanced TDD:
- Think about API design through tests
- Write multiple test cases before implementation
- Refactor with confidence
- Use TDD to drive architecture

**Aim for Advanced!**

---

## 🆘 When Tests Fail

### Debug Strategy:

```bash
# 1. Run just the failing test
pytest tests/test_file.py::test_failing -v

# 2. Add print statements (or use debugger)
def test_failing():
    result = function_under_test()
    print(f"Result: {result}")  # Debug
    assert result == expected

# 3. Check the traceback
# Read from bottom up - where did it actually fail?

# 4. Simplify the test
# Remove complexity until you find the issue

# 5. Check your assumptions
# Is the test correct? Is the code correct?
```

---

## 📊 TDD Benefits (Why It's Worth It)

### Short Term:
- ✅ Catch bugs immediately
- ✅ Clear requirements
- ✅ Faster debugging

### Long Term:
- ✅ Refactor without fear
- ✅ Living documentation
- ✅ Easier onboarding
- ✅ Higher code quality
- ✅ Less technical debt

### Career:
- ✅ Professional skill
- ✅ Confidence in interviews
- ✅ Trusted by teams
- ✅ Ship faster with less bugs

**Companies LOVE developers who write tests!**

---

**Master TDD and you'll ship better code, faster, with more confidence!** 🚀

---

*For git workflow, see [GIT_WORKFLOW.md](GIT_WORKFLOW.md)*
*For logging, see [LOGGING.md](LOGGING.md)*
