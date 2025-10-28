# 📅 Course Structure: Your 2-Week Journey

**Total Time:** 2 weeks (4 sessions, 90 min each) + Week 3 demo presentations  
**Format:** Hands-on, project-based learning  
**Outcome:** Working AI application + professional development skills

---

## Overview

> Use the [Learning Path Map](LEARNING_PATH.md) alongside this schedule to track labs, milestones, and reflections.

### Week 1: Learn by Reading Production Code
Understand how professional software is structured, tested, and documented.

### Week 2: Build Your Own Application
Apply patterns from Week 1 to create your unique AI project.

### Week 3: Demo Presentations
Present your work professionally. Students present in assigned slots while working quietly on final polish between presentations.

---

## Week 1: Foundation

### Session 1 (90 min): Architecture & Patterns

**Learning Objective:** Understand how production applications are structured

#### Part 1: Setup & Exploration (25 min)

**Do This:**
1. Complete [Getting Started Guide](GETTING_STARTED.md) if not done
2. Run the application with different queries
3. Observe the output format
4. Look at the file structure

**Try:**
```bash
python -m src.main "machine learning basics"
python -m src.main "latest SpaceX news" --verbose
python -m src.main "Python tutorials" --model gpt-4o
```

**Observe:**
- How does it format results?
- What happens with `--verbose`?
- How are sources cited?

#### Part 2: Read Code as Narrative (45 min in-class, continue at home)

**Do This:**
1. Open `src/models.py`
2. Read it like a textbook chapter
3. Try the examples in Python REPL
4. Ask yourself: "Why did they do it this way?"

_Note: If you need more time to fully understand the patterns, that's normal! Continue exploration at home._

**Example Interactive Reading:**
```bash
# Open Python REPL
python

# Try the examples from comments
>>> from src.models import SearchOptions
>>> options = SearchOptions()
>>> print(options)
>>> options = SearchOptions(model="gpt-4o", reasoning_effort="high")
>>> print(options)
```

5. Continue with `src/client.py`
6. Notice the design patterns explained in comments

#### Part 3: Hands-On Experimentation (20 min)

**Do This:**
1. Make a small change to `SearchOptions`
2. Run tests to see what breaks
3. Fix it and see tests pass

**Exercise:**
```python
# In src/models.py, add a new field
@dataclass
class SearchOptions:
    model: str = "gpt-4o-mini"
    # Add this:
    max_results: int = 10
```

```bash
# Run tests
pytest tests/test_models.py -v

# What happened? Why?
```

#### Homework (2-3 hours)

**Reading:**
- [ ] Complete `src/parser.py` (30 min)
- [ ] Complete `src/search_service.py` (30 min)
- [ ] Skim `src/main.py` (20 min)

**Planning:**
- [ ] Read [Project Ideas](PROJECT_IDEAS.md) (20 min)
- [ ] Choose your project idea
- [ ] Read [OpenAI API docs](openai_tools_research_oct2025.md) for your chosen APIs (30 min)
- [ ] Sketch your application architecture on paper

**Questions to Answer:**
1. What problem will your app solve?
2. Which OpenAI APIs will you use?
3. What are 3-5 core features?
4. Who is the user?

---

### Session 2 (90 min): Test-Driven Development

**Learning Objective:** Master TDD methodology and understand how tests prove correctness

#### Part 1: Understand Test Patterns (30 min)

**Do This:**
1. Read [TDD Workflow Guide](TDD_WORKFLOW.md)
2. Open `tests/conftest.py`
3. See how fixtures provide reusable test data
4. Open `tests/test_models.py`
5. Notice the pattern: Arrange → Act → Assert

**Example Test Anatomy:**
```python
def test_create_default_search_options():
    """Test that default options are created correctly."""
    # ARRANGE: Setup (often empty for simple tests)
    
    # ACT: Create the object
    options = SearchOptions()
    
    # ASSERT: Verify expectations
    assert options.model == "gpt-4o-mini"
    assert options.allowed_domains is None
```

6. Continue with `tests/test_client.py`
7. See how we mock external API calls

#### Part 2: Practice TDD (40 min)

**Do This - RED-GREEN-REFACTOR:**

**RED - Write Failing Test:**
```bash
# Create new test file
touch tests/test_my_feature.py
```

```python
# tests/test_my_feature.py
def test_feature_that_doesnt_exist_yet():
    """Test that my new feature works."""
    # This will fail because feature doesn't exist
    from src.models import MyNewFeature
    feature = MyNewFeature(value=42)
    assert feature.doubled() == 84
```

```bash
# Run it - should fail
pytest tests/test_my_feature.py -v
# ✅ Good! Failing test = RED phase
```

**GREEN - Make It Pass:**
```python
# src/models.py
@dataclass
class MyNewFeature:
    value: int
    
    def doubled(self) -> int:
        return self.value * 2
```

```bash
# Run again - should pass
pytest tests/test_my_feature.py -v
# ✅ Good! Passing test = GREEN phase
```

**REFACTOR - Improve Code:**
```python
# Make it more Pythonic
@dataclass
class MyNewFeature:
    value: int
    
    @property
    def doubled(self) -> int:
        """Return double the value."""
        return self.value * 2
```

```bash
# Tests still pass after refactoring
pytest tests/test_my_feature.py -v
# ✅ Good! Tests protect your refactoring
```

#### Part 3: Group Discussion (20 min)

**Discuss:**
1. Why write tests before code?
2. What did you find surprising?
3. How would you test your project idea?

**Share:**
- Your project idea
- One concern about testing it
- Get feedback from peers

#### Homework (3-4 hours)

**Practice:**
- [ ] Read remaining test files (1 hour)
- [ ] Write 3 test cases for your project (pseudocode is fine) (1 hour)

**Planning:**
- [ ] Read [AI Collaboration Guide](AI_COLLABORATION.md) (15 min)
- [ ] Read [Git Workflow Guide](GIT_WORKFLOW.md) (15 min)
- [ ] Create GitHub repo for your project (15 min)
- [ ] Write initial README for your project (30 min)
- [ ] Draft your data models on paper (30 min)

---

## Week 2: Building Your Application

### Session 3 (90 min): Core Development

**Learning Objective:** Build your application using TDD and professional patterns

#### Part 1: Project Initialization (25 min)

**Do This:**
```bash
# Create your project
mkdir my-ai-project
cd my-ai-project

# Initialize git
git init
git branch -M main

# Copy structure from reference
cp -r ../enterprise_ai_demo1_websearch/.github .
cp ../enterprise_ai_demo1_websearch/.gitignore .
cp ../enterprise_ai_demo1_websearch/.pylintrc .
cp ../enterprise_ai_demo1_websearch/pytest.ini .
cp ../enterprise_ai_demo1_websearch/.coveragerc .
cp ../enterprise_ai_demo1_websearch/requirements.txt .

# Create structure
mkdir src tests docs
touch src/__init__.py tests/__init__.py
touch README.md .env.example

# Install dependencies
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# First commit
git add .
git commit -m "chore: Initial project structure"
```

#### Part 2: Build Core Models with TDD (45 min)

**RED Phase - Write Tests First:**
```python
# tests/test_models.py
"""Tests for data models."""
import pytest
from src.models import MyMainModel

def test_create_my_model():
    """Test creating the main model."""
    model = MyMainModel(name="test", value=42)
    assert model.name == "test"
    assert model.value == 42

def test_model_validation():
    """Test that model validates input."""
    with pytest.raises(ValueError):
        MyMainModel(name="", value=-1)
```

```bash
# Should fail - feature doesn't exist
pytest tests/test_models.py -v
```

**GREEN Phase - Implement:**
```python
# src/models.py
"""Data models for my AI application."""
from dataclasses import dataclass

@dataclass
class MyMainModel:
    name: str
    value: int
    
    def __post_init__(self):
        if not self.name:
            raise ValueError("Name cannot be empty")
        if self.value < 0:
            raise ValueError("Value must be positive")
```

```bash
# Should pass now
pytest tests/test_models.py -v
```

**Commit:**
```bash
git add src/models.py tests/test_models.py
git commit -m "feat: Add core data models with validation"
```

#### Part 3: Peer Code Review (20 min)

**Do This:**
1. Share your screen
2. Walk through your tests and code
3. Get feedback on:
   - Test coverage
   - Code clarity
   - Design decisions

**Give Feedback:**
- What's clear?
- What's confusing?
- Suggest improvements

#### Homework (5-7 hours)

**Must Complete:**
- [ ] Build API client layer (2-3 hours)
- [ ] Write client tests with mocking (1-2 hours)
- [ ] Build parser/service layer (2 hours)
- [ ] Achieve >70% test coverage (check with `pytest --cov=src`)
- [ ] Make atomic git commits (feat:, test:, fix:)

**Quality Checklist:**
- [ ] All tests pass
- [ ] Code follows reference patterns
- [ ] Type hints on all functions
- [ ] Docstrings on all public methods
- [ ] Error handling implemented
- [ ] Logging added

---

### Session 4 (90 min): Polish & Demo Preparation

**Learning Objective:** Complete features, ensure quality, prepare for demo

#### Part 1: Feature Completion Sprint (40 min)

**Focus:**
1. Finish any incomplete features
2. Add error handling
3. Improve user experience
4. Hit 80%+ coverage target

**Check Quality:**
```bash
# Run all tests
pytest -v

# Check coverage
pytest --cov=src --cov-report=html
open htmlcov/index.html  # See what's missing

# Check code quality
pylint src/ tests/

# Run actual application
python -m src.main [your args]
```

#### Part 2: Documentation & Polish (30 min)

**Do This:**
1. Write/update README with:
   - What it does
   - How to install
   - How to use
   - Example output
2. Add usage examples
3. Create demo script
4. Test on fresh clone

**README Template:**
```markdown
# My AI Project

## What It Does
[One paragraph explanation]

## Quick Start
[Installation steps]

## Usage
[Example commands]

## Demo
[Screenshot or sample output]

## Technical Details
[Architecture, APIs used, testing approach]
```

#### Part 3: Demo Preparation (20 min)

**Prepare Your 5-Minute Demo:**

**Structure:**
```
0:00-0:30  Problem statement
0:30-2:30  Live demo (show it working!)
2:30-4:00  Technical deep-dive (pick ONE interesting challenge)
4:00-4:30  TDD showcase (show a test)
4:30-5:00  Q&A
```

**Practice:**
1. Time yourself
2. Have backup plan if live demo fails
3. Prepare for questions:
   - "Why did you choose X over Y?"
   - "How did you test this?"
   - "What was hardest?"

#### Homework (2-3 hours)

**Final Polish:**
- [ ] Fix any remaining bugs
- [ ] Achieve 80%+ coverage
- [ ] Update documentation
- [ ] Practice demo 3 times
- [ ] Prepare answers to likely questions
- [ ] Have backup screenshots in case demo fails

---

## Week 3: Demo Day

### Presentation Day

**Arrival:**
- Come 10 minutes early
- Have your project running
- Test on presentation machine if possible

**Your 5-Minute Slot:**
| Time | Section | Tips |
|------|---------|------|
| 0:00-0:30 | **Problem** | Start with "Imagine you need to..." |
| 0:30-2:30 | **Demo** | Actually run it live, show real output |
| 2:30-4:00 | **Technical** | Explain ONE decision: "I chose X because..." |
| 4:00-4:30 | **Testing** | Show a test, explain how it proves correctness |
| 4:30-5:00 | **Q&A** | Listen carefully, it's ok to say "I don't know" |

**What Judges Look For:**
1. ✅ Does it work? (live demo)
2. ✅ Is it tested? (show coverage)
3. ✅ Can you explain decisions? (technical knowledge)
4. ✅ Did you use TDD? (process matters)
5. ✅ Is it creative/useful? (project quality)

**After Your Demo:**
- Watch other presentations
- Take notes on good ideas
- Provide constructive feedback to peers

---

## Timeline Summary

| Week | Focus | Deliverables |
|------|-------|--------------|
| **Week 1** | Learn patterns | Architecture understanding, project chosen |
| **Week 2** | Build project | Working application, >80% coverage |
| **Week 3** | Present | 5-minute demo, Q&A |

---

## Tips for Success

### Time Management
- **Don't build too much** - 5-7 features is plenty
- **Start simple** - You can always add more
- **Leave time for testing** - Tests take longer than you think
- **Polish matters** - Better to have 3 great features than 10 broken ones

### Technical Strategy
- **Copy patterns** - Use reference code as template
- **Test first** - RED-GREEN-REFACTOR always
- **Commit often** - Small, atomic commits
- **Ask for help early** - Don't struggle for hours alone

### Demo Preparation
- **Practice timing** - 5 minutes goes fast
- **Have backup plan** - Screenshots if live demo fails
- **Focus on one thing** - Don't try to show everything
- **Tell a story** - Problem → Solution → How you built it

---

## Common Pitfalls to Avoid

❌ **"I'll add tests later"** → You won't. Tests first!  
❌ **"I'll build everything"** → Scope creep kills projects  
❌ **"I'll figure out git later"** → Commit from day one  
❌ **"I don't need to practice demo"** → Yes you do!  
❌ **"It works on my machine"** → Test on fresh clone  

---

## Resources You'll Need

### Every Session
- Laptop with working setup
- Notebook for sketching
- [Code as Textbook](CODE_AS_TEXTBOOK.md) open
- Reference code accessible

### Week 1
- [TDD Workflow](TDD_WORKFLOW.md)
- [AI Collaboration](AI_COLLABORATION.md)
- [Git Workflow](GIT_WORKFLOW.md)

### Week 2
- [OpenAI API Reference](openai_tools_research_oct2025.md)
- [Project Ideas](PROJECT_IDEAS.md) for inspiration
- [Logging Guide](LOGGING.md)

### Demo Day
- Working laptop
- Backup slides/screenshots
- Demo script
- 5-minute timer

---

## Questions?

**About the course structure:** Ask your instructor

**About your project:** Post in class discussion or ask during office hours

**About specific code:** Check the reference implementation or [Code as Textbook](CODE_AS_TEXTBOOK.md)

**About OpenAI APIs:** See [comprehensive API guide](openai_tools_research_oct2025.md)

---

**[← Back to README](../README.md)** • **[Getting Started →](GETTING_STARTED.md)** • **[View Grading Rubric →](GRADING.md)**
