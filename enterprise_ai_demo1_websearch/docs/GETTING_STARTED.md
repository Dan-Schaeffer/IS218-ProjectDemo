# 🚀 Getting Started Guide

**Goal:** Get the application running on your machine in 5-10 minutes.

---

## Prerequisites

- **Python 3.11+** installed ([Download](https://python.org))
- **Git** installed ([Download](https://git-scm.com))
- **OpenAI API Key** ([Get one](https://platform.openai.com/api-keys))
- **Text editor** (VS Code recommended)

---

## Step 1: Clone the Repository (1 min)

```bash
# Clone
git clone https://github.com/kaw393939/enterprise_ai_demo1_websearch.git
cd enterprise_ai_demo1_websearch

# Or if you prefer SSH:
git clone git@github.com:kaw393939/enterprise_ai_demo1_websearch.git
cd enterprise_ai_demo1_websearch
```

---

## Step 2: Create Virtual Environment (2 min)

```bash
# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate          # Mac/Linux
# OR
venv\Scripts\activate             # Windows

# You should see (venv) in your terminal prompt
```

**Why a virtual environment?**
- Isolates project dependencies
- Prevents conflicts with other Python projects
- Standard professional practice

---

## Step 3: Install Dependencies (1-2 min)

```bash
pip install -r requirements.txt
```

**What gets installed:**
- `openai` - Official OpenAI Python library
- `python-dotenv` - Environment variable management
- `pytest` - Testing framework
- `pytest-cov` - Coverage reporting
- `pytest-mock` - Mocking utilities
- `pylint` - Code quality checker

---

## Step 4: Configure API Key (2 min)

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your favorite editor
nano .env      # or vim, code, etc.
```

**Add your OpenAI API key:**
```bash
OPENAI_API_KEY=sk-your-actual-key-here
```

**Important:**
- Replace `sk-your-actual-key-here` with your real key
- Keys start with `sk-`
- Never commit `.env` to git (it's already in `.gitignore`)
- Keep your key secret!

**Don't have a key yet?**
1. Go to [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. Sign in or create account
3. Click "Create new secret key"
4. Copy and save it (you won't see it again!)

---

## Step 5: Verify Installation (2 min)

### Run Tests

```bash
pytest
```

**Expected output:**
```
====== test session starts ======
...
69 passed in 2.5s
====== coverage: 100% ======
```

✅ **All tests pass?** Great! Your environment is set up correctly.

❌ **Tests fail?** See [Troubleshooting](#troubleshooting) below.

### Run the Application

```bash
python -m src.main "latest developments in AI"
```

**Expected output:**
```
🔍 Searching: latest developments in AI

📝 AI-Generated Summary:
Recent developments in artificial intelligence include...
[... AI-generated response ...]

📚 Sources Consulted:
[1] Title (url)
[2] Title (url)
...
```

✅ **Got results?** Perfect! You're ready to start learning.

❌ **Error?** See [Troubleshooting](#troubleshooting) below.

---

## What You Just Did

🎉 **Congratulations!** You've:

1. ✅ Cloned a professional Python project
2. ✅ Set up a virtual environment (industry standard)
3. ✅ Installed dependencies from requirements.txt
4. ✅ Configured environment variables securely
5. ✅ Run a test suite (69 tests!)
6. ✅ Executed a production-quality application

This is **exactly** how professional development teams start projects.

---

## Next Steps

### Path 1: Understand the Code (Recommended)
� First, visit the **[Learning Path Map](LEARNING_PATH.md)** to see how onboarding, labs, and the project studio interlock.
�📖 **[Read: Code as Textbook Guide](CODE_AS_TEXTBOOK.md)**
- Learn how to read code with narrative comments
- Choose your learning path (beginner/intermediate/advanced)
- Understand the chapter structure

### Path 2: See the Full Course
📅 **[Read: Course Structure](COURSE_STRUCTURE.md)**
- 2-week session-by-session breakdown
- What you'll do each class
- Homework assignments
- Demo day preparation

### Path 3: Start Building
🛠️ **[Read: Project Ideas](PROJECT_IDEAS.md)**
- Browse 60+ project ideas
- Pick something that excites you
- See difficulty ratings and time estimates

---

## Troubleshooting

### Tests Fail: "ModuleNotFoundError: No module named 'openai'"

**Problem:** Virtual environment not activated or dependencies not installed.

**Solution:**
```bash
# Make sure venv is activated (you should see (venv) in prompt)
source venv/bin/activate    # Mac/Linux
venv\Scripts\activate       # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

### Tests Fail: "401 Unauthorized" or "Invalid API key"

**Problem:** API key not set or incorrect.

**Solution:**
```bash
# Check .env file exists
ls -la .env

# Verify it has your key
cat .env

# Make sure it looks like:
# OPENAI_API_KEY=sk-your-actual-key

# Verify key format (should start with sk- and be ~48 characters)
```

### Application Runs but Gets Error: "No API key found"

**Problem:** .env file not loaded or in wrong location.

**Solution:**
```bash
# Make sure .env is in project root
pwd  # Check you're in project directory
ls .env  # Should show the file

# Verify .env content
cat .env

# Try running from project root
cd /path/to/enterprise_ai_demo1_websearch
python -m src.main "test query"
```

### Import Errors: "ImportError: attempted relative import with no known parent package"

**Problem:** Running files directly instead of as modules.

**Solution:**
```bash
# ❌ Don't do this:
python src/main.py

# ✅ Do this:
python -m src.main
```

### Virtual Environment Won't Activate (Windows)

**Problem:** PowerShell execution policy.

**Solution:**
```powershell
# Run PowerShell as Administrator, then:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Or use Command Prompt instead:
venv\Scripts\activate.bat
```

### Coverage Shows Less Than 100%

**Problem:** This is normal! We exclude some files.

**Solution:**
```bash
# Check .coveragerc to see what's excluded
cat .coveragerc

# Run coverage with report to see details
pytest --cov=src --cov-report=term-missing
```

### Python Version Error

**Problem:** Need Python 3.11 or higher.

**Solution:**
```bash
# Check your version
python --version

# If too old, install newer version:
# - Mac: brew install python@3.11
# - Windows: Download from python.org
# - Linux: Use package manager (apt, yum, etc.)

# Create venv with specific version
python3.11 -m venv venv
```

---

## Common Questions

### Do I need to pay for OpenAI API?

**Yes, but:** OpenAI gives new users $5 free credit. For this course:
- Most projects cost $0.10 - $2.00 total
- Use `gpt-4o-mini` (cheapest model) while learning
- Monitor usage at [platform.openai.com/usage](https://platform.openai.com/usage)

### Can I use a different AI API?

**For learning purposes:** No, the reference code uses OpenAI specifically.

**For your project:** Yes! But you'll need to adapt patterns. Ask instructor.

### What IDE should I use?

**Recommended:** VS Code with these extensions:
- Python
- Pylance
- GitHub Copilot (optional, but helpful)

**Also good:** PyCharm, Sublime Text, Vim/Neovim

### Should I use git for my project?

**Absolutely yes!** Professional development requires version control.

```bash
# For your project
mkdir my-ai-project
cd my-ai-project
git init
# Copy structure from reference repo
```

See [Git Workflow Guide](GIT_WORKFLOW.md) for best practices.

---

## Quick Command Reference

### Every Day Commands

```bash
# Activate environment (do this first, every time!)
source venv/bin/activate

# Run tests
pytest

# Run with coverage
pytest --cov=src

# Run application
python -m src.main "your query"

# Check code quality
pylint src/ tests/

# Git workflow
git status
git add .
git commit -m "feat: add feature name"
git push
```

### When Things Go Wrong

```bash
# See detailed test output
pytest -v

# See which lines aren't covered
pytest --cov=src --cov-report=term-missing

# Run specific test file
pytest tests/test_client.py

# Run specific test
pytest tests/test_client.py::TestWebSearchClient::test_search_with_basic_query

# Debug mode (drops into debugger on error)
pytest --pdb
```

---

## You're Ready!

✅ Environment is working  
✅ Tests pass  
✅ Application runs  
✅ You know how to troubleshoot  

### Choose Your Next Step:

**📖 [Read Code as Textbook](CODE_AS_TEXTBOOK.md)** - Learn how this code teaches

**📅 [See Course Structure](COURSE_STRUCTURE.md)** - View 2-week plan

**🛠️ [Browse Project Ideas](PROJECT_IDEAS.md)** - Find your project

**📚 [Back to README](../README.md)** - Main page

---

**Still stuck?** Ask your instructor or check with classmates. Everyone hits setup issues—it's part of learning!
