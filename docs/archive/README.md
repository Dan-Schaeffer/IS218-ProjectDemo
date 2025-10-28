# 🚀 OpenAI Web Search Demo

**Production-quality Python CLI built with TDD | Perfect learning example for students**

```bash
# Quick start (with virtual environment)
python -m venv venv
source venv/bin/activate  # Mac/Linux (or venv\Scripts\activate on Windows)
pip install -r requirements.txt
echo "OPENAI_API_KEY=your-key" > .env
pytest  # All tests pass! (no -m needed)
python -m src.main "AI news today"
```

---

## 📚 Documentation (Read In Order)

### 🎯 For Students (Start Here!)

1. **[START_HERE.md](START_HERE.md)** - 5 min overview
2. **[HOW_TO_WORK_WITH_AI.md](HOW_TO_WORK_WITH_AI.md)** - 10 min on AI collaboration
3. **[QUICK_TDD_GUIDE.md](QUICK_TDD_GUIDE.md)** - 15 min on testing
4. **[GIT_GUIDE.md](GIT_GUIDE.md)** - 5 min on professional commits
5. **[LOGGING_GUIDE.md](LOGGING_GUIDE.md)** - 10 min on enterprise logging
6. **[PROJECT_IDEAS.md](PROJECT_IDEAS.md)** - 5 min for inspiration

**Total reading time: 50 minutes** ⏱️

### 🔧 For Quick Reference

- **[QUICKSTART.md](QUICKSTART.md)** - Installation & usage commands
- **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** - Technical overview
- **[docs/architecture.md](docs/architecture.md)** - System design

### 📖 Detailed Guides (If You Want Deep Dive)

- **[docs/archive/TUTORIAL.md](docs/archive/TUTORIAL.md)** - 28KB complete guide
- **[docs/archive/AI_PROMPTS.md](docs/archive/AI_PROMPTS.md)** - All prompts used
- **[docs/archive/DOCS_INDEX.md](docs/archive/DOCS_INDEX.md)** - Full doc navigation

---

## ⚡ What This Project Demonstrates

✅ **Test-Driven Development** (69 tests, 90% coverage)  
✅ **Clean Architecture** (Models → Client → Parser → Service → CLI)  
✅ **Real OpenAI Integration** (Web Search API)  
✅ **Enterprise Patterns** (Error handling, fixtures, mocking)  
✅ **Professional Documentation** (You're reading it!)

---

## 🎯 Use This Project To:

1. **Learn TDD** - See how tests drive development
2. **Understand Architecture** - Study the layered design
3. **Build Your Own** - Use as template for your project
4. **Work With AI** - Follow our process with Claude/Copilot

---

## 📁 Project Structure

```
src/
├── models.py         # Data models
├── client.py         # OpenAI API client
├── parser.py         # Response parser
├── search_service.py # Business logic
└── main.py           # CLI interface

tests/
├── conftest.py       # Shared fixtures
├── test_models.py    # 16 tests
├── test_client.py    # 13 tests
├── test_parser.py    # 13 tests
├── test_service.py   # 14 tests
└── test_main.py      # 13 tests
```

---

## 🎓 For Instructors

This project demonstrates:
- TDD methodology (RED-GREEN-REFACTOR)
- Clean code principles
- API integration patterns
- Comprehensive testing strategies
- Professional documentation practices

Students learn to collaborate with AI while understanding every line of code they write.

---

## 🆘 Quick Help

**First time setup:**
```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
pip install -r requirements.txt
```

**Tests failing?**
```bash
pytest -v  # See which tests fail (no -m needed!)
```

**Need to understand the code?**
```bash
# Read the source files in order:
cat src/models.py      # 1. Data structures
cat src/parser.py      # 2. Response parsing
cat src/client.py      # 3. API calls
cat src/search_service.py  # 4. Business logic
cat src/main.py        # 5. CLI interface
```

**Want to build your own?**
1. Read [START_HERE.md](START_HERE.md)
2. Pick idea from [PROJECT_IDEAS.md](PROJECT_IDEAS.md)
3. Ask Claude to help design it

---

## 📊 Stats

- **69 tests** (all passing ✅)
- **90.25% coverage** (exceeds 90% requirement)
- **318 lines** of source code
- **500+ lines** of test code
- **Built in ~6 hours** using TDD + AI

---

## 🔗 Links

- [OpenAI Platform Docs](https://platform.openai.com/docs)
- [OpenAI Python Library](https://github.com/openai/openai-python)
- [pytest Documentation](https://docs.pytest.org/)

---

## 💬 Questions?

**For Students:** Start with [START_HERE.md](START_HERE.md) then ask Claude!

**For Instructors:** See [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) for technical details

---

**Built with ❤️ using TDD + Claude | Perfect example of human-AI collaboration**
