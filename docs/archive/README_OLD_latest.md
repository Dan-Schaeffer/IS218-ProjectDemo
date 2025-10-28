# 🚀 OpenAI Web Search Demo

**Build enterprise-quality AI applications using TDD + Claude**

[![Tests](https://img.shields.io/badge/tests-69%20passing-brightgreen)]()
[![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)]()

---

## � About This Assignment

This is a **2-week project-based learning experience** where you'll build a creative application using OpenAI's APIs while learning professional software development practices. This repository serves as your **reference implementation**—a real-world example of how enterprise development teams write production-quality code. You'll learn to collaborate with AI tools (like Claude) not as a crutch, but as a **creative partner** in the development process, using Test-Driven Development (TDD), professional git workflows, enterprise logging, and automated testing.

By the end, you won't just have a working project—you'll have learned the **vocabulary, patterns, and practices** that professional development teams use daily. You'll understand how to leverage AI to accelerate development while maintaining high code quality standards. Most importantly, you'll gain confidence in your ability to build, test, and demonstrate real software that solves real problems.

---

## �🎯 Your Mission (2 Weeks)

**Pick an OpenAI tool → Build something creative → Present on Demo Day**

### 📖 OpenAI Tools Available (October 2025)
See [**OpenAI Tools Research**](docs/openai_tools_research_oct2025.md) for comprehensive details on:
- **GPT-5 Pro** - Advanced reasoning for complex tasks
- **Chat Completion** - Conversational AI and content generation
- **Vision API** - Image analysis and understanding
- **DALL-E 3** - Advanced image generation
- **Whisper** - Speech-to-text transcription
- **TTS** - Text-to-speech with 6 voices
- **Embeddings** - Semantic search and recommendations
- **Assistants API** - Build persistent AI agents
- **Sora 2** - Video generation with audio (NEW!)
- **AgentKit** - Multi-agent workflow builder (NEW!)

You'll learn:
- ✅ Test-Driven Development (TDD)
- ✅ Working with AI coding assistants
- ✅ Enterprise logging & code quality
- ✅ Professional git workflows
- ✅ CI/CD with GitHub Actions

---

## 🔍 What This Web Search Tool Does

This repository contains a **production-quality web search application** built using OpenAI's latest APIs. When you run it, the tool takes your search query, sends it to OpenAI's search-enhanced models, and returns intelligent, sourced results with citations—think of it as a smarter web search that understands context and provides AI-generated summaries.

**Example:**
```bash
$ python -m src.main "latest developments in quantum computing"

# Returns: AI-generated summary with citations from recent sources
# Shows: Web sources, structured analysis, and referenced URLs
```

**What makes this "enterprise-quality"?**
- 🧪 **100% test coverage** - Every feature is tested
- 📊 **Enterprise logging** - Production-ready monitoring with log rotation
- 🔄 **CI/CD pipeline** - Automated testing on every commit
- 🎯 **Clean architecture** - Separation of concerns (models, client, parser, service)
- ✅ **Error handling** - Graceful failures with informative messages
- 📝 **Type hints & documentation** - Self-documenting code

We built this tool using the **same techniques you'll use** for your project. In fact, we even used this very tool to research OpenAI's APIs and create the [comprehensive OpenAI Tools Research document](docs/openai_tools_research_oct2025.md) that you'll reference throughout your project. This demonstrates the power of building practical tools with AI—you can use what you create to accelerate your own learning!

**Your turn:** Pick a different OpenAI API or combination of APIs and build something equally impressive.

---

## ⚡ Quick Start (5 minutes)

```bash
# 1. Clone and setup
git clone https://github.com/kaw393939/enterprise_ai_demo1_websearch.git
cd enterprise_ai_demo1_websearch
python -m venv venv
source venv/bin/activate  # Mac/Linux
# or: venv\Scripts\activate  # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 4. Test it works
pytest
python -m src.main "AI news today"
```

**✅ Working? Great! Now read the guides below.**

---

## � Code as Textbook: Learn by Reading

**This codebase is designed as a living textbook.** Every file is heavily commented with explanations, examples, and design decisions. Instead of just showing you WHAT the code does, the comments explain WHY and teach you the underlying concepts.

### 🎓 Start Here: [Code as Textbook Guide](docs/CODE_AS_TEXTBOOK.md)

This guide provides:
- **📚 Chapter-by-chapter reading order** (each source file is a chapter)
- **🎯 Multiple learning paths** based on your experience level
- **💡 Concept explanations** woven into the code comments
- **🔍 Cross-references** between related concepts
- **📖 Story-based narrative** that makes code feel like a book

**Reading Paths Available:**
1. **"I'm New to Python"** - Learn Python through a real project
2. **"Teach Me Architecture"** - Understand application structure
3. **"Master Testing"** - Deep dive into TDD
4. **"Full Story"** - Read cover-to-cover like a textbook

**Example of what you'll find in the code:**
```python
# From src/models.py:
"""
📖 CHAPTER 1: THE FOUNDATION - DATA MODELS

STORY: Building Blocks of Our Application
------------------------------------------
Imagine you're building a house. Before you start construction, 
you need blueprints. In software, these blueprints are called 
"data models"...

LEARNING OBJECTIVES:
✓ Understand Python dataclasses
✓ Learn type hints for better documentation
✓ Master properties (computed attributes)
...
"""
```

Each file teaches you:
- **What** the code does (functionality)
- **Why** it's written this way (design decisions)  
- **How** to use it (examples)
- **Alternatives** we didn't choose (and why)

💡 **This is unique:** Most codebases have minimal comments. We've turned ours into a narrative textbook where each chapter builds on the last, teaching you professional software engineering through storytelling.

---

## �📚 Learning Path (Read in Order)

### 🎓 Week 1: Foundation

**Day 1 - Read These and Get Started (40 min total):**
1. [STUDENT_GUIDE.md](docs/STUDENT_GUIDE.md) - Your 2-week roadmap (10 min)
2. [AI_COLLABORATION.md](docs/AI_COLLABORATION.md) - How to work with Claude (10 min)
3. [OpenAI Tools Research](docs/openai_tools_research_oct2025.md) - All available APIs (10 min skim)
4. [PROJECT_IDEAS.md](docs/PROJECT_IDEAS.md) - Pick your project (10 min)

**Day 2 - Development Practices (30 min total):**
5. [TDD_WORKFLOW.md](docs/TDD_WORKFLOW.md) - Test-driven development (15 min)
6. [GIT_WORKFLOW.md](docs/GIT_WORKFLOW.md) - Professional commits (10 min)
7. [LOGGING.md](docs/LOGGING.md) - Enterprise logging (5 min)

### 🛠️ Week 2: Build & Polish

**Day 3-4:**
- Keep building with TDD
- Reference guides as needed
- Ask Claude for help
- Make atomic git commits

**Day 5:**
- Polish features
- Write documentation
- Prepare 5-minute demo

### 🎤 Week 3: Demo Day

Present your project:
- What you built (2 min)
- Live demo (2 min)
- One technical challenge you solved (1 min)

---

## 🏗️ What This Repo Shows You

This is a **reference implementation** of a production-quality OpenAI web search app.

**Structure:**
```
src/
├── models.py           # Data models (what TDD looks like)
├── client.py           # API client (error handling examples)
├── parser.py           # Response parsing (clean code)
├── search_service.py   # Business logic (architecture)
├── main.py             # CLI interface (logging examples)
└── logging_config.py   # Enterprise logging setup

tests/
├── test_*.py           # 69 tests showing TDD patterns
└── conftest.py         # Shared test fixtures

.github/workflows/
└── ci.yml              # Automated testing (CI/CD)
```

**Key Features to Study:**
- ✅ 100% test coverage (with smart `pragma: no cover`)
- ✅ Structured logging with rotation
- ✅ Clean git history with conventional commits
- ✅ Professional error handling
- ✅ Type hints and docstrings
- ✅ Automated CI/CD pipeline

---

## 🎨 Your Project Should Have

**Minimum (C grade):**
- Works (no crashes)
- Uses 1-2 OpenAI tools
- Has some tests (>50% coverage)
- Basic README

**Good (B grade):**
- Uses TDD (tests first)
- 80%+ test coverage
- Conventional git commits
- Good documentation
- Handles errors

**Excellent (A grade):**
- Full TDD workflow
- 90%+ test coverage
- Enterprise logging
- Conventional commits
- Great documentation
- Creative/useful project
- Polished demo

---

## 🆘 Getting Help

### During Class:
- Ask your instructor
- Pair with classmates
- Use Claude in VS Code

### Outside Class:
- Read the guides in `docs/`
- Check the reference code in `src/`
- Look at tests in `tests/`
- Ask Claude: "I'm trying to [goal]. Here's my code: [paste]. What's wrong?"

### Common Issues:
```bash
# Tests failing?
pytest -v  # See details

# Import errors?
source venv/bin/activate  # Activate venv first

# Coverage not 100%?
pytest --cov=src --cov-report=term-missing  # See what's missing

# Git confused?
git status  # See what's going on
```

---

## 📊 Grading Rubric

| Criteria | Points | What We're Looking For |
|----------|--------|------------------------|
| **Functionality** | 25 | Works without crashes, uses OpenAI API correctly |
| **Testing** | 25 | TDD approach, 80%+ coverage, meaningful tests |
| **Code Quality** | 20 | Clean code, logging, error handling, type hints |
| **Git Workflow** | 10 | Atomic commits, conventional format, clear history |
| **Documentation** | 10 | README, setup instructions, usage examples |
| **Demo** | 10 | Clear presentation, live demo, explains challenge |

**Total: 100 points**

---

## 🔗 Useful Links

### OpenAI Resources
- [**OpenAI Tools Research (Oct 2025)**](docs/openai_tools_research_oct2025.md) - Comprehensive guide to all APIs
- [OpenAI Platform Docs](https://platform.openai.com/docs) - Official documentation
- [OpenAI Python Library](https://github.com/openai/openai-python) - Python SDK
- [OpenAI Cookbook](https://cookbook.openai.com/) - Code examples

### Development Tools
- [pytest Documentation](https://docs.pytest.org/) - Testing framework
- [Conventional Commits](https://www.conventionalcommits.org/) - Commit standards
- [Python Type Hints](https://docs.python.org/3/library/typing.html) - Type annotations

---

## 💬 Questions?

1. Read the relevant guide in `docs/`
2. Check the example code in `src/`
3. Ask Claude for help
4. Ask your instructor during class

---

## 🎉 Example Projects from Students

*After Demo Day, we'll add the best projects here!*

---

**Built with TDD + Claude | Your turn to build something awesome!** 🚀
