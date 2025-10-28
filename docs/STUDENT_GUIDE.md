# 🎓 Student Guide - Your 2-Week Roadmap

**⏱️ 5 min read | Everything you need to succeed**

---

## 🗓️ The Timeline

**Week 1: Foundation & Start**
- Session 1: Setup + Pick Project
- Session 2: Design + First Tests

**Week 2: Build & Polish**
- Session 3: Keep Building (TDD)
- Session 4: Polish + Prep Demo

**Week 3: Demo Presentations** 🎤
_Students present their projects. Work quietly on polish/fixes while waiting for your demo slot._

---

## 📋 Session 1: Setup & Choose Project (90 min)

### Before Class:
- [ ] Read [README.md](../README.md)
- [ ] Read [AI_COLLABORATION.md](AI_COLLABORATION.md)
- [ ] Skim [OpenAI Tools Research](openai_tools_research_oct2025.md) - Latest APIs (October 2025)
- [ ] Clone the repo

### In Class:

**Part 1: Setup (20 min)**
```bash
# Follow README Quick Start
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add your OPENAI_API_KEY to .env
pytest  # Should see 69 tests pass
```

**Part 2: Explore the Example (30 min)**
- Run: `python -m src.main "AI news"`
- Look at `src/models.py` - see the data structures
- Look at `tests/test_models.py` - see the tests
- Notice: tests are simple, focused, one thing at a time

**Part 3: Pick Your Project (40 min)**
- Read [PROJECT_IDEAS.md](PROJECT_IDEAS.md)
- Review [OpenAI Tools Research](openai_tools_research_oct2025.md) for your chosen API
- Research 2-3 OpenAI tools that interest you (NEW in Oct 2025: GPT-5 Pro, Sora 2, AgentKit!)
- Brainstorm 3 project ideas
- Discuss with instructor
- **Decision:** Write down your chosen project

**Homework:**
- [ ] Research your chosen OpenAI tool (read official docs)
- [ ] Read [TDD_WORKFLOW.md](TDD_WORKFLOW.md)
- [ ] Think about your project architecture

---

## 📋 Session 2: Design & First Tests (90 min)

### Before Class:
- [ ] Read [TDD_WORKFLOW.md](TDD_WORKFLOW.md)
- [ ] Read [GIT_WORKFLOW.md](GIT_WORKFLOW.md)
- [ ] Have project idea ready

### In Class:

**Part 1: Architecture with Claude (30 min)**
```
"Hey Claude! I want to build [your project description].

I'll use OpenAI's [tool name] API.

Help me:
1. Break this into testable components
2. Define the data models I need
3. Suggest what to build first

I want to use TDD like the enterprise_ai_demo1_websearch example."
```

**Part 2: Setup Your Repo (20 min)**
```bash
# Create new repo on GitHub
# Clone it locally
cd your-project
python -m venv venv
source venv/bin/activate

# Copy structure from demo
cp ../enterprise_ai_demo1_websearch/pytest.ini .
cp ../enterprise_ai_demo1_websearch/.pylintrc .
cp ../enterprise_ai_demo1_websearch/.coveragerc .
cp ../enterprise_ai_demo1_websearch/requirements.txt .
pip install -r requirements.txt

# Create structure
mkdir src tests
touch src/__init__.py tests/__init__.py
```

**Part 3: Write First Test (40 min)**

Follow TDD:
1. Write test for your first model
2. Run it (should FAIL)
3. Write code to pass it
4. Run it (should PASS)
5. Commit with: `test: Add tests for [model] model`
6. Commit code: `feat: Implement [model] model`

**Homework:**
- [ ] Create 2-3 more models with tests
- [ ] Keep using TDD (test first!)
- [ ] Make atomic commits

---

## 📋 Session 3: Keep Building (90 min)

### Before Class:
- [ ] Have at least 3 models with tests
- [ ] Read [LOGGING.md](LOGGING.md)
- [ ] All tests passing

### In Class:

**Part 1: Progress Check (15 min)**
- Show your models to instructor
- Run: `pytest --cov=src`
- Check coverage (aim for 80%+)

**Part 2: Build API Client (45 min)**
- Look at `src/client.py` in demo for patterns
- Write tests first for your API client
- Implement to pass tests
- Handle errors (see demo's error handling)
- Test with real API

Ask Claude:
```
"I need to call OpenAI's [API name]. 
Here's what I want to do: [describe]

Show me:
1. Tests for the API client
2. Implementation with error handling
3. How to mock the API in tests

Use the pattern from enterprise_ai_demo1_websearch."
```

**Part 3: Add Logging (30 min)**
- Copy `src/logging_config.py` from demo
- Add logging to your main file
- Test: check `logs/app.log` file is created

**Homework:**
- [ ] Build parser/service layer
- [ ] Keep coverage above 80%
- [ ] Make conventional commits

---

## 📋 Session 4: Polish & Prep (90 min)

### Before Class:
- [ ] Core functionality working
- [ ] Tests passing
- [ ] Coverage 80%+

### In Class:

**Part 1: Feature Complete (30 min)**
- Finish last features
- Fix any bugs
- Get coverage to 90%+

**Part 2: Documentation (30 min)**

Create README.md:
```markdown
# Your Project Name

Brief description

## Setup
[Installation steps]

## Usage
[How to use it]

## Features
- Feature 1
- Feature 2

## Architecture
[Brief overview]

## Demo
[Screenshot or example output]
```

**Part 3: Prepare Demo (30 min)**

Prepare 5-minute presentation:
1. **Intro (30 sec):** "I built [X] using [Y] API"
2. **Demo (2 min):** Show it working live
3. **Code (1 min):** Show one interesting test or feature
4. **Challenge (1 min):** "The hardest part was [X]. I solved it by [Y]"
5. **Questions (30 sec)**

Practice with a classmate!

**Homework:**
- [ ] Practice demo
- [ ] Test your demo flow
- [ ] Be ready to present!

---

## 🎤 Demo Day Checklist

### Before Demo:
- [ ] Code pushed to GitHub
- [ ] All tests passing
- [ ] README complete
- [ ] Demo script ready
- [ ] API key set up
- [ ] Internet connection works

### During Demo:
1. Stay calm (everyone is rooting for you!)
2. If something breaks, explain what should happen
3. Show enthusiasm for your project
4. Take questions confidently

### After Demo:
- Celebrate! You built something real! 🎉

---

## 🆘 Common Problems & Solutions

### "I'm stuck on [X]"
**Solution:** Ask Claude:
```
"I'm trying to [goal]. Here's my code: [paste]
Here's the error: [paste error]
What's wrong and how do I fix it?"
```

### "My tests are failing"
**Solution:**
```bash
pytest -v  # See which test is failing
pytest tests/test_specific.py -v  # Run one file
pytest tests/test_specific.py::test_name -v  # Run one test
```

### "Coverage is low"
**Solution:**
```bash
pytest --cov=src --cov-report=html
open htmlcov/index.html  # See what's not covered
# Write tests for red lines
```

### "Git is confusing"
**Solution:** Follow the pattern:
```bash
git status  # See what changed
git add file.py  # Stage one file
git commit -m "feat: Add feature X"  # Commit
git push  # Push to GitHub
```

### "I'm behind schedule"
**Solution:**
- Cut features if needed (better to demo something working)
- Focus on core functionality
- Ask for help early
- Pair with a classmate

---

## 💡 Pro Tips

1. **Commit often** - Every 15-30 minutes when something works
2. **Run tests constantly** - After every small change
3. **Ask Claude for help** - But understand what it suggests
4. **Don't copy-paste blindly** - Read and understand the code
5. **Start simple** - Build the simplest version first
6. **Take breaks** - Stepping away helps solve problems
7. **Help classmates** - Teaching others helps you learn

---

## 📊 Self-Assessment

After each session, check:

**Session 1:**
- [ ] Environment works
- [ ] Project chosen
- [ ] Architecture planned

**Session 2:**
- [ ] First models with tests
- [ ] TDD working
- [ ] Git commits looking good

**Session 3:**
- [ ] API client working
- [ ] 50%+ coverage
- [ ] Logging added

**Session 4:**
- [ ] Features complete
- [ ] 80%+ coverage
- [ ] Demo ready

---

## 🎯 Learning Outcomes

By the end, you'll be able to say:
- ✅ "I built a production-quality application using TDD"
- ✅ "I collaborated with AI to design and implement features"
- ✅ "I wrote comprehensive tests with 90%+ coverage"
- ✅ "I used enterprise logging and error handling patterns"
- ✅ "I followed professional git workflows"
- ✅ "I can explain technical challenges I solved"

**These skills are what employers want!**

---

## 🚀 After the Course

Want to keep building?

**Portfolio:**
- Add to your resume
- Link in LinkedIn
- Show in interviews

**Keep Learning:**
- Try other OpenAI tools
- Build more complex projects
- Contribute to open source

**Share:**
- Write a blog post about what you built
- Share on Twitter/LinkedIn
- Help future students

---

**You've got this! See you on Demo Day!** 🎉

---

*Questions during the project? Read the other guides in `docs/` or ask Claude!*
