# 🎓 Instructor Guide: 2-Week AI Project Sprint

**Course:** Enterprise AI Development  
**Duration:** 2 weeks (4 sessions) + Demo Day  
**Format:** 90-minute sessions, twice per week  

---

## 📋 Assignment Overview

### Learning Objectives

By the end of this assignment, students will be able to:

1. **Collaborate with AI** to design and build software
2. **Apply Test-Driven Development** (TDD) methodology
3. **Use enterprise patterns** (logging, error handling, architecture)
4. **Follow professional Git workflows** (atomic commits, conventional commits)
5. **Integrate OpenAI APIs** into applications
6. **Present technical work** to an audience

### Deliverables

Students must submit:
1. ✅ GitHub repository with clean commit history
2. ✅ Working application using OpenAI tool(s)
3. ✅ 100% test coverage on core logic
4. ✅ Comprehensive README with setup instructions
5. ✅ 5-minute demo presentation

### Grading Rubric (100 points)

| Category | Points | Criteria |
|----------|--------|----------|
| **Functionality** | 25 | Application works, uses OpenAI API, has useful features |
| **Code Quality** | 25 | Clean code, follows patterns, well-structured |
| **Testing** | 20 | 100% coverage, meaningful tests, TDD approach |
| **Git/Documentation** | 15 | Atomic commits, conventional commits, clear README |
| **Logging** | 10 | Enterprise logging, appropriate levels, context |
| **Demo** | 5 | Clear presentation, answers questions |

---

## 📅 Session-by-Session Plan

### **Week 1, Session 1: Setup & Planning** (90 min)

**Your Goals:**
- Get everyone set up
- Help students pick feasible projects
- Ensure they understand TDD

**Session Outline:**

**[0-15 min] Introduction**
- Explain the assignment (deliverables, timeline, demo day)
- Show your demo of the web search example
- Emphasize: "You'll build YOUR OWN project, not copy mine"

**[15-50 min] Reading & Setup**
- Students work through:
  - START_HERE.md (5 min)
  - HOW_TO_WORK_WITH_AI.md (10 min)
  - QUICK_TDD_GUIDE.md (15 min)
  - GIT_GUIDE.md (5 min)
  - LOGGING_GUIDE.md (10 min)
- While they read, circulate and answer questions

**[50-75 min] Project Selection**
- Students browse PROJECT_IDEAS.md
- Each student picks 2-3 ideas they like
- **Walk around:** Ask each student about their ideas
  - "What interests you?"
  - "Which APIs will you use?"
  - "Is this doable in 2 weeks?"
- Help them narrow down to ONE project

**[75-90 min] Environment Setup**
- Clone the demo repo
- Create virtual environment
- Run tests
- Make their first search
- **Troubleshoot:** Help anyone stuck

**Homework:**
- Finalize project idea
- Research OpenAI tool(s) they'll use
- Create GitHub repo
- Make first commit: "chore: Initialize project"

---

### **Week 1, Session 2: Architecture & First Feature** (90 min)

**Your Goals:**
- Students have working code by end of class
- They understand TDD cycle
- They're using AI effectively

**Session Outline:**

**[0-10 min] Check-In**
- Quick status: "Who has their GitHub repo set up?"
- "Who researched their OpenAI tools?"
- Address any blockers

**[10-40 min] Architecture Design**
- Students work with Claude to design architecture
- Prompt: "Help me design [project] using [OpenAI tools]. I need: data models, components, testing strategy"
- **Walk around:** Review architectures
  - "Do you have clear components?"
  - "How will you test this?"
  - "Is this too complex for 2 weeks?"
- Students create architecture.md in their repo

**[40-70 min] First TDD Cycle**
- Pick simplest feature
- Write test FIRST
- Run test (should fail - RED)
- Write minimal code
- Run test (should pass - GREEN)
- Commit: "test: Add tests for [feature]"
- Commit: "feat: Implement [feature]"
- **Walk around:** Ensure they're doing RED-GREEN-REFACTOR
  - "Show me your failing test"
  - "Now make it pass"
  - "Great! Commit it!"

**[70-90 min] Second Feature (TDD)**
- Repeat TDD cycle for another feature
- Students should commit 2-4 times this session
- **Goal:** Students feel confident with TDD

**Homework:**
- Build 2-3 more features using TDD
- Aim for 50% of core functionality done
- Keep committing (aim for 10+ commits by next session)

---

### **Week 2, Session 3: Core Features & Quality** (90 min)

**Your Goals:**
- Students have most features working
- Coverage approaching 100%
- Logging added

**Session Outline:**

**[0-10 min] Progress Check**
- "Show me your app working!"
- Quick demos at their desks
- Celebrate progress!

**[10-50 min] Feature Completion**
- Students continue building features
- Focus on core functionality
- **Walk around:** 
  - "What's your MVP (Minimum Viable Product)?"
  - "What's essential vs. nice-to-have?"
  - "How's your test coverage?"

**[50-70 min] Add Enterprise Logging**
- Follow LOGGING_GUIDE.md
- Add logging to key functions
- Test that logs are being created
- Commit: "feat: Add enterprise logging"

**[70-90 min] Coverage Push**
- Run coverage report
- Identify uncovered lines
- Add tests or pragma: no cover
- **Goal:** 90%+ coverage by end of class

**Homework:**
- Polish features
- Get to 100% coverage
- Write README.md
- Practice explaining your project

---

### **Week 2, Session 4: Polish & Demo Prep** (90 min)

**Your Goals:**
- Projects are complete
- Documentation is solid
- Students are ready to demo

**Session Outline:**

**[0-30 min] Final Features**
- Students finish any remaining features
- Add error handling
- Test edge cases
- **This is the deadline for code**

**[30-60 min] Documentation Sprint**
- Write comprehensive README:
  - What does it do?
  - How to install?
  - How to use?
  - Example output
  - Architecture overview
- Add screenshots/GIFs if possible
- Commit: "docs: Add comprehensive README"

**[60-75 min] Demo Dry Run**
- Partner up students
- Each person does 2-minute demo to partner
- Partner gives feedback:
  - "Did I understand what it does?"
  - "Was it clear?"
  - "What would make it better?"

**[75-90 min] Final Push**
- Make final commits
- Push everything to GitHub
- Verify tests pass
- Check README displays correctly on GitHub

**Homework:**
- Prepare 5-minute demo presentation
- Practice! Time yourself!
- Test your demo (make sure it works)

---

## 🎤 Week 3: Demo Day

### Format (2 hours recommended)

**Setup:**
- Students present in random order (draw names)
- 5 minutes per student:
  - 3 min demo
  - 2 min Q&A
- Project displays on big screen

**Presentation Structure:**
Students should cover:
1. **Problem** (30 sec): What does it solve?
2. **Demo** (2 min): Show it working
3. **Technical** (30 sec): What OpenAI tools? How did you build it?
4. **Q&A** (2 min): Answer questions

**Questions You Might Ask:**
- "Why did you choose this architecture?"
- "How did you use AI to help build this?"
- "What was the hardest part?"
- "If you had another week, what would you add?"
- "Show me your test coverage report"
- "Show me your logs"

**Awards (Optional but Fun!)** 🏆
- **Most Creative:** Most unique idea
- **Best Code Quality:** Cleanest code, best patterns
- **Best Testing:** Highest coverage, best test design
- **Best Demo:** Clearest presentation
- **Most Ambitious:** Tackled hardest challenge
- **Best Use of AI:** Most effective AI collaboration

---

## 🚨 Common Issues & Solutions

### Issue 1: "My API calls are too slow"
**Solution:** 
- Use mocking in tests (don't call real API in tests!)
- Show them the conftest.py fixtures
- Explain caching strategies

### Issue 2: "I can't get 100% coverage"
**Solution:**
- Review what's uncovered: `pytest --cov=src --cov-report=html`
- Explain `pragma: no cover` for defensive code
- Help them write tests for the tricky parts

### Issue 3: "I'm stuck and don't know what to do"
**Solution:**
- "What's the smallest next step?"
- "Can you write a test for it?"
- "Ask Claude for help with this specific thing"

### Issue 4: "My project is too complex"
**Solution:**
- "What's the core feature?"
- "Can you cut features and still have something cool?"
- "Scope down to MVP"

### Issue 5: "I don't understand how to use Claude"
**Solution:**
- Show them example prompts from AI_PROMPTS.md
- Model a conversation with Claude
- Emphasize: "Give Claude context, be specific"

### Issue 6: "Git is confusing"
**Solution:**
- Show them the basic workflow:
  ```bash
  # After writing code
  git status                  # See what changed
  git add file.py            # Stage the file
  git commit -m "feat: Add X"  # Commit with message
  git push                   # Push to GitHub
  ```
- Reference GIT_GUIDE.md

---

## 📊 What Success Looks Like

### Minimum Success (C grade):
- ✅ Application works
- ✅ Uses at least one OpenAI API
- ✅ Has some tests (70%+ coverage)
- ✅ Basic README
- ✅ Can demo it

### Good Success (B grade):
- ✅ Application is useful/interesting
- ✅ Uses OpenAI APIs effectively
- ✅ 90%+ test coverage
- ✅ Good commit history
- ✅ Enterprise logging
- ✅ Clear demo

### Excellent Success (A grade):
- ✅ Creative, polished application
- ✅ Combines multiple APIs
- ✅ 100% test coverage
- ✅ Professional Git workflow
- ✅ Comprehensive logging
- ✅ Excellent documentation
- ✅ Engaging demo, answers questions well

---

## 💬 Conversation Starters (While Circulating)

**Week 1:**
- "Tell me about your project idea"
- "Which OpenAI tools are you using?"
- "Show me your first test"
- "Walk me through your TDD cycle"

**Week 2:**
- "Demo what you have so far"
- "What's left to build?"
- "Show me your test coverage"
- "Show me your logs"
- "Walk me through your architecture"

**General:**
- "What did you learn this week?"
- "What's been the hardest part?"
- "How has AI helped you?"
- "What would you do differently?"

---

## 🎯 Key Teaching Moments

### Emphasize These Concepts:

1. **TDD is not "testing"** - it's a design methodology
   - Write test FIRST
   - Think about API before implementation
   - Tests as documentation

2. **AI is a partner, not a crutch**
   - You should understand every line
   - Ask "why?" questions
   - Verify AI's suggestions

3. **Enterprise patterns matter**
   - Logging helps you debug
   - Clean code is maintainable
   - Tests give confidence

4. **Git tells a story**
   - Commit messages matter
   - Atomic commits are professional
   - Your GitHub is your portfolio

5. **Scope management**
   - MVP first, polish later
   - Done is better than perfect
   - You can always add features later

---

## 📈 Assessment Tips

### Code Review Checklist:

**Browse their GitHub:**
- [ ] Has meaningful commit messages? (feat:, fix:, test:, docs:)
- [ ] Commits are atomic? (one logical change each)
- [ ] README is clear and complete?
- [ ] Tests are comprehensive?
- [ ] Code is clean and well-structured?

**Run their code:**
```bash
git clone [their-repo]
cd [their-repo]
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pytest --cov=src
python -m src.main [command]
```

**Check logs:**
```bash
ls -la logs/
tail logs/app.log
```

### Red Flags:
- ❌ All commits on last day (they procrastinated)
- ❌ Commit messages like "fixed stuff" (didn't follow guide)
- ❌ No tests or very low coverage (didn't use TDD)
- ❌ No logs directory (didn't add logging)
- ❌ Can't explain their code (copied without understanding)

### Green Flags:
- ✅ Commits spread across 2 weeks (good time management)
- ✅ Clean commit history (followed Git guide)
- ✅ 100% coverage (took testing seriously)
- ✅ Logs with context (understood logging)
- ✅ Can explain decisions (understood the work)

---

## 🎁 Bonus Challenges (For Advanced Students)

If a student finishes early:

1. **Add a Web UI** (Flask/FastAPI)
2. **Deploy to Cloud** (Heroku/Render/Railway)
3. **Add Caching** (Redis/in-memory)
4. **Create Docker Container**
5. **Add CI/CD Badge** to README
6. **Write Blog Post** explaining the project
7. **Combine 3+ OpenAI Tools**
8. **Add Database** (SQLite/Postgres)
9. **Create Interactive Demo** (Streamlit)
10. **Open Source It** (accept contributions)

---

## 📚 Resources for Students

Point struggling students to:
- **START_HERE.md** - If overwhelmed
- **HOW_TO_WORK_WITH_AI.md** - If stuck on using AI
- **QUICK_TDD_GUIDE.md** - If tests are confusing
- **GIT_GUIDE.md** - If Git is confusing
- **LOGGING_GUIDE.md** - If logging is unclear
- **PROJECT_IDEAS.md** - If can't think of ideas
- **docs/architecture.md** - Example of good architecture

---

## 🎬 Your Demo Day Role

As instructor, during demos:

**Before each demo:**
- "Let's welcome [Student Name]!"
- *[Applause]*

**During demo:**
- Take notes
- Let them present without interruption

**After demo:**
- Ask 1-2 technical questions
- Invite class questions
- Thank them: "Great work!"

**After all demos:**
- Congratulate the class
- Highlight common themes
- Announce awards (if doing them)
- Remind them: "This is portfolio material - add it to your resume!"

---

## 💡 Final Tips

1. **Be available** - Have office hours or be on Slack
2. **Show enthusiasm** - Your energy is contagious
3. **Celebrate progress** - Acknowledge milestones
4. **Be flexible** - Some students will struggle, help them scope down
5. **Document it** - Take photos/videos of demos (with permission)
6. **Follow up** - Post-assignment reflection survey

---

## 📊 Post-Assignment Survey

Ask students:
1. How many hours did you spend on this project?
2. What was the most valuable thing you learned?
3. What was the most challenging part?
4. How helpful was AI (Claude) in your development?
5. Would you continue working on this project?
6. What would you change about the assignment?
7. Rate your confidence with TDD (1-10)
8. Rate your confidence with Git (1-10)
9. Rate your confidence working with AI (1-10)

---

**Good luck! This is going to be an amazing learning experience for your students.** 🚀

**Questions during the assignment? Ask Claude!** 💬
