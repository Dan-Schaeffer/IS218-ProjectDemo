# 🚀 START HERE - Quick Guide

**⏱️ 5 min read | Everything you need to start**

---

## 🎯 Your Mission

Build a cool project using OpenAI tools + TDD + AI coding assistant (Claude/Copilot)

**Why?** Learn to work WITH AI, not just use it. This skill = job security.

---

## ⚡ Quick Setup (3 minutes)

```bash
# 1. Clone this repo (you already did this)
cd enterprise_ai_demo1_websearch

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
source venv/bin/activate  # On Mac/Linux
# OR
venv\Scripts\activate     # On Windows

# 4. Install dependencies
pip install -r requirements.txt

# 5. Add your OpenAI API key
echo "OPENAI_API_KEY=your-key-here" > .env

# 6. Test it works
pytest                    # No -m needed!
python -m src.main "AI news"
```

**✅ If tests pass, you're ready!**

**💡 Always activate venv before working:** `source venv/bin/activate`

---

## 📚 The Only 4 Docs You Need

### 1️⃣ **[HOW_TO_WORK_WITH_AI.md](HOW_TO_WORK_WITH_AI.md)** (10 min)
**READ THIS FIRST!** Claude explains:
- ⚠️ Why Claude's knowledge is outdated (critical!)
- 🤝 How to collaborate (not just copy-paste)
- 🎨 9 project ideas you can steal
- 💬 Prompts that actually work

### 2️⃣ **[QUICK_TDD_GUIDE.md](QUICK_TDD_GUIDE.md)** (15 min)
**The process:**
- Write test → Run (fails) → Write code → Run (passes) → Repeat
- Why it matters: No broken code = happy professor

### 3️⃣ **[GIT_GUIDE.md](GIT_GUIDE.md)** (5 min)
**Professional commits:**
- Use conventional commits (feat:, fix:, test:, docs:)
- Make atomic commits (one thing per commit)
- Clean history = professional developer

### 4️⃣ **[PROJECT_IDEAS.md](PROJECT_IDEAS.md)** (5 min)
**Get inspired:**
- Starter projects (1-2 weeks)
- Advanced projects (3-4 weeks)
- How to pick one

---

## 🎓 How To Use This Repo

### Option A: Learn The Process (Recommended)
1. Read the 3 docs above
2. Look at how THIS project is structured
3. Build YOUR OWN project using the same process
4. Use our architecture as a template

### Option B: Fork & Modify
1. Fork this repo
2. Replace web search with a different OpenAI tool
3. Modify features
4. Keep the TDD structure

---

## 🆘 Common Questions

**Q: Can I just copy this code for my assignment?**  
A: No. Your professor wants YOU to build something. Use this as a learning example.

**Q: What if I get stuck?**  
A: Ask Claude/Copilot! Show them the error and ask "Why is this failing and how do I fix it?"

**Q: Do I need to read all the docs?**  
A: No! Start with the 3 above. Only read others if you need deeper info.

**Q: How long will this take?**  
A: 
- Understanding this repo: 2-3 hours
- Building your own project: 15-20 hours
- Getting to 90% test coverage: Worth it for your grade

---

## 🔥 Pro Tips

1. **Clone OpenAI docs locally** so Claude can read them:
```bash
mkdir references
cd references
git clone https://github.com/openai/openai-python.git
# Now ask Claude to read from references/
```

2. **Start small** - Build one feature at a time with tests

3. **Commit often** - Git save = life save

4. **Ask "why"** - Don't just copy code. Understand it.

---

## 📁 What's In This Repo?

```
src/           ← The actual code (read this to understand structure)
tests/         ← Tests for every piece of code (90% coverage!)
docs/          ← Technical docs (optional reading)
*.md files     ← Guides for students (you!)
```

---

## ⏭️ Next Steps

1. ✅ Read [HOW_TO_WORK_WITH_AI.md](HOW_TO_WORK_WITH_AI.md)
2. ✅ Run the code and tests
3. ✅ Pick a project idea
4. ✅ Start building!

---

**Questions? Open this repo in VS Code and ask Claude!** 💬

*"I read START_HERE.md. I want to build [idea]. Where do I start?"*
