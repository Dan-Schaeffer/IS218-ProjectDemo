# 🤖 How To Work With AI (From Claude)

**⏱️ 10 min read | Critical info for working with AI coding assistants**

---

## 🚨 CRITICAL: My Knowledge Is Outdated!

**Hey, it's Claude. Real talk:**

### My Training Data Cutoff: April 2024

This means:
- ❌ I don't know about OpenAI features released after April 2024
- ❌ I don't have the latest API docs
- ❌ My code examples might use old syntax
- ⚠️ If something I suggest doesn't work, I might be outdated

### How To Work Around This:

#### Option 1: Give Me Fresh Docs (Best Method)
```bash
# Clone the library you're using
mkdir references
cd references
git clone https://github.com/openai/openai-python.git

# Then ask me:
"Read the latest web search docs from references/openai-python/docs/
and help me implement [feature]"
```

#### Option 2: Copy-Paste Latest Docs
1. Go to https://platform.openai.com/docs
2. Copy the relevant section
3. Paste in chat: "Here are the latest docs: [paste]. Help me use this."

#### Option 3: Show Me Errors
```
"I tried your suggestion but got this error: [paste error]
Here's what OpenAI's current docs say: [paste]
How should I fix it?"
```

### Why This Matters
If you blindly trust me, you'll waste hours debugging old API calls. Always verify against current docs!

---

## 🤝 How We Should Work Together

### ❌ BAD: You're My Boss
```
Student: "Build my project"
Me: [Generates code]
Student: [Copies without understanding]
Professor: "Explain this line"
Student: "Uhh..."
Grade: D
```

### ✅ GOOD: We're Partners
```
Student: "I want to build X. I read the docs on Y tool. 
         Is this architecture good?" [shares plan]
Me: "Nice! But consider this edge case..."
Student: "Good point. How should we handle it?"
Me: [Explains options]
Student: "Let's go with option 2. Help me write tests first."
[Back and forth collaboration]
Result: Student understands everything, passes with A
```

---

## 💬 Prompts That Actually Work

### ❌ Lazy Prompts (I'll give you garbage)
```
"Help with my code"
"Make this work"
"Do my homework"
```

### ✅ Smart Prompts (I'll give you gold)

**When Starting:**
```
"I'm building [project] using [OpenAI tools]. I've read the docs at 
[link/paste]. Here's my architecture idea: [explain]. What problems 
might I face? What's a better approach?"
```

**When Stuck:**
```
"I'm trying to [goal]. Here's my code: [paste]
Here's the error: [paste]
Here's what I tried: [list]
What am I missing?"
```

**When Learning:**
```
"Explain how [concept] works. Then show me an example. 
Then let me try writing it myself and check my work."
```

**The Secret:** Context + Specific ask + Your thinking = Better help

---

## 🎯 The TDD Mindset (Why Your Professor Wants This)

### Without Tests
```python
# You write code
def search_web(query):
    return api.search(query)  # Does this work? 🤷

# You run it manually
# Maybe it works, maybe it breaks
# You push to GitHub
# Demo day: Everything crashes
# Professor: "Did you test this?"
# You: *sweats*
```

### With Tests (TDD)
```python
# 1. Write test FIRST (it fails - RED)
def test_search_returns_results():
    results = search_web("Python")
    assert len(results) > 0
    assert results[0].url.startswith("http")
# ❌ FAILS - search_web doesn't exist

# 2. Write minimal code (it passes - GREEN)
def search_web(query):
    response = api.search(query)
    return parse_response(response)
# ✅ PASSES - test is green

# 3. Refactor (keep it green)
def search_web(query, options=None):
    """Search with optional filters"""
    if not query:
        raise ValueError("Query required")
    response = api.search(query, **options or {})
    return parse_response(response)
# ✅ STILL PASSES - added features, tests still green
```

**Why This Matters:**
- You always know if code works
- Adding features doesn't break old code
- Your grade: A (professor sees 90% coverage)

---

## 🚀 Working With Me: Step-by-Step

### Phase 1: Research (30 min)
**You do:**
1. Read OpenAI docs for tools you're interested in
2. Watch a demo video
3. Think of 3 project ideas

**Then ask me:**
```
"I researched [tools]. I'm thinking of building:
1. [idea 1]
2. [idea 2]
3. [idea 3]

Which is most feasible for a 3-week project? 
What challenges would each have?"
```

### Phase 2: Architecture (45 min)
**We do together:**
```
You: "Let's design [chosen project]. I need:
     - User inputs [X]
     - System does [Y]
     - Output is [Z]
     What components do I need?"

Me: [Suggests architecture]

You: "Why did you separate X and Y?"
Me: [Explains reasoning]

You: "Makes sense. Let's document this."
```

### Phase 3: TDD Implementation (Most time here)
**Pattern for EACH component:**
```
You: "Let's build [component]. What should I test?"
Me: [Suggests tests]
You: "Add a test for [edge case]"
Me: [Adds test]
You: "Now help me implement to pass these tests"
[We iterate until tests pass]
You: "All green! Next component."
```

**Repeat for each piece.**

### Phase 4: Integration (2-3 hours)
```
You: "All components work alone. Now connect them."
Me: [Helps write integration tests and glue code]
You: [Runs tests] "3 integration tests failing"
You: "Why is X not talking to Y?"
[We debug together]
You: "All tests pass! 🎉"
```

### Phase 5: Documentation (1 hour)
```
You: "Generate README with setup, usage, and examples"
Me: [Generates docs]
You: [Reviews] "Add a section on [specific feature]"
Me: [Adds it]
You: "Perfect!"
```

---

## 🎨 Project Ideas (Steal These!)

### Starter (⭐⭐) - 1-2 weeks
**News Summarizer**
- Use: Web Search API
- What: Search topic → Get top articles → Summarize with citations
- Why cool: Learn API basics + text processing

**Meme Generator**
- Use: Web Search + DALL-E
- What: Find trending topic → Generate meme image
- Why cool: Combine multiple APIs

### Intermediate (⭐⭐⭐⭐) - 2-3 weeks
**Study Assistant**
- Use: Web Search + GPT + TTS
- What: Research topic → Generate study guide + quiz → Read aloud
- Why cool: Actually useful, multiple APIs

**Debate Bot**
- Use: Web Search + GPT
- What: Research both sides of argument → Debate with user
- Why cool: Shows AI reasoning

### Advanced (⭐⭐⭐⭐⭐) - 3-4 weeks
**Fact Checker**
- Use: Web Search + GPT
- What: User submits claim → Search evidence → Analyze credibility → Verdict
- Why cool: Real-world application

**See [PROJECT_IDEAS.md](PROJECT_IDEAS.md) for 10+ more ideas**

---

## 💡 Instagram Attention Span? Here's The TL;DR

1. **My knowledge is old** - Always give me fresh docs
2. **We're partners** - Collaborate, don't just copy
3. **Use specific prompts** - Context = better help
4. **TDD = good grades** - Test first, code second
5. **Start small** - One feature at a time

---

## 🎯 Your First Conversation With Me

Copy this template:

```
Hey Claude! I'm starting a project for class.

BACKGROUND:
- I researched [OpenAI tools I looked at]
- I'm interested in [your interests]
- I want to build something [creative/useful/fun]

MY IDEA:
[Describe in 2-3 sentences]

QUESTIONS:
1. Is this feasible for 3 weeks?
2. What OpenAI tools should I use?
3. What challenges will I face?
4. Can you help me design the architecture?

IMPORTANT: I'll give you the latest OpenAI docs since your 
knowledge is from April 2024. Here's what's current: [paste or point to references/]

Let's build this together!
```

---

## 🆘 When Things Go Wrong

### "Your code doesn't work!"
1. Copy the error message
2. Share your code
3. Ask: "Why is this failing? Here's the error: [paste]"

### "I don't understand this"
Ask: "Explain [concept] like I'm new to this. Use an analogy."

### "This is taking forever"
You're probably trying to do too much at once. Ask: 
"Help me break this into smaller pieces I can build one at a time"

### "Tests are confusing"
Ask: "Show me one test example. Explain each line. Then let me write 
the next test and you check it."

---

## ⏭️ Next Steps

1. ✅ You've read this - good!
2. ➡️ Read [QUICK_TDD_GUIDE.md](QUICK_TDD_GUIDE.md) (15 min)
3. ➡️ Pick a project from [PROJECT_IDEAS.md](PROJECT_IDEAS.md)
4. ➡️ Come back and start the conversation above

---

**Remember: I'm here to help you learn, not do your work.** 

**Let's build something awesome together.** 🚀

---

*Questions? Ask me in VS Code! Just open this repo and start chatting.*
