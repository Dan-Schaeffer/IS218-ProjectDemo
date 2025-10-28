# 🤖 Working with Claude - A Practical Guide

**⏱️ 10 min read | How to collaborate effectively with AI**

---

## ⚠️ Critical First Thing: Claude's Knowledge Cutoff

**Claude's training data ends: April 2024**

This means Claude doesn't know about:
- OpenAI features released after April 2024 (like GPT-5 Pro, Sora 2, AgentKit!)
- Library updates after April 2024
- New APIs or breaking changes

### 🔥 ALWAYS Give Claude Fresh Documentation

**✨ WE'VE DONE THIS FOR YOU!**

This repo includes comprehensive research on all OpenAI tools as of **October 2025**:
- 📄 [**OpenAI Tools Research (Oct 2025)**](openai_tools_research_oct2025.md)

**Tell Claude:**
```
"I'm building a project with OpenAI APIs. 

Please read the comprehensive research at:
docs/openai_tools_research_oct2025.md

This has all the latest tools including GPT-5 Pro, Sora 2, AgentKit, 
and updated documentation from October 2025."
```

**For additional/specific APIs, you can also:**

```bash
# Download latest docs from OpenAI
# Visit: https://platform.openai.com/docs/api-reference
# Save specific sections to docs/ folder
# (Note: references/ folder is in .gitignore for student clones)
```

**Why this matters:**
- Claude might suggest deprecated methods
- API parameters change frequently
- New features exist that Claude doesn't know about (GPT-5 Pro, Sora 2!)
- You'll waste time debugging outdated code
- The research document has October 2025 updates!

---

## 🤝 Partnership, Not Magic

### What Claude IS:
- ✅ A pair programmer
- ✅ A rubber duck that talks back
- ✅ A pattern suggester
- ✅ A syntax helper
- ✅ A test generator

### What Claude IS NOT:
- ❌ A replacement for thinking
- ❌ Always correct
- ❌ Aware of your specific requirements
- ❌ Able to run your code
- ❌ Connected to the internet

**Golden Rule:** You're the architect, Claude is the construction crew.

---

## 💬 How to Prompt Effectively

### ❌ BAD Prompts (Vague)

```
"Write my project"
"Fix this code" [paste 500 lines]
"Make it better"
"Add tests"
```

**Why bad:** Claude doesn't know what you want, what "better" means, or what context matters.

### ✅ GOOD Prompts (Specific + Context)

```
"I'm building a CLI that analyzes images using OpenAI Vision API.

I need:
1. A test for the ImageAnalyzer class
2. It should handle network errors gracefully
3. Use the pattern from enterprise_ai_demo1_websearch

Here's my current model: [paste 20 lines]"
```

**Why good:** Clear goal, specific requirements, reference example, minimal code.

---

## 🎯 Effective Prompt Formula

```
SITUATION: [What are you building?]
CONTEXT: [What have you done so far?]
GOAL: [What do you need help with?]
CONSTRAINTS: [Any requirements or patterns?]
REFERENCE: [Point to example code]
```

### Example:

```
SITUATION: I'm building a sentiment analyzer using OpenAI Chat API

CONTEXT: I have a SentimentAnalyzer class that calls the API,
but I need to add error handling for rate limits.

GOAL: Show me how to implement exponential backoff retry logic.

CONSTRAINTS: 
- Use pytest for tests
- Follow the error handling pattern in enterprise_ai_demo1_websearch
- Max 3 retries

REFERENCE: Look at src/client.py lines 80-100 in the demo
```

---

## 🔄 The Collaboration Loop

### 1. Plan Together
**You:** "I need to build X. How should I structure it?"
**Claude:** [Suggests architecture]
**You:** "I like it, but let's change Y because Z"

### 2. Test First (TDD)
**You:** "Write a test for the User model with name and email"
**Claude:** [Writes test]
**You:** Run test → RED (fails as expected)

### 3. Implement
**You:** "Now write the model to pass that test"
**Claude:** [Writes code]
**You:** Run test → GREEN (passes!)

### 4. Refactor
**You:** "This works but looks messy. Can we simplify?"
**Claude:** [Suggests refactor]
**You:** Run test → Still GREEN!

### 5. Commit
```bash
git add tests/test_models.py src/models.py
git commit -m "feat: Add User model with validation"
```

**Repeat!**

---

## 🚨 Common Mistakes (And How to Avoid Them)

### Mistake #1: Copy-Paste Without Understanding

**Bad:**
```python
# Claude suggested this, no idea what it does
@functools.lru_cache(maxsize=128)
def process_data(x: str) -> list[dict[str, Any]]:
    return [{'data': json.loads(x)}]
```

**Good:**
Ask first: "What does `@functools.lru_cache` do and why would I use it here?"

**Rule:** If you can't explain it, don't commit it.

---

### Mistake #2: Letting Claude Make All Decisions

**Bad:**
"Build my entire project for me."

**Good:**
"I decided to use a Pipeline pattern. Help me implement the first stage."

**Rule:** You decide WHAT, Claude helps with HOW.

---

### Mistake #3: Not Providing Context

**Bad:**
```
"This doesn't work" [paste error]
```

**Good:**
```
"I'm trying to parse OpenAI's response. Getting KeyError: 'choices'.

Here's my code: [paste 15 lines]
Here's the error: [paste traceback]
Here's what I expect: [describe expected behavior]

The API response structure should be: [paste docs]"
```

**Rule:** Context = Better answers.

---

### Mistake #4: Asking to Debug 500 Lines

**Bad:**
[Paste entire file] "Why doesn't this work?"

**Good:**
"Line 47 in my parser raises IndexError. Here's the function: [paste 10 lines]
And here's the test that fails: [paste 5 lines]"

**Rule:** Isolate the problem first.

---

### Mistake #5: Trusting Outdated Info

**Bad:**
Follow Claude's suggestion without checking docs.

**Good:**
"You suggested using `response.data['content']` but the OpenAI docs show `response.choices[0].message.content`. Which is correct?"

**Rule:** Verify against official docs (especially for APIs).

---

## 🎓 Real Conversation Examples

### Example 1: Starting a Feature

**You:**
```
"I need to add a feature that translates text using OpenAI.

Current architecture:
- TranslationClient (calls API)
- TranslationService (business logic)
- TranslationParser (formats output)

I want to start with the model. What should my Translation dataclass look like?"
```

**Claude:** [Suggests model with fields]

**You:** "Good, but also include detected_language. Now write the test for it."

---

### Example 2: Stuck on a Test

**You:**
```
"My test is failing. Here's the test: [paste]

Error: AttributeError: 'Mock' object has no attribute 'choices'

I'm mocking OpenAI's response. What am I doing wrong?"
```

**Claude:** [Explains mock structure]

**You:** "Ah! I need to mock the nested structure. Show me the correct mock setup."

---

### Example 3: Refactoring

**You:**
```
"This function works but it's 50 lines and does 3 things:
1. Calls API
2. Parses response
3. Handles errors

Help me split it into smaller functions."
```

**Claude:** [Suggests separation]

**You:** "Good! Now help me write tests for each new function."

---

## 🧪 Using Claude for TDD

### Perfect TDD Flow:

**Step 1: Describe what you want**
```
"I need a validator that checks if email is valid format.
Should return True/False.
Write the test first."
```

**Step 2: Run the test (RED)**
```bash
pytest tests/test_validators.py
# ImportError: cannot import name 'validate_email'
```

**Step 3: Ask for implementation**
```
"Now write the validate_email function to pass that test.
Keep it simple - just check for @ symbol and domain."
```

**Step 4: Run test (GREEN)**
```bash
pytest tests/test_validators.py
# 1 passed
```

**Step 5: Commit**
```bash
git add tests/test_validators.py src/validators.py
git commit -m "feat: Add email validation"
```

---

## 📚 When to Use References

### Always Give Claude These:

1. **API Documentation**
   - Endpoints
   - Parameters
   - Response formats
   - Error codes

2. **Library Docs**
   - If using new library
   - If using specific version

3. **Your Existing Code**
   - "Follow the pattern in src/client.py"
   - "Use same error handling as src/parser.py"

4. **Requirements**
   - Grading rubric
   - Project specification

### How to Provide References:

**Option 1: Upload file**
```
"I uploaded openai_api_docs.md to references/
Please review before helping."
```

**Option 2: Paste relevant section**
```
"According to the docs:
[paste 10-20 lines of relevant docs]

How do I implement this?"
```

**Option 3: Point to working code**
```
"Look at how enterprise_ai_demo1_websearch handles this
in src/client.py lines 50-70.

Help me adapt that pattern for my use case."
```

---

## 🔍 Debugging with Claude

### Effective Debug Prompts:

**Template:**
```
PROBLEM: [One sentence description]
CODE: [Paste minimal relevant code]
ERROR: [Paste error message or unexpected behavior]
EXPECTED: [What should happen]
TRIED: [What you already attempted]
```

**Example:**
```
PROBLEM: Parser crashes on empty OpenAI response

CODE:
def parse_response(response):
    return response.choices[0].message.content  # Line 23

ERROR:
IndexError: list index out of range

EXPECTED: Should handle empty responses gracefully

TRIED: Added if statement but still crashes
```

---

## ✅ Quality Checks (Before You Commit)

Ask Claude to review:

1. **"Does this follow the project pattern?"**
2. **"Are there edge cases I'm missing?"**
3. **"Should I add more tests?"**
4. **"Is this code too complex?"**
5. **"Any security concerns?"**

### Example Review Prompt:

```
"Here's my new feature: [paste code]
And the tests: [paste tests]

Before I commit:
1. Does this follow TDD best practices?
2. Am I missing test cases?
3. Is the error handling sufficient?
4. Any code smells?"
```

---

## 🎯 Best Practices Summary

### Do:
- ✅ Provide fresh documentation for APIs
- ✅ Start with small, specific asks
- ✅ Understand every line Claude suggests
- ✅ Test Claude's code before committing
- ✅ Ask "why" when you don't understand
- ✅ Iterate in small steps
- ✅ Point to reference examples

### Don't:
- ❌ Trust Claude blindly
- ❌ Copy-paste without understanding
- ❌ Ask vague questions
- ❌ Paste 500 lines of code
- ❌ Assume Claude knows latest APIs
- ❌ Skip testing suggested code
- ❌ Let Claude make all decisions

---

## 🚀 Level Up Your Collaboration

### Beginner:
"Write a function that does X"

### Intermediate:
"Write a test for a function that does X, then implement it"

### Advanced:
"I want to implement X. Given the architecture in src/, suggest how to structure this feature. Then help me write tests and implementation following TDD."

**Aim for Advanced!**

---

## 💡 Pro Tips

1. **Start sessions with context**
   ```
   "I'm working on [project]. So far I've built [X, Y, Z].
   Now I need help with [specific thing]."
   ```

2. **End sessions with summary**
   ```
   "Summarize what we built this session and what to do next."
   ```

3. **Save good prompts**
   - Keep a "prompts.txt" file
   - Reuse patterns that work

4. **Pair with humans too**
   - Claude + classmate = powerful
   - Explain Claude's suggestions to others

5. **Check the demo repo**
   - Before asking Claude, see if demo solves it
   - Reference demo patterns in prompts

---

## 🆘 When Claude Can't Help

Sometimes you need:
- **Human instructor** - For project direction, grading questions
- **Official docs** - For latest API changes
- **Stack Overflow** - For community solutions
- **Debugger** - For complex runtime issues
- **Break** - Step away, come back fresh

**Claude is a tool, not the only tool.**

---

## 📊 Self-Check

After using Claude, ask yourself:

- [ ] Do I understand what Claude suggested?
- [ ] Did I test the code?
- [ ] Does it match our project patterns?
- [ ] Can I explain this to someone else?
- [ ] Did I verify against official docs?

**If you answered NO to any: Don't commit yet!**

---

**Remember: The best developers know how AND when to use AI assistance. You're learning both!** 🚀

---

*For TDD workflow, see [TDD_WORKFLOW.md](TDD_WORKFLOW.md)*
