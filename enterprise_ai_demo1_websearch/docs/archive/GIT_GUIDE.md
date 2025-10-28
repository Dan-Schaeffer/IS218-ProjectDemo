# 📝 Git Best Practices for Students

**⏱️ 5 min read | How to use Git like a professional**

---

## 🎯 Why This Matters

Employers look at your GitHub. Clean commit history = professional developer.

**Bad commits:**
```
"fixed stuff"
"asdfjkl"
"IT WORKS!!!"
"final final FINAL version"
```

**Good commits:**
```
feat: Add web search functionality with citations
test: Add unit tests for search parser (90% coverage)
fix: Handle empty API responses gracefully
docs: Update README with setup instructions
```

**Which developer would you hire?**

---

## 🏆 Conventional Commits (Industry Standard)

Format: `<type>: <description>`

### Common Types:

| Type | When to Use | Example |
|------|-------------|---------|
| **feat** | New feature | `feat: Add domain filtering to searches` |
| **fix** | Bug fix | `fix: Handle missing API credentials error` |
| **test** | Adding/updating tests | `test: Add edge cases for empty queries` |
| **docs** | Documentation only | `docs: Add architecture diagram to README` |
| **refactor** | Code improvement (no new features) | `refactor: Extract parsing logic into separate class` |
| **style** | Formatting (no code change) | `style: Format code with black` |
| **chore** | Maintenance tasks | `chore: Update dependencies to latest versions` |
| **build** | Build system changes | `build: Add pytest coverage configuration` |
| **ci** | CI/CD changes | `ci: Add GitHub Actions workflow` |
| **perf** | Performance improvement | `perf: Cache API responses for 5 minutes` |

---

## ⚡ Atomic Commits (One Thing Per Commit)

### ❌ Bad (Everything in one commit):
```bash
git add .
git commit -m "did homework"
# Changed: tests, docs, 5 features, bug fixes, config
```

**Problem:** Can't undo one change without undoing everything.

### ✅ Good (Atomic - one logical change per commit):
```bash
# Commit 1: Setup
git add pytest.ini requirements.txt
git commit -m "build: Configure pytest with coverage settings"

# Commit 2: Models
git add src/models.py tests/test_models.py
git commit -m "feat: Add SearchResult and Citation models"

# Commit 3: Tests
git add tests/test_models.py
git commit -m "test: Add edge cases for SearchResult validation"

# Commit 4: Docs
git add README.md
git commit -m "docs: Add installation and usage instructions"
```

**Benefit:** Each commit can be reverted independently.

---

## 📋 Your Commit Workflow

### Step 1: Check What Changed
```bash
git status                  # See modified files
git diff                    # See exact changes
```

### Step 2: Stage Related Changes
```bash
# Add specific files (not git add .)
git add src/models.py tests/test_models.py
```

### Step 3: Write Good Commit Message
```bash
git commit -m "feat: Add Citation model with length validation"
```

### Step 4: Push When Ready
```bash
git push origin main
```

---

## 📖 Commit Message Rules

### Rule 1: Start with Type
```bash
✅ feat: Add web search API client
❌ Add web search API client
```

### Rule 2: Use Imperative Mood (Give a command)
```bash
✅ fix: Handle null response from API
❌ fixed null response handling
❌ fixes null response
```

### Rule 3: Be Specific
```bash
✅ test: Add unit tests for parser edge cases
❌ test: Add tests
```

### Rule 4: Keep Under 50 Characters (for first line)
```bash
✅ feat: Add domain filtering to search
❌ feat: Add the ability to filter search results by domain name which allows users to specify...
```

### Rule 5: No Period at End
```bash
✅ docs: Update README with examples
❌ docs: Update README with examples.
```

---

## 🎯 Real-World Examples

### Scenario 1: Starting Your Project
```bash
# First commit - setup
git add .gitignore requirements.txt
git commit -m "chore: Initialize project structure"

# Virtual environment setup
git add pytest.ini
git commit -m "build: Configure pytest with 90% coverage requirement"

# Add models
git add src/models.py
git commit -m "feat: Add core data models (SearchResult, Citation, Source)"

# Add tests
git add tests/test_models.py tests/conftest.py
git commit -m "test: Add comprehensive model tests with fixtures"
```

### Scenario 2: Fixing a Bug
```bash
# You found a bug and fixed it
git add src/client.py
git commit -m "fix: Handle missing 'created' attribute in API response"

# Add test to prevent regression
git add tests/test_client.py
git commit -m "test: Add test for API response without created field"
```

### Scenario 3: Adding a Feature
```bash
# TDD: Write test first
git add tests/test_search.py
git commit -m "test: Add tests for domain filtering feature"

# Implement feature
git add src/search_service.py
git commit -m "feat: Implement domain filtering for search results"

# Document it
git add README.md
git commit -m "docs: Add domain filtering examples to README"
```

### Scenario 4: Refactoring
```bash
# Improve code without changing behavior
git add src/parser.py
git commit -m "refactor: Extract citation parsing into separate method"

# Tests should still pass!
git status  # Make sure tests didn't change
```

---

## 🚫 Common Mistakes (Don't Do These!)

### Mistake 1: Vague Messages
```bash
❌ git commit -m "update"
❌ git commit -m "changes"
❌ git commit -m "working on it"

✅ git commit -m "feat: Add citation parsing to search results"
```

### Mistake 2: Too Much in One Commit
```bash
❌ git add .
❌ git commit -m "complete project"
# Contains: tests, docs, 10 features, config changes

✅ Make 15 atomic commits instead
```

### Mistake 3: Committing Secrets
```bash
❌ git add .env
# Now your API key is on GitHub forever!

✅ Add .env to .gitignore FIRST
✅ Only commit .env.example (without real keys)
```

### Mistake 4: Not Testing Before Commit
```bash
❌ git commit -m "feat: Add new feature"
❌ git push
# Oh no, tests are failing!

✅ pytest                    # Run tests first
✅ git commit -m "feat: Add new feature"
✅ git push                  # Now push
```

---

## 📊 Your Project Commit History Should Look Like:

```bash
git log --oneline

feat: Add CLI argument parsing
test: Add integration tests for search flow
feat: Implement search service layer
test: Add parser tests with mock responses
feat: Add response parser with citation extraction
test: Add client tests with mocked API
feat: Add OpenAI web search client
test: Add model validation tests
feat: Add core data models
build: Configure pytest with coverage
chore: Initialize project with dependencies
```

**See the story?** Each commit is one logical change.

---

## 🎓 For Your Class Project

### Week 1: Setup
```bash
chore: Initialize project structure
build: Configure pytest and coverage
docs: Add README with project overview
```

### Week 2: Core Features (TDD)
```bash
test: Add tests for [feature] models
feat: Implement [feature] models
test: Add tests for [feature] API client
feat: Implement [feature] API client
test: Add tests for [feature] parser
feat: Implement [feature] parser
```

### Week 3: Polish
```bash
test: Increase coverage to 90%
fix: Handle edge case in [component]
refactor: Improve error handling in [component]
docs: Add architecture documentation
docs: Add usage examples to README
```

### Week 4: Final Touches
```bash
docs: Create tutorial for students
docs: Add API documentation
chore: Update dependencies to latest versions
docs: Final README polish
```

---

## 🔥 Pro Tips

### Tip 1: Commit Often
```bash
# Don't wait until everything is perfect
# Commit after each small working piece
git commit -m "feat: Add basic search function"
# 10 minutes later...
git commit -m "feat: Add error handling to search"
# 15 minutes later...
git commit -m "test: Add tests for search errors"
```

### Tip 2: Use Git Diff Before Committing
```bash
git diff src/models.py    # Review your changes
git add src/models.py     # Stage if good
git commit -m "feat: Add validation to Citation model"
```

### Tip 3: Amend Last Commit (If Not Pushed)
```bash
# Oops, forgot something in last commit
git add forgotten_file.py
git commit --amend --no-edit   # Adds to previous commit

# Or change message
git commit --amend -m "feat: Add Citation model with validation"
```

### Tip 4: Write Commit Body for Complex Changes
```bash
git commit -m "feat: Add domain filtering to searches" -m "
- Added domain parameter to SearchOptions
- Updated client to pass domains to API
- Added validation for max 3 domains
- Updated tests to cover new functionality
"
```

---

## 🆘 Undo Mistakes

### Unstage File
```bash
git add wrong_file.py
git reset wrong_file.py      # Unstage it
```

### Undo Last Commit (Keep Changes)
```bash
git reset --soft HEAD~1      # Undo commit, keep files staged
```

### Discard Local Changes
```bash
git checkout -- file.py      # Discard changes in file
git reset --hard             # Discard ALL local changes (careful!)
```

---

## 📋 Checklist Before Each Commit

- [ ] Tests pass: `pytest`
- [ ] Code is clean (no debugging print statements)
- [ ] Only related files staged (`git status`)
- [ ] Commit message follows convention
- [ ] Commit is atomic (one logical change)
- [ ] No secrets committed (.env in .gitignore)

---

## 🎯 Your First Commits Template

```bash
# 1. Initial setup
git init
git add .gitignore README.md
git commit -m "chore: Initialize repository"

# 2. Setup virtual environment
python -m venv venv
source venv/bin/activate
git add requirements.txt pytest.ini
git commit -m "build: Add dependencies and test configuration"

# 3. First feature (TDD)
git add tests/test_models.py
git commit -m "test: Add tests for SearchResult model"

git add src/models.py
git commit -m "feat: Implement SearchResult model"

# 4. Keep going!
```

---

## 💬 Ask Claude for Commit Messages

If you're stuck on how to describe a commit:

```
"Claude, I changed these files: [list]
Here's what I did: [explain]
What's a good conventional commit message?"
```

I'll suggest something like:
```
feat: Add citation extraction to parser
test: Add edge case tests for empty responses
fix: Handle timeout errors in API client
```

---

## 🏆 Goal: Professional GitHub Profile

Your professor (and future employers) will see:

✅ Clear commit history  
✅ Conventional commit messages  
✅ Atomic commits  
✅ Good testing practices  
✅ Professional documentation  

**This shows you're a serious developer.**

---

## ⏭️ Next Steps

1. ✅ Read this guide
2. ➡️ Start your project with good first commit
3. ➡️ Make atomic commits as you build
4. ➡️ Review your `git log` - does it tell a story?

---

**Remember: Commits are documentation of your thought process. Make them count!** 📝

---

*Need help writing a commit message? Ask Claude: "What's a good commit message for [your changes]?"*
