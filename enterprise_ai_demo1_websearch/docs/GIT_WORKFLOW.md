# 📝 Git Workflow - Professional Commits

**⏱️ 10 min read | Level up your git game**

---

## 🎯 Why Git Matters

**Bad git history:**
```
fix stuff
more changes
forgot to commit yesterday
final version
final version 2
actually final
```

**Good git history:**
```
feat: Add user authentication
test: Add tests for login validation
fix: Handle empty email in validator
docs: Update README with setup steps
```

**Which developer would you hire?**

---

## 🏗️ The Foundation: Atomic Commits

### What is an Atomic Commit?

**One commit = One logical change**

### ❌ BAD (Too Big):
```bash
git commit -m "Add user system, fix bugs, update docs"
# Changes: 15 files, 500 lines
```

**Why bad:**
- Hard to review
- Hard to debug
- Hard to revert
- What if you only want the bug fix?

### ✅ GOOD (Atomic):
```bash
# Commit 1
git commit -m "feat: Add User model with validation"
# Changes: src/models.py, tests/test_models.py

# Commit 2
git commit -m "fix: Handle null email in validator"
# Changes: src/validators.py, tests/test_validators.py

# Commit 3
git commit -m "docs: Add User model to README"
# Changes: README.md
```

**Why good:**
- Easy to review each change
- Easy to find which commit broke something
- Easy to revert just one change

---

## 📋 Conventional Commits

### Format:

```
<type>: <description>

[optional body]

[optional footer]
```

### Types (Use These):

| Type | When to use | Example |
|------|------------|---------|
| `feat:` | New feature | `feat: Add email validation` |
| `fix:` | Bug fix | `fix: Handle missing API key` |
| `test:` | Add or update tests | `test: Add tests for parser` |
| `docs:` | Documentation | `docs: Update setup guide` |
| `refactor:` | Code restructure | `refactor: Extract validation logic` |
| `style:` | Formatting (no logic change) | `style: Fix indentation` |
| `chore:` | Maintenance | `chore: Update dependencies` |
| `perf:` | Performance improvement | `perf: Cache API responses` |
| `build:` | Build system | `build: Add pytest to requirements` |
| `ci:` | CI/CD changes | `ci: Add GitHub Actions workflow` |

---

## ✍️ Writing Good Commit Messages

### The Formula:

```
<type>: <what changed>

Why it changed (if not obvious)
How it works (if complex)
```

### ❌ BAD Messages:

```bash
git commit -m "fix"
git commit -m "update code"
git commit -m "changes"
git commit -m "wip"
git commit -m "asdfasdf"
```

**Why bad:** Future you has no idea what changed!

### ✅ GOOD Messages:

```bash
git commit -m "feat: Add user authentication

Implements JWT-based auth for API endpoints.
Tokens expire after 24 hours."

git commit -m "fix: Handle empty search results

Parser was crashing on empty array.
Now returns empty list instead."

git commit -m "test: Add edge cases for validator

Tests empty string, null, and special characters."
```

---

## 🔄 The Git Workflow

### Daily Flow:

```bash
# 1. Start fresh
git pull origin main

# 2. Check status
git status
# See: what files changed

# 3. Stage specific files
git add src/models.py tests/test_models.py

# 4. Check what you're committing
git diff --staged

# 5. Commit with message
git commit -m "feat: Add User model with validation"

# 6. Push to GitHub
git push origin main

# Repeat!
```

---

## 📚 Real Workflow Example

### Building a Feature (Step by Step):

**Step 1: Write test (RED)**
```bash
# Edit: tests/test_parser.py

git status
# modified: tests/test_parser.py

git add tests/test_parser.py
git commit -m "test: Add test for parse_user function"
git push
```

**Step 2: Implement code (GREEN)**
```bash
# Edit: src/parser.py

git status
# modified: src/parser.py

git add src/parser.py
git commit -m "feat: Implement user parsing"
git push
```

**Step 3: Fix bug you found**
```bash
# Edit: src/parser.py

git add src/parser.py
git commit -m "fix: Handle missing name field in parser"
git push
```

**Step 4: Add documentation**
```bash
# Edit: README.md

git add README.md
git commit -m "docs: Add parser usage example"
git push
```

---

## 🎯 Commit Frequency

### How Often to Commit?

**Rule of thumb:** Commit when you complete ONE thing

### Good Times to Commit:
- ✅ Test passes (after writing code)
- ✅ Fixed a bug
- ✅ Added a feature
- ✅ Refactored successfully
- ✅ Updated documentation

### Bad Times to Commit:
- ❌ Code doesn't run
- ❌ Tests are failing
- ❌ Haven't tested yet
- ❌ In the middle of a change
- ❌ End of the day with broken code

**Aim for: 3-5 commits per session**

---

## 🚨 Common Git Mistakes

### Mistake #1: Committing Everything

**Bad:**
```bash
git add .
git commit -m "stuff"
```

**Why bad:**
- Might include temporary files
- Mixing multiple unrelated changes
- No thought about what's being committed

**Good:**
```bash
git status  # See what changed
git add src/specific_file.py tests/test_specific.py
git commit -m "feat: Add specific feature"
```

---

### Mistake #2: Vague Messages

**Bad:**
```bash
git commit -m "update"
git commit -m "fix bug"
git commit -m "changes"
```

**Good:**
```bash
git commit -m "refactor: Extract validation to separate function"
git commit -m "fix: Handle APIError in client.chat()"
git commit -m "test: Add edge cases for empty responses"
```

**Test:** Can you understand the commit without looking at the code?

---

### Mistake #3: Committing Broken Code

**Bad:**
```bash
# Tests failing
pytest
# 5 failed

git commit -m "feat: Add feature"  # DON'T!
```

**Good:**
```bash
# Tests failing
pytest
# 5 failed

# Fix the tests first!
# Edit code until tests pass

pytest
# All passed

git commit -m "feat: Add feature"  # Now commit!
```

**Rule:** Never commit broken code (unless it's a WIP branch)

---

### Mistake #4: Forgetting to Pull

**Bad:**
```bash
# Make changes
git commit -m "feat: Add feature"
git push
# ERROR: updates were rejected
```

**Good:**
```bash
git pull origin main  # Start with this!
# Make changes
git commit -m "feat: Add feature"
git push  # Works!
```

**Rule:** Pull before you start working

---

## 🔧 Essential Git Commands

### Daily Commands:

```bash
# Check what changed
git status

# See changes in files
git diff

# See changes you're about to commit
git diff --staged

# Stage files
git add file.py

# Unstage files
git restore --staged file.py

# Commit
git commit -m "feat: Add feature"

# Push
git push origin main

# Pull
git pull origin main

# View history
git log --oneline

# View history with graph
git log --oneline --graph
```

---

## 🛠️ Fixing Mistakes

### Mistake: Bad commit message

**Fix:**
```bash
git commit --amend -m "better: Better commit message"
```

**Warning:** Only do this BEFORE pushing!

---

### Mistake: Forgot to add a file

**Fix:**
```bash
git add forgotten_file.py
git commit --amend --no-edit
```

**This adds the file to the last commit**

---

### Mistake: Committed to wrong branch

**Fix:**
```bash
# Save the changes
git stash

# Switch to correct branch
git checkout correct-branch

# Apply the changes
git stash pop

# Now commit
git add .
git commit -m "feat: Add feature"
```

---

### Mistake: Need to undo last commit

**Option 1: Keep changes, undo commit**
```bash
git reset --soft HEAD~1
# Changes still in your files
# Commit is gone
```

**Option 2: Discard everything**
```bash
git reset --hard HEAD~1
# Changes DELETED
# Commit DELETED
# Be careful!
```

---

## 📖 Reading Git History

### View commits:

```bash
# Last 10 commits
git log --oneline -10

# Output:
# a1b2c3d feat: Add user validation
# d4e5f6g fix: Handle empty email
# g7h8i9j test: Add validator tests
```

### See what changed in a commit:

```bash
git show a1b2c3d

# Shows:
# - Commit message
# - Files changed
# - Actual diff
```

### Search history:

```bash
# Find commits with "validation" in message
git log --grep="validation"

# Find commits that changed specific file
git log -- src/validators.py
```

---

## 📋 Commit Message Template

Save this in your notes:

```
<type>: <short description>

<Longer explanation if needed>
<Why was this change necessary?>
<How does it work?>

<References if any>
```

### Example:

```
feat: Add email validation to User model

Users can now provide email during registration.
Validation checks for @ symbol and domain format.
Invalid emails raise ValidationError.

Uses regex pattern from IETF RFC 5322.
```

---

## ✅ Pre-Commit Checklist

Before every commit, check:

- [ ] Tests pass (`pytest`)
- [ ] Code runs without errors
- [ ] Only relevant files staged (`git status`)
- [ ] Reviewed changes (`git diff --staged`)
- [ ] Good commit message
- [ ] One logical change only

---

## 🎯 Quick Reference

### Your Go-To Commands:

```bash
# Every time you sit down:
git pull origin main

# While working:
git status           # What changed?
git diff             # How did it change?

# When ready to commit:
git add file.py      # Stage files
git diff --staged    # Review staged changes
git commit -m "type: description"

# When done:
git push origin main

# If you mess up:
git status           # See the situation
git log --oneline    # See history
# Ask Claude or instructor!
```

---

## 💡 Pro Tips

### Tip #1: Write commit messages for your future self
Ask: "Will I understand this in 3 months?"

### Tip #2: Commit small and often
Better 10 small commits than 1 giant commit

### Tip #3: Use the imperative mood
- ✅ "Add feature"
- ❌ "Added feature"
- ❌ "Adds feature"

### Tip #4: Keep subject line short
Max 50 characters for the first line

### Tip #5: Use the body for details
```
feat: Add caching to API client

Responses are cached for 5 minutes to reduce API calls.
Cache key is request URL + parameters.
Uses Python's lru_cache decorator.
```

---

## 🚀 Level Up Your Git

### Beginner:
```bash
git add .
git commit -m "update"
git push
```

### Intermediate:
```bash
git add specific_files.py
git commit -m "feat: Add specific feature"
git push
```

### Advanced:
```bash
git status  # Check state
git diff    # Review changes
git add src/file.py tests/test_file.py  # Stage related files
git diff --staged  # Verify staged changes
git commit -m "feat: Add well-documented feature

Implements X using Y pattern.
Includes comprehensive tests."
git push origin main
```

**Aim for Advanced!**

---

## 📊 Git for This Project

### Expected Commit Types:

**Week 1:**
```
test: Add test for [model]
feat: Implement [model]
test: Add test for [client]
feat: Implement [client]
docs: Add project README
```

**Week 2:**
```
feat: Add [feature]
test: Add tests for [feature]
fix: Handle [edge case]
refactor: Simplify [function]
docs: Update README with usage
```

**Grading looks for:**
- Atomic commits ✅
- Conventional commit format ✅
- Clear messages ✅
- Frequent commits (20+) ✅

---

## 🆘 When Git Gets Confusing

### First: Don't Panic

```bash
# See where you are
git status

# See what happened
git log --oneline
```

### Ask for Help:

**Claude:**
```
"I tried to [what you did] and got [error message].
Here's my git status: [paste output]
How do I fix this?"
```

**Instructor:**
"Can you help me understand what happened with git?"

### Emergency Reset:

```bash
# If everything is broken and you have backup:
git reset --hard HEAD
# Back to last commit (loses changes!)
```

---

## 📚 Resources

### Learning More:

```bash
# Git help
git help commit
git help log

# Git docs
# https://git-scm.com/doc
```

### GitHub:

- View commit history: Your repo → "Commits"
- Compare changes: Click on any commit
- Review code: See file diffs

---

## ✅ Git Workflow Summary

```bash
# 1. Start
git pull origin main

# 2. Work (repeat)
# - Edit files
# - Run tests
# - See tests pass

# 3. Commit (repeat)
git status
git add changed_files.py
git commit -m "type: description"

# 4. Push
git push origin main

# 5. Repeat!
```

---

**Good git habits = Professional developer!** 🚀

---

*For TDD workflow, see [TDD_WORKFLOW.md](TDD_WORKFLOW.md)*
*For logging, see [LOGGING.md](LOGGING.md)*
