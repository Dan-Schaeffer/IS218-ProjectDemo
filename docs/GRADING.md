# 📊 Grading Rubric

**Total Points: 100**

This rubric evaluates both your final product AND the professional practices you used to create it.

---

## Functionality (30 points)

### Does It Work? (10 points)

| Score | Criteria |
|-------|----------|
| **10** | Works flawlessly, handles edge cases, no crashes |
| **8** | Works well with minor issues, handles most cases |
| **6** | Works but has noticeable bugs or limitations |
| **4** | Partially works, significant issues |
| **2** | Barely functional |
| **0** | Doesn't run or crashes immediately |

**What We Test:**
- Run your application with various inputs
- Try edge cases (empty input, very long input, special characters)
- Check error messages
- Verify it handles failures gracefully

### Uses OpenAI APIs Correctly (10 points)

| Score | Criteria |
|-------|----------|
| **10** | Proper API usage, follows best practices, efficient |
| **8** | Correct usage, minor inefficiencies |
| **6** | Works but not optimal, some issues |
| **4** | Misuses API or has significant problems |
| **2** | Barely integrates API |
| **0** | No API integration or completely wrong |

**What We Check:**
- Proper authentication
- Correct API parameters
- Error handling for API failures
- Efficient API usage (not wasteful)
- Follows OpenAI's guidelines

### Solves Real Problem / Creative Value (10 points)

| Score | Criteria |
|-------|----------|
| **10** | Highly creative, solves real problem, useful |
| **8** | Creative and useful, good execution |
| **6** | Decent idea, acceptable execution |
| **4** | Simple/obvious idea, basic execution |
| **2** | Trivial project, minimal effort |
| **0** | Just copied example without modification |

**Examples:**
- 10: AI tutor that adapts to learning style, with progress tracking
- 8: Recipe generator from food photos with dietary preferences
- 6: Simple chatbot with personality
- 4: Basic text completion interface
- 2: Minimal wrapper around API with no added value

---

## Code Quality (30 points)

### Test Coverage >80% (10 points)

| Score | Criteria |
|-------|----------|
| **10** | 90-100% coverage, comprehensive tests |
| **9** | 85-89% coverage, good tests |
| **8** | 80-84% coverage, decent tests |
| **6** | 70-79% coverage, some gaps |
| **4** | 50-69% coverage, significant gaps |
| **2** | <50% coverage, minimal testing |
| **0** | No tests |

**How to Check:**
```bash
pytest --cov=src --cov-report=term
```

**What Makes Tests "Good":**
- Test actual behavior, not implementation
- Cover edge cases
- Clear test names
- Use fixtures properly
- Test error conditions

### Clean Architecture (10 points)

| Score | Criteria |
|-------|----------|
| **10** | Clear separation, follows patterns, maintainable |
| **8** | Well structured, minor issues |
| **6** | Acceptable structure, some mixing of concerns |
| **4** | Poor organization, hard to follow |
| **2** | Everything in one file, no structure |
| **0** | Unintelligible code |

**We Look For:**
- Separation of concerns (models, API client, business logic, UI)
- Clear file organization
- Logical module structure
- Follows patterns from reference code
- Type hints throughout
- Docstrings on public methods

### Error Handling & Validation (10 points)

| Score | Criteria |
|-------|----------|
| **10** | Comprehensive error handling, helpful messages |
| **8** | Good error handling, clear messages |
| **6** | Basic error handling, acceptable messages |
| **4** | Minimal error handling, cryptic errors |
| **2** | Crashes with unhelpful errors |
| **0** | No error handling |

**Examples:**

❌ **Bad (0-2 points):**
```python
def process(data):
    result = api.call(data)  # Crashes if API fails
    return result
```

⚠️ **Acceptable (6 points):**
```python
def process(data):
    try:
        return api.call(data)
    except Exception as e:
        print(f"Error: {e}")
        return None
```

✅ **Good (10 points):**
```python
def process(data: str) -> Result:
    """Process data using API.
    
    Args:
        data: Input data to process
        
    Returns:
        Processed result
        
    Raises:
        ValidationError: If data is invalid
        APIError: If API call fails
    """
    if not data or not data.strip():
        raise ValidationError("Data cannot be empty")
        
    try:
        return api.call(data)
    except AuthenticationError as e:
        raise APIError(
            code="AUTH_FAILED",
            message="Invalid API key",
            details={"hint": "Check your .env file"}
        ) from e
    except RateLimitError as e:
        raise APIError(
            code="RATE_LIMIT",
            message="Too many requests",
            details={"retry_after": 60}
        ) from e
```

---

## Professional Practices (25 points)

### Test-Driven Development (10 points)

| Score | Criteria |
|-------|----------|
| **10** | Clear TDD workflow, tests written first, evidence in git history |
| **8** | Mostly TDD, occasional deviations |
| **6** | Some TDD, but many tests written after code |
| **4** | Tests exist but clearly written after code |
| **2** | Tests are afterthought, minimal coverage |
| **0** | No evidence of TDD |

**How We Verify:**
```bash
# Look at git history
git log --oneline --all

# We look for commits like:
# test: add tests for user authentication
# feat: implement user authentication
# (test commit BEFORE feat commit = TDD)
```

**Evidence of TDD:**
- Test commits before feature commits
- Tests for features that don't exist yet
- Commit messages like "test: ..." followed by "feat: ..."
- Can explain your TDD process in demo

### Git Commits (5 points)

| Score | Criteria |
|-------|----------|
| **5** | Atomic commits, conventional format, clear history |
| **4** | Good commits, minor issues |
| **3** | Acceptable commits, some poor messages |
| **2** | Poor commit messages or too few commits |
| **1** | One giant commit with everything |
| **0** | No git usage |

**Conventional Commit Format:**
```
feat: add image upload feature
fix: correct API key validation
test: add tests for parser module
docs: update README with examples
chore: update dependencies
refactor: simplify error handling
```

**Good Example:**
```bash
$ git log --oneline
abc1234 docs: add usage examples to README
def5678 feat: implement image analysis
901abc2 test: add tests for image analysis
345def6 feat: add API client for vision
789012a test: add client tests with mocking
bcd3456 chore: initial project structure
```

**Bad Example:**
```bash
$ git log --oneline
xyz9876 fixed stuff
abc1234 more changes
def5678 wip
```

### Logging (5 points)

| Score | Criteria |
|-------|----------|
| **5** | Proper logging levels, helpful messages, structured |
| **4** | Good logging, minor issues |
| **3** | Basic logging, acceptable |
| **2** | Minimal or poor logging |
| **0** | No logging or only print statements |

**Good Logging:**
```python
import logging

logger = logging.getLogger(__name__)

def process_image(image_path: str) -> Result:
    logger.info(f"Processing image: {image_path}")
    
    try:
        logger.debug(f"Image size: {os.path.getsize(image_path)} bytes")
        result = analyze(image_path)
        logger.info(f"Analysis complete: {result.confidence}% confidence")
        return result
    except Exception as e:
        logger.error(f"Failed to process image: {e}", exc_info=True)
        raise
```

### CI/CD Pipeline (5 points)

| Score | Criteria |
|-------|----------|
| **5** | Working GitHub Actions, tests run on push, badges |
| **4** | CI/CD works, minor issues |
| **3** | Basic CI/CD, some problems |
| **2** | CI/CD attempted but broken |
| **0** | No CI/CD |

**We Check:**
- `.github/workflows/` exists
- Tests run automatically
- Status badges in README
- Pipeline passes

---

## Documentation & Demo (15 points)

### README (5 points)

| Score | Criteria |
|-------|----------|
| **5** | Professional, clear setup, examples, screenshots |
| **4** | Good README, minor gaps |
| **3** | Acceptable, covers basics |
| **2** | Minimal, missing key info |
| **0** | No README or template only |

**Must Include:**
- What the project does (1-2 paragraphs)
- How to install/setup
- How to use (with examples)
- Example output (text or screenshot)
- Technologies used
- How to run tests

**Bonus Points:**
- Architecture diagram
- Demo GIF
- API documentation
- Contributing guide

### Code Comments (5 points)

| Score | Criteria |
|-------|----------|
| **5** | Explains WHY, clear decisions, helpful context |
| **4** | Good comments, minor issues |
| **3** | Adequate comments, some obvious ones |
| **2** | Minimal or poor comments |
| **0** | No comments or only comment obvious code |

**Good vs Bad Comments:**

❌ **Bad:**
```python
# Increment counter
counter = counter + 1

# Loop through items
for item in items:
    # Process item
    process(item)
```

✅ **Good:**
```python
# We cache results for 5 minutes to reduce API costs
# while still providing reasonably fresh data
CACHE_TTL = 300

# Vision API has strict rate limits (20 req/min)
# so we batch process images to stay under limit
MAX_BATCH_SIZE = 10
```

### Presentation (5 points)

| Score | Criteria |
|-------|----------|
| **5** | Clear, engaging, good timing, live demo works, answered questions well |
| **4** | Good presentation, minor issues |
| **3** | Acceptable, hit main points |
| **2** | Disorganized or ran out of time |
| **0** | No presentation |

**5-Minute Structure:**
1. **Problem** (30 sec) - What does this solve?
2. **Demo** (2 min) - Show it working LIVE
3. **Technical** (1.5 min) - Explain ONE interesting decision
4. **Testing** (30 sec) - Show coverage/test
5. **Q&A** (30 sec)

**We Evaluate:**
- Stays within time limit
- Live demo (backup if needed)
- Can explain technical decisions
- Answers questions confidently
- Professional presentation

---

## Grade Boundaries

| Grade | Points | What This Looks Like |
|-------|--------|----------------------|
| **A** | 90-100 | Exceeds expectations, production-quality, full TDD, creative |
| **B** | 80-89 | Meets all requirements, solid practices, good quality |
| **C** | 70-79 | Meets basic requirements, works, some tests |
| **D** | 60-69 | Incomplete, poor quality, minimal testing |
| **F** | <60 | Doesn't work, no tests, missing key requirements |

### What Each Grade Means

#### A (90-100): Exceptional
- Application is polished and creative
- 90%+ test coverage with comprehensive tests
- Clear TDD workflow in git history
- Clean, well-structured code
- Professional documentation
- Excellent demo presentation
- Goes beyond requirements

**Example A Project:**
"AI Study Buddy that creates personalized quizzes from uploaded documents, adapts difficulty based on performance, tracks progress over time, 95% coverage, clean architecture, well-documented."

#### B (80-89): Strong
- Application works well
- 80%+ test coverage
- Evidence of TDD usage
- Good code quality
- Clear documentation
- Solid demo
- Meets all requirements

**Example B Project:**
"Recipe generator from food photos, handles various image formats, provides ingredient lists and instructions, 82% coverage, good structure, clear README."

#### C (70-79): Adequate
- Application mostly works
- 70%+ test coverage
- Some tests, may not be TDD
- Acceptable code quality
- Basic documentation
- Demo shows functionality
- Meets minimum requirements

**Example C Project:**
"Simple chatbot with personality, basic conversation handling, 72% coverage, works but limited features, minimal documentation."

#### D (60-69): Below Standard
- Application partially works
- <70% coverage
- Tests written after code
- Poor code organization
- Incomplete documentation
- Weak demo
- Missing key requirements

#### F (<60): Unacceptable
- Doesn't work reliably
- No or minimal tests
- No evidence of TDD
- Poor quality throughout
- No meaningful documentation
- Can't demo or explain work

---

## Bonus Points (up to +5)

### Exceptional Creativity (+2)
Project is highly innovative or solves important problem creatively

### Goes Beyond Requirements (+2)
Significantly exceeds expectations in scope or quality

### Helps Classmates (+1)
Actively helps others learn and debug

### Example:
Base score: 88/100 (B)  
Bonus: +3 (creative project, helped classmates)  
Final: 91/100 (A)

---

## Self-Assessment Checklist

Before submission, verify:

### Functionality
- [ ] Application runs without crashes
- [ ] Handles edge cases and errors gracefully
- [ ] Uses OpenAI API correctly and efficiently
- [ ] Solves a real problem or demonstrates creativity

### Code Quality
- [ ] Test coverage >80% (`pytest --cov=src`)
- [ ] All tests pass (`pytest -v`)
- [ ] Code is well-organized (separate files/modules)
- [ ] Type hints on all functions
- [ ] Docstrings on public methods
- [ ] Comprehensive error handling

### Professional Practices
- [ ] Git history shows TDD workflow
- [ ] Commits are atomic with conventional format
- [ ] Logging implemented with appropriate levels
- [ ] CI/CD pipeline works (tests run on push)
- [ ] No hardcoded secrets (uses .env)

### Documentation
- [ ] README explains what, why, how
- [ ] Setup instructions are clear
- [ ] Usage examples provided
- [ ] Code comments explain decisions (not obvious code)
- [ ] Example output shown

### Demo
- [ ] 5-minute presentation prepared
- [ ] Live demo tested and working
- [ ] Can explain one technical decision
- [ ] Can answer questions about choices
- [ ] Have backup plan if demo fails

---

## Common Mistakes to Avoid

### High-Impact Mistakes
These can cost you 10+ points:

❌ **No tests** - Automatic 30 point deduction  
❌ **Doesn't run** - Automatic 10+ point deduction  
❌ **Hardcoded API keys** - Security issue, 5 point deduction  
❌ **One giant commit** - Not professional, 5 point deduction  
❌ **No README** - Can't evaluate properly, 5 point deduction  

### Medium-Impact Mistakes
These cost 3-5 points each:

⚠️ Poor error messages  
⚠️ No logging  
⚠️ Bad commit messages  
⚠️ Tests clearly written after code  
⚠️ No type hints  

### Low-Impact Mistakes
These cost 1-2 points each:

⚠️ Minor PEP 8 violations  
⚠️ Spelling errors in docs  
⚠️ Missing docstrings on some methods  
⚠️ Demo slightly over time  

---

## Sample Evaluation

**Project:** AI-powered meeting transcriber

| Category | Score | Notes |
|----------|-------|-------|
| Works reliably | 9/10 | Works well, minor formatting issue |
| Uses API correctly | 10/10 | Proper Whisper API usage, efficient |
| Creative/useful | 9/10 | Solves real problem, good features |
| Test coverage | 8/10 | 82% coverage, good tests |
| Clean architecture | 9/10 | Well organized, clear structure |
| Error handling | 8/10 | Good errors, could be more specific |
| TDD workflow | 9/10 | Clear TDD in git history |
| Git commits | 5/5 | Perfect conventional commits |
| Logging | 4/5 | Good logging, missing some debug statements |
| CI/CD | 5/5 | GitHub Actions working, badges shown |
| README | 5/5 | Excellent documentation with examples |
| Code comments | 4/5 | Explains decisions, some obvious comments |
| Presentation | 5/5 | Clear demo, good timing, answered questions |
| **TOTAL** | **90/100** | **Grade: A** |

**Feedback:** Excellent project with production-quality code. Minor improvements in test coverage and error specificity would make it perfect. Great use of TDD and professional practices throughout.

---

## Getting Your Grade

### When
Grades posted within 1 week of Demo Day

### What You'll Receive
1. Point breakdown by category
2. Written feedback on strengths
3. Suggestions for improvement
4. Overall grade

### Appeals
If you believe grading was unfair:
1. Review the rubric carefully
2. Gather specific evidence
3. Email instructor within 48 hours
4. Request meeting to discuss

---

## Questions About Grading?

**Before starting:** Review this rubric, plan accordingly

**During development:** Self-assess using checklist above

**Before demo:** Verify you meet all requirements

**After grading:** Ask for clarification if needed

---

**[← Back to Course Structure](COURSE_STRUCTURE.md)** • **[View Full Course →](COURSE_STRUCTURE.md)** • **[README](../README.md)**
