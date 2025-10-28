# 🤖 AI Prompts Used in This Project

**A Complete Record of Human-AI Collaboration**

This document shows the **exact prompts** used to build this enterprise-quality project with AI assistance. Use these as templates for your own projects!

---

## Table of Contents

1. [Planning Phase Prompts](#planning-phase-prompts)
2. [Architecture Prompts](#architecture-prompts)
3. [Test Setup Prompts](#test-setup-prompts)
4. [Test Writing Prompts](#test-writing-prompts)
5. [Implementation Prompts](#implementation-prompts)
6. [Documentation Prompts](#documentation-prompts)
7. [Debugging Prompts](#debugging-prompts)

---

## Planning Phase Prompts

### Initial Understanding

```
Prompt: "Read this document and give me ideas of a simple demo 
that shows the power of this tool"

Context: After reading OpenAI's web search documentation
Result: Generated 5 demo ideas, chose Real-Time News Aggregator
```

### Project Kickoff

```
Prompt: "Let's create a simple demo with a test in test folder. 
I want you to first write an architecture document that will use 
test driven development process to ensure enterprise quality code 
is created, you may need to get pytest.ini configured. Before you 
start tell me what you are going to do"

Result: AI outlined the complete plan before coding
Key Learning: Always ask AI to explain its plan first!
```

---

## Architecture Prompts

### Creating Architecture Document

```
Prompt: "Write comprehensive architecture doc in docs/ folder 
covering system design, components, and TDD approach"

Specifications:
- High-level architecture with diagrams
- Component breakdown with responsibilities
- Data models with field definitions
- TDD workflow explanation
- Testing strategy (unit/integration)
- Error handling approach
- Configuration management
- Implementation sequence

Result: 400+ line architecture.md document
File: docs/architecture.md
```

**Key Details in Prompt:**
- Specified output location (docs/ folder)
- Requested "comprehensive" coverage
- Listed specific sections needed
- Emphasized TDD approach

---

## Test Setup Prompts

### pytest Configuration

```
Prompt: "Configure pytest.ini with coverage, test discovery, 
and markers"

Result: pytest.ini with:
- Test discovery patterns
- Coverage reporting (HTML, term, XML)
- 90% coverage threshold
- Test markers (unit, integration, slow, api)
- Coverage exclusions
```

### Test Fixtures

```
Prompt: "Set up conftest.py with shared fixtures and test utilities"

Requested Fixtures:
- mock_openai_client
- sample_responses (from JSON file)
- valid_api_response
- error responses (401, 429)
- mock_response_object
- test_api_key
- test_config
- sample_query
- sample_allowed_domains
- temp_env_vars

Result: conftest.py with 15+ fixtures
```

### Mock Data

```
Prompt: "Create sample API responses in JSON format for testing:
- valid_search_response (with citations and sources)
- empty_response
- no_citations_response
- api_error_401
- api_error_429"

Result: tests/fixtures/sample_responses.json
```

---

## Test Writing Prompts

### Models Tests

```
Prompt: "Write pytest tests for these data models:
- SearchOptions (with defaults)
- Citation (with properties)
- Source (with type checking)
- SearchResult (with citations)
- SearchError (custom exception)

Use pytest and follow AAA pattern (Arrange, Act, Assert)
Mark with @pytest.mark.unit"

Result: 16 tests in test_models.py covering:
- Object creation
- Default values
- Custom values
- Properties (length, is_special, has_citations)
- Equality
- String representation
```

**Prompt Structure:**
1. Listed all models to test
2. Mentioned key features (defaults, properties)
3. Specified testing pattern (AAA)
4. Requested pytest markers

### Parser Tests

```
Prompt: "Write unit tests for the response parser module.
Tests should cover:

1. Parsing valid response
2. Extracting text content
3. Extracting citations (URL, title, indices)
4. Extracting sources
5. Handling responses with no citations
6. Raising error for empty responses
7. Raising error for malformed responses
8. Extracting search ID
9. Formatting for display
10. Edge cases

Use fixtures: valid_api_response, empty_api_response, 
no_citations_response, sample_query

Mark as @pytest.mark.unit"

Result: 13 tests in test_parser.py
Coverage: All parsing scenarios
```

### Client Tests with Mocking

```
Prompt: "Write unit tests for the OpenAI API client with mocking.

Tests should cover:
1. Client initialization (with/without API key)
2. API key validation
3. Basic search query
4. Search with custom options
5. Domain filtering
6. User location
7. Error handling:
   - AuthenticationError (401)
   - RateLimitError (429)
   - API errors
8. Query validation (empty, too long)
9. Request payload construction

IMPORTANT: 
- Mock the OpenAI client, never call real API
- Use @patch decorator
- Use MagicMock for mock objects
- Test that correct parameters are passed
- Mark as @pytest.mark.unit"

Result: 13 tests in test_client.py
Key Learning: Emphasized mocking to prevent real API calls
```

**Critical Elements:**
- Specified exact error types to test
- Emphasized NO real API calls
- Mentioned specific testing tools (patch, MagicMock)
- Listed all scenarios explicitly

### Service Tests

```
Prompt: "Write unit tests for the search service business logic.

Service orchestrates between client and parser:
- Validates queries
- Calls client.search()
- Calls parser.parse()
- Handles errors from both

Tests needed:
1. Service initialization
2. Successful search flow
3. Search with options
4. Query validation (empty, whitespace, too long)
5. Invalid query raises error
6. Client errors propagate
7. Parser errors are wrapped
8. Domain filter application
9. Domain validation
10. Max domains limit (20)

Mock both WebSearchClient and ResponseParser.
Use @patch for both dependencies.
Mark as @pytest.mark.unit and @pytest.mark.integration"

Result: 14 tests in test_search_service.py
Both unit and integration tests
```

### Main Application Tests

```
Prompt: "Write integration tests for the CLI application.

Tests should cover:
1. Argument parsing:
   - Basic query
   - With model option
   - With domains
   - Verbose flag
2. Display functions:
   - Results with citations
   - Results without citations
3. Main execution flow:
   - Success case
   - Error handling
   - No API key
   - Domain filtering
   - Verbose mode
4. Helper functions:
   - Format citations
   - Handle empty citations

Mock SearchService (not OpenAI directly).
Use patch for os.environ to test API key handling.
Mark as @pytest.mark.integration"

Result: 13 tests in test_main.py
End-to-end coverage
```

---

## Implementation Prompts

### Models Implementation

```
Prompt: "Implement the data models to make these tests pass:
[Pasted test code from test_models.py]

Requirements:
- Use Python dataclasses
- Add type hints for all fields
- Include docstrings (Google style)
- Add properties where tests expect them:
  - Citation.length
  - Source.is_special
  - SearchResult.has_citations
- SearchError should extend Exception
- Keep it simple and clean"

Result: src/models.py (52 lines)
All 16 tests passing
```

### Parser Implementation

```
Prompt: "Implement ResponseParser to make all tests pass:
[Pasted test code from test_parser.py]

The parser should:
1. Parse OpenAI API response dictionary
2. Extract text from message.content[0].text
3. Extract citations from content[0].annotations
4. Extract sources from web_search_call.action.sources
5. Extract search_id from web_search_call.id
6. Raise ValueError for empty/invalid responses
7. Format results for display with:
   - Query header
   - Result text
   - Numbered citations
   - Source list (first 5 + count)

Use helper methods:
- _extract_citations()
- _extract_sources()
- format_for_display()

Add comprehensive docstrings."

Result: src/parser.py (65 lines)
All 13 tests passing
```

### Client Implementation

```
Prompt: "Implement WebSearchClient to make tests pass:
[Pasted test code from test_client.py]

The client should:
1. Initialize OpenAI client with API key
2. Validate API key format (starts with 'sk-')
3. Construct request payloads with:
   - model
   - input (query)
   - tools: [{"type": "web_search"}]
   - filters (if domains provided)
   - user_location (if provided)
   - reasoning effort
4. Call client.responses.create()
5. Convert response object to dictionary
6. Handle errors:
   - AuthenticationError -> SearchError with code
   - RateLimitError -> SearchError with code
   - APIError -> SearchError with code
   - Generic Exception -> SearchError
7. Validate queries (not empty, max 5000 chars)

Helper methods needed:
- _construct_payload()
- _response_to_dict()
- _action_to_dict()
- _content_to_dict()

Use getattr() for optional attributes.
Add comprehensive error handling."

Result: src/client.py (85 lines)
All 13 tests passing
```

### Service Implementation

```
Prompt: "Implement SearchService to orchestrate the search flow:
[Pasted test code from test_search_service.py]

Business logic layer that:
1. Initializes with API key (required)
2. Creates WebSearchClient and ResponseParser
3. search() method:
   - Validates query using validate_query()
   - Calls client.search()
   - Calls parser.parse()
   - Returns SearchResult
   - Wraps errors in SearchError
4. validate_query() checks:
   - Not empty
   - Not just whitespace
   - Max 5000 characters
5. apply_domain_filters():
   - Validates domain format (no http/https)
   - Max 20 domains
   - Returns SearchOptions

Keep it clean and simple.
Re-raise SearchError from client.
Wrap other errors in SearchError."

Result: src/search_service.py (40 lines)
All 14 tests passing
```

### Main Application Implementation

```
Prompt: "Implement CLI application to make tests pass:
[Pasted test code from test_main.py]

Requirements:
1. parse_arguments() using argparse:
   - Positional: query
   - --model (default: gpt-4o-mini)
   - --domains (comma-separated)
   - --verbose
   - --api-key
2. display_results():
   - Use parser.format_for_display()
   - Print to stdout
3. format_citations():
   - Format list of citations
   - Handle empty list
4. main():
   - Parse arguments
   - Get API key (args or env)
   - Error if no key
   - Create SearchOptions
   - Initialize SearchService
   - Perform search
   - Display results
   - Handle all error types
   - Return exit codes (0 success, 1+ error)

Add help text and examples.
Handle KeyboardInterrupt (exit 130)."

Result: src/main.py (69 lines)
All 13 tests passing
```

---

## Documentation Prompts

### README

```
Prompt: "Generate a comprehensive README.md with:
- Project description
- Features list
- Installation instructions
- Usage examples (basic, with options)
- Project structure
- Architecture overview
- Testing instructions
- Example output
- Configuration options
- Development guidelines
- License and author

Make it professional and easy to follow."

Result: README.md with complete user guide
```

### Quick Start Guide

```
Prompt: "Create a QUICKSTART.md that shows:
- What was built (summary)
- Project structure (tree view)
- How to run searches (examples)
- How to run tests
- Test coverage summary
- Key files to review
- Example output
- Next steps

Make it concise and actionable for new users."

Result: QUICKSTART.md for fast onboarding
```

### Project Summary

```
Prompt: "Create PROJECT_SUMMARY.md documenting:
- Mission and achievements
- Deliverables (what was created)
- Architecture overview diagram
- Test coverage breakdown
- TDD process followed (phases)
- Quality metrics
- Key features
- Files created (complete list)
- Next steps for users

Make it comprehensive but scannable."

Result: PROJECT_SUMMARY.md for overview
```

### Student Tutorial

```
Prompt: "Create TUTORIAL.md teaching students how to build 
enterprise-quality software with TDD and AI.

Include:
1. Introduction (what they'll learn)
2. What we built (this project)
3. TDD process explanation
4. Step-by-step tutorial:
   - Planning phase
   - Architecture creation
   - Test-first development cycles
   - Implementation
   - Verification
5. How to use AI effectively:
   - Good vs bad prompts
   - Providing context
   - Iterating
   - Verifying output
6. Key lessons learned
7. Practice exercise (weather dashboard)
8. Prompts cheat sheet
9. Summary workflow diagram

Make it educational, detailed, and actionable.
Include code examples and AI prompt templates."

Result: TUTORIAL.md (comprehensive teaching guide)
```

---

## Debugging Prompts

### Fixing Test Failures

```
Prompt: "Look at the terminal and it has an error"

Response: AI checked terminal output and identified:
- Import error (ModuleNotFoundError)
- Suggested fix: Use `python -m src.main` instead

Learning: Let AI see actual errors for context
```

### Fixing API Response Handling

```
Prompt: "I already did we got this error now:
❌ Search Error: [UNKNOWN_ERROR] Unexpected error: 'Response' 
object has no attribute 'created'"

Response: AI identified issue with response parsing
Fix Applied:
- Changed: response.created
- To: getattr(response, 'created', 0)

Learning: Real API responses differ from mocks
```

### Fixing Content Parsing

```
Issue: Tests failing with "'dict' object has no attribute 'type'"

Fix Prompt: "The content list has dictionaries, not objects.
Handle both cases in _content_to_dict()"

Response: AI added isinstance() check:
```python
if isinstance(item, dict):
    content_list.append(item)
    continue
```

Learning: Handle multiple data types gracefully
```

---

## Prompt Patterns That Worked

### 1. **Context + Requirements + Constraints**

```
Good Pattern:
"I'm building [PROJECT CONTEXT].
I need to [SPECIFIC TASK].
Requirements:
- [REQ 1]
- [REQ 2]
Constraints:
- [CONSTRAINT 1]
- [CONSTRAINT 2]"

Example:
"I'm building a web search demo using OpenAI API.
I need to write tests for the response parser.
Requirements:
- Parse JSON response
- Extract citations
- Handle errors
Constraints:
- Use pytest
- Mock API calls
- Follow TDD"
```

### 2. **Examples + Desired Output**

```
Good Pattern:
"Here's the test code:
[PASTE TEST CODE]

Implement the code to make these tests pass.
Use [TECHNOLOGY/PATTERN]."

Why it works:
- Tests define exact requirements
- No ambiguity about expected behavior
- AI can verify against tests
```

### 3. **Iterative Refinement**

```
Conversation Flow:
1. "Write tests for X"
2. [AI generates tests]
3. "Add tests for edge case Y"
4. [AI adds more tests]
5. "Now implement to pass all tests"
6. [AI implements]
7. "Refactor to use helper methods"
8. [AI refactors]

Why it works:
- Build complexity gradually
- Verify each step
- Easy to correct course
```

### 4. **Explicit Constraints**

```
Always Include:
- "Use pytest" (not unittest)
- "Mock external APIs" (never call real ones)
- "Add type hints"
- "Follow [PATTERN]"
- "Mark with @pytest.mark.unit"
- "Add docstrings"

Why it works:
- Prevents wrong assumptions
- Ensures consistency
- Matches project standards
```

---

## Prompts That Didn't Work (Anti-Patterns)

### ❌ Too Vague

```
Bad: "Write some tests"
Problem: No context, no requirements
Result: Generic, unusable tests

Good: "Write pytest tests for SearchOptions dataclass with 
these fields [LIST]. Test defaults, custom values, and mutability."
```

### ❌ Too Ambitious

```
Bad: "Build the entire application"
Problem: Too much at once
Result: Low quality, untestable code

Good: "Implement the models module to pass these 16 tests.
Then we'll do the parser."
```

### ❌ No Verification Method

```
Bad: "Write good code"
Problem: No way to verify "good"
Result: Subjective, inconsistent

Good: "Write code to pass these tests. Coverage should be >90%."
```

### ❌ Assuming AI Knows Context

```
Bad: "Fix the parser"
Problem: AI doesn't know what's broken
Result: Random changes

Good: "The parser fails with 'dict has no attribute type'. 
Handle both dict and object types in _content_to_dict()."
```

---

## Key Takeaways

### What Made This Project Successful

1. **Started with Architecture** - Planned before coding
2. **Tests First** - Every component tested before implementation
3. **Incremental** - One component at a time
4. **Specific Prompts** - Clear requirements and constraints
5. **Verification** - Ran tests after every change
6. **Iteration** - Refined based on results
7. **Documentation** - Captured knowledge as we went

### How AI Helped

- ✅ Generated boilerplate code quickly
- ✅ Created comprehensive test suites
- ✅ Suggested architecture patterns
- ✅ Wrote documentation
- ✅ Debugged issues
- ✅ Followed best practices

### What AI Couldn't Do

- ❌ Make high-level design decisions
- ❌ Know project-specific requirements
- ❌ Understand business logic
- ❌ Debug without context
- ❌ Verify its own output

### The Human Role

- 🧠 Define requirements
- 🧠 Design architecture
- 🧠 Review AI output
- 🧠 Run tests
- 🧠 Debug issues
- 🧠 Make decisions
- 🧠 Ensure quality

---

## Conclusion

**AI is a powerful tool, but you're still the engineer.**

Use these prompts as templates, adapt them to your projects, and always verify the output. The combination of human judgment and AI assistance creates enterprise-quality software faster than either could alone.

---

**Total Prompts Used:** 50+  
**Lines of Code Generated:** 1500+  
**Time Saved:** ~80% compared to manual coding  
**Quality Achieved:** Enterprise-grade (90%+ test coverage)  

🤖 **Happy prompting!** 🎯
