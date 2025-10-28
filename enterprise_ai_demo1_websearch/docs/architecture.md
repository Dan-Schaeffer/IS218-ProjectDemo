# Web Search Demo - Architecture Document

**Version:** 1.0  
**Date:** October 10, 2025  
**Author:** Enterprise Development Team

---

## 1. Executive Summary

This document outlines the architecture for a Real-Time News Research Assistant that leverages OpenAI's Web Search capability through the Responses API. The system follows Test-Driven Development (TDD) principles and enterprise-grade software engineering practices.

### 1.1 Purpose
To demonstrate the power of OpenAI's web search tool by creating a maintainable, testable, and production-ready application that retrieves real-time information with proper citations.

### 1.2 Key Features
- Real-time web search via OpenAI Responses API
- Citation extraction and formatting
- Comprehensive error handling
- Fully tested with high code coverage
- Clean separation of concerns
- Configuration management

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Application Layer                    │
│  (CLI Interface / Main Entry Point)                     │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                   Service Layer                          │
│  (Business Logic / Search Orchestration)                │
└────────────────────┬────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
┌────────────────────┐  ┌────────────────────┐
│   API Client       │  │  Response Parser   │
│   (OpenAI SDK)     │  │  (Data Transform)  │
└────────────────────┘  └────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│              OpenAI Responses API                        │
│              (External Service)                          │
└─────────────────────────────────────────────────────────┘
```

### 2.2 Component Breakdown

#### 2.2.1 **API Client Layer** (`src/client.py`)
**Responsibility:** Direct interaction with OpenAI API

**Responsibilities:**
- Manage OpenAI API authentication
- Construct web search requests
- Handle API communication
- Manage timeouts and retries
- Validate API responses

**Key Methods:**
- `create_web_search_request(query, model, filters)` → Raw API response
- `validate_api_key()` → bool
- `handle_api_errors(exception)` → Structured error

**Testing Strategy:**
- Mock OpenAI client
- Test error handling (401, 429, 500)
- Validate request construction
- Test timeout scenarios

---

#### 2.2.2 **Search Service Layer** (`src/search_service.py`)
**Responsibility:** Business logic and orchestration

**Responsibilities:**
- Validate user input
- Coordinate between client and parser
- Apply business rules
- Cache results (future enhancement)
- Log search operations

**Key Methods:**
- `search(query: str, options: SearchOptions) → SearchResult`
- `validate_query(query: str) → ValidationResult`
- `apply_domain_filters(domains: List[str]) → FilterConfig`

**Testing Strategy:**
- Unit tests with mocked client
- Test validation logic
- Test error propagation
- Test business rule application

---

#### 2.2.3 **Response Parser** (`src/parser.py`)
**Responsibility:** Transform API responses into domain models

**Responsibilities:**
- Extract text content from responses
- Parse citations and annotations
- Extract search sources
- Format output for display
- Handle malformed responses

**Key Methods:**
- `parse_response(raw_response) → ParsedResult`
- `extract_citations(content) → List[Citation]`
- `extract_sources(response) → List[Source]`
- `format_for_display(parsed_result) → str`

**Testing Strategy:**
- Test with various response structures
- Test citation extraction logic
- Test malformed/missing data handling
- Test edge cases (no citations, empty content)

---

#### 2.2.4 **Main Application** (`src/main.py`)
**Responsibility:** User interface and program flow

**Responsibilities:**
- Parse command-line arguments
- Initialize components
- Handle user interaction
- Display results
- Manage application lifecycle

**Key Methods:**
- `main(args) → int`
- `display_results(result) → None`
- `handle_errors(error) → None`

**Testing Strategy:**
- Integration tests
- Test CLI argument parsing
- Test error display
- Test end-to-end flows

---

## 3. Data Models

### 3.1 Domain Models (`src/models.py`)

```python
@dataclass
class SearchOptions:
    model: str = "gpt-4o-mini"
    allowed_domains: Optional[List[str]] = None
    user_location: Optional[dict] = None
    reasoning_effort: str = "low"

@dataclass
class Citation:
    url: str
    title: str
    start_index: int
    end_index: int

@dataclass
class Source:
    url: str
    type: str  # 'web', 'oai-sports', 'oai-weather', etc.

@dataclass
class SearchResult:
    query: str
    text: str
    citations: List[Citation]
    sources: List[Source]
    search_id: str
    timestamp: datetime

@dataclass
class SearchError:
    code: str
    message: str
    details: Optional[dict] = None
```

---

## 4. Test-Driven Development Approach

### 4.1 TDD Workflow

```
1. Write Test (RED)
   ↓
2. Run Test → FAIL
   ↓
3. Write Minimal Code (GREEN)
   ↓
4. Run Test → PASS
   ↓
5. Refactor (REFACTOR)
   ↓
6. Repeat
```

### 4.2 Test Structure

```
tests/
├── conftest.py              # Shared fixtures
├── test_client.py           # API client tests
├── test_search_service.py   # Service layer tests
├── test_parser.py           # Parser tests
├── test_models.py           # Data model tests
├── test_main.py             # Integration tests
└── fixtures/
    └── sample_responses.json # Mock API responses
```

### 4.3 Test Categories

#### Unit Tests
- Test individual functions in isolation
- Mock all external dependencies
- Fast execution (< 1ms per test)
- High coverage target (>90%)

#### Integration Tests
- Test component interactions
- Mock only external APIs
- Test real data flows
- Validate end-to-end scenarios

#### Contract Tests
- Validate API response structures
- Ensure OpenAI API compatibility
- Test against real API (optional, marked)

### 4.4 Testing Standards

- **Naming Convention:** `test_<component>_<scenario>_<expected_result>`
- **Arrange-Act-Assert Pattern:** Clear test structure
- **One Assertion Per Test:** Focus on single behavior
- **Descriptive Test Names:** Self-documenting tests
- **Fixtures Over Setup:** Reusable test data

---

## 5. Configuration Management

### 5.1 Environment Variables (`.env`)

```bash
OPENAI_API_KEY=sk-...
DEFAULT_MODEL=gpt-4o-mini
LOG_LEVEL=INFO
ENABLE_CACHING=false
REQUEST_TIMEOUT=30
```

### 5.2 Configuration Hierarchy

1. Environment variables (highest priority)
2. `.env` file
3. Default values in code
4. Command-line arguments override all

---

## 6. Error Handling Strategy

### 6.1 Error Categories

1. **User Input Errors** (400-level)
   - Empty query
   - Invalid options
   - Handle with validation

2. **API Errors** (500-level)
   - Authentication failures
   - Rate limits
   - Service unavailable
   - Handle with retries and clear messages

3. **System Errors**
   - Network issues
   - Parsing failures
   - Handle with graceful degradation

### 6.2 Error Response Format

```python
{
    "success": False,
    "error": {
        "code": "RATE_LIMIT_EXCEEDED",
        "message": "API rate limit exceeded. Please try again later.",
        "details": {
            "retry_after": 60
        }
    }
}
```

---

## 7. Code Quality Standards

### 7.1 Python Standards
- **Style Guide:** PEP 8
- **Type Hints:** Mandatory for all functions
- **Docstrings:** Google style for all public methods
- **Max Line Length:** 88 characters (Black formatter)

### 7.2 Quality Metrics
- **Test Coverage:** Minimum 90%
- **Cyclomatic Complexity:** Max 10 per function
- **Code Duplication:** Max 3%
- **Documentation:** All public APIs documented

### 7.3 Tools
- **Linting:** pylint, flake8
- **Formatting:** black, isort
- **Type Checking:** mypy
- **Testing:** pytest with coverage
- **Pre-commit hooks:** Enforce standards

---

## 8. Dependency Management

### 8.1 Core Dependencies
```
openai>=2.0.0          # OpenAI SDK
python-dotenv>=1.0.0   # Environment management
pydantic>=2.0.0        # Data validation
```

### 8.2 Development Dependencies
```
pytest>=8.0.0          # Testing framework
pytest-cov>=4.0.0      # Coverage reporting
pytest-mock>=3.0.0     # Mocking utilities
pytest-asyncio>=0.21   # Async test support
```

---

## 9. Testing Infrastructure

### 9.1 pytest Configuration (`pytest.ini`)

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --verbose
    --cov=src
    --cov-report=html
    --cov-report=term-missing
    --cov-fail-under=90
markers =
    unit: Unit tests
    integration: Integration tests
    slow: Slow running tests
    api: Tests that call real API
```

### 9.2 Test Fixtures (Shared)

- `mock_openai_client`: Mocked OpenAI client
- `sample_response`: Valid API response
- `sample_search_result`: Parsed result object
- `config`: Test configuration
- `temp_env`: Temporary environment variables

---

## 10. Implementation Sequence (TDD)

### Phase 1: Foundation
1. ✅ Create project structure
2. ✅ Write architecture document
3. ✅ Configure pytest
4. ✅ Create data models + tests
5. ✅ Implement models

### Phase 2: Core Components
6. ✅ Write parser tests
7. ✅ Implement parser
8. ✅ Write client tests
9. ✅ Implement client
10. ✅ Verify tests pass

### Phase 3: Business Logic
11. ✅ Write service tests
12. ✅ Implement service
13. ✅ Write integration tests
14. ✅ Verify all tests pass

### Phase 4: Application
15. ✅ Write main app tests
16. ✅ Implement main app
17. ✅ Run full test suite
18. ✅ Achieve >90% coverage

---

## 11. Future Enhancements

### 11.1 Phase 2 Features
- Result caching with Redis
- Rate limiting middleware
- Async/concurrent searches
- Web UI with FastAPI
- Streaming responses

### 11.2 Observability
- Structured logging (JSON)
- Metrics collection (Prometheus)
- Distributed tracing (OpenTelemetry)
- Performance monitoring

### 11.3 Advanced Features
- Custom domain presets (medical, legal, etc.)
- Result comparison across models
- Historical search tracking
- Export to various formats (JSON, CSV, PDF)

---

## 12. Deployment Considerations

### 12.1 Environment Setup
- Python 3.10+ required
- Virtual environment isolation
- Secure API key storage
- Environment-specific configs

### 12.2 CI/CD Pipeline
```
1. Lint & Format Check
2. Type Check (mypy)
3. Run Test Suite
4. Coverage Report
5. Security Scan
6. Build Package
7. Deploy (if main branch)
```

---

## 13. Success Criteria

### 13.1 Functional Requirements
- ✅ Successfully search web with various queries
- ✅ Extract and display citations
- ✅ Handle errors gracefully
- ✅ Configurable via CLI and environment

### 13.2 Non-Functional Requirements
- ✅ 90%+ test coverage
- ✅ All tests pass
- ✅ < 3 second response time (typical)
- ✅ Clear documentation
- ✅ Enterprise code quality

---

## 14. Conclusion

This architecture provides a solid foundation for an enterprise-grade web search demo. By following TDD principles and maintaining clean separation of concerns, we ensure:

1. **Testability:** Every component is easily testable
2. **Maintainability:** Clear structure and documentation
3. **Extensibility:** Easy to add new features
4. **Reliability:** Comprehensive error handling
5. **Quality:** High standards enforced through tests

The system is designed to be a reference implementation that showcases both the power of OpenAI's web search capability and best practices in Python software engineering.
