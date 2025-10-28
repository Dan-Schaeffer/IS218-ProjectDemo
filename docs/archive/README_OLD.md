# Web Search Demo

A demonstration of OpenAI's Web Search capability using enterprise-grade Python with Test-Driven Development (TDD).

# OpenAI Web Search Demo

**A production-quality Python CLI application showcasing OpenAI's web search capabilities with comprehensive testing and documentation.**

🎓 **Students Start Here:** [DEAR_STUDENTS.md](DEAR_STUDENTS.md) - Direct advice from Claude on working with AI coding tools  
📖 **Tutorial:** [TUTORIAL.md](TUTORIAL.md) - Complete guide on how we built this using TDD and AI  
📚 **All Docs:** [DOCS_INDEX.md](DOCS_INDEX.md) - Complete guide to all documentation

## Features

- 🔍 Real-time web search using OpenAI's Responses API
- 📚 Citation extraction with source attribution
- 🎯 Domain filtering for targeted searches
- 🧪 90%+ test coverage with enterprise-quality code
- 🏗️ Clean architecture with separation of concerns
- 📝 Comprehensive documentation

## Quick Start

### Installation

1. Clone the repository and navigate to the project:
```bash
cd /Users/kwilliams/Desktop/Projects/demo
```

2. Create and activate a virtual environment (if not already done):
```bash
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your OpenAI API key:
```bash
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

### Usage

Basic search:
```bash
python -m src.main "What are the latest AI developments?"
```

Search with specific model:
```bash
python -m src.main "Python 3.12 features" --model gpt-5
```

Search with domain filtering:
```bash
python -m src.main "climate news" --domains bbc.com,cnn.com,theguardian.com
```

Verbose output:
```bash
python -m src.main "latest tech news" --verbose
```

## Running Tests

Run all tests:
```bash
pytest
```

Run with coverage report:
```bash
pytest --cov=src --cov-report=html
```

Run specific test categories:
```bash
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
```

Run tests with verbose output:
```bash
pytest -v
```

## Project Structure

```
demo/
├── src/
│   ├── __init__.py         # Package initialization
│   ├── models.py           # Domain models (SearchOptions, SearchResult, etc.)
│   ├── client.py           # OpenAI API client
│   ├── parser.py           # Response parser
│   ├── search_service.py   # Business logic layer
│   └── main.py             # CLI application
├── tests/
│   ├── conftest.py         # Shared test fixtures
│   ├── test_models.py      # Model tests
│   ├── test_client.py      # Client tests
│   ├── test_parser.py      # Parser tests
│   ├── test_search_service.py  # Service tests
│   ├── test_main.py        # Integration tests
│   └── fixtures/
│       └── sample_responses.json  # Mock API responses
├── docs/
│   ├── architecture.md     # System architecture document
│   └── web_search_openai.md  # OpenAI web search documentation
├── pytest.ini              # Pytest configuration
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Architecture

The application follows a layered architecture with clean separation of concerns:

- **Models Layer**: Domain models and data structures
- **Client Layer**: Direct OpenAI API interaction
- **Parser Layer**: Response transformation and parsing
- **Service Layer**: Business logic and orchestration
- **Application Layer**: CLI interface and user interaction

See [docs/architecture.md](docs/architecture.md) for detailed architecture documentation.

## Test-Driven Development

This project was built using TDD principles:

1. ✅ Tests written first (RED)
2. ✅ Implementation written to pass tests (GREEN)
3. ✅ Code refactored for quality (REFACTOR)
4. ✅ 90%+ test coverage achieved

Test categories:
- **Unit Tests**: Fast, isolated tests for individual components
- **Integration Tests**: Test component interactions
- **Fixtures**: Reusable test data and mocks

## Example Output

```
================================================================================
Query: What are the latest AI developments?
================================================================================

Result:
Recent technology news highlights several exciting developments. A major AI 
breakthrough was announced today, revolutionizing natural language processing[1]. 
Additionally, quantum computing reached a new milestone with improved error 
correction[2].

Citations:
  [1] Major AI Breakthrough Announced - TechCrunch
      https://techcrunch.com/2025/10/10/ai-breakthrough
  [2] Quantum Computing Milestone - The Verge
      https://theverge.com/2025/10/10/innovation

Sources (2 total):
  - https://techcrunch.com/2025/10/10/ai-breakthrough (web)
  - https://theverge.com/2025/10/10/innovation (web)
================================================================================
```

## Configuration

### Environment Variables

Create a `.env` file with:

```bash
OPENAI_API_KEY=sk-your-api-key-here
DEFAULT_MODEL=gpt-4o-mini
LOG_LEVEL=INFO
```

### Command-Line Options

```
usage: main.py [-h] [--model MODEL] [--domains DOMAINS] [--verbose] [--api-key API_KEY] query

positional arguments:
  query                 The search query

optional arguments:
  --model MODEL         OpenAI model to use (default: gpt-4o-mini)
  --domains DOMAINS     Comma-separated list of allowed domains
  --verbose             Enable verbose output
  --api-key API_KEY     OpenAI API key (can also use OPENAI_API_KEY env var)
```

## Development

### Code Quality Standards

- **Style**: PEP 8 compliant
- **Type Hints**: Required for all functions
- **Docstrings**: Google style for all public methods
- **Test Coverage**: Minimum 90%

### Running Quality Checks

Format code (if you have black installed):
```bash
black src/ tests/
```

Type checking (if you have mypy installed):
```bash
mypy src/
```

## 📚 Learning Resources

- **[TUTORIAL.md](TUTORIAL.md)** - 🎓 Complete step-by-step guide for students
- **[docs/architecture.md](docs/architecture.md)** - 📐 System design and TDD approach
- **[QUICKSTART.md](QUICKSTART.md)** - ⚡ Quick reference guide
- **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** - 📊 Project overview and metrics

## License

This project is for demonstration purposes.

## Author

Enterprise Development Team - October 10, 2025
