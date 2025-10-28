"""
Pytest configuration and shared fixtures for the web search demo test suite.

This module provides reusable test fixtures, mock objects, and utilities
for testing the web search application.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any
from unittest.mock import Mock, MagicMock

import pytest
from dotenv import load_dotenv


# Load test environment variables
load_dotenv()


@pytest.fixture
def fixtures_dir() -> Path:
    """Return the path to the test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_responses(fixtures_dir: Path) -> Dict[str, Any]:
    """Load sample API responses from JSON fixture file."""
    with open(fixtures_dir / "sample_responses.json", "r") as f:
        return json.load(f)


@pytest.fixture
def valid_api_response(sample_responses: Dict[str, Any]) -> Dict[str, Any]:
    """Provide a valid web search API response."""
    return sample_responses["valid_search_response"]


@pytest.fixture
def empty_api_response(sample_responses: Dict[str, Any]) -> Dict[str, Any]:
    """Provide an empty API response."""
    return sample_responses["empty_response"]


@pytest.fixture
def no_citations_response(sample_responses: Dict[str, Any]) -> Dict[str, Any]:
    """Provide a response with no citations."""
    return sample_responses["no_citations_response"]


@pytest.fixture
def api_error_401(sample_responses: Dict[str, Any]) -> Dict[str, Any]:
    """Provide a 401 authentication error response."""
    return sample_responses["api_error_401"]


@pytest.fixture
def api_error_429(sample_responses: Dict[str, Any]) -> Dict[str, Any]:
    """Provide a 429 rate limit error response."""
    return sample_responses["api_error_429"]


@pytest.fixture
def mock_openai_client():
    """Provide a mocked OpenAI client for testing."""
    mock_client = MagicMock()
    mock_client.responses = MagicMock()
    mock_client.responses.create = MagicMock()
    return mock_client


@pytest.fixture
def mock_response_object(valid_api_response: Dict[str, Any]):
    """Create a mock response object that mimics OpenAI response structure."""
    mock_response = Mock()
    
    # Set basic attributes
    mock_response.id = valid_api_response["id"]
    mock_response.model = valid_api_response["model"]
    mock_response.created = valid_api_response["created"]
    
    # Mock output items
    mock_response.output = []
    for item in valid_api_response["output"]:
        mock_item = Mock()
        for key, value in item.items():
            setattr(mock_item, key, value)
        mock_response.output.append(mock_item)
    
    # Mock helper methods
    def get_output_text():
        for item in mock_response.output:
            if hasattr(item, 'type') and item.type == 'message':
                for content in item.content:
                    if hasattr(content, 'type') and content.type == 'output_text':
                        return content.text
        return ""
    
    mock_response.get_output_text = get_output_text
    
    return mock_response


@pytest.fixture
def test_api_key() -> str:
    """Provide a test API key."""
    return os.getenv("OPENAI_API_KEY", "sk-test-key-123")


@pytest.fixture
def test_config() -> Dict[str, Any]:
    """Provide test configuration."""
    return {
        "model": "gpt-4o-mini",
        "timeout": 30,
        "max_retries": 3,
        "log_level": "DEBUG"
    }


@pytest.fixture
def sample_query() -> str:
    """Provide a sample search query."""
    return "What are the latest developments in AI?"


@pytest.fixture
def sample_allowed_domains() -> list[str]:
    """Provide sample allowed domains for filtering."""
    return [
        "techcrunch.com",
        "theverge.com",
        "arstechnica.com"
    ]


@pytest.fixture
def temp_env_vars(monkeypatch):
    """Fixture to temporarily set environment variables for tests."""
    def _set_env(**kwargs):
        for key, value in kwargs.items():
            monkeypatch.setenv(key, str(value))
    return _set_env


@pytest.fixture(autouse=True)
def reset_environment(monkeypatch):
    """Automatically reset environment variables after each test."""
    # This ensures tests don't interfere with each other
    yield
    # Cleanup happens automatically with monkeypatch


@pytest.fixture
def mock_datetime():
    """Provide a mock datetime for consistent testing."""
    from datetime import datetime
    return datetime(2025, 10, 10, 12, 0, 0)


# Pytest hooks for custom behavior

def pytest_configure(config):
    """Configure pytest with custom settings."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "api: mark test as requiring real API access"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers and skip conditions."""
    skip_api = pytest.mark.skip(reason="Requires OPENAI_API_KEY environment variable")
    
    for item in items:
        # Skip API tests if no API key is set
        if "api" in item.keywords:
            if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY").startswith("sk-test"):
                item.add_marker(skip_api)
