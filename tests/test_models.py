"""
Unit tests for data models.

Tests the domain models used throughout the application.
"""

from datetime import datetime
from typing import List

import pytest

from src.models import (
    SearchOptions,
    Citation,
    Source,
    SearchResult,
    SearchError
)


@pytest.mark.unit
class TestSearchOptions:
    """Test the SearchOptions data model."""
    
    def test_create_default_search_options(self):
        """Test creating SearchOptions with default values."""
        options = SearchOptions()
        
        assert options.model == "gpt-4o-mini"
        assert options.allowed_domains is None
        assert options.user_location is None
        assert options.reasoning_effort == "low"
    
    def test_create_custom_search_options(self):
        """Test creating SearchOptions with custom values."""
        options = SearchOptions(
            model="gpt-5",
            allowed_domains=["example.com"],
            user_location={"city": "London", "country": "GB"},
            reasoning_effort="high"
        )
        
        assert options.model == "gpt-5"
        assert options.allowed_domains == ["example.com"]
        assert options.user_location["city"] == "London"
        assert options.reasoning_effort == "high"
    
    def test_search_options_immutability(self):
        """Test that SearchOptions fields can be modified (dataclass is mutable by default)."""
        options = SearchOptions()
        options.model = "gpt-5"
        
        assert options.model == "gpt-5"


@pytest.mark.unit
class TestCitation:
    """Test the Citation data model."""
    
    def test_create_citation(self):
        """Test creating a Citation with all required fields."""
        citation = Citation(
            url="https://example.com",
            title="Example Article",
            start_index=100,
            end_index=200
        )
        
        assert citation.url == "https://example.com"
        assert citation.title == "Example Article"
        assert citation.start_index == 100
        assert citation.end_index == 200
    
    def test_citation_equality(self):
        """Test that two citations with same data are equal."""
        citation1 = Citation(
            url="https://example.com",
            title="Test",
            start_index=0,
            end_index=10
        )
        citation2 = Citation(
            url="https://example.com",
            title="Test",
            start_index=0,
            end_index=10
        )
        
        assert citation1 == citation2
    
    def test_citation_length_property(self):
        """Test calculating the length of cited text."""
        citation = Citation(
            url="https://example.com",
            title="Test",
            start_index=100,
            end_index=150
        )
        
        assert citation.length == 50


@pytest.mark.unit
class TestSource:
    """Test the Source data model."""
    
    def test_create_web_source(self):
        """Test creating a web source."""
        source = Source(
            url="https://techcrunch.com/article",
            type="web"
        )
        
        assert source.url == "https://techcrunch.com/article"
        assert source.type == "web"
    
    def test_create_special_source(self):
        """Test creating a special OpenAI source."""
        source = Source(
            url="oai-weather://london",
            type="oai-weather"
        )
        
        assert source.url == "oai-weather://london"
        assert source.type == "oai-weather"
    
    def test_source_is_special_property(self):
        """Test identifying special OpenAI sources."""
        web_source = Source(url="https://example.com", type="web")
        special_source = Source(url="oai-sports://data", type="oai-sports")
        
        assert not web_source.is_special
        assert special_source.is_special


@pytest.mark.unit
class TestSearchResult:
    """Test the SearchResult data model."""
    
    def test_create_search_result(self, mock_datetime):
        """Test creating a complete SearchResult."""
        citations = [
            Citation(
                url="https://example.com",
                title="Example",
                start_index=0,
                end_index=10
            )
        ]
        sources = [
            Source(url="https://example.com", type="web")
        ]
        
        result = SearchResult(
            query="test query",
            text="Result text",
            citations=citations,
            sources=sources,
            search_id="search_123",
            timestamp=mock_datetime
        )
        
        assert result.query == "test query"
        assert result.text == "Result text"
        assert len(result.citations) == 1
        assert len(result.sources) == 1
        assert result.search_id == "search_123"
        assert result.timestamp == mock_datetime
    
    def test_search_result_with_no_citations(self, mock_datetime):
        """Test creating SearchResult with empty citations."""
        result = SearchResult(
            query="test",
            text="No sources found",
            citations=[],
            sources=[],
            search_id="search_456",
            timestamp=mock_datetime
        )
        
        assert len(result.citations) == 0
        assert len(result.sources) == 0
    
    def test_search_result_has_citations_property(self, mock_datetime):
        """Test property to check if result has citations."""
        result_with_citations = SearchResult(
            query="test",
            text="text",
            citations=[Citation("url", "title", 0, 10)],
            sources=[],
            search_id="id",
            timestamp=mock_datetime
        )
        
        result_without_citations = SearchResult(
            query="test",
            text="text",
            citations=[],
            sources=[],
            search_id="id",
            timestamp=mock_datetime
        )
        
        assert result_with_citations.has_citations
        assert not result_without_citations.has_citations


@pytest.mark.unit
class TestSearchError:
    """Test the SearchError data model."""
    
    def test_create_basic_error(self):
        """Test creating a basic error."""
        error = SearchError(
            code="API_ERROR",
            message="Something went wrong"
        )
        
        assert error.code == "API_ERROR"
        assert error.message == "Something went wrong"
        assert error.details is None
    
    def test_create_error_with_details(self):
        """Test creating an error with additional details."""
        error = SearchError(
            code="RATE_LIMIT",
            message="Rate limit exceeded",
            details={"retry_after": 60}
        )
        
        assert error.code == "RATE_LIMIT"
        assert error.details["retry_after"] == 60
    
    def test_error_string_representation(self):
        """Test string representation of error."""
        error = SearchError(
            code="TEST_ERROR",
            message="Test message"
        )
        
        error_str = str(error)
        assert "TEST_ERROR" in error_str
        assert "Test message" in error_str
