"""
Unit tests for the response parser module.

Tests the parsing and transformation of OpenAI API responses.
"""

import pytest
from unittest.mock import Mock

from src.parser import ResponseParser
from src.models import SearchResult, Citation, Source


@pytest.mark.unit
class TestResponseParser:
    """Test the ResponseParser class."""
    
    def test_parser_initialization(self):
        """Test creating a ResponseParser instance."""
        parser = ResponseParser()
        assert parser is not None
    
    def test_parse_valid_response(self, valid_api_response, sample_query):
        """Test parsing a valid API response."""
        parser = ResponseParser()
        result = parser.parse(valid_api_response, sample_query)
        
        assert isinstance(result, SearchResult)
        assert result.query == sample_query
        assert len(result.text) > 0
        assert len(result.citations) > 0
        assert len(result.sources) > 0
    
    def test_parse_response_extracts_text(self, valid_api_response, sample_query):
        """Test that parser correctly extracts text from response."""
        parser = ResponseParser()
        result = parser.parse(valid_api_response, sample_query)
        
        expected_text = "Recent technology news highlights several exciting developments"
        assert expected_text in result.text
    
    def test_parse_response_extracts_citations(self, valid_api_response, sample_query):
        """Test that parser extracts all citations."""
        parser = ResponseParser()
        result = parser.parse(valid_api_response, sample_query)
        
        assert len(result.citations) == 2
        
        # Check first citation
        citation1 = result.citations[0]
        assert citation1.url == "https://techcrunch.com/2025/10/10/ai-breakthrough"
        assert citation1.title == "Major AI Breakthrough Announced - TechCrunch"
        assert citation1.start_index == 95
        assert citation1.end_index == 150
        
        # Check second citation
        citation2 = result.citations[1]
        assert citation2.url == "https://theverge.com/2025/10/10/innovation"
    
    def test_parse_response_extracts_sources(self, valid_api_response, sample_query):
        """Test that parser extracts all sources."""
        parser = ResponseParser()
        result = parser.parse(valid_api_response, sample_query)
        
        assert len(result.sources) == 2
        assert result.sources[0].url == "https://techcrunch.com/2025/10/10/ai-breakthrough"
        assert result.sources[0].type == "web"
    
    def test_parse_response_with_no_citations(self, no_citations_response, sample_query):
        """Test parsing response with no citations."""
        parser = ResponseParser()
        result = parser.parse(no_citations_response, sample_query)
        
        assert len(result.citations) == 0
        assert len(result.text) > 0
    
    def test_parse_empty_response_raises_error(self, empty_api_response, sample_query):
        """Test that parsing empty response raises an error."""
        parser = ResponseParser()
        
        with pytest.raises(ValueError, match="No output in response"):
            parser.parse(empty_api_response, sample_query)
    
    def test_parse_malformed_response_raises_error(self, sample_query):
        """Test that parsing malformed response raises an error."""
        parser = ResponseParser()
        malformed_response = {"invalid": "structure"}
        
        with pytest.raises(ValueError):
            parser.parse(malformed_response, sample_query)
    
    def test_extract_search_id(self, valid_api_response, sample_query):
        """Test extracting web search call ID."""
        parser = ResponseParser()
        result = parser.parse(valid_api_response, sample_query)
        
        assert result.search_id == "ws_67c9fa0502748190b7dd390736892e100be649c1a5ff9609"
    
    def test_format_for_display(self, valid_api_response, sample_query):
        """Test formatting search result for display."""
        parser = ResponseParser()
        result = parser.parse(valid_api_response, sample_query)
        
        formatted = parser.format_for_display(result)
        
        assert "Query:" in formatted
        assert sample_query in formatted
        assert "Result:" in formatted
        assert result.text in formatted
        assert "Citations:" in formatted
        # Sources may appear as "Sources (X total):" in formatted output
        assert "Sources" in formatted or "sources" in formatted
    
    def test_format_display_with_no_citations(self, no_citations_response, sample_query):
        """Test formatting when there are no citations."""
        parser = ResponseParser()
        result = parser.parse(no_citations_response, sample_query)
        
        formatted = parser.format_for_display(result)
        
        assert "No citations found" in formatted or "Citations: None" in formatted
    
    def test_extract_citations_from_annotations(self):
        """Test helper method to extract citations from annotations."""
        parser = ResponseParser()
        
        annotations = [
            {
                "type": "url_citation",
                "url": "https://example.com",
                "title": "Example",
                "start_index": 0,
                "end_index": 10
            }
        ]
        
        citations = parser._extract_citations(annotations)
        
        assert len(citations) == 1
        assert citations[0].url == "https://example.com"
    
    def test_extract_sources_from_search_action(self):
        """Test helper method to extract sources from search action."""
        parser = ResponseParser()
        
        action = {
            "type": "search",
            "sources": [
                {"url": "https://example.com", "type": "web"},
                {"url": "oai-weather://london", "type": "oai-weather"}
            ]
        }
        
        sources = parser._extract_sources(action)
        
        assert len(sources) == 2
        assert sources[0].type == "web"
        assert sources[1].type == "oai-weather"
