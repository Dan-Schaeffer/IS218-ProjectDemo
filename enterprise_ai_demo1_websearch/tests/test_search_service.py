"""
Unit tests for the search service module.

Tests the business logic and orchestration layer.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.search_service import SearchService
from src.models import SearchOptions, SearchResult, SearchError, Citation, Source


@pytest.mark.unit
class TestSearchService:
    """Test the SearchService class."""
    
    def test_service_initialization(self, test_api_key):
        """Test creating a SearchService instance."""
        service = SearchService(api_key=test_api_key)
        
        assert service is not None
    
    def test_service_initialization_without_api_key_raises_error(self):
        """Test that creating service without API key raises error."""
        with pytest.raises(ValueError):
            SearchService(api_key=None)
    
    @patch('src.search_service.WebSearchClient')
    @patch('src.search_service.ResponseParser')
    def test_search_success(self, mock_parser_class, mock_client_class,
                           test_api_key, sample_query, valid_api_response,
                           mock_datetime):
        """Test successful search operation."""
        # Setup mocks
        mock_client = MagicMock()
        mock_client.search.return_value = valid_api_response
        mock_client_class.return_value = mock_client
        
        mock_parser = MagicMock()
        expected_result = SearchResult(
            query=sample_query,
            text="Result text",
            citations=[],
            sources=[],
            search_id="test_id",
            timestamp=mock_datetime
        )
        mock_parser.parse.return_value = expected_result
        mock_parser_class.return_value = mock_parser
        
        service = SearchService(api_key=test_api_key)
        result = service.search(sample_query)
        
        assert result is not None
        assert isinstance(result, SearchResult)
        mock_client.search.assert_called_once()
        mock_parser.parse.assert_called_once()
    
    @patch('src.search_service.WebSearchClient')
    def test_search_with_options(self, mock_client_class, test_api_key,
                                 sample_query, sample_allowed_domains):
        """Test search with custom options."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        service = SearchService(api_key=test_api_key)
        options = SearchOptions(
            model="gpt-5",
            allowed_domains=sample_allowed_domains
        )
        
        service.search(sample_query, options)
        
        call_args = mock_client.search.call_args
        assert call_args[0][0] == sample_query
        assert call_args[0][1].model == "gpt-5"
    
    def test_validate_query_success(self, test_api_key):
        """Test validation of a valid query."""
        service = SearchService(api_key=test_api_key)
        
        is_valid = service.validate_query("What is the weather today?")
        
        assert is_valid is True
    
    def test_validate_query_empty_fails(self, test_api_key):
        """Test validation fails for empty query."""
        service = SearchService(api_key=test_api_key)
        
        is_valid = service.validate_query("")
        
        assert is_valid is False
    
    def test_validate_query_whitespace_fails(self, test_api_key):
        """Test validation fails for whitespace-only query."""
        service = SearchService(api_key=test_api_key)
        
        is_valid = service.validate_query("   ")
        
        assert is_valid is False
    
    def test_validate_query_too_long_fails(self, test_api_key):
        """Test validation fails for overly long query."""
        service = SearchService(api_key=test_api_key)
        very_long_query = "a" * 10000
        
        is_valid = service.validate_query(very_long_query)
        
        assert is_valid is False
    
    @patch('src.search_service.WebSearchClient')
    def test_search_with_invalid_query_raises_error(self, mock_client_class, test_api_key):
        """Test that search with invalid query raises error."""
        service = SearchService(api_key=test_api_key)
        
        with pytest.raises(ValueError, match="Invalid query"):
            service.search("")
    
    @patch('src.search_service.WebSearchClient')
    @patch('src.search_service.ResponseParser')
    def test_search_handles_client_error(
        self, mock_parser_class, mock_client_class, test_api_key, sample_query
    ):
        """Test that service properly handles client errors."""
        mock_client = MagicMock()
        mock_client.search.side_effect = SearchError(
            code="API_ERROR",
            message="API request failed"
        )
        mock_client_class.return_value = mock_client
        
        service = SearchService(api_key=test_api_key)
        
        with pytest.raises(SearchError) as exc_info:
            service.search(sample_query)
        
        assert exc_info.value.code == "API_ERROR"
    
    @patch('src.search_service.WebSearchClient')
    @patch('src.search_service.ResponseParser')
    def test_search_handles_parser_error(self, mock_parser_class, mock_client_class,
                                         test_api_key, sample_query, valid_api_response):
        """Test that service properly handles parser errors."""
        mock_client = MagicMock()
        mock_client.search.return_value = valid_api_response
        mock_client_class.return_value = mock_client
        
        mock_parser = MagicMock()
        mock_parser.parse.side_effect = ValueError("Invalid response structure")
        mock_parser_class.return_value = mock_parser
        
        service = SearchService(api_key=test_api_key)
        
        # Service wraps ValueError in SearchError
        with pytest.raises(SearchError):
            service.search(sample_query)
    
    def test_apply_domain_filters(self, test_api_key, sample_allowed_domains):
        """Test applying domain filters to options."""
        service = SearchService(api_key=test_api_key)
        
        options = service.apply_domain_filters(sample_allowed_domains)
        
        assert options.allowed_domains == sample_allowed_domains
    
    def test_apply_domain_filters_validates_domains(self, test_api_key):
        """Test that domain filter validation works."""
        service = SearchService(api_key=test_api_key)
        
        invalid_domains = ["not a url", "http://example.com"]
        
        with pytest.raises(ValueError, match="Invalid domain"):
            service.apply_domain_filters(invalid_domains)
    
    def test_max_domains_limit(self, test_api_key):
        """Test that domain limit is enforced (max 20)."""
        service = SearchService(api_key=test_api_key)
        
        too_many_domains = [f"domain{i}.com" for i in range(25)]
        
        with pytest.raises(ValueError, match="Too many domains"):
            service.apply_domain_filters(too_many_domains)


@pytest.mark.integration
class TestSearchServiceIntegration:
    """Integration tests for SearchService."""
    
    @patch('src.search_service.WebSearchClient')
    @patch('src.search_service.ResponseParser')
    def test_end_to_end_search_flow(self, mock_parser_class, mock_client_class,
                                     test_api_key, sample_query, valid_api_response,
                                     mock_datetime):
        """Test complete search flow from query to result."""
        # Setup realistic mocks
        mock_client = MagicMock()
        mock_client.search.return_value = valid_api_response
        mock_client_class.return_value = mock_client
        
        mock_parser = MagicMock()
        result = SearchResult(
            query=sample_query,
            text="Full result text with information",
            citations=[
                Citation(
                    url="https://example.com",
                    title="Example Article",
                    start_index=0,
                    end_index=20
                )
            ],
            sources=[
                Source(url="https://example.com", type="web")
            ],
            search_id="search_123",
            timestamp=mock_datetime
        )
        mock_parser.parse.return_value = result
        mock_parser_class.return_value = mock_parser
        
        # Execute search
        service = SearchService(api_key=test_api_key)
        search_result = service.search(sample_query)
        
        # Verify complete flow
        assert search_result.query == sample_query
        assert len(search_result.text) > 0
        assert len(search_result.citations) > 0
        assert search_result.citations[0].url == "https://example.com"
        mock_client.search.assert_called_once()
        mock_parser.parse.assert_called_once_with(valid_api_response, sample_query)
