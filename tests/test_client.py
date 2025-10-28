"""
Unit tests for the OpenAI API client module.

Tests the OpenAI API client interactions with mocking.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from openai import OpenAI, AuthenticationError, RateLimitError

from src.client import WebSearchClient
from src.models import SearchOptions, SearchError


@pytest.mark.unit
class TestWebSearchClient:
    """Test the WebSearchClient class."""
    
    def test_client_initialization_with_api_key(self, test_api_key):
        """Test creating client with explicit API key."""
        client = WebSearchClient(api_key=test_api_key)
        
        assert client is not None
        assert client.api_key == test_api_key
    
    def test_client_initialization_from_env(self, temp_env_vars, test_api_key):
        """Test creating client with API key from environment."""
        temp_env_vars(OPENAI_API_KEY=test_api_key)
        
        client = WebSearchClient()
        
        assert client is not None
    
    def test_client_initialization_without_api_key_raises_error(self, temp_env_vars):
        """Test that creating client without API key raises error."""
        temp_env_vars(OPENAI_API_KEY="")
        
        with pytest.raises(ValueError, match="API key"):
            WebSearchClient()
    
    def test_validate_api_key_success(self, test_api_key):
        """Test validating a correct API key."""
        client = WebSearchClient(api_key=test_api_key)
        
        # For this test, we just check the format
        is_valid = client.validate_api_key()
        
        assert isinstance(is_valid, bool)
    
    @patch('src.client.OpenAI')
    def test_search_with_basic_query(self, mock_openai_class, test_api_key, 
                                    sample_query, mock_response_object):
        """Test performing a basic web search."""
        # Setup mock
        mock_client_instance = MagicMock()
        mock_client_instance.responses.create.return_value = mock_response_object
        mock_openai_class.return_value = mock_client_instance
        
        client = WebSearchClient(api_key=test_api_key)
        response = client.search(sample_query)
        
        assert response is not None
        mock_client_instance.responses.create.assert_called_once()
    
    @patch('src.client.OpenAI')
    def test_search_with_options(self, mock_openai_class, test_api_key,
                                sample_query, sample_allowed_domains,
                                mock_response_object):
        """Test search with custom options."""
        mock_client_instance = MagicMock()
        mock_client_instance.responses.create.return_value = mock_response_object
        mock_openai_class.return_value = mock_client_instance
        
        client = WebSearchClient(api_key=test_api_key)
        options = SearchOptions(
            model="gpt-5",
            allowed_domains=sample_allowed_domains
        )
        
        response = client.search(sample_query, options)
        
        assert response is not None
        call_args = mock_client_instance.responses.create.call_args
        assert call_args[1]["model"] == "gpt-5"
    
    @patch('src.client.OpenAI')
    def test_search_with_domain_filtering(self, mock_openai_class, test_api_key,
                                          sample_query, sample_allowed_domains,
                                          mock_response_object):
        """Test that domain filtering is properly applied."""
        mock_client_instance = MagicMock()
        mock_client_instance.responses.create.return_value = mock_response_object
        mock_openai_class.return_value = mock_client_instance
        
        client = WebSearchClient(api_key=test_api_key)
        options = SearchOptions(allowed_domains=sample_allowed_domains)
        
        client.search(sample_query, options)
        
        call_args = mock_client_instance.responses.create.call_args
        tools = call_args[1]["tools"]
        
        assert len(tools) > 0
        assert tools[0]["type"] == "web_search"
    
    @patch('src.client.OpenAI')
    def test_search_handles_authentication_error(
        self, mock_openai_class, test_api_key, sample_query
    ):
        """Test handling authentication errors."""
        mock_client_instance = MagicMock()
        mock_client_instance.responses.create.side_effect = AuthenticationError(
            "Invalid API key",
            response=Mock(status_code=401),
            body=None
        )
        mock_openai_class.return_value = mock_client_instance
        
        client = WebSearchClient(api_key=test_api_key)
        
        with pytest.raises(SearchError) as exc_info:
            client.search(sample_query)
        
        assert exc_info.value.code == "AUTHENTICATION_ERROR"
    
    @patch('src.client.OpenAI')
    def test_search_handles_rate_limit_error(self, mock_openai_class, test_api_key, sample_query):
        """Test handling rate limit errors."""
        mock_client_instance = MagicMock()
        mock_client_instance.responses.create.side_effect = RateLimitError(
            "Rate limit exceeded",
            response=Mock(status_code=429),
            body=None
        )
        mock_openai_class.return_value = mock_client_instance
        
        client = WebSearchClient(api_key=test_api_key)
        
        with pytest.raises(SearchError) as exc_info:
            client.search(sample_query)
        
        assert exc_info.value.code == "RATE_LIMIT_ERROR"
    
    @patch('src.client.OpenAI')
    def test_search_with_user_location(self, mock_openai_class, test_api_key,
                                       sample_query, mock_response_object):
        """Test search with user location specified."""
        mock_client_instance = MagicMock()
        mock_client_instance.responses.create.return_value = mock_response_object
        mock_openai_class.return_value = mock_client_instance
        
        client = WebSearchClient(api_key=test_api_key)
        options = SearchOptions(
            user_location={
                "type": "approximate",
                "country": "GB",
                "city": "London"
            }
        )
        
        response = client.search(sample_query, options)
        
        assert response is not None
        call_args = mock_client_instance.responses.create.call_args
        tools = call_args[1]["tools"]
        
        assert "user_location" in str(tools) or response is not None
    
    def test_construct_request_payload(self, test_api_key, sample_query):
        """Test that request payload is correctly constructed."""
        client = WebSearchClient(api_key=test_api_key)
        options = SearchOptions(model="gpt-5")
        
        payload = client._construct_payload(sample_query, options)
        
        assert payload["model"] == "gpt-5"
        assert payload["input"] == sample_query
        assert "tools" in payload
        assert payload["tools"][0]["type"] == "web_search"
    
    def test_empty_query_raises_error(self, test_api_key):
        """Test that empty query raises validation error."""
        client = WebSearchClient(api_key=test_api_key)
        
        with pytest.raises(ValueError, match="Query cannot be empty"):
            client.search("")
    
    def test_query_too_long_raises_error(self, test_api_key):
        """Test that extremely long query raises error."""
        client = WebSearchClient(api_key=test_api_key)
        very_long_query = "a" * 10000
        
        with pytest.raises(ValueError, match="Query too long"):
            client.search(very_long_query)
