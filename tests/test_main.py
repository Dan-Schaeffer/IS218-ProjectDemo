"""
Integration tests for the main application.

Tests the CLI interface and end-to-end functionality.
"""

import pytest
from unittest.mock import patch, MagicMock
from io import StringIO
import sys

from src.main import main, parse_arguments, display_results
from src.models import SearchResult, Citation, Source
from datetime import datetime


@pytest.mark.integration
class TestMainApplication:
    """Test the main application entry point."""
    
    def test_parse_arguments_basic_query(self):
        """Test parsing basic command line arguments."""
        test_args = ["prog", "What is AI?"]
        
        with patch.object(sys, 'argv', test_args):
            args = parse_arguments()
        
        assert args.query == "What is AI?"
        assert args.model == "gpt-4o-mini"  # default
    
    def test_parse_arguments_with_model(self):
        """Test parsing arguments with custom model."""
        test_args = ["prog", "test query", "--model", "gpt-5"]
        
        with patch.object(sys, 'argv', test_args):
            args = parse_arguments()
        
        assert args.query == "test query"
        assert args.model == "gpt-5"
    
    def test_parse_arguments_with_domains(self):
        """Test parsing arguments with domain filtering."""
        test_args = ["prog", "test", "--domains", "example.com,test.com"]
        
        with patch.object(sys, 'argv', test_args):
            args = parse_arguments()
        
        assert args.domains == "example.com,test.com"
    
    def test_parse_arguments_verbose_mode(self):
        """Test parsing verbose flag."""
        test_args = ["prog", "test", "--verbose"]
        
        with patch.object(sys, 'argv', test_args):
            args = parse_arguments()
        
        assert args.verbose is True
    
    def test_display_results_with_citations(self, mock_datetime):
        """Test displaying search results with citations."""
        result = SearchResult(
            query="test query",
            text="Result text",
            citations=[
                Citation(
                    url="https://example.com",
                    title="Example",
                    start_index=0,
                    end_index=10
                )
            ],
            sources=[Source(url="https://example.com", type="web")],
            search_id="id_123",
            timestamp=mock_datetime
        )
        
        # Capture output
        captured_output = StringIO()
        with patch('sys.stdout', captured_output):
            display_results(result)
        
        output = captured_output.getvalue()
        
        assert "test query" in output
        assert "Result text" in output
        assert "https://example.com" in output
    
    def test_display_results_without_citations(self, mock_datetime):
        """Test displaying results when no citations are found."""
        result = SearchResult(
            query="test query",
            text="No results found",
            citations=[],
            sources=[],
            search_id="id_456",
            timestamp=mock_datetime
        )
        
        captured_output = StringIO()
        with patch('sys.stdout', captured_output):
            display_results(result)
        
        output = captured_output.getvalue()
        
        assert "No results found" in output or "test query" in output
    
    @patch('src.main.SearchService')
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_main_success_flow(self, mock_service_class, mock_datetime):
        """Test successful main execution flow."""
        # Setup mock service
        mock_service = MagicMock()
        result = SearchResult(
            query="AI news",
            text="Latest AI developments",
            citations=[],
            sources=[],
            search_id="test_id",
            timestamp=mock_datetime
        )
        mock_service.search.return_value = result
        mock_service_class.return_value = mock_service
        
        test_args = ["prog", "AI news"]
        
        with patch.object(sys, 'argv', test_args):
            exit_code = main()
        
        assert exit_code == 0
        mock_service.search.assert_called_once()
    
    @patch('src.main.SearchService')
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_main_handles_search_error(self, mock_service_class):
        """Test main handles search errors gracefully."""
        mock_service = MagicMock()
        mock_service.search.side_effect = Exception("Search failed")
        mock_service_class.return_value = mock_service
        
        test_args = ["prog", "test query"]
        
        with patch.object(sys, 'argv', test_args):
            exit_code = main()
        
        assert exit_code != 0
    
    @patch.dict('os.environ', {}, clear=True)
    def test_main_without_api_key(self):
        """Test main exits when no API key is provided."""
        test_args = ["prog", "test query"]
        
        with patch.object(sys, 'argv', test_args):
            exit_code = main()
        
        assert exit_code != 0
    
    @patch('src.main.SearchService')
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_main_with_domain_filtering(self, mock_service_class, mock_datetime):
        """Test main with domain filtering enabled."""
        mock_service = MagicMock()
        result = SearchResult(
            query="test",
            text="result",
            citations=[],
            sources=[],
            search_id="id",
            timestamp=mock_datetime
        )
        mock_service.search.return_value = result
        mock_service_class.return_value = mock_service
        
        test_args = ["prog", "test", "--domains", "example.com"]
        
        with patch.object(sys, 'argv', test_args):
            exit_code = main()
        
        assert exit_code == 0
        # Verify search was called with options containing domains
        call_args = mock_service.search.call_args
        assert call_args is not None
    
    @patch('src.main.SearchService')
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_main_verbose_mode(self, mock_service_class, mock_datetime):
        """Test main with verbose output."""
        mock_service = MagicMock()
        result = SearchResult(
            query="test",
            text="result",
            citations=[],
            sources=[],
            search_id="id",
            timestamp=mock_datetime
        )
        mock_service.search.return_value = result
        mock_service_class.return_value = mock_service
        
        test_args = ["prog", "test", "--verbose"]
        
        captured_output = StringIO()
        with patch.object(sys, 'argv', test_args):
            with patch('sys.stdout', captured_output):
                exit_code = main()
        
        assert exit_code == 0


@pytest.mark.unit
class TestHelperFunctions:
    """Test helper functions in main module."""
    
    def test_format_citations_list(self):
        """Test formatting a list of citations."""
        from src.main import format_citations
        
        citations = [
            Citation("https://example.com", "Example", 0, 10),
            Citation("https://test.com", "Test", 20, 30)
        ]
        
        formatted = format_citations(citations)
        
        assert "example.com" in formatted.lower()
        assert "test.com" in formatted.lower()
    
    def test_format_empty_citations_list(self):
        """Test formatting an empty citations list."""
        from src.main import format_citations
        
        formatted = format_citations([])
        
        assert "no citations" in formatted.lower() or formatted == ""
