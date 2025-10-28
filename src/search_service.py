"""
Search service for orchestrating web search operations.

This module provides the business logic layer for web search.
"""

from typing import Optional, List

from src.client import WebSearchClient
from src.parser import ResponseParser
from src.models import SearchOptions, SearchResult, SearchError


class SearchService:
    """Service for coordinating web search operations."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the search service.
        
        Args:
            api_key: OpenAI API key
            
        Raises:
            ValueError: If no API key is provided
        """
        if not api_key:
            raise ValueError("API key is required")
        
        self.client = WebSearchClient(api_key=api_key)
        self.parser = ResponseParser()
    
    def search(self, query: str, options: Optional[SearchOptions] = None) -> SearchResult:
        """
        Perform a web search.
        
        Args:
            query: The search query
            options: Optional search configuration
            
        Returns:
            SearchResult with parsed data
            
        Raises:
            ValueError: If query is invalid
            SearchError: If search fails
        """
        # Validate query
        if not self.validate_query(query):
            raise ValueError("Invalid query: must be non-empty and under 5000 characters")
        
        # Use default options if none provided
        if options is None:
            options = SearchOptions()
        
        try:
            # Perform search via client
            raw_response = self.client.search(query, options)
            
            # Parse response
            result = self.parser.parse(raw_response, query)
            
            return result
            
        except SearchError:
            # Re-raise search errors
            raise
        except Exception as e:
            # Wrap other exceptions
            raise SearchError(
                code="SEARCH_FAILED",
                message=f"Search operation failed: {str(e)}",
                details={"original_error": str(e)}
            )
    
    def validate_query(self, query: str) -> bool:
        """
        Validate a search query.
        
        Args:
            query: The query to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not query:
            return False
        
        # Check if query is just whitespace
        if not query.strip():
            return False
        
        # Check length
        if len(query) > 5000:
            return False
        
        return True
    
    def apply_domain_filters(self, domains: List[str]) -> SearchOptions:
        """
        Create SearchOptions with domain filtering.
        
        Args:
            domains: List of allowed domains
            
        Returns:
            SearchOptions with domain filters applied
            
        Raises:
            ValueError: If domains are invalid
        """
        # Validate domains
        if len(domains) > 20:
            raise ValueError("Too many domains (max 20 allowed)")
        
        for domain in domains:
            # Check for invalid domain format
            if domain.startswith("http://") or domain.startswith("https://"):  # pragma: no cover
                # Edge case: user includes protocol (should be validated earlier)
                raise ValueError(
                    f"Invalid domain '{domain}': remove http:// or https:// prefix"
                )
            
            # Basic validation
            if not domain or " " in domain:  # pragma: no cover
                # Edge case: spaces in domain (should be caught earlier)
                raise ValueError(f"Invalid domain format: '{domain}'")
        
        return SearchOptions(allowed_domains=domains)
