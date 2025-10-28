"""
Web Search Demo - A demonstration of OpenAI's web search capabilities.

This package provides a CLI tool for performing web searches using OpenAI's API.
"""

__version__ = "1.0.0"
__author__ = "Enterprise Development Team"

from src.models import SearchOptions, SearchResult, Citation, Source, SearchError
from src.client import WebSearchClient
from src.parser import ResponseParser
from src.search_service import SearchService

__all__ = [
    "SearchOptions",
    "SearchResult",
    "Citation",
    "Source",
    "SearchError",
    "WebSearchClient",
    "ResponseParser",
    "SearchService",
]
