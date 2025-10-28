"""
Response parser for OpenAI web search API responses.

This module handles parsing and transforming API responses into domain models.
"""

from datetime import datetime
from typing import Dict, Any, List

from src.models import SearchResult, Citation, Source


class ResponseParser:
    """Parser for OpenAI web search API responses."""
    
    def parse(self, response: Dict[str, Any], query: str) -> SearchResult:
        """
        Parse an OpenAI API response into a SearchResult.
        
        Args:
            response: Raw API response dictionary
            query: The original search query
            
        Returns:
            SearchResult object with parsed data
            
        Raises:
            ValueError: If response structure is invalid
        """
        if not response.get("output"):
            raise ValueError("No output in response")
        
        output_items = response["output"]
        
        # Extract text content and citations
        text = ""
        citations = []
        
        for item in output_items:
            if item.get("type") == "message":
                content = item.get("content", [])
                for content_item in content:
                    if content_item.get("type") == "output_text":
                        text = content_item.get("text", "")
                        annotations = content_item.get("annotations", [])
                        citations = self._extract_citations(annotations)
        
        # Extract sources and search ID
        sources = []
        search_id = ""
        
        for item in output_items:
            if item.get("type") == "web_search_call":
                search_id = item.get("id", "")
                action = item.get("action", {})
                if action:
                    sources = self._extract_sources(action)
        
        return SearchResult(
            query=query,
            text=text,
            citations=citations,
            sources=sources,
            search_id=search_id,
            timestamp=datetime.now()
        )
    
    def _extract_citations(self, annotations: List[Dict[str, Any]]) -> List[Citation]:
        """
        Extract citations from annotations.
        
        Args:
            annotations: List of annotation dictionaries
            
        Returns:
            List of Citation objects
        """
        citations = []
        
        for annotation in annotations:
            if annotation.get("type") == "url_citation":
                citation = Citation(
                    url=annotation.get("url", ""),
                    title=annotation.get("title", ""),
                    start_index=annotation.get("start_index", 0),
                    end_index=annotation.get("end_index", 0)
                )
                citations.append(citation)
        
        return citations
    
    def _extract_sources(self, action: Dict[str, Any]) -> List[Source]:
        """
        Extract sources from search action.
        
        Args:
            action: Search action dictionary
            
        Returns:
            List of Source objects
        """
        sources = []
        
        sources_data = action.get("sources", [])
        for source_data in sources_data:
            source = Source(
                url=source_data.get("url", ""),
                type=source_data.get("type", "web")
            )
            sources.append(source)
        
        return sources
    
    def format_for_display(self, result: SearchResult) -> str:
        """
        Format a SearchResult for display to users.
        
        Args:
            result: SearchResult to format
            
        Returns:
            Formatted string for display
        """
        lines = []
        lines.append("=" * 80)
        lines.append(f"Query: {result.query}")
        lines.append("=" * 80)
        lines.append("")
        lines.append("Result:")
        lines.append(result.text)
        lines.append("")
        
        if result.citations:
            lines.append("Citations:")
            for i, citation in enumerate(result.citations, 1):
                lines.append(f"  [{i}] {citation.title}")
                lines.append(f"      {citation.url}")
        else:
            lines.append("Citations: None")
        
        lines.append("")
        
        if result.sources:
            lines.append(f"Sources ({len(result.sources)} total):")
            for source in result.sources[:5]:  # Show first 5
                lines.append(f"  - {source.url} ({source.type})")
            if len(result.sources) > 5:  # pragma: no cover
                lines.append(f"  ... and {len(result.sources) - 5} more")
        
        lines.append("=" * 80)
        
        return "\n".join(lines)
