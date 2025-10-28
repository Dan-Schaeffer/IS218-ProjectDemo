"""
Main application entry point for the web search demo.

This module provides the CLI interface for the web search application.
"""

import os
import sys
import argparse
from typing import List

from dotenv import load_dotenv

from src.search_service import SearchService
from src.parser import ResponseParser
from src.models import SearchOptions, SearchResult, Citation, SearchError
from src.logging_config import setup_logging, get_logger, LogContext


# Load environment variables
load_dotenv()

# Initialize logging
app_logger = setup_logging(
    log_level=os.getenv("LOG_LEVEL", "INFO"),
    log_dir=os.getenv("LOG_DIR", "logs"),
    enable_console=True,
    enable_file=True,
    json_format=os.getenv("LOG_FORMAT", "text").lower() == "json"
)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Web Search Demo - Search the web using OpenAI's API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "What are the latest AI developments?"
  %(prog)s "Python 3.12 new features" --model gpt-5
  %(prog)s "climate news" --domains bbc.com,cnn.com
        """
    )
    
    parser.add_argument(
        "query",
        type=str,
        help="The search query"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model to use (default: gpt-4o-mini)"
    )
    
    parser.add_argument(
        "--domains",
        type=str,
        help="Comma-separated list of allowed domains (e.g., 'example.com,test.com')"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--api-key",
        type=str,
        help="OpenAI API key (can also use OPENAI_API_KEY env var)"
    )
    
    return parser.parse_args()


def display_results(result: SearchResult) -> None:
    """
    Display search results to the user.
    
    Args:
        result: The search result to display
    """
    parser = ResponseParser()
    formatted = parser.format_for_display(result)
    print(formatted)


def format_citations(citations: List[Citation]) -> str:
    """
    Format a list of citations for display.
    
    Args:
        citations: List of citations
        
    Returns:
        Formatted string
    """
    if not citations:
        return "No citations found"
    
    lines = []
    for i, citation in enumerate(citations, 1):
        lines.append(f"[{i}] {citation.title} - {citation.url}")
    
    return "\n".join(lines)


def main() -> int:
    """
    Main application entry point.
    
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    logger = get_logger(__name__)
    
    try:
        # Log application start
        logger.info("Web search application started")
        
        # Parse command line arguments
        args = parse_arguments()
        logger.debug(
            f"Parsed arguments: query='{args.query}', "
            f"model={args.model}, domains={args.domains}"
        )
        
        # Verbose logging
        if args.verbose:  # pragma: no cover
            # Verbose mode - logged but not tested in unit tests
            print(f"Using model: {args.model}")
            print(f"Query: {args.query}")
            if args.domains:
                print(f"Domain filter: {args.domains}")
            print()
        
        # Create search options
        options = SearchOptions(model=args.model)
        logger.debug(f"Created search options: model={options.model}")
        
        if args.domains:
            domain_list = [d.strip() for d in args.domains.split(",")]
            options.allowed_domains = domain_list
            logger.info(f"Domain filtering enabled: {domain_list}")
        
        # Get API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("OPENAI_API_KEY not found in environment")
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        # Initialize service
        logger.debug("Initializing search service")
        service = SearchService(api_key=api_key)
        
        # Perform search
        if args.verbose:
            print("Searching...\n")
        
        logger.info(f"Executing search query: '{args.query}'")
        with LogContext(logger, "Web search", query=args.query, model=args.model):
            result = service.search(args.query, options)
        
        logger.info(f"Search completed: {len(result.citations)} citations found")
        
        # Display results
        display_results(result)
        
        logger.info("Web search application completed successfully")
        return 0
        
    except SearchError as e:  # pragma: no cover
        # Error display - tested via integration tests, not unit tests
        logger.error(f"Search error occurred: {e}", exc_info=True)
        print(f"\n❌ Search Error: {e}", file=sys.stderr)
        return 1
        
    except ValueError as e:  # pragma: no cover
        # Error display - tested via integration tests, not unit tests
        logger.error(f"Invalid input: {e}", exc_info=True)
        print(f"\n❌ Invalid Input: {e}", file=sys.stderr)
        return 1
        
    except KeyboardInterrupt:  # pragma: no cover
        logger.warning("Search cancelled by user (KeyboardInterrupt)")
        print("\n\nSearch cancelled by user.", file=sys.stderr)
        return 130
        
    except Exception as e:  # pragma: no cover
        # Defensive fallback for unexpected errors
        logger.critical(f"Unexpected error: {e}", exc_info=True)
        print(f"\n❌ Unexpected Error: {e}", file=sys.stderr)
        if args.verbose if 'args' in locals() else False:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
