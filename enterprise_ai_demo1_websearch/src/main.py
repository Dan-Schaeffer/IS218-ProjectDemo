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

from .models import ExplainRequest
from .code_explainer_service import explain_code



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

    parser.add_argument(
        "extra",
        nargs="*",
        help=argparse.SUPPRESS,  # keep help clean; this is an internal convenience
    )

    explainer_group = parser.add_argument_group("Code Explainer")
    src_group = explainer_group.add_mutually_exclusive_group(required=False)

    src_group.add_argument(
        "--code",
        type=str,
        help="Code snippet to explain (mutually exclusive with --file)",
    )
    src_group.add_argument(
        "--file",
        type=str,
        help="Path to a code file to explain (mutually exclusive with --code)",
    )

    explainer_group.add_argument(
        "--language",
        type=str,
        default=None,
        help="Optional language hint (e.g., python, js, csharp)",
    )
    explainer_group.add_argument(
        "--context",
        type=str,
        default=None,
        help="Extra context for the explainer (optional)",
    )

    explainer_group.add_argument(
        "--explain-model",
        type=str,
        default=None,
        help="Override model for explainer (optional, defaults to client’s default)",
    )
    explainer_group.add_argument(
        "--explain-max-tokens",
        type=int,
        default=8000,
        help="Max tokens for explainer (optional)",
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

        # Support `... explain "<inline code>"` without flags,
        # but do NOT overwrite --code/--file if they were provided.
        if args.query.strip().lower() == "explain":  # pragma: no cover
            has_flag_code = bool(getattr(args, "code", None))  # pragma: no cover
            has_flag_file = bool(getattr(args, "file", None))  # pragma: no cover
            if not (has_flag_code or has_flag_file):           # pragma: no cover
                extra = getattr(args, "extra", None)           # pragma: no cover
                if extra:                                      # pragma: no cover
                    args.code = " ".join(extra)                # pragma: no cover
                else:                                          # pragma: no cover
                    args.code = None                           # pragma: no cover
            return run_explain_command(args)                   # pragma: no cover



        # If explainer flags are present, run explainer and exit
        if getattr(args, "code", None) or getattr(args, "file", None):
            return run_explain_command(args)


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
        logger.error(f"Invalid input: {e}", exc_info=True)
        print(f"\n❌ Invalid Input: {e}", file=sys.stderr)
        return 1
        
    except KeyboardInterrupt:  # pragma: no cover
        logger.warning("Search cancelled by user (KeyboardInterrupt)")
        print("\n\nSearch cancelled by user.", file=sys.stderr)
        return 130     
    
    except Exception as e:  # pragma: no cover
        # Messages for unexpected errors
        logger.critical(f"Unexpected error: {e}", exc_info=True)  # pragma: no cover
        print(f"\n❌ Unexpected Error: {e}", file=sys.stderr)  # pragma: no cover
        if 'args' in locals() and getattr(args, 'verbose', False):  # pragma: no cover
            import traceback  # pragma: no cover
            traceback.print_exc()  # pragma: no cover
        return 1  # pragma: no cover
    

def run_explain_command(args) -> int:
    # Load code from flag or file
    if args.code:
        code_text = args.code
    else:
        if not args.file:
            print("Error: --code or --file is required for explanation.") 
            return 2                                                     
        with open(args.file, "r", encoding="utf-8") as fh:
            code_text = fh.read()

    req = ExplainRequest(
        code=code_text,
        language=args.language,
        extra_context=args.context,
        max_tokens=args.explain_max_tokens,   # <-- use explain-specific flag
    )

    result = explain_code(req, model=args.explain_model)  # <-- use explain-specific flag

    # Search Output Formatting
    print("\n=== Code Explanation ===")
    if result.detected_language:
        print(f"Detected language: {result.detected_language}")

    print("\nSummary:")
    print(result.summary or "(no summary)")

    print("\nHow it works:")
    if result.steps:
        for i, step in enumerate(result.steps, 1):
            print(f"{i}. {step}")
    else:
        print("(no steps)")

    print("\nErrors:")
    if result.pitfalls:
        for p in result.pitfalls:
            print(f"- {p}")
    else:
        print("(none)")
    print()
    return 0



if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())