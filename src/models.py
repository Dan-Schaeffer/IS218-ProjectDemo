"""
📖 CHAPTER 1: THE FOUNDATION - DATA MODELS
==========================================

STORY: Building Blocks of Our Application
------------------------------------------
Imagine you're building a house. Before you start construction, you need blueprints.
In software, these blueprints are called "data models" - they define the shape and
structure of the information flowing through your application.

This is where our story begins. We'll create five essential blueprints:
1. SearchOptions - What the user wants to search for (the request)
2. Citation - Where information came from (the evidence)
3. Source - Which websites were consulted (the references)
4. SearchResult - The complete answer (the response)
5. SearchError - When things go wrong (the exception handling)

LEARNING OBJECTIVES:
-------------------
✓ Understand Python dataclasses (automatic constructors & repr)
✓ Learn type hints for better code documentation
✓ Master properties (computed attributes)
✓ See how custom exceptions work
✓ Appreciate immutability and data validation
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Dict, Any


# ============================================================================
# BLUEPRINT 1: SearchOptions - Configuring the Search
# ============================================================================

@dataclass
class SearchOptions:
    """
    Configuration options for web search requests.
    
    📚 CONCEPT: Dataclasses
    -----------------------
    The @dataclass decorator is Python's way of saying "this is just data."
    It automatically generates __init__, __repr__, and __eq__ methods.
    
    Think of it like filling out a form:
    - model: Which AI model to use (like choosing your search engine)
    - allowed_domains: Only search these websites (like Google site:example.com)
    - user_location: Where you're searching from (for local results)
    - reasoning_effort: How hard the AI should think ("low", "medium", "high")
    
    📝 DESIGN DECISION: Default Values
    ----------------------------------
    Notice the = signs? These are default values. If you don't specify a model,
    it defaults to "gpt-4o-mini" (the fastest, cheapest option for learning).
    
    EXAMPLE USAGE:
    >>> # Use all defaults
    >>> options = SearchOptions()
    >>> 
    >>> # Customize for your needs
    >>> options = SearchOptions(
    ...     model="gpt-4o",
    ...     allowed_domains=["python.org", "docs.python.org"],
    ...     reasoning_effort="high"
    ... )
    """
    
    # Which AI model processes the search (default: fastest option)
    model: str = "gpt-4o-mini"
    
    # Optional: restrict search to specific domains (None = search anywhere)
    allowed_domains: Optional[List[str]] = None
    
    # Optional: user's location for localized results
    user_location: Optional[Dict[str, Any]] = None
    
    # How much computational effort to use ("low", "medium", "high")
    reasoning_effort: str = "low"


# ============================================================================
# BLUEPRINT 2: Citation - Tracking Where Information Came From
# ============================================================================

@dataclass
class Citation:
    """
    Represents a citation from a web search result.
    
    📚 CONCEPT: Academic Integrity Meets Code
    ------------------------------------------
    Just like citing sources in a research paper, our AI tells us exactly
    where each piece of information came from. This builds trust and allows
    fact-checking.
    
    Think of this like highlighting text in a book:
    - url: Which website (the book's title)
    - title: Page title (the chapter name)
    - start_index: Where the quote starts (page number, line 5)
    - end_index: Where the quote ends (page number, line 12)
    
    📝 DESIGN PATTERN: Properties
    ------------------------------
    The @property decorator creates a "computed attribute" - it looks like
    a variable but actually runs code. This is cleaner than calling a method.
    
    Compare:
    ❌ citation.get_length()  # Verbose, feels like a function call
    ✅ citation.length         # Clean, feels like accessing data
    
    EXAMPLE USAGE:
    >>> citation = Citation(
    ...     url="https://python.org",
    ...     title="Python Tutorial",
    ...     start_index=100,
    ...     end_index=250
    ... )
    >>> print(citation.length)  # Computes: 250 - 100 = 150
    150
    >>> print(citation)  # Uses __str__ method
    [Python Tutorial](https://python.org)
    """
    
    # The full URL of the source
    url: str
    
    # Human-readable title of the page
    title: str
    
    # Character position where citation starts in the response text
    start_index: int
    
    # Character position where citation ends in the response text
    end_index: int
    
    @property
    def length(self) -> int:
        """
        Calculate the length of the cited text.
        
        💡 WHY THIS MATTERS:
        Longer citations = more thorough evidence
        Shorter citations = quick facts
        """
        return self.end_index - self.start_index
    
    def __str__(self) -> str:
        """
        String representation using Markdown link format.
        
        📝 CONCEPT: Dunder Methods
        ---------------------------
        Methods with double underscores (__str__, __init__, etc.) are "magic methods"
        that Python calls automatically. __str__ is used when you print() something.
        """
        return f"[{self.title}]({self.url})"


# ============================================================================
# BLUEPRINT 3: Source - The Websites We Consulted
# ============================================================================

@dataclass
class Source:
    """
    Represents a source consulted during web search.
    
    📚 CONCEPT: Bibliography vs Citations
    --------------------------------------
    While Citations tell us WHERE in the text info came from, Sources tell us
    WHICH websites were consulted (even if not directly quoted).
    
    Think of it like writing a research paper:
    - Bibliography: All books you looked at (Sources)
    - Footnotes: Specific quotes from those books (Citations)
    
    🎯 OPENAI SPECIAL SOURCES:
    ---------------------------
    OpenAI has special data sources for real-time info:
    - oai-sports: Live sports scores
    - oai-weather: Current weather data
    - oai-finance: Stock market data
    - web: Regular internet websites
    
    EXAMPLE USAGE:
    >>> # Regular website
    >>> source1 = Source(url="https://wikipedia.org", type="web")
    >>> print(source1.is_special)  # False
    >>> 
    >>> # Special OpenAI source
    >>> source2 = Source(url="", type="oai-weather")
    >>> print(source2.is_special)  # True
    """
    
    # The URL of the source (empty string for special OpenAI sources)
    url: str
    
    # Type of source: 'web', 'oai-sports', 'oai-weather', 'oai-finance', etc.
    type: str
    
    @property
    def is_special(self) -> bool:
        """
        Check if this is a special OpenAI data source.
        
        💡 PATTERN: Boolean Properties
        -------------------------------
        Properties that return True/False should start with "is_", "has_", or "can_"
        This makes code read like English:
        
        ✅ if source.is_special:     # Reads naturally
        ❌ if source.special():       # Less clear
        """
        return self.type.startswith("oai-")
    
    def __str__(self) -> str:
        """Format source for display with type information."""
        return f"{self.url} ({self.type})"


# ============================================================================
# BLUEPRINT 4: SearchResult - The Complete Answer Package
# ============================================================================

@dataclass
class SearchResult:
    """
    Represents the complete result of a web search operation.
    
    📚 CONCEPT: Aggregation - Combining Multiple Data Types
    --------------------------------------------------------
    This is where everything comes together! A SearchResult is like a gift box
    that contains multiple items:
    - The answer text (what you asked for)
    - Citations (proof for claims)
    - Sources (websites consulted)
    - Metadata (tracking information)
    
    🎯 REAL-WORLD ANALOGY:
    ----------------------
    Imagine asking a librarian a question. They return:
    - query: Your original question (so you remember what you asked)
    - text: Their answer in paragraph form
    - citations: Footnotes showing which book each fact came from
    - sources: List of all books they looked at
    - search_id: Transaction number (for tracking)
    - timestamp: When you asked (for your records)
    
    📝 DESIGN: Why Lists?
    ---------------------
    citations and sources are Lists because:
    - One answer might need multiple sources (scholarly rigor)
    - Order matters (primary sources first)
    - Empty list [] is valid (AI gave answer from training data)
    
    EXAMPLE USAGE:
    >>> result = SearchResult(
    ...     query="What is Python?",
    ...     text="Python is a high-level programming language...",
    ...     citations=[citation1, citation2],
    ...     sources=[source1, source2, source3],
    ...     search_id="abc-123",
    ...     timestamp=datetime.now()
    ... )
    >>> 
    >>> # Check if answer is backed by sources
    >>> if result.has_citations:
    ...     print(f"Answer backed by {len(result.citations)} sources")
    """
    
    # The original search query from the user
    query: str
    
    # The AI's answer text (may contain citation markers like [1], [2])
    text: str
    
    # List of all citations supporting claims in the text
    citations: List[Citation]
    
    # List of all sources consulted (broader than citations)
    sources: List[Source]
    
    # Unique identifier for this search (for logging/debugging)
    search_id: str
    
    # When this search was performed (for caching/expiry)
    timestamp: datetime
    
    @property
    def has_citations(self) -> bool:
        """
        Check if the result has any citations.
        
        💡 WHY CHECK THIS?
        ------------------
        Results without citations might be:
        - From the AI's training data (pre-October 2023)
        - General knowledge that doesn't need sources
        - A sign something went wrong (no sources found)
        
        You can use this to add warnings in your UI:
        "⚠️ This answer doesn't cite any sources"
        """
        return len(self.citations) > 0
    
    def __str__(self) -> str:
        """
        Concise string representation for logging.
        
        📝 DESIGN: Why Not Print Everything?
        ------------------------------------
        We only show query and citation count because:
        - Logs should be scannable (not walls of text)
        - You can always access full data via attributes
        - This appears in error messages and debug logs
        """
        return f"SearchResult(query='{self.query}', citations={len(self.citations)})"


# ============================================================================
# BLUEPRINT 5: SearchError - When Things Go Wrong
# ============================================================================

@dataclass
class SearchError(Exception):
    """
    Custom exception for search-related errors.
    
    📚 CONCEPT: Exception Hierarchies
    ---------------------------------
    Python has built-in exceptions (ValueError, TypeError, etc.) but sometimes
    you need your own. By inheriting from Exception, we create a custom error
    type that can carry more information than a simple error message.
    
    🎯 WHY CUSTOM EXCEPTIONS?
    --------------------------
    Compare these two approaches:
    
    ❌ BAD: Generic exception
    raise Exception("Something went wrong with code 401")
    - Hard to catch specific errors
    - No structured data
    - Can't handle different errors differently
    
    ✅ GOOD: Custom exception
    raise SearchError(code="AUTHENTICATION_ERROR", message="Invalid API key")
    - Can catch SearchError specifically
    - Machine-readable error codes
    - Can include debug details
    
    📝 ERROR CODES WE USE:
    ----------------------
    - AUTHENTICATION_ERROR: Invalid API key
    - RATE_LIMIT_ERROR: Too many requests
    - API_ERROR: OpenAI service problems
    - VALIDATION_ERROR: Invalid input data
    - PARSING_ERROR: Couldn't understand API response
    - UNKNOWN_ERROR: Unexpected issues
    
    EXAMPLE USAGE:
    >>> # Raising an error with context
    >>> if not api_key:
    ...     raise SearchError(
    ...         code="AUTHENTICATION_ERROR",
    ...         message="API key is required",
    ...         details={"hint": "Set OPENAI_API_KEY environment variable"}
    ...     )
    >>> 
    >>> # Catching and handling
    >>> try:
    ...     result = search("Python")
    ... except SearchError as e:
    ...     if e.code == "RATE_LIMIT_ERROR":
    ...         print("Slow down! Try again in 60 seconds")
    ...     elif e.code == "AUTHENTICATION_ERROR":
    ...         print("Check your API key")
    ...     else:
    ...         print(f"Error: {e}")
    """
    
    # Machine-readable error code (for programmatic handling)
    code: str
    
    # Human-readable error message (for display to users)
    message: str
    
    # Optional extra information for debugging (logs, stack traces, etc.)
    details: Optional[Dict[str, Any]] = None
    
    def __str__(self) -> str:
        """
        Format error for display.
        
        💡 PATTERN: Structured Error Messages
        --------------------------------------
        Format: [CODE] Message
        Example: [RATE_LIMIT_ERROR] Too many requests
        
        This makes logs searchable:
        $ grep "RATE_LIMIT_ERROR" logs.txt
        """
        error_str = f"[{self.code}] {self.message}"
        if self.details:
            error_str += f" | Details: {self.details}"
        return error_str
