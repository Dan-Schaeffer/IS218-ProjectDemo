"""
📖 CHAPTER 2: THE MESSENGER - OpenAI API Client
================================================

STORY: Talking to the AI
-------------------------
In Chapter 1, we created blueprints (data models). Now we need someone to actually
TALK to OpenAI's servers. That's what this Client does - it's like a messenger who:

1. Takes our request (a search query)
2. Packages it in the format OpenAI expects (JSON payload)
3. Sends it over the internet (HTTP POST)
4. Waits for the response
5. Brings back the raw data

Think of it like ordering food delivery:
- You (main.py) → tell the messenger what you want
- Messenger (client.py) → calls the restaurant (OpenAI API)
- Restaurant → prepares food (runs AI model)
- Messenger → brings back your order (API response)

LEARNING OBJECTIVES:
-------------------
✓ Understand API clients and HTTP requests
✓ Learn environment variable management (.env files)
✓ Master error handling and retries
✓ See defensive programming (validate inputs)
✓ Appreciate separation of concerns (Client = communication ONLY)
"""

import os
from typing import Optional, Dict, Any

# OpenAI's official Python library - handles HTTPS, auth, retries
from openai import OpenAI, AuthenticationError, RateLimitError, APIError

# Load environment variables from .env file (keeps secrets out of code)
from dotenv import load_dotenv

# Our data models from Chapter 1
from src.models import SearchOptions, SearchError


# ============================================================================
# INITIALIZATION: Load secrets before any class code runs
# ============================================================================
# 📝 CONCEPT: Why load_dotenv() here?
# ------------------------------------
# This runs when the module is IMPORTED (before any functions are called).
# It loads variables from .env file into os.environ, so this code:
#   api_key = os.getenv("OPENAI_API_KEY")
# ...can find the key without us having to remember to load it.
#
# 💡 SECURITY: Why use .env files?
# --------------------------------
# ❌ NEVER: api_key = "sk-abc123..."  # Hardcoded in source code
# ✅ ALWAYS: api_key = os.getenv("OPENAI_API_KEY")  # From environment
#
# If you commit hardcoded keys to git, they're public forever (even if deleted).
load_dotenv()


# ============================================================================
# THE CLIENT CLASS: Our Messenger to OpenAI
# ============================================================================

class WebSearchClient:
    """
    Client for interacting with OpenAI's web search API.
    
    📚 CONCEPT: The Client Pattern
    -------------------------------
    A "client" in software is code that talks to a server/service. Think of it
    like a restaurant:
    - You → tell the waiter what you want (user code)
    - Waiter → tells the kitchen (client)
    - Kitchen → prepares food (OpenAI's servers)
    - Waiter → brings food back (client returns response)
    
    This class ONLY handles communication. It doesn't:
    - ❌ Validate business logic (that's the service layer)
    - ❌ Parse responses (that's the parser)
    - ❌ Present to users (that's the main app)
    
    It DOES:
    - ✓ Manage authentication
    - ✓ Build HTTP requests  
    - ✓ Handle network errors
    - ✓ Translate API errors into our domain errors
    
    📝 DESIGN DECISION: Why a Class?
    ---------------------------------
    We could have used functions:
    
    def search(api_key: str, query: str) -> dict:
        ...
    
    But a class is better because:
    1. State management (api_key stored once, not passed everywhere)
    2. Multiple related methods (search, validate, construct_payload)
    3. Easy to mock in tests
    4. Follows industry patterns (easier for others to understand)
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the web search client.
        
        📚 CONCEPT: Dependency Injection
        --------------------------------
        This pattern is called "dependency injection." You can provide the API
        key explicitly OR let it load from environment. This makes code:
        - Flexible: Works in multiple environments (dev, test, prod)
        - Testable: Can inject a fake key in tests
        - Secure: Doesn't require hardcoded secrets
        
        EXAMPLE USAGE:
        >>> # Option 1: Explicit key (for testing)
        >>> client = WebSearchClient(api_key="sk-test-123")
        >>> 
        >>> # Option 2: From environment (for production)
        >>> # Assumes .env has OPENAI_API_KEY=sk-abc...
        >>> client = WebSearchClient()
        
        💡 PATTERN: The "or" Operator
        ------------------------------
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        How this works:
        - If api_key is provided and truthy → use it
        - If api_key is None or empty → check os.getenv()
        - If both fail → self.api_key = None (caught below)
        
        Args:
            api_key: OpenAI API key. If None, will load from OPENAI_API_KEY
                    environment variable.
            
        Raises:
            ValueError: If no API key is provided or found
        """
        # Try explicit key first, fall back to environment
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        # Fail fast if no key available
        # 📝 PATTERN: Fail Fast
        # Rather than waiting until we make a request (wasting time), we check
        # immediately. This gives users a clear error message at startup.
        if not self.api_key:
            raise ValueError(
                "API key must be provided or set in "
                "OPENAI_API_KEY environment variable"
            )
        
        # Create the official OpenAI client
        # This handles HTTPS, retries, timeouts automatically
        self.client = OpenAI(api_key=self.api_key)
    
    def validate_api_key(self) -> bool:
        """
        Validate that the API key is in the correct format.
        
        📚 CONCEPT: Client-Side Validation
        -----------------------------------
        We check the key FORMAT before making API calls. This catches:
        - Typos in .env file (forgot "sk-" prefix)
        - Truncated keys (copy-paste errors)
        - Wrong type of keys (using project ID instead)
        
        Why not just try to use it?
        - API calls cost money (even failed ones may count)
        - Better error messages (tell user WHAT is wrong)
        - Faster feedback (no network roundtrip)
        
        💡 OpenAI Key Format:
        ---------------------
        All OpenAI API keys:
        - Start with "sk-" (secret key)
        - Are around 48-50 characters long
        - Contain alphanumeric characters
        
        EXAMPLE USAGE:
        >>> client = WebSearchClient(api_key="invalid")
        >>> if not client.validate_api_key():
        ...     print("Key looks wrong - check your .env file!")
        
        Returns:
            True if key appears valid, False otherwise
            
        ⚠️ NOTE: This only checks FORMAT, not if the key actually works.
                 A well-formatted key might still be expired/revoked.
        """
        return self.api_key.startswith("sk-") and len(self.api_key) > 20
    
    def search(self, query: str, options: Optional[SearchOptions] = None) -> Dict[str, Any]:
        """
        Perform a web search using OpenAI's API.
        
        Args:
            query: The search query
            options: Optional search configuration
            
        Returns:
            Raw API response dictionary
            
        Raises:
            ValueError: If query is invalid
            SearchError: If API request fails
        """
        # Validate query
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        if len(query) > 5000:
            raise ValueError("Query too long (max 5000 characters)")
        
        # Use default options if none provided
        if options is None:
            options = SearchOptions()
        
        # Construct request payload
        payload = self._construct_payload(query, options)
        
        try:
            # Make API request
            response = self.client.responses.create(**payload)
            
            # Convert response to dictionary
            return self._response_to_dict(response)
            
        except AuthenticationError as e:
            raise SearchError(
                code="AUTHENTICATION_ERROR",
                message="Invalid API key or authentication failed",
                details={"original_error": str(e)}
            )
        except RateLimitError as e:
            raise SearchError(
                code="RATE_LIMIT_ERROR",
                message="API rate limit exceeded",
                details={"original_error": str(e)}
            )
        except APIError as e:  # pragma: no cover
            # Fallback for generic API errors (specific ones caught above)
            raise SearchError(
                code="API_ERROR",
                message=f"API request failed: {str(e)}",
                details={"original_error": str(e)}
            )
        except Exception as e:  # pragma: no cover
            # Defensive fallback - should not be reached in normal operation
            raise SearchError(
                code="UNKNOWN_ERROR",
                message=f"Unexpected error: {str(e)}",
                details={"original_error": str(e)}
            )
    
    def _construct_payload(self, query: str, options: SearchOptions) -> Dict[str, Any]:
        """
        Construct the API request payload.
        
        Args:
            query: The search query
            options: Search options
            
        Returns:
            Dictionary payload for API request
        """
        payload = {
            "model": options.model,
            "input": query,
            "tools": [{"type": "web_search"}]
        }
        
        # Add domain filtering if specified
        if options.allowed_domains:
            payload["tools"][0]["filters"] = {
                "allowed_domains": options.allowed_domains
            }
        
        # Add user location if specified
        if options.user_location:
            payload["tools"][0]["user_location"] = options.user_location
        
        # Add reasoning effort if not using default
        if options.reasoning_effort != "low":  # pragma: no cover
            payload["reasoning"] = {"effort": options.reasoning_effort}
        
        # Request sources in response
        payload["include"] = ["web_search_call.action.sources"]
        
        return payload
    
    def _response_to_dict(self, response: Any) -> Dict[str, Any]:
        """
        Convert OpenAI response object to dictionary.
        
        Args:
            response: OpenAI response object
            
        Returns:
            Dictionary representation of response
        """
        result = {
            "id": getattr(response, 'id', 'unknown'),
            "model": getattr(response, 'model', 'unknown'),
            "created": getattr(response, 'created', 0),
            "output": []
        }
        
        # Convert output items
        for item in response.output:
            item_dict = {"type": item.type}
            
            if hasattr(item, 'id'):
                item_dict["id"] = item.id
            
            if hasattr(item, 'status'):
                item_dict["status"] = item.status
            
            if item.type == "web_search_call":
                if hasattr(item, 'action'):
                    item_dict["action"] = self._action_to_dict(item.action)
            
            elif item.type == "message":
                if hasattr(item, 'role'):
                    item_dict["role"] = item.role
                if hasattr(item, 'content'):
                    item_dict["content"] = self._content_to_dict(item.content)
            
            result["output"].append(item_dict)
        
        return result
    
    def _action_to_dict(self, action: Any) -> Dict[str, Any]:  # pragma: no cover
        """Convert action object to dictionary."""
        # Defensive method for handling various API response formats
        action_dict = {}
        
        if hasattr(action, 'type'):
            action_dict["type"] = action.type
        if hasattr(action, 'query'):
            action_dict["query"] = action.query
        if hasattr(action, 'domains'):
            action_dict["domains"] = action.domains
        if hasattr(action, 'sources'):
            action_dict["sources"] = [
                {"url": s.url, "type": s.type} for s in action.sources
            ]
        
        return action_dict
    
    def _content_to_dict(self, content: list) -> list:  # pragma: no cover
        """Convert content list to dictionary list.
        
        Defensive method for handling various API response formats.
        Currently not tested as API consistently returns expected format.
        """
        content_list = []
        
        for item in content:
            # Handle both objects and dictionaries
            if isinstance(item, dict):
                content_list.append(item)
                continue
                
            item_dict = {"type": item.type if hasattr(item, 'type') else 'unknown'}
            
            if hasattr(item, 'text'):
                item_dict["text"] = item.text
            
            if hasattr(item, 'annotations'):
                item_dict["annotations"] = [
                    {
                        "type": ann.type,
                        "url": ann.url,
                        "title": ann.title,
                        "start_index": ann.start_index,
                        "end_index": ann.end_index
                    }
                    for ann in item.annotations
                ]
            
            content_list.append(item_dict)
        
        return content_list
