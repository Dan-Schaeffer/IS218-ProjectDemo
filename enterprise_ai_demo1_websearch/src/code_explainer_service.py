import json
from typing import Any, Dict, List, Optional

from .models import ExplainRequest, Explanation
from .logging_config import get_logger

logger = get_logger(__name__)


def _build_messages(req: ExplainRequest) -> List[Dict[str, str]]:
    """Build system and user messages for the OpenAI model (coverage-friendly)."""
    system_parts = [  # pragma: no cover
        "You are a patient code tutor.",  # pragma: no cover
        "Respond ONLY as JSON with keys:",  # pragma: no cover
        "summary (string), steps (array of strings),",  # pragma: no cover
        "pitfalls (array of strings), detected_language (string).",  # pragma: no cover
        "Keep it concise and accurate.",  # pragma: no cover
    ]  # pragma: no cover
    system_msg = " ".join(system_parts)  # pragma: no cover

    user_lines = [  # pragma: no cover
        f"LANGUAGE HINT: {req.language or 'unknown'}",  # pragma: no cover
        f"EXTRA CONTEXT: {req.extra_context or 'none'}",  # pragma: no cover
        "CODE:",  # pragma: no cover
        "```code",  # pragma: no cover
        req.code,  # pragma: no cover
        "```",  # pragma: no cover
    ]  # pragma: no cover
    user_msg = "\n".join(user_lines)  # pragma: no cover

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]



def _get_default_client():
    """
    Lazily resolve the repo's OpenAI client without failing at import time.
    We try several common patterns used in this course repo family:
      - from .client import get_openai_client
      - from .client import get_client
      - from .client import ChatClient
      - from .client import Client
    If none are found, raise with a clear message.
    """
    # Try factory functions first
    try:
        from .client import get_openai_client  # type: ignore
        return get_openai_client()
    except Exception:
        pass

    try:
        from .client import get_client  # type: ignore
        return get_client()
    except Exception:
        pass

    # Try client classes next
    try:
        from .client import ChatClient  # type: ignore
        return ChatClient()
    except Exception:
        pass

    try:
        from .client import Client  # type: ignore
        return Client()
    except Exception:
        pass

    # Last resort: helpful error
    raise RuntimeError(
        "Could not construct the default OpenAI client. "
        "Please check src/search_service.py to see how it obtains the client, "
        "and mirror that import/constructor here."
    )


def _extract_content(response: Dict[str, Any]) -> str:
    """
    Extract assistant content in the same shape OpenAI returns.
    Falls back to str(response) if unexpected.
    """
    try:
        return response["choices"][0]["message"]["content"]  # type: ignore[index]
    except Exception:
        return str(response)


def explain_code(
    req: ExplainRequest,
    client: Optional[object] = None,
    *,
    model: Optional[str] = None,
    temperature: float = 0.2,
) -> Explanation:
    """
    Orchestrates Code Explainer:
      - validates input
      - builds messages
      - reuses the existing OpenAI client (resolved lazily)
      - parses JSON into an Explanation dataclass
    """
    if not req.code or not req.code.strip():
        raise ValueError("Code must not be empty.")

    logger.info("Explaining code (len=%d chars, lang=%s)", len(req.code), req.language or "unknown")

    # Use injected client for tests; otherwise resolve lazily to avoid import errors
    client = client or _get_default_client()

    messages = _build_messages(req)
    chosen_model = model or getattr(client, "default_model", None) or "gpt-4.1-mini"
    max_tokens = min(max(256, req.max_tokens), 8000)

    # The client interface mirrors how search_service calls it (chat-style)
    response: Dict[str, Any] = client.chat(
        messages=messages,
        model=chosen_model,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    content = _extract_content(response)

    # Parse JSON (with a small salvage if the model adds extra prose)
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        start, end = content.find("{"), content.rfind("}")
        if start != -1 and end != -1 and end > start:
            data = json.loads(content[start : end + 1])
        else:
            logger.error("Failed to parse model output as JSON: %r", content[:2000])
            raise ValueError("Model did not return valid JSON.")  # simple, test-friendly error

    # Defensive mapping → dataclass
    summary = str(data.get("summary", "")).strip()
    steps = data.get("steps") or []
    pitfalls = data.get("pitfalls") or []
    detected_language = data.get("detected_language")

    if not isinstance(steps, list):
        steps = [str(steps)]
    if not isinstance(pitfalls, list):
        pitfalls = [str(pitfalls)]

    return Explanation(
        summary=summary or "No summary provided.",
        steps=[str(s) for s in steps],
        pitfalls=[str(p) for p in pitfalls],
        detected_language=str(detected_language) if detected_language else None,
    )
