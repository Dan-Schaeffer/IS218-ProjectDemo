import json
import sys
import types
import pytest

from src.models import ExplainRequest, Explanation
from src import code_explainer_service as ces


# ---------------------------
# Helpers / Fakes
# ---------------------------
class FakeClient:
    def __init__(self, content_json=None, *, expose_default_model=True):
        if expose_default_model:
            self.default_model = "gpt-4.1-mini"
        self.calls = []
        self.content_json = content_json or {
            "summary": "Adds two numbers.",
            "steps": ["Define function", "Add inputs", "Return result"],
            "pitfalls": ["Inputs must be numbers"],
            "detected_language": "python",
        }

    def chat(self, *, messages, model, temperature, max_tokens, **kwargs):
        self.calls.append(
            {"messages": messages, "model": model, "temperature": temperature, "max_tokens": max_tokens}
        )
        return {"choices": [{"message": {"content": json.dumps(self.content_json)}}]}


# ---------------------------
# Core success / validation
# ---------------------------
def test_explain_code_returns_structured_explanation():
    req = ExplainRequest(code="def add(a,b): return a+b", language="python")
    fake = FakeClient()
    out = ces.explain_code(req, client=fake)
    assert isinstance(out, Explanation)
    assert "Adds two numbers" in out.summary
    assert out.steps and isinstance(out.steps, list)
    assert out.detected_language == "python"
    call = fake.calls[0]
    assert call["model"] == "gpt-4.1-mini"
    assert call["max_tokens"] >= 256


def test_explain_code_rejects_empty_input():
    with pytest.raises(ValueError):
        ces.explain_code(ExplainRequest(code="   "))


# ---------------------------
# Branches / error handling
# ---------------------------
def test_json_salvage_path_extra_prose_around_json():
    inner = {
        "summary": "Salvaged.",
        "steps": ["One", "Two"],
        "pitfalls": [],
        "detected_language": "python",
    }

    class SalvageClient:
        def chat(self, **kwargs):
            return {"choices": [{"message": {"content": "NOTE...\n" + json.dumps(inner) + "\nEOF"}}]}

    out = ces.explain_code(ExplainRequest(code="x=1"), client=SalvageClient())
    assert out.summary == "Salvaged."
    assert out.steps == ["One", "Two"]


def test_bad_json_raises_value_error():
    class BadJSONClient:
        def chat(self, **kwargs):
            return {"choices": [{"message": {"content": "not-json"}}]}

    with pytest.raises(ValueError):
        ces.explain_code(ExplainRequest(code="x=1"), client=BadJSONClient())


def test_coerce_nonlist_steps_and_pitfalls():
    payload = {"summary": "Coercion", "steps": "single-step", "pitfalls": "single-pitfall"}
    out = ces.explain_code(ExplainRequest(code="x=1"), client=FakeClient(payload))
    assert out.steps == ["single-step"]
    assert out.pitfalls == ["single-pitfall"]


def test_explicit_model_argument_overrides_client_default():
    client = FakeClient()
    _ = ces.explain_code(ExplainRequest(code="print('hi')"), client=client, model="gpt-4o-mini")
    assert client.calls[-1]["model"] == "gpt-4o-mini"


def test_fallback_model_when_client_has_no_default_model():
    client = FakeClient(expose_default_model=False)
    _ = ces.explain_code(ExplainRequest(code="print('hi')"), client=client)
    assert client.calls[-1]["model"] == "gpt-4.1-mini"  # service fallback


def test_extract_content_fallback_nonstandard_response_shape():
    class OddClient:
        def chat(self, **kwargs):
            return {"foo": "bar"}  # no choices/message/content → will become str(dict)

    with pytest.raises(ValueError):
        ces.explain_code(ExplainRequest(code="x=1"), client=OddClient())


# ---------------------------
# _get_default_client coverage
# ---------------------------
def test_get_default_client_raises_runtimeerror_when_no_known_symbols(monkeypatch):
    # Create a fake 'src.client' with none of the expected names
    fake_client_mod = types.ModuleType("src.client")
    sys.modules["src.client"] = fake_client_mod
    from importlib import reload
    reload(ces)
    with pytest.raises(RuntimeError):
        ces._get_default_client()
    # cleanup
    del sys.modules["src.client"]
    reload(ces)


def test_get_default_client_works_with_factory(monkeypatch):
    # Provide a factory symbol get_openai_client() that returns a sentinel
    sentinel = object()
    fake_client_mod = types.ModuleType("src.client")
    def get_openai_client():
        return sentinel
    fake_client_mod.get_openai_client = get_openai_client
    sys.modules["src.client"] = fake_client_mod
    from importlib import reload
    reload(ces)
    got = ces._get_default_client()
    assert got is sentinel
    # cleanup
    del sys.modules["src.client"]
    reload(ces)


def test_get_default_client_works_with_class(monkeypatch):
    # Provide a class ChatClient that can be constructed
    class ChatClient:
        pass
    
def test__build_messages_formats_system_and_user():
    req = ExplainRequest(
        code="def add(a,b): return a+b",
        language="python",
        extra_context="unit test"
    )
    msgs = ces._build_messages(req)
    # two messages in order
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert "Respond ONLY as JSON" in msgs[0]["content"]
    assert msgs[1]["role"] == "user"
    uc = msgs[1]["content"]
    # user content includes the hint, context, and code block
    assert "LANGUAGE HINT: python" in uc
    assert "EXTRA CONTEXT: unit test" in uc
    assert "```code" in uc and "def add(a,b): return a+b" in uc

def test_get_default_client_works_with_get_client(monkeypatch):
    """Covers the 'from .client import get_client' branch."""
    import types, sys
    # Fake src.client with get_client() factory
    fake_client_mod = types.ModuleType("src.client")
    sentinel = object()
    def get_client():
        return sentinel
    fake_client_mod.get_client = get_client
    sys.modules["src.client"] = fake_client_mod

    from importlib import reload
    import src.code_explainer_service as ces
    reload(ces)

    got = ces._get_default_client()
    assert got is sentinel

    # cleanup
    del sys.modules["src.client"]
    reload(ces)


def test_get_default_client_works_with_Client_class(monkeypatch):
    """Covers the 'from .client import Client' class branch."""
    import types, sys
    fake_client_mod = types.ModuleType("src.client")
    class Client:
        pass
    fake_client_mod.Client = Client
    sys.modules["src.client"] = fake_client_mod

    from importlib import reload
    import src.code_explainer_service as ces
    reload(ces)

    got = ces._get_default_client()
    assert isinstance(got, Client)

    # cleanup
    del sys.modules["src.client"]
    reload(ces)

def test_get_default_client_prefers_ChatClient_when_present(monkeypatch):
    """
    Cover the explicit 'return ChatClient()' branch in _get_default_client.
    We provide ONLY ChatClient in src.client (no factories, no Client),
    so the function must land on that return line.
    """
    import sys, types
    from importlib import reload
    import src.code_explainer_service as ces

    # Fake src.client with ONLY ChatClient defined
    fake_client_mod = types.ModuleType("src.client")
    class ChatClient:
        pass
    fake_client_mod.ChatClient = ChatClient
    # Ensure no factories/classes that would short-circuit earlier branches
    for name in ("get_openai_client", "get_client", "Client"):
        if hasattr(fake_client_mod, name):
            delattr(fake_client_mod, name)

    sys.modules["src.client"] = fake_client_mod
    reload(ces)  # make ces see our fake module

    got = ces._get_default_client()
    assert isinstance(got, ChatClient)

    # cleanup
    del sys.modules["src.client"]
    reload(ces)

