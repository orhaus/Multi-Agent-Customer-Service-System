"""Tests for the routing layer.

The model call is replaced at llm.generate, so these test the router's own
decisions: what it asks for, and what it does when asking fails. How the call
actually reaches Gemini is covered by test_llm.py.
"""

from unittest.mock import MagicMock, patch

import pytest

from customer_service import llm
from customer_service.config import Settings
from customer_service.router import SYSTEM_PROMPT, THINKING, Router
from customer_service.schemas import Category, Conversation, Message, Route

SETTINGS = Settings.from_env({"GEMINI_API_KEY": "test-gemini-key"})
BILLING = Route(category=Category.BILLING, confidence=0.93, reasoning="refund request")


def conversation(*turns: tuple[str, str]) -> Conversation:
    return Conversation(
        id="c1",
        messages=[Message(role=role, content=content) for role, content in turns],
    )


@pytest.fixture
def generate():
    with patch.object(llm, "generate", return_value=BILLING) as fake:
        yield fake


def route(conv: Conversation) -> Route:
    return Router(client=MagicMock(), settings=SETTINGS).route(conv)


def test_returns_the_route_the_model_produced(generate):
    assert route(conversation(("customer", "I want a refund"))) == BILLING


def test_classifies_the_latest_customer_message_only(generate):
    """History is not sent - the router is deliberately cheap and stateless."""
    route(
        conversation(
            ("customer", "I was charged twice"),
            ("assistant", "Let me look"),
            ("customer", "Actually the app crashes on login"),
        )
    )
    assert generate.call_args.kwargs["messages"] == [
        Message(role="customer", content="Actually the app crashes on login")
    ]


def test_asks_the_router_model_for_a_route_with_minimal_thinking(generate):
    route(conversation(("customer", "invoice?")))
    kwargs = generate.call_args.kwargs
    assert kwargs["model"] == SETTINGS.router_model
    assert kwargs["system"] == SYSTEM_PROMPT
    assert kwargs["schema"] is Route
    assert kwargs["thinking_level"] == THINKING == "MINIMAL"


def test_empty_conversation_is_undecided_without_calling_the_model(generate):
    """Nothing to classify should cost nothing."""
    result = route(Conversation(id="c2"))
    assert result.category is Category.UNKNOWN
    assert result.confidence == 0.0
    generate.assert_not_called()


def test_a_conversation_with_no_customer_turn_is_undecided(generate):
    result = route(conversation(("assistant", "Hello, how can I help?")))
    assert result.category is Category.UNKNOWN
    generate.assert_not_called()


def test_any_model_failure_escalates_instead_of_raising(generate):
    """Quota, network, bad JSON, out-of-range values: all arrive as LLMError."""
    generate.side_effect = llm.LLMError("quota exhausted")
    result = route(conversation(("customer", "I want a refund")))
    assert result.category is Category.UNKNOWN
    assert result.confidence == 0.0
