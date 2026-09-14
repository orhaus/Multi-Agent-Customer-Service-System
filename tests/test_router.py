"""Tests for the routing layer.

None of these make a real API call. The Anthropic client is replaced with a
mock, so the suite is free, offline, and gives the same answer every run - which
a real model would not.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import anthropic
import httpx2

from customer_service.config import Settings
from customer_service.router import SYSTEM_PROMPT, Router
from customer_service.schemas import Category, Conversation, Message, Route

SETTINGS = Settings.from_env({"ANTHROPIC_API_KEY": "sk-ant-test"})


def fake_client(route: Route | None = None, error: Exception | None = None) -> MagicMock:
    """A stand-in for anthropic.Anthropic that returns `route` or raises `error`."""
    client = MagicMock()
    if error is not None:
        client.messages.parse.side_effect = error
    else:
        client.messages.parse.return_value = SimpleNamespace(parsed_output=route)
    return client


def conversation(*turns: tuple[str, str]) -> Conversation:
    return Conversation(
        id="c1",
        messages=[Message(role=role, content=content) for role, content in turns],
    )


def test_returns_the_route_the_model_produced():
    expected = Route(category=Category.BILLING, confidence=0.93, reasoning="refund request")
    router = Router(client=fake_client(expected), settings=SETTINGS)

    assert router.route(conversation(("customer", "I want a refund"))) == expected


def test_classifies_the_latest_customer_message_only():
    """History is not sent - the router is deliberately cheap and stateless."""
    client = fake_client(Route(category=Category.TECHNICAL, confidence=0.8, reasoning="ok"))
    router = Router(client=client, settings=SETTINGS)

    router.route(
        conversation(
            ("customer", "I was charged twice"),
            ("assistant", "Let me look"),
            ("customer", "Actually the app crashes on login"),
        )
    )

    sent = client.messages.parse.call_args.kwargs["messages"]
    assert sent == [{"role": "user", "content": "Actually the app crashes on login"}]


def test_uses_the_configured_router_model_and_system_prompt():
    client = fake_client(Route(category=Category.BILLING, confidence=0.8, reasoning="ok"))
    Router(client=client, settings=SETTINGS).route(conversation(("customer", "invoice?")))

    kwargs = client.messages.parse.call_args.kwargs
    assert kwargs["model"] == SETTINGS.router_model
    assert kwargs["system"] == SYSTEM_PROMPT
    assert kwargs["output_format"] is Route


def test_empty_conversation_is_undecided_without_calling_the_api():
    """Nothing to classify should cost nothing."""
    client = fake_client(Route(category=Category.BILLING, confidence=1.0, reasoning="x"))
    route = Router(client=client, settings=SETTINGS).route(Conversation(id="c2"))

    assert route.category is Category.UNKNOWN
    assert route.confidence == 0.0
    client.messages.parse.assert_not_called()


def test_a_conversation_with_no_customer_turn_is_undecided():
    client = fake_client(Route(category=Category.BILLING, confidence=1.0, reasoning="x"))
    route = Router(client=client, settings=SETTINGS).route(
        conversation(("assistant", "Hello, how can I help?"))
    )

    assert route.category is Category.UNKNOWN
    client.messages.parse.assert_not_called()


def test_api_failure_escalates_instead_of_raising():
    """An unreachable API must not take the whole request down with it."""
    error = anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.anthropic.com"))
    router = Router(client=fake_client(error=error), settings=SETTINGS)

    route = router.route(conversation(("customer", "I want a refund")))

    assert route.category is Category.UNKNOWN
    assert route.confidence == 0.0

