"""Tests for the shared contract.

These assert the guarantees other components are allowed to rely on. If one
fails, some module downstream is about to receive a shape it didn't expect.
"""

import pytest
from pydantic import ValidationError

from customer_service.schemas import Category, Conversation, Message, Route


def test_confidence_must_be_a_probability():
    """The router cannot hand escalation a number outside 0..1."""
    with pytest.raises(ValidationError):
        Route(category=Category.BILLING, confidence=1.4, reasoning="over")


def test_route_rejects_unexpected_fields():
    """Claude returning an extra key is an error we see, not one we ignore."""
    with pytest.raises(ValidationError):
        Route.model_validate(
            {
                "category": "billing",
                "confidence": 0.9,
                "reasoning": "refund request",
                "agent": "billing_v2",
            }
        )


def test_latest_customer_message_ignores_assistant_turns():
    conversation = Conversation(
        id="c1",
        messages=[
            Message(role="customer", content="I was charged twice"),
            Message(role="assistant", content="Let me check that"),
            Message(role="customer", content="Thanks, it's invoice 42"),
        ],
    )
    assert conversation.latest_customer_message == "Thanks, it's invoice 42"


def test_latest_customer_message_is_none_when_customer_has_not_spoken():
    assert Conversation(id="c2").latest_customer_message is None


def test_assistant_turns_counts_only_the_assistant():
    conversation = Conversation(
        id="c3",
        messages=[
            Message(role="customer", content="hi"),
            Message(role="assistant", content="hello"),
            Message(role="customer", content="still there?"),
            Message(role="assistant", content="yes"),
        ],
    )
    assert conversation.assistant_turns == 2


def test_category_serialises_as_its_string_value():
    """StrEnum keeps logs and JSON readable: 'billing', not 'Category.BILLING'."""
    route = Route(category=Category.BILLING, confidence=0.9, reasoning="refund")
    assert route.model_dump()["category"] == "billing"
