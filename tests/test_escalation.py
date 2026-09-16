"""Tests for the handoff rules.

Escalation decides when a human takes over, so every rule and every boundary
gets a case. No mocking needed - this is pure logic over plain objects.
"""

from customer_service.config import Settings
from customer_service.escalation import EscalationReason, escalation_reason
from customer_service.schemas import Category, Conversation, Message, Route

# threshold 0.7, at most 3 assistant turns
SETTINGS = Settings.from_env(
    {
        "GEMINI_API_KEY": "test-gemini-key",
        "ROUTER_CONFIDENCE_THRESHOLD": "0.7",
        "MAX_AGENT_TURNS": "3",
    }
)


def route(category: Category = Category.BILLING, confidence: float = 0.9) -> Route:
    return Route(category=category, confidence=confidence, reasoning="test")


def conversation(assistant_turns: int = 0) -> Conversation:
    messages = [Message(role="customer", content="hello")]
    for _ in range(assistant_turns):
        messages.append(Message(role="assistant", content="working on it"))
    return Conversation(id="c1", messages=messages)


def test_a_confident_route_on_a_fresh_conversation_is_not_escalated():
    assert escalation_reason(route(), conversation(), SETTINGS) is None


def test_confidence_below_the_threshold_escalates():
    reason = escalation_reason(route(confidence=0.69), conversation(), SETTINGS)
    assert reason is EscalationReason.LOW_CONFIDENCE


def test_confidence_exactly_at_the_threshold_does_not_escalate():
    """The threshold is inclusive: 0.7 means 'at least 0.7 is good enough'."""
    assert escalation_reason(route(confidence=0.7), conversation(), SETTINGS) is None


def test_unknown_category_escalates_even_when_confident():
    """A router certain it cannot tell is still a router that cannot tell."""
    reason = escalation_reason(route(category=Category.UNKNOWN, confidence=1.0), conversation(), SETTINGS)
    assert reason is EscalationReason.UNKNOWN_CATEGORY


def test_reaching_the_turn_limit_escalates():
    reason = escalation_reason(route(), conversation(assistant_turns=3), SETTINGS)
    assert reason is EscalationReason.TURN_LIMIT_REACHED


def test_one_turn_below_the_limit_does_not_escalate():
    """The agent gets its full budget - the limit is a ceiling, not a fence."""
    assert escalation_reason(route(), conversation(assistant_turns=2), SETTINGS) is None


def test_routing_problems_are_reported_before_turn_limits():
    """Both rules fire; the earliest cause is the one worth fixing."""
    reason = escalation_reason(route(confidence=0.2), conversation(assistant_turns=5), SETTINGS)
    assert reason is EscalationReason.LOW_CONFIDENCE


def test_unknown_is_reported_before_low_confidence():
    """The router's failure route is UNKNOWN at 0.0 - both rules match it."""
    reason = escalation_reason(route(category=Category.UNKNOWN, confidence=0.0), conversation(), SETTINGS)
    assert reason is EscalationReason.UNKNOWN_CATEGORY


def test_the_reason_is_a_plain_string_in_logs_and_json():
    reason = escalation_reason(route(confidence=0.1), conversation(), SETTINGS)
    assert f"{reason}" == "low_confidence"
