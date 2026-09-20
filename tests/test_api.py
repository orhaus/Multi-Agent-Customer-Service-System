"""Tests for the HTTP layer.

The orchestrator is replaced through FastAPI's dependency_overrides, so these
test the endpoint's own job: parsing the request, threading the conversation
through, and handing the updated one back. Routing and escalation are covered
in test_orchestrator.py.
"""

import pytest
from fastapi.testclient import TestClient

from customer_service.api import app, get_orchestrator
from customer_service.escalation import EscalationReason
from customer_service.orchestrator import HANDOFF_MESSAGE
from customer_service.schemas import (
    AgentStep,
    Category,
    Conversation,
    Message,
    Resolution,
    Route,
    ToolCall,
    Trace,
)


class FakeOrchestrator:
    """Keeps Orchestrator.turn's contract: record the question, record the
    answer only when nobody was escalated to, and fill in the trace."""

    def __init__(self, *results: Resolution):
        self.results = list(results)
        self.seen: list[Conversation] = []

    def turn(self, conversation: Conversation, message: str, trace: Trace | None = None) -> Resolution:
        conversation.messages.append(Message(role="customer", content=message))
        result = self.results.pop(0)

        if not result.escalated:
            conversation.messages.append(Message(role="assistant", content=result.reply))
            conversation.assigned_category = result.category

        if trace is not None:
            trace.route = Route(category=result.category, confidence=0.91, reasoning="test")
            trace.escalation_reason = result.escalation_reason
            trace.agents = [
                AgentStep(
                    category=result.category,
                    handled=not result.escalated,
                    tools=[ToolCall(name="list_invoices", arguments={"customer_id": "cus_demo"},
                                    result={"invoices": []})],
                )
            ]
            trace.seconds = 1.25

        self.seen.append(conversation.model_copy(deep=True))
        return result


def answered(reply: str, category: Category = Category.BILLING, **extra) -> Resolution:
    return Resolution(conversation_id="x", category=category, reply=reply, **extra)


def handed_off(reason: EscalationReason) -> Resolution:
    return Resolution(
        conversation_id="x", category=Category.BILLING, reply=HANDOFF_MESSAGE,
        escalated=True, escalation_reason=reason,
    )


@pytest.fixture
def client_for():
    """Builds a TestClient whose orchestrator is the fake, and cleans up after."""
    def build(*results: Resolution) -> tuple[TestClient, FakeOrchestrator]:
        fake = FakeOrchestrator(*results)
        app.dependency_overrides[get_orchestrator] = lambda: fake
        return TestClient(app), fake

    yield build
    app.dependency_overrides.clear()


def test_health_does_not_touch_the_model(client_for):
    """It answers 'is the process up', not 'is Gemini up'."""
    client, fake = client_for()
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert fake.seen == []


def test_a_first_message_starts_a_conversation(client_for):
    client, _ = client_for(answered("I've checked your invoice."))
    body = client.post("/chat", json={"message": "I was charged twice"}).json()

    assert body["resolution"]["reply"] == "I've checked your invoice."
    assert body["conversation"]["id"]  # generated for the caller
    assert [m["content"] for m in body["conversation"]["messages"]] == [
        "I was charged twice",
        "I've checked your invoice.",
    ]


def test_each_new_conversation_gets_its_own_id(client_for):
    client, _ = client_for(answered("one"), answered("two"))
    first = client.post("/chat", json={"message": "hello"}).json()
    second = client.post("/chat", json={"message": "hello"}).json()

    assert first["conversation"]["id"] != second["conversation"]["id"]


def test_sending_the_conversation_back_continues_it(client_for):
    """The stateless contract: the client carries the history, the server doesn't."""
    client, fake = client_for(answered("Which OS?", Category.TECHNICAL), answered("Try reinstalling."))

    first = client.post("/chat", json={"message": "app crashes"}).json()
    second = client.post(
        "/chat", json={"message": "android", "conversation": first["conversation"]}
    ).json()

    assert second["resolution"]["reply"] == "Try reinstalling."
    assert [m["content"] for m in second["conversation"]["messages"]] == [
        "app crashes", "Which OS?", "android", "Try reinstalling.",
    ]
    assert fake.seen[1].id == first["conversation"]["id"]


def test_the_assigned_specialist_survives_the_round_trip(client_for):
    """Without this the next turn would re-route a bare reply like 'android'."""
    client, fake = client_for(answered("Which OS?", Category.TECHNICAL), answered("ok", Category.TECHNICAL))

    first = client.post("/chat", json={"message": "app crashes"}).json()
    assert first["conversation"]["assigned_category"] == "technical"

    client.post("/chat", json={"message": "android", "conversation": first["conversation"]})
    assert fake.seen[1].assigned_category is Category.TECHNICAL


def test_an_escalation_is_reported_with_its_reason(client_for):
    client, _ = client_for(handed_off(EscalationReason.LOW_CONFIDENCE))
    body = client.post("/chat", json={"message": "hmm"}).json()

    assert body["resolution"]["escalated"] is True
    assert body["resolution"]["escalation_reason"] == "low_confidence"
    assert body["resolution"]["reply"] == HANDOFF_MESSAGE


def test_an_escalated_turn_leaves_no_assistant_message_behind(client_for):
    """A person owns the conversation now; the handoff line is not history."""
    client, _ = client_for(handed_off(EscalationReason.AGENT_DECLINED))
    body = client.post("/chat", json={"message": "refund me"}).json()

    assert [m["role"] for m in body["conversation"]["messages"]] == ["customer"]


def test_a_reroute_is_visible_to_the_caller(client_for):
    client, _ = client_for(
        answered("Let's get you logged in.", Category.TECHNICAL, rerouted_from=Category.BILLING)
    )
    body = client.post("/chat", json={"message": "charged but can't log in"}).json()

    assert body["resolution"]["category"] == "technical"
    assert body["resolution"]["rerouted_from"] == "billing"


@pytest.mark.parametrize("payload", [{}, {"message": ""}, {"message": "   ", "conversation": "nope"}])
def test_bad_requests_are_rejected_before_the_model(client_for, payload):
    client, fake = client_for()
    assert client.post("/chat", json=payload).status_code == 422
    assert fake.seen == []


def test_the_response_explains_how_the_reply_was_reached(client_for):
    """The trace is what a caller renders to show the routing, not take it on trust."""
    client, _ = client_for(answered("I've checked your invoice."))
    trace = client.post("/chat", json={"message": "charged twice"}).json()["trace"]

    assert trace["route"]["category"] == "billing"
    assert trace["route"]["confidence"] == 0.91
    assert trace["route"]["reasoning"] == "test"
    assert trace["agents"][0]["handled"] is True
    assert trace["agents"][0]["tools"][0]["name"] == "list_invoices"
    assert trace["seconds"] == 1.25


def test_the_trace_carries_the_escalation_reason(client_for):
    client, _ = client_for(handed_off(EscalationReason.LOW_CONFIDENCE))
    trace = client.post("/chat", json={"message": "hmm"}).json()["trace"]

    assert trace["escalation_reason"] == "low_confidence"
    assert trace["agents"][0]["handled"] is False
