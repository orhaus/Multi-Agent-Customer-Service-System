"""Tests for the orchestrator: routing, escalation and re-routing together.

The router and agents are faked at their own interfaces - their internals are
covered by their own tests. What's under test here is the decision flow.
"""

from unittest.mock import MagicMock

from customer_service.agents.base import AgentError
from customer_service.config import Settings
from customer_service.escalation import EscalationReason
from customer_service.orchestrator import HANDOFF_MESSAGE, Orchestrator
from customer_service.schemas import AgentReply, Category, Conversation, Message, Route

SETTINGS = Settings.from_env(
    {"GEMINI_API_KEY": "test-gemini-key", "ROUTER_CONFIDENCE_THRESHOLD": "0.7", "MAX_AGENT_TURNS": "3"}
)
CONVERSATION = Conversation(id="c1", messages=[Message(role="customer", content="help")])


def router_picks(category: Category, confidence: float = 0.9) -> MagicMock:
    router = MagicMock()
    router.route.return_value = Route(category=category, confidence=confidence, reasoning="test")
    return router


def agent(*answers) -> MagicMock:
    """A fake agent that returns each answer in turn, raising any that are exceptions."""
    fake = MagicMock()
    fake.reply.side_effect = list(answers)
    return fake


def handles(text: str = "Sorted.") -> AgentReply:
    return AgentReply(handled=True, suggested_category=Category.BILLING, reply=text)


def declines(to: Category, text: str = "not mine") -> AgentReply:
    return AgentReply(handled=False, suggested_category=to, reply=text)


def run(router, billing=None, technical=None, extra_agents=None):
    agents = {Category.BILLING: billing or agent(), Category.TECHNICAL: technical or agent()}
    agents.update(extra_agents or {})
    return Orchestrator(settings=SETTINGS, router=router, agents=agents).handle(CONVERSATION)


def test_the_routed_specialist_answers():
    billing, technical = agent(handles("Refund noted.")), agent()
    result = run(router_picks(Category.BILLING), billing, technical)

    assert result.reply == "Refund noted."
    assert result.category is Category.BILLING
    assert not result.escalated
    assert result.rerouted_from is None
    technical.reply.assert_not_called()


def test_router_level_escalation_never_reaches_an_agent():
    billing, technical = agent(), agent()
    result = run(router_picks(Category.BILLING, confidence=0.3), billing, technical)

    assert result.escalated
    assert result.escalation_reason == EscalationReason.LOW_CONFIDENCE
    assert result.reply == HANDOFF_MESSAGE
    billing.reply.assert_not_called()
    technical.reply.assert_not_called()


def test_a_misrouted_conversation_moves_to_the_suggested_specialist():
    """The case that started this: billing gets a login problem."""
    billing = agent(declines(Category.TECHNICAL))
    technical = agent(handles("Let's get you logged in."))
    result = run(router_picks(Category.BILLING), billing, technical)

    assert result.reply == "Let's get you logged in."
    assert result.category is Category.TECHNICAL
    assert result.rerouted_from is Category.BILLING
    assert not result.escalated


def test_a_declined_reply_is_never_shown_to_the_customer():
    result = run(router_picks(Category.BILLING), agent(declines(Category.UNKNOWN, "INTERNAL")))
    assert "INTERNAL" not in result.reply
    assert result.reply == HANDOFF_MESSAGE


def test_declining_as_unknown_escalates_without_trying_anyone_else():
    """Unknown means 'needs a person', e.g. a refund that must be approved."""
    technical = agent()
    result = run(router_picks(Category.BILLING), agent(declines(Category.UNKNOWN)), technical)

    assert result.escalated
    assert result.escalation_reason == EscalationReason.AGENT_DECLINED
    technical.reply.assert_not_called()


def test_an_agent_that_declines_in_favour_of_itself_escalates():
    billing = agent(declines(Category.BILLING))
    result = run(router_picks(Category.BILLING), billing)

    assert result.escalation_reason == EscalationReason.AGENT_DECLINED
    assert billing.reply.call_count == 1


def test_two_agents_cannot_bounce_a_customer_between_them():
    billing = agent(declines(Category.TECHNICAL))
    technical = agent(declines(Category.BILLING))
    result = run(router_picks(Category.BILLING), billing, technical)

    assert result.escalation_reason == EscalationReason.AGENT_DECLINED
    assert result.category is Category.TECHNICAL
    assert result.rerouted_from is Category.BILLING
    assert billing.reply.call_count == 1
    assert technical.reply.call_count == 1


def test_the_reroute_cap_holds_even_when_there_is_somewhere_new_to_go():
    """Registers a third agent under UNKNOWN purely to give a second re-route a target."""
    third = agent(handles("should never run"))
    result = run(
        router_picks(Category.BILLING),
        agent(declines(Category.TECHNICAL)),
        agent(declines(Category.UNKNOWN)),
        extra_agents={Category.UNKNOWN: third},
    )

    assert result.escalation_reason == EscalationReason.AGENT_DECLINED
    third.reply.assert_not_called()


def test_an_agent_failure_escalates():
    result = run(router_picks(Category.BILLING), agent(AgentError("boom")))

    assert result.escalated
    assert result.escalation_reason == EscalationReason.AGENT_FAILED
    assert result.reply == HANDOFF_MESSAGE


def test_a_failure_after_rerouting_still_records_the_original_route():
    result = run(
        router_picks(Category.BILLING),
        agent(declines(Category.TECHNICAL)),
        agent(AgentError("boom")),
    )

    assert result.escalation_reason == EscalationReason.AGENT_FAILED
    assert result.category is Category.TECHNICAL
    assert result.rerouted_from is Category.BILLING
