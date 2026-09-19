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


def conversation(*turns: tuple[str, str], assigned_category: Category | None = None) -> Conversation:
    """A fresh Conversation per call - handle() now mutates assigned_category,
    so sharing one instance across tests would leak state between them."""
    turns = turns or (("customer", "help"),)
    return Conversation(
        id="c1",
        messages=[Message(role=role, content=content) for role, content in turns],
        assigned_category=assigned_category,
    )


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


def run(router, billing=None, technical=None, extra_agents=None, conv=None):
    agents = {Category.BILLING: billing or agent(), Category.TECHNICAL: technical or agent()}
    agents.update(extra_agents or {})
    return Orchestrator(settings=SETTINGS, router=router, agents=agents).handle(conv or conversation())


def test_the_routed_specialist_answers():
    conv = conversation()
    billing, technical = agent(handles("Refund noted.")), agent()
    result = run(router_picks(Category.BILLING), billing, technical, conv=conv)

    assert result.reply == "Refund noted."
    assert result.category is Category.BILLING
    assert not result.escalated
    assert result.rerouted_from is None
    technical.reply.assert_not_called()
    assert conv.assigned_category is Category.BILLING  # so the next turn skips the router


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


# --- follow-up turns: once a specialist owns the conversation ---------------
#
# The bug this fixes: a router asked to classify a lone reply like "android"
# has nothing to go on and comes back UNKNOWN, throwing away a conversation
# that was going fine. Once assigned_category is set, the router is skipped
# entirely - the owning specialist keeps the conversation, and its own decline
# is what notices if that later stops being the right fit.


def test_a_follow_up_turn_skips_the_router_entirely():
    router = router_picks(Category.BILLING)  # would answer if called
    technical = agent(handles("Let's get that Android crash sorted."))
    conv = conversation(
        ("customer", "app crashes in settings"),
        ("assistant", "which OS?"),
        ("customer", "android"),
        assigned_category=Category.TECHNICAL,
    )

    result = run(router, technical=technical, conv=conv)

    assert result.reply == "Let's get that Android crash sorted."
    assert result.category is Category.TECHNICAL
    assert not result.escalated
    router.route.assert_not_called()


def test_a_follow_up_turn_sees_the_full_conversation():
    technical = agent(handles("ok"))
    conv = conversation(
        ("customer", "app crashes in settings"),
        ("assistant", "which OS?"),
        ("customer", "android"),
        assigned_category=Category.TECHNICAL,
    )

    run(router_picks(Category.BILLING), technical=technical, conv=conv)

    sent = technical.reply.call_args.args[0]
    assert len(sent.messages) == 3


def test_a_follow_up_turn_still_respects_the_turn_limit():
    router = router_picks(Category.BILLING)
    billing = agent()
    conv = conversation(
        ("customer", "still stuck"), ("assistant", "try X"),
        ("customer", "no"), ("assistant", "try Y"),
        ("customer", "no"), ("assistant", "try Z"),
        assigned_category=Category.BILLING,
    )

    result = run(router, billing=billing, conv=conv)

    assert result.escalated
    assert result.escalation_reason == EscalationReason.TURN_LIMIT_REACHED
    assert result.category is Category.BILLING
    router.route.assert_not_called()
    billing.reply.assert_not_called()


def test_a_declined_follow_up_turn_can_still_reroute():
    """The specialist, not the router, is what notices a later pivot."""
    billing = agent(declines(Category.TECHNICAL))
    technical = agent(handles("Sure, let's look at the crash."))
    conv = conversation(
        ("customer", "when's my invoice due"),
        ("assistant", "the 5th - anything else?"),
        ("customer", "also the app keeps crashing"),
        assigned_category=Category.BILLING,
    )

    result = run(router_picks(Category.BILLING), billing, technical, conv=conv)

    assert result.reply == "Sure, let's look at the crash."
    assert result.category is Category.TECHNICAL
    assert result.rerouted_from is Category.BILLING


# --- turn(): the shared rules for recording a turn ------------------------
#
# Both the terminal client and the HTTP API go through this, so the rules about
# what ends up in the history live here rather than in each caller.


def take_turn(router, billing=None, technical=None, conv=None, message="help"):
    agents = {Category.BILLING: billing or agent(), Category.TECHNICAL: technical or agent()}
    orchestrator = Orchestrator(settings=SETTINGS, router=router, agents=agents)
    conversation = conv if conv is not None else Conversation(id="c1")
    return orchestrator.turn(conversation, message), conversation


def test_a_turn_records_the_question_and_the_answer():
    _, conversation = take_turn(
        router_picks(Category.BILLING),
        agent(handles("Refund noted.")),
        message="I want a refund",
    )

    assert [(m.role, m.content) for m in conversation.messages] == [
        ("customer", "I want a refund"),
        ("assistant", "Refund noted."),
    ]


def test_an_escalated_turn_records_the_question_but_not_a_reply():
    """A person owns the conversation from here; the handoff line is not history."""
    _, conversation = take_turn(
        router_picks(Category.BILLING, confidence=0.2), message="something vague"
    )

    assert [(m.role, m.content) for m in conversation.messages] == [
        ("customer", "something vague")
    ]


def test_a_turn_remembers_the_specialist_for_the_next_one():
    _, conversation = take_turn(router_picks(Category.TECHNICAL), technical=agent(handles("ok")))
    assert conversation.assigned_category is Category.TECHNICAL


def test_a_second_turn_builds_on_the_first():
    router = router_picks(Category.TECHNICAL)
    technical = agent(handles("which OS?"), handles("Try reinstalling."))
    _, conversation = take_turn(router, technical=technical, message="app crashes")

    orchestrator = Orchestrator(
        settings=SETTINGS,
        router=router,
        agents={Category.BILLING: agent(), Category.TECHNICAL: technical},
    )
    resolution = orchestrator.turn(conversation, "android")

    assert resolution.reply == "Try reinstalling."
    assert [m.content for m in conversation.messages] == [
        "app crashes", "which OS?", "android", "Try reinstalling.",
    ]
    assert router.route.call_count == 1  # the second turn never re-classified
