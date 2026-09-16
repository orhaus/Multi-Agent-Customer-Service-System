"""Tests for the specialist agents.

The model call is replaced at llm.generate, so these test what each agent asks
for and how it treats the answer. How the call reaches Gemini is covered by
test_llm.py; whether replies are any *good* is an eval question.
"""

from unittest.mock import MagicMock, patch

import pytest

from customer_service import llm
from customer_service.agents.base import SHARED_RULES, THINKING, Agent, AgentError
from customer_service.agents.billing import BillingAgent
from customer_service.agents.technical import TechnicalAgent
from customer_service.config import Settings
from customer_service.schemas import AgentReply, Category, Conversation, Message

SETTINGS = Settings.from_env({"GEMINI_API_KEY": "test-gemini-key"})
HANDLED = AgentReply(handled=True, suggested_category=Category.BILLING, reply="I have checked your invoice.")
CONVERSATION = Conversation(
    id="c1",
    messages=[
        Message(role="customer", content="app crashes"),
        Message(role="assistant", content="which version?"),
        Message(role="customer", content="3.2"),
    ],
)


@pytest.fixture
def generate():
    with patch.object(llm, "generate", return_value=HANDLED) as fake:
        yield fake


def reply(agent_cls: type[Agent] = BillingAgent) -> AgentReply:
    return agent_cls(client=MagicMock(), settings=SETTINGS).reply(CONVERSATION)


def test_a_handled_reply_is_returned(generate):
    assert reply() == HANDLED


def test_a_decline_comes_back_as_data_not_prose(generate):
    """The orchestrator can *read* that the agent said no."""
    generate.return_value = AgentReply(handled=False, suggested_category=Category.TECHNICAL, reply="")
    answer = reply()
    assert answer.handled is False
    assert answer.suggested_category is Category.TECHNICAL


def test_the_whole_conversation_is_sent_not_just_the_last_message(generate):
    """Unlike the router, an agent needs the history to hold a conversation."""
    reply(TechnicalAgent)
    assert generate.call_args.kwargs["messages"] == CONVERSATION.messages


def test_each_specialist_sends_its_own_prompt_its_specialism_and_the_shared_rules(generate):
    reply(BillingAgent)
    billing_system = generate.call_args.kwargs["system"]
    reply(TechnicalAgent)
    technical_system = generate.call_args.kwargs["system"]

    assert BillingAgent.system_prompt in billing_system
    assert "Your specialism is billing." in billing_system
    assert TechnicalAgent.system_prompt in technical_system
    assert "Your specialism is technical." in technical_system
    assert SHARED_RULES in billing_system and SHARED_RULES in technical_system


def test_asks_the_agent_model_for_an_agent_reply_at_low_thinking(generate):
    reply()
    kwargs = generate.call_args.kwargs
    assert kwargs["model"] == SETTINGS.agent_model
    assert kwargs["schema"] is AgentReply
    assert kwargs["thinking_level"] == THINKING == "LOW"


def test_a_model_failure_becomes_an_agent_error(generate):
    cause = llm.LLMError("quota exhausted")
    generate.side_effect = cause
    with pytest.raises(AgentError) as info:
        reply()
    assert info.value.__cause__ is cause  # the original is kept for the logs


def test_handling_a_conversation_with_an_empty_reply_is_an_error(generate):
    """Handled plus empty would send the customer a blank message."""
    generate.return_value = AgentReply(handled=True, suggested_category=Category.BILLING, reply="   ")
    with pytest.raises(AgentError):
        reply()


def test_declining_with_an_empty_reply_is_fine(generate):
    """A declined reply is never shown, so it may be empty."""
    generate.return_value = AgentReply(handled=False, suggested_category=Category.UNKNOWN, reply="")
    assert reply().handled is False


def test_every_specialist_declares_a_distinct_category():
    """The orchestrator looks agents up by category, so collisions must not happen."""
    specialists = [BillingAgent, TechnicalAgent]
    categories = [cls.category for cls in specialists]

    assert len(set(categories)) == len(specialists)
    assert Category.UNKNOWN not in categories
    assert all(issubclass(cls, Agent) for cls in specialists)
