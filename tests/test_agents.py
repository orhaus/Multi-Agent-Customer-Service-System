"""Tests for the specialist agents.

The client is mocked, but responses go through the SDK's real parsing, so these
check our assumptions about the library as well as our own plumbing. Whether
Claude's replies are any *good* is an eval question, not a unit-test question.
"""

from unittest.mock import MagicMock

import anthropic
import httpx2
import pytest
from anthropic.lib._parse._response import parse_response
from anthropic.types import Message as SDKMessage
from pydantic import ValidationError

from customer_service.agents.base import SHARED_RULES, Agent, AgentError, to_api_messages
from customer_service.agents.billing import BillingAgent
from customer_service.agents.technical import TechnicalAgent
from customer_service.config import Settings
from customer_service.schemas import AgentReply, Category, Conversation, Message

SETTINGS = Settings.from_env({"ANTHROPIC_API_KEY": "sk-ant-test"})

HANDLED = '{"handled": true, "suggested_category": "billing", "reply": "I have checked your invoice."}'
DECLINED = '{"handled": false, "suggested_category": "technical", "reply": ""}'


def sdk_message(text: str) -> SDKMessage:
    """A real anthropic.types.Message, validated the way a live response is."""
    return SDKMessage.model_validate(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "content": [{"type": "text", "text": text}] if text else [],
        }
    )


def parsed(text: str):
    return parse_response(response=sdk_message(text), output_format=AgentReply)


def fake_client(text: str | None = None, error: Exception | None = None) -> MagicMock:
    client = MagicMock()
    if error is not None:
        client.messages.parse.side_effect = error
    else:
        client.messages.parse.return_value = parsed(text)
    return client


def conversation(*turns: tuple[str, str]) -> Conversation:
    return Conversation(
        id="c1",
        messages=[Message(role=role, content=content) for role, content in turns],
    )


def ask(agent_cls=BillingAgent, **client_kwargs) -> tuple[AgentReply, MagicMock]:
    client = fake_client(**client_kwargs)
    answer = agent_cls(client=client, settings=SETTINGS).reply(conversation(("customer", "?")))
    return answer, client


def test_customer_and_assistant_roles_map_to_the_api_vocabulary():
    messages = to_api_messages(
        conversation(("customer", "hi"), ("assistant", "hello"), ("customer", "still broken"))
    )
    assert messages == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "still broken"},
    ]


def test_a_handled_reply_comes_back_structured():
    answer, _ = ask(text=HANDLED)
    assert answer == AgentReply(
        handled=True, suggested_category=Category.BILLING, reply="I have checked your invoice."
    )


def test_a_decline_comes_back_as_data_not_prose():
    """This is the fix: the orchestrator can now *read* that the agent said no."""
    answer, _ = ask(text=DECLINED)
    assert answer.handled is False
    assert answer.suggested_category is Category.TECHNICAL


def test_the_whole_conversation_is_sent_not_just_the_last_message():
    client = fake_client(HANDLED)
    TechnicalAgent(client=client, settings=SETTINGS).reply(
        conversation(("customer", "app crashes"), ("assistant", "which version?"), ("customer", "3.2"))
    )
    assert len(client.messages.parse.call_args.kwargs["messages"]) == 3


def test_each_specialist_sends_its_own_prompt_its_specialism_and_the_shared_rules():
    _, billing = ask(BillingAgent, text=HANDLED)
    _, technical = ask(TechnicalAgent, text=HANDLED)
    billing_system = billing.messages.parse.call_args.kwargs["system"]
    technical_system = technical.messages.parse.call_args.kwargs["system"]

    assert BillingAgent.system_prompt in billing_system
    assert "Your specialism is billing." in billing_system
    assert TechnicalAgent.system_prompt in technical_system
    assert "Your specialism is technical." in technical_system
    assert SHARED_RULES in billing_system and SHARED_RULES in technical_system


def test_the_request_asks_for_an_agent_reply_on_the_agent_model_at_low_effort():
    _, client = ask(text=HANDLED)
    kwargs = client.messages.parse.call_args.kwargs
    assert kwargs["model"] == SETTINGS.agent_model
    assert kwargs["output_format"] is AgentReply
    assert kwargs["output_config"] == {"effort": "low"}


def test_an_api_failure_becomes_an_agent_error():
    cause = anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.anthropic.com"))
    with pytest.raises(AgentError) as info:
        ask(error=cause)
    assert info.value.__cause__ is cause  # the original is kept for the logs


def test_invalid_structured_output_becomes_an_agent_error():
    with pytest.raises(ValidationError) as bad:
        parsed('{"handled": "maybe", "suggested_category": "billing", "reply": "x"}')
    with pytest.raises(AgentError):
        ask(error=bad.value)


def test_a_response_with_no_text_becomes_an_agent_error():
    with pytest.raises(AgentError):
        ask(text="")


def test_handling_a_conversation_with_an_empty_reply_is_an_error():
    """Handled plus empty would send the customer a blank message."""
    with pytest.raises(AgentError):
        ask(text='{"handled": true, "suggested_category": "billing", "reply": "   "}')


def test_every_specialist_declares_a_distinct_category():
    """The orchestrator looks agents up by category, so collisions must not happen."""
    specialists = [BillingAgent, TechnicalAgent]
    categories = [cls.category for cls in specialists]

    assert len(set(categories)) == len(specialists)
    assert Category.UNKNOWN not in categories
    assert all(issubclass(cls, Agent) for cls in specialists)
