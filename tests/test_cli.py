"""Tests for the terminal chat loop.

The orchestrator is faked and input is scripted, so these test the loop itself:
what gets printed, what goes into the conversation history, and when it stops.
"""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from customer_service import __main__ as cli
from customer_service.config import Settings
from customer_service.escalation import EscalationReason
from customer_service.schemas import Category, Resolution


def scripted(*lines: str):
    """Stands in for input(): returns each line in turn, then behaves like Ctrl+D."""
    remaining = list(lines)

    def read(prompt: str) -> str:
        if not remaining:
            raise EOFError
        return remaining.pop(0)

    return read


def answered(reply: str, category: Category = Category.BILLING, **extra) -> Resolution:
    return Resolution(conversation_id="x", category=category, reply=reply, **extra)


def handed_off(reason: EscalationReason) -> Resolution:
    return Resolution(
        conversation_id="x", category=Category.BILLING, reply="A person will help.",
        escalated=True, escalation_reason=reason,
    )


def fake_orchestrator(*results: Resolution):
    """Returns each result in turn, recording what the conversation looked like at each call."""
    seen = []

    def handle(conversation):
        seen.append((conversation.id, [(m.role, m.content) for m in conversation.messages]))
        return results[len(seen) - 1]

    orchestrator = MagicMock()
    orchestrator.handle.side_effect = handle
    return orchestrator, seen


def run(orchestrator, *lines: str) -> list[str]:
    output: list[str] = []
    cli.chat(orchestrator, read=scripted(*lines), write=output.append)
    return output


def test_a_reply_is_printed_with_its_routing_detail():
    orchestrator, _ = fake_orchestrator(answered("Refund noted."))
    output = run(orchestrator, "I want a refund")

    assert "support> Refund noted." in output
    assert any("[answered by billing]" in line for line in output)


def test_replies_become_history_for_the_next_turn():
    orchestrator, seen = fake_orchestrator(answered("Which invoice?"), answered("Found it."))
    run(orchestrator, "charged twice", "invoice 42")

    _, second_call = seen[1]
    assert second_call == [
        ("customer", "charged twice"),
        ("assistant", "Which invoice?"),
        ("customer", "invoice 42"),
    ]


def test_a_handoff_starts_a_fresh_conversation():
    orchestrator, seen = fake_orchestrator(
        handed_off(EscalationReason.LOW_CONFIDENCE), answered("Hello again.")
    )
    output = run(orchestrator, "something vague", "new question")

    (first_id, _), (second_id, second_messages) = seen
    assert first_id != second_id
    assert second_messages == [("customer", "new question")]
    assert any("handed to a human: low_confidence" in line for line in output)


def test_blank_lines_are_ignored_and_exit_stops_the_loop():
    orchestrator, seen = fake_orchestrator(answered("ok"))
    run(orchestrator, "", "   ", "hello", "exit", "never read")

    assert len(seen) == 1


def test_end_of_input_stops_cleanly():
    orchestrator, seen = fake_orchestrator()
    run(orchestrator)  # no lines at all: the first read raises EOFError

    assert seen == []


def test_rerouting_is_shown_alongside_the_final_specialist():
    result = answered("Let's get you logged in.", Category.TECHNICAL, rerouted_from=Category.BILLING)
    assert cli.describe(result) == "[answered by technical, rerouted from billing]"


def test_missing_configuration_exits_with_a_message_instead_of_a_traceback(capsys):
    with pytest.raises(ValidationError) as missing_key:
        Settings.from_env({})

    with patch.object(cli, "get_settings", side_effect=missing_key.value):
        assert cli.main() == 1

    assert "check your .env against .env.example" in capsys.readouterr().out
