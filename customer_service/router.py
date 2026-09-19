"""Classifies an inbound message and picks the specialist agent to handle it.

The router is deliberately the cheapest part of the system: one short call to a
small model, no tools, no conversation history beyond the latest customer
message. Everything expensive happens after it has decided who should answer.
"""

import logging

from google import genai

from customer_service import llm
from customer_service.config import Settings, get_settings
from customer_service.schemas import Category, Conversation, Message, Route

logger = logging.getLogger(__name__)

# Classification is a narrow task; the lightest thinking setting is enough.
THINKING: llm.ThinkingLevel = "MINIMAL"

SYSTEM_PROMPT = """\
You are the routing layer of a customer support system. Classify the customer's \
message into exactly one category.

billing - invoices, charges, refunds, payment methods, subscriptions, plan \
changes, pricing.
technical - errors, bugs, outages, login failures, configuration, integrations \
and how-to questions.
unknown - only when the message states no problem at all: a greeting, a \
thank-you, or a request to be contacted with no topic attached.

When a message touches both specialisms, pick the one the customer most wants \
solved. Do not answer unknown because two categories apply. A wrong pick is \
cheap - the specialist who receives it can hand the conversation to the other \
one - while unknown takes up a person who could be doing work only a person can \
do.

Confidence is your probability that the category is correct, from 0 to 1. Report \
it honestly: low confidence sends the conversation to a human instead of a \
specialist. Reserve that for messages you genuinely cannot place, not for ones \
where two specialisms both apply.

Reasoning is one short sentence naming the signal you classified on.\
"""


def _undecided(reason: str) -> Route:
    """Returned when we cannot classify at all.

    Confidence 0.0 is below every possible threshold, so escalation always
    picks this up.
    """
    return Route(category=Category.UNKNOWN, confidence=0.0, reasoning=reason)


class Router:
    """Decides which specialist agent should handle a conversation."""

    def __init__(
        self,
        client: genai.Client | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._client = client or llm.make_client(self._settings)

    def route(self, conversation: Conversation) -> Route:
        """Classify the latest customer message.

        Never raises on a model failure. A router that cannot get an answer -
        quota exhausted, network down, output that doesn't validate - returns
        UNKNOWN at zero confidence, which escalates to a human. Losing the whole
        request because routing failed would be worse.
        """
        message = conversation.latest_customer_message
        if message is None:
            return _undecided("No customer message to classify.")

        try:
            return llm.generate(
                self._client,
                model=self._settings.router_model,
                system=SYSTEM_PROMPT,
                messages=[Message(role="customer", content=message)],
                schema=Route,
                thinking_level=THINKING,
            )
        except llm.LLMError:
            logger.exception("Routing failed for conversation %s", conversation.id)
            return _undecided("Routing failed; handing off to a human.")
