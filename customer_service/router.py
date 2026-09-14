"""Classifies an inbound message and picks the specialist agent to handle it.

The router is deliberately the cheapest part of the system: one short call to a
small model, no tools, no conversation history beyond the latest customer
message. Everything expensive happens after it has decided who should answer.
"""

import logging

import anthropic
from pydantic import ValidationError

from customer_service.config import Settings, get_settings
from customer_service.schemas import Category, Conversation, Route

logger = logging.getLogger(__name__)

# The reply is one category, one number and one sentence. This is generous.
MAX_TOKENS = 512

SYSTEM_PROMPT = """\
You are the routing layer of a customer support system. Classify the customer's \
message into exactly one category.

billing - invoices, charges, refunds, payment methods, subscriptions, plan \
changes, pricing.
technical - errors, bugs, outages, login failures, configuration, integrations \
and how-to questions.
unknown - anything that fits neither, fits both, or is too vague to tell apart.

Confidence is your probability that the category is correct, from 0 to 1. Report \
it honestly. A message that could plausibly belong to either category should \
score low, not high: low confidence sends the conversation to a human, which is \
the right outcome when you are unsure.

Reasoning is one short sentence naming the signal you classified on.\
"""

# Returned when we cannot classify at all. Confidence 0.0 is below every
# possible threshold, so escalation will always pick this up.
def _undecided(reason: str) -> Route:
    return Route(category=Category.UNKNOWN, confidence=0.0, reasoning=reason)


class Router:
    """Decides which specialist agent should handle a conversation."""

    def __init__(
        self,
        client: anthropic.Anthropic | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._client = client or anthropic.Anthropic()

    def route(self, conversation: Conversation) -> Route:
        """Classify the latest customer message.

        Never raises on an API failure - a router that cannot reach Claude
        returns UNKNOWN at zero confidence, which escalates to a human. Losing
        the whole request because the routing call timed out would be worse.
        """
        message = conversation.latest_customer_message
        if message is None:
            return _undecided("No customer message to classify.")

        try:
            response = self._client.messages.parse(
                model=self._settings.router_model,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": message}],
                output_format=Route,
            )
        except anthropic.AnthropicError:
            logger.exception("Routing call failed for conversation %s", conversation.id)
            return _undecided("Routing call failed; handing off to a human.")
        except ValidationError:
            # The API is told about Route's bounds but does not enforce them -
            # the SDK moves `minimum`/`maximum` into the schema description. A
            # model that answers with confidence 1.4 lands here, not upstream.
            logger.exception("Router returned an invalid Route for conversation %s", conversation.id)
            return _undecided("Router produced an unusable classification; handing off to a human.")

        route = response.parsed_output
        if route is None:
            # parsed_output is Optional: a response with no text block yields None.
            logger.error("Router returned no classification for conversation %s", conversation.id)
            return _undecided("Router returned no classification; handing off to a human.")

        return route
