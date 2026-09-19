"""Entry point that wires router -> escalation rules -> specialist agents.

    router -> escalation rules -> specialist -> handled?  -> reply
                   |                               |
                   +-> human                  declined -> other specialist (once)
                                                               |
                                                               +-> human
"""

import logging
from collections.abc import Mapping

from google import genai

from customer_service import llm
from customer_service.agents.base import Agent, AgentError
from customer_service.agents.billing import BillingAgent
from customer_service.agents.technical import TechnicalAgent
from customer_service.config import Settings, get_settings
from customer_service.escalation import EscalationReason, escalation_reason, turn_limit_reached
from customer_service.router import Router
from customer_service.schemas import Category, Conversation, Message, Resolution

logger = logging.getLogger(__name__)

SPECIALISTS: tuple[type[Agent], ...] = (BillingAgent, TechnicalAgent)

# How many times a conversation may move to a different specialist after the
# first declines. One fixes a misroute; more lets agents bounce a customer around.
MAX_REROUTES = 1

HANDOFF_MESSAGE = (
    "Thanks for your patience. I'm passing this to a member of our support team, "
    "who will pick it up from here."
)


class Orchestrator:
    """Takes a conversation and returns what the customer should see next."""

    def __init__(
        self,
        client: genai.Client | None = None,
        settings: Settings | None = None,
        router: Router | None = None,
        agents: Mapping[Category, Agent] | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        if router is None or agents is None:
            client = client or llm.make_client(self._settings)

        self._router = router or Router(client=client, settings=self._settings)
        self._agents = dict(agents) if agents is not None else {
            cls.category: cls(client=client, settings=self._settings) for cls in SPECIALISTS
        }

    def handle(self, conversation: Conversation) -> Resolution:
        if conversation.assigned_category is None:
            # First turn: nothing has been classified yet, so ask the router.
            route = self._router.route(conversation)
            reason = escalation_reason(route, conversation, self._settings)
            if reason is not None:
                return self._handoff(conversation, route.category, reason)
            category = route.category
        else:
            # A specialist already owns this conversation. Re-running the router
            # on a single reply like "android" has nothing to classify - it
            # isn't a fresh support request, it's an answer to whatever the
            # specialist just asked. The specialist keeps the conversation; if a
            # later message turns out not to be theirs after all, its own
            # decline (below) is what notices - not the router guessing blind.
            if turn_limit_reached(conversation, self._settings):
                return self._handoff(
                    conversation, conversation.assigned_category, EscalationReason.TURN_LIMIT_REACHED
                )
            category = conversation.assigned_category

        tried: list[Category] = []

        while True:
            tried.append(category)
            rerouted_from = tried[0] if len(tried) > 1 else None

            try:
                answer = self._agents[category].reply(conversation)
            except AgentError:
                logger.exception("%s agent failed on conversation %s", category, conversation.id)
                return self._handoff(conversation, category, EscalationReason.AGENT_FAILED, rerouted_from)

            if answer.handled:
                # Remembered so the *next* turn skips the router entirely.
                conversation.assigned_category = category
                return Resolution(
                    conversation_id=conversation.id,
                    category=category,
                    reply=answer.reply,
                    rerouted_from=rerouted_from,
                )

            suggested = answer.suggested_category
            can_reroute = (
                suggested in self._agents  # a real specialist, not UNKNOWN
                and suggested not in tried  # not itself, and no ping-pong
                and len(tried) <= MAX_REROUTES
            )
            if not can_reroute:
                logger.info(
                    "%s agent declined conversation %s (suggested %s); escalating",
                    category, conversation.id, suggested,
                )
                return self._handoff(conversation, category, EscalationReason.AGENT_DECLINED, rerouted_from)

            logger.info("Rerouting conversation %s from %s to %s", conversation.id, category, suggested)
            category = suggested

    def turn(self, conversation: Conversation, message: str) -> Resolution:
        """Record the customer's message, answer it, and record the reply.

        The conversation is updated in place, so the next turn sees the history
        and the specialist that took it. An escalated turn deliberately records
        no assistant message: a person owns the conversation from there, and
        what they say next is not ours to put words around.

        Every caller needs these rules - the terminal client and the HTTP API
        both go through here so they cannot drift apart.
        """
        conversation.messages.append(Message(role="customer", content=message))
        resolution = self.handle(conversation)

        if not resolution.escalated:
            conversation.messages.append(Message(role="assistant", content=resolution.reply))

        return resolution

    def _handoff(
        self,
        conversation: Conversation,
        category: Category,
        reason: EscalationReason,
        rerouted_from: Category | None = None,
    ) -> Resolution:
        return Resolution(
            conversation_id=conversation.id,
            category=category,
            reply=HANDOFF_MESSAGE,
            escalated=True,
            escalation_reason=reason,
            rerouted_from=rerouted_from,
        )
