"""Entry point that wires router -> escalation rules -> specialist agents.

    router -> escalation rules -> specialist -> handled?  -> reply
                   |                               |
                   +-> human                  declined -> other specialist (once)
                                                               |
                                                               +-> human
"""

import logging
from collections.abc import Mapping

import anthropic

from customer_service.agents.base import Agent, AgentError
from customer_service.agents.billing import BillingAgent
from customer_service.agents.technical import TechnicalAgent
from customer_service.config import Settings, get_settings
from customer_service.escalation import EscalationReason, escalation_reason
from customer_service.router import Router
from customer_service.schemas import Category, Conversation, Resolution

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
        client: anthropic.Anthropic | None = None,
        settings: Settings | None = None,
        router: Router | None = None,
        agents: Mapping[Category, Agent] | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        if router is None or agents is None:
            client = client or anthropic.Anthropic()

        self._router = router or Router(client=client, settings=self._settings)
        self._agents = dict(agents) if agents is not None else {
            cls.category: cls(client=client, settings=self._settings) for cls in SPECIALISTS
        }

    def handle(self, conversation: Conversation) -> Resolution:
        route = self._router.route(conversation)

        reason = escalation_reason(route, conversation, self._settings)
        if reason is not None:
            return self._handoff(conversation, route.category, reason)

        # escalation_reason() has ruled out UNKNOWN, so this is a real specialist.
        category = route.category
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
