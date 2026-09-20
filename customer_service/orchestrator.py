"""Entry point that wires router -> escalation rules -> specialist agents.

    router -> escalation rules -> specialist -> handled?  -> reply
                   |                               |
                   +-> human                  declined -> other specialist (once)
                                                               |
                                                               +-> human
"""

import logging
import time
from collections.abc import Mapping

from google import genai

from customer_service import llm
from customer_service.agents.base import Agent, AgentError
from customer_service.agents.billing import BillingAgent
from customer_service.agents.technical import TechnicalAgent
from customer_service.config import Settings, get_settings
from customer_service.escalation import EscalationReason, escalation_reason, turn_limit_reached
from customer_service.router import Router
from customer_service.schemas import (
    AgentStep,
    Category,
    Conversation,
    Message,
    Resolution,
    ToolCall,
    Trace,
)

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

    def handle(self, conversation: Conversation, trace: Trace | None = None) -> Resolution:
        """Answer one message. Pass a Trace to find out how it was answered."""
        trace = trace if trace is not None else Trace()
        started = time.monotonic()
        try:
            return self._handle(conversation, trace)
        finally:
            trace.seconds = round(time.monotonic() - started, 3)

    def _handle(self, conversation: Conversation, trace: Trace) -> Resolution:
        if conversation.assigned_category is None:
            # First turn: nothing has been classified yet, so ask the router.
            route = self._router.route(conversation)
            trace.route = route
            reason = escalation_reason(route, conversation, self._settings)
            if reason is not None:
                return self._handoff(conversation, route.category, reason, trace=trace)
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
                    conversation,
                    conversation.assigned_category,
                    EscalationReason.TURN_LIMIT_REACHED,
                    trace=trace,
                )
            category = conversation.assigned_category

        tried: list[Category] = []

        while True:
            tried.append(category)
            rerouted_from = tried[0] if len(tried) > 1 else None

            # Collected per agent, not per turn: a re-routed conversation has
            # two specialists, and each one's lookups belong to it.
            tools: list[ToolCall] = []

            try:
                answer = self._agents[category].reply(conversation, on_tool_call=tools.append)
            except AgentError:
                trace.agents.append(
                    AgentStep(category=category, handled=False, failed=True, tools=tools)
                )
                logger.exception("%s agent failed on conversation %s", category, conversation.id)
                return self._handoff(
                    conversation, category, EscalationReason.AGENT_FAILED, rerouted_from, trace=trace
                )

            trace.agents.append(
                AgentStep(
                    category=category,
                    handled=answer.handled,
                    suggested_category=None if answer.handled else answer.suggested_category,
                    tools=tools,
                )
            )

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
                return self._handoff(
                    conversation, category, EscalationReason.AGENT_DECLINED, rerouted_from, trace=trace
                )

            logger.info("Rerouting conversation %s from %s to %s", conversation.id, category, suggested)
            category = suggested

    def turn(
        self, conversation: Conversation, message: str, trace: Trace | None = None
    ) -> Resolution:
        """Record the customer's message, answer it, and record the reply.

        The conversation is updated in place, so the next turn sees the history
        and the specialist that took it. An escalated turn deliberately records
        no assistant message: a person owns the conversation from there, and
        what they say next is not ours to put words around.

        Every caller needs these rules - the terminal client and the HTTP API
        both go through here so they cannot drift apart.
        """
        conversation.messages.append(Message(role="customer", content=message))
        resolution = self.handle(conversation, trace)

        if not resolution.escalated:
            conversation.messages.append(Message(role="assistant", content=resolution.reply))

        return resolution

    def _handoff(
        self,
        conversation: Conversation,
        category: Category,
        reason: EscalationReason,
        rerouted_from: Category | None = None,
        trace: Trace | None = None,
    ) -> Resolution:
        if trace is not None:
            trace.escalation_reason = reason

        return Resolution(
            conversation_id=conversation.id,
            category=category,
            reply=HANDOFF_MESSAGE,
            escalated=True,
            escalation_reason=reason,
            rerouted_from=rerouted_from,
        )
