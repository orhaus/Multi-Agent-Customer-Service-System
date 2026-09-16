"""Rules and signals that decide when a conversation goes to a human.

Deliberately free of model calls. Escalation is the safety net under the parts
that can be wrong, so it has to be something you can read, reason about, and
test exhaustively - not another thing that might misjudge.
"""

from enum import StrEnum

from customer_service.config import Settings, get_settings
from customer_service.schemas import Category, Conversation, Route


class EscalationReason(StrEnum):
    """Why a conversation was handed off.

    A named reason rather than a free-text string: these end up in logs and
    metrics, and "what fraction of handoffs were low confidence" is a question
    you can only answer if the answers are a fixed, countable set.
    """

    UNKNOWN_CATEGORY = "unknown_category"
    LOW_CONFIDENCE = "low_confidence"
    TURN_LIMIT_REACHED = "turn_limit_reached"

    # Set by the orchestrator after an agent has run. escalation_reason() never
    # returns these, because it runs before any agent does.
    AGENT_DECLINED = "agent_declined"
    AGENT_FAILED = "agent_failed"


def turn_limit_reached(conversation: Conversation, settings: Settings | None = None) -> bool:
    """Has this conversation used up its budget of agent turns?

    Its own function because the orchestrator needs it on turns that never call
    the router at all - the turn limit applies regardless of how a specialist
    was chosen, unlike the two checks below which are about the routing
    decision itself.
    """
    settings = settings or get_settings()
    return conversation.assistant_turns >= settings.max_agent_turns


def escalation_reason(
    route: Route,
    conversation: Conversation,
    settings: Settings | None = None,
) -> EscalationReason | None:
    """Return why this conversation needs a human, or None to let an agent run.

    Only used on a conversation's first turn, when there is a fresh Route to
    judge. Checks run earliest-cause-first: a conversation that was badly
    routed *and* had already run out of turns reports the routing problem,
    because that is the one worth fixing.
    """
    settings = settings or get_settings()

    if route.category is Category.UNKNOWN:
        return EscalationReason.UNKNOWN_CATEGORY

    if route.confidence < settings.router_confidence_threshold:
        return EscalationReason.LOW_CONFIDENCE

    if turn_limit_reached(conversation, settings):
        return EscalationReason.TURN_LIMIT_REACHED

    return None
