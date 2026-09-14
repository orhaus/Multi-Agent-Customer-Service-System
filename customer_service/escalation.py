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


def escalation_reason(
    route: Route,
    conversation: Conversation,
    settings: Settings | None = None,
) -> EscalationReason | None:
    """Return why this conversation needs a human, or None to let an agent run.

    Checks run earliest-cause-first: a conversation that was badly routed *and*
    then ran out of turns reports the routing problem, because that is the one
    worth fixing.
    """
    settings = settings or get_settings()

    if route.category is Category.UNKNOWN:
        return EscalationReason.UNKNOWN_CATEGORY

    if route.confidence < settings.router_confidence_threshold:
        return EscalationReason.LOW_CONFIDENCE

    if conversation.assistant_turns >= settings.max_agent_turns:
        return EscalationReason.TURN_LIMIT_REACHED

    return None
