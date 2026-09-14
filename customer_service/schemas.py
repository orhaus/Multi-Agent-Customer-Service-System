"""Pydantic models shared across the system.

These are the only shapes that cross a component boundary. The router emits a
Route, the orchestrator turns it into a Resolution, and escalation reads both.
Keeping them in one file means a change to the contract is one diff, not a
hunt through four modules.
"""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Category(StrEnum):
    """Which specialist should handle a conversation.

    UNKNOWN is not a failure mode to fix later - it is a real outcome. A router
    that cannot tell billing from technical should say so and let escalation
    decide, rather than pick one and be confidently wrong.
    """

    BILLING = "billing"
    TECHNICAL = "technical"
    UNKNOWN = "unknown"


class Message(BaseModel):
    """One turn in a conversation."""

    role: Literal["customer", "assistant"]
    content: str = Field(min_length=1)


class Conversation(BaseModel):
    """Everything said so far, plus the ID we log and trace against."""

    id: str
    messages: list[Message] = Field(default_factory=list)

    @property
    def latest_customer_message(self) -> str | None:
        """What the router classifies. None if the customer hasn't spoken yet."""
        for message in reversed(self.messages):
            if message.role == "customer":
                return message.content
        return None

    @property
    def assistant_turns(self) -> int:
        """Turns spent so far, checked against MAX_AGENT_TURNS."""
        return sum(1 for message in self.messages if message.role == "assistant")


class Route(BaseModel):
    """The router's decision.

    This crosses the boundary to Claude - it doubles as the JSON schema the
    router asks the model to fill in - so it forbids extra fields.
    """

    model_config = ConfigDict(extra="forbid")

    category: Category
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(min_length=1)


class AgentReply(BaseModel):
    """A specialist's answer, plus whether it should have been the one answering.

    Like Route, Claude fills this in, so extra fields are forbidden. Field order
    matters: the model writes `handled` before `reply`, so it decides whether
    the problem is its own before drafting an answer to it.
    """

    model_config = ConfigDict(extra="forbid")

    handled: bool
    suggested_category: Category  # only read when handled is False
    reply: str


class Resolution(BaseModel):
    """What the orchestrator hands back to the caller."""

    conversation_id: str
    category: Category  # the specialist that finally answered, or would have
    reply: str
    escalated: bool = False
    escalation_reason: str | None = None
    rerouted_from: Category | None = None  # set when the router's first pick declined
