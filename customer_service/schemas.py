"""Pydantic models shared across the system.

These are the only shapes that cross a component boundary. The router emits a
Route, the orchestrator turns it into a Resolution, and escalation reads both.
Keeping them in one file means a change to the contract is one diff, not a
hunt through four modules.
"""

import uuid
from enum import StrEnum
from typing import Any, Literal

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

    # Short rather than a full uuid: it exists to correlate log lines by eye.
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    messages: list[Message] = Field(default_factory=list)

    # None until a specialist has taken the conversation. The orchestrator sets
    # this after the first turn so later turns skip the router - re-classifying
    # a one-word reply like "android" in isolation is meaningless and was
    # sending healthy conversations to a human. The specialist itself is still
    # free to notice it's the wrong fit and hand off, via the existing decline.
    assigned_category: Category | None = None

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

    This crosses the boundary to the model - it doubles as the JSON schema the
    router asks the model to fill in - so it forbids extra fields.
    """

    model_config = ConfigDict(extra="forbid")

    category: Category
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(min_length=1)


class ToolCall(BaseModel):
    """One lookup the model asked for, and what came back.

    Part of the trace rather than the answer: it records how a reply was
    reached, which is what you need when a reply looks wrong. `result` holds
    the tool's error dict too, since a failed lookup is a step worth seeing.
    """

    name: str
    arguments: dict[str, Any]
    result: dict[str, Any]


class AgentReply(BaseModel):
    """A specialist's answer, plus whether it should have been the one answering.

    Like Route, the model fills this in, so extra fields are forbidden. `handled`
    is listed before `reply` so that a model writing keys in schema order decides
    whether the problem is its own before drafting an answer to it.
    """

    model_config = ConfigDict(extra="forbid")

    handled: bool
    suggested_category: Category  # only read when handled is False
    reply: str


class AgentStep(BaseModel):
    """One specialist's attempt at a conversation."""

    category: Category
    handled: bool
    suggested_category: Category | None = None  # set when it declined
    tools: list[ToolCall] = Field(default_factory=list)
    failed: bool = False  # the call itself broke, as opposed to declining


class Trace(BaseModel):
    """How a reply was arrived at, as opposed to what the reply was.

    Separate from Resolution on purpose: a Resolution is what the customer
    sees and should stay small, while a trace is for whoever has to explain or
    debug the answer. `agents` has more than one entry when a conversation was
    re-routed; `route` is None on later turns, which skip the router entirely.
    """

    route: Route | None = None
    escalation_reason: str | None = None
    agents: list[AgentStep] = Field(default_factory=list)
    seconds: float = 0.0

    # What the router's confidence was actually measured against. Without it a
    # reader only sees "0.62" and cannot tell whether that was good enough.
    confidence_threshold: float | None = None


class Resolution(BaseModel):
    """What the orchestrator hands back to the caller."""

    conversation_id: str
    category: Category  # the specialist that finally answered, or would have
    reply: str
    escalated: bool = False
    escalation_reason: str | None = None
    rerouted_from: Category | None = None  # set when the router's first pick declined
