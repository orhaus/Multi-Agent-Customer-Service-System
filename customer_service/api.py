"""HTTP interface to the support system.

    uvicorn customer_service.api:app --reload

Deliberately stateless: the client sends the whole conversation with every
request and gets the updated one back. The alternative - keeping sessions in
memory here - would work on one machine and break the moment a second replica
exists, because a customer's second message could land on the replica that
never saw the first.

The cost of statelessness is that the conversation is client-controlled: a
caller can edit the history, or set assigned_category to skip routing. That is
fine for a demo and would not be for a real deployment, which would sign the
conversation or keep it in a store keyed by a session id.
"""

from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from customer_service.orchestrator import Orchestrator
from customer_service.schemas import Conversation, Resolution, Trace

app = FastAPI(
    title="Multi-Agent Customer Service",
    description="Routes support conversations to specialist agents, or to a human.",
    version="0.1.0",
)


@lru_cache(maxsize=1)
def get_orchestrator() -> Orchestrator:
    """Built once and reused.

    Per request would mean a new API client and a re-read of settings every
    time. Cached rather than built at import so that a missing GEMINI_API_KEY
    surfaces on the first request instead of breaking collection in tests that
    never call the API. Tests replace this through app.dependency_overrides.
    """
    return Orchestrator()


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, description="What the customer just said.")
    conversation: Conversation | None = Field(
        default=None,
        description="The conversation so far, as returned by the previous call. "
        "Omit it to start a new one.",
    )


class ChatResponse(BaseModel):
    resolution: Resolution = Field(description="What the customer should see next.")
    trace: Trace = Field(
        description="How that reply was reached: the routing decision, every specialist "
        "tried, the lookups each one made, and why a conversation was escalated."
    )
    conversation: Conversation = Field(
        description="Send this back with the next message so the conversation continues. "
        "After an escalation, start a new one instead - a person has this conversation."
    )


@app.get("/health", summary="Liveness check")
def health() -> dict[str, str]:
    """Deliberately does not call the model: this answers 'is the process up',
    not 'is Gemini up'. Mixing the two makes a deploy look unhealthy during an
    upstream outage it is designed to survive."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse, summary="Send a customer message")
def chat(
    request: ChatRequest,
    orchestrator: Annotated[Orchestrator, Depends(get_orchestrator)],
) -> ChatResponse:
    """Answer one customer message.

    Never fails because of the model: a quota error, an unreachable API or an
    unusable answer all come back as an escalation with a reason, which is what
    a customer waiting on support should get.
    """
    conversation = request.conversation or Conversation()
    trace = Trace()
    resolution = orchestrator.turn(conversation, request.message, trace)
    return ChatResponse(resolution=resolution, trace=trace, conversation=conversation)


# Mounted last, and at "/", so it catches everything the routes above didn't.
# Serving the page from the same app means one process and one container - no
# CORS, no second deploy, nothing to build.
app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True))
