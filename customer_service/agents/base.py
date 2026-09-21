"""Contract every specialist agent implements, plus the shared model call.

A specialist is a system prompt and a category. Everything else - the client,
the model, getting a structured answer back - is identical between them and
lives here, so adding a third specialist means writing a prompt, not another
model call.
"""

from collections.abc import Callable, Sequence
from typing import ClassVar

from google import genai

from customer_service import llm
from customer_service.config import Settings, get_settings
from customer_service.schemas import AgentReply, Category, Conversation, ToolCall
from customer_service.tools import DEMO_CUSTOMER_ID

# Customer support is not a hard reasoning task. Starting low keeps replies fast
# and within free-tier limits; raise it if the eval set shows quality suffering.
THINKING: llm.ThinkingLevel = "LOW"

SHARED_RULES = """\
You are replying directly to a customer. Be concise, specific and warm.

Never invent an account detail, a charge, a date or a policy. When you have a \
tool that can find it, use the tool and answer from what it returns. When you \
do not, say what you need and ask the customer for it.

Answer in the structured format you have been given:
- handled: true if you are the right specialist for this problem, false if you \
are not.
- suggested_category: when handled is false, the specialism the problem really \
belongs to - billing or technical - or unknown if it needs a person rather than \
a different specialist. When handled is true, your own specialism.
- reply: your message to the customer. What it should contain depends on the \
other two fields:
  - handled true: your answer.
  - handled false, suggested_category unknown: what you established, in a \
sentence or two. This IS shown to the customer before a person takes over, and \
it is what that colleague starts from, so give them the facts you found.
  - handled false, naming another specialism: not shown to anyone. Keep it short.

Never tell the customer you are passing them on or that someone will follow up \
- the system says that itself, and saying it twice reads badly.\
"""


class AgentError(Exception):
    """An agent could not produce a usable reply.

    Wraps every model failure, plus the one domain failure the model layer
    can't see - handling a conversation with an empty reply - so the
    orchestrator has exactly one thing to catch.
    """


class Agent:
    """Base class for a specialist. Subclasses supply a category and a prompt."""

    category: ClassVar[Category]
    system_prompt: ClassVar[str]

    # What this specialist may look up. Empty means it answers from the
    # conversation alone, which is where every specialist starts.
    tools: ClassVar[Sequence[Callable[..., dict]]] = ()

    def __init__(
        self,
        client: genai.Client | None = None,
        settings: Settings | None = None,
        customer_id: str = DEMO_CUSTOMER_ID,
    ) -> None:
        self._settings = settings or get_settings()
        self._client = client or llm.make_client(self._settings)
        # Tools are per-customer, so the agent has to know who it is helping.
        # Working that out from the conversation would be its own feature; for
        # now the demo customer is the default and callers can override it.
        self._customer_id = customer_id

    def reply(
        self,
        conversation: Conversation,
        on_tool_call: Callable[[ToolCall], None] | None = None,
    ) -> AgentReply:
        """Answer the customer, or say this problem belongs to someone else.

        Declining is a normal outcome and comes back as data (handled=False).
        Failing is not, and raises AgentError. The orchestrator decides what
        either means, so that logic lives in one place.

        `on_tool_call` is passed straight through to the model layer: this
        agent doesn't collect its own lookups, it just lets the caller watch.
        """
        system = (
            f"{self.system_prompt}\n\n"
            f"Your specialism is {self.category}.\n"
            f"You are helping customer {self._customer_id}.\n\n"
            f"{SHARED_RULES}"
        )
        try:
            answer = llm.generate(
                self._client,
                model=self._settings.agent_model,
                system=system,
                messages=conversation.messages,
                schema=AgentReply,
                thinking_level=THINKING,
                tools=self.tools or None,
                on_tool_call=on_tool_call,
            )
        except llm.LLMError as exc:
            raise AgentError(f"{type(self).__name__} could not reply") from exc

        if answer.handled and not answer.reply.strip():
            raise AgentError(f"{type(self).__name__} handled the conversation with an empty reply")

        return answer
