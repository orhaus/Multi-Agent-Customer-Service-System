"""Contract every specialist agent implements, plus the shared model call.

A specialist is a system prompt and a category. Everything else - the client,
the model, getting a structured answer back - is identical between them and
lives here, so adding a third specialist means writing a prompt, not another
model call.
"""

from typing import ClassVar

from google import genai

from customer_service import llm
from customer_service.config import Settings, get_settings
from customer_service.schemas import AgentReply, Category, Conversation

# Customer support is not a hard reasoning task. Starting low keeps replies fast
# and within free-tier limits; raise it if the eval set shows quality suffering.
THINKING: llm.ThinkingLevel = "LOW"

SHARED_RULES = """\
You are replying directly to a customer. Be concise, specific and warm.

Never invent an account detail, a charge, a date or a policy. If answering \
needs information you do not have, say what you need and ask for it.

Answer in the structured format you have been given:
- handled: true if you are the right specialist for this problem, false if you \
are not.
- suggested_category: when handled is false, the specialism the problem really \
belongs to - billing or technical - or unknown if it needs a person rather than \
a different specialist. When handled is true, your own specialism.
- reply: your message to the customer. It is shown only when handled is true, \
so never tell the customer you are passing them on - the system does that.\
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

    def __init__(
        self,
        client: genai.Client | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._client = client or llm.make_client(self._settings)

    def reply(self, conversation: Conversation) -> AgentReply:
        """Answer the customer, or say this problem belongs to someone else.

        Declining is a normal outcome and comes back as data (handled=False).
        Failing is not, and raises AgentError. The orchestrator decides what
        either means, so that logic lives in one place.
        """
        try:
            answer = llm.generate(
                self._client,
                model=self._settings.agent_model,
                system=f"{self.system_prompt}\n\nYour specialism is {self.category}.\n\n{SHARED_RULES}",
                messages=conversation.messages,
                schema=AgentReply,
                thinking_level=THINKING,
            )
        except llm.LLMError as exc:
            raise AgentError(f"{type(self).__name__} could not reply") from exc

        if answer.handled and not answer.reply.strip():
            raise AgentError(f"{type(self).__name__} handled the conversation with an empty reply")

        return answer
