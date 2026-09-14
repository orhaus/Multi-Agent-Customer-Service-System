"""Contract every specialist agent implements, plus shared Claude call plumbing.

A specialist is a system prompt and a category. Everything else - the client,
the model, turning a Conversation into API messages, getting a structured
answer back - is identical between them and lives here, so adding a third
specialist means writing a prompt, not another API call.
"""

from typing import ClassVar

import anthropic
from pydantic import ValidationError

from customer_service.config import Settings, get_settings
from customer_service.schemas import AgentReply, Category, Conversation

# Support replies are a few paragraphs at most.
MAX_TOKENS = 1024

# Customer support is not a hard reasoning task. Starting low keeps replies
# fast and cheap; raise it if the eval set shows quality suffering.
EFFORT = "low"

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

    One exception type for every way a reply can fail - the API was unreachable,
    the structured output didn't validate, or it came back empty - so the
    orchestrator has exactly one thing to catch.
    """


def to_api_messages(conversation: Conversation) -> list[dict[str, str]]:
    """Translate our roles into the API's.

    We say "customer"; the API says "user". Keeping our own vocabulary means
    the domain model reads like the domain, not like a vendor's wire format.
    """
    return [
        {"role": "user" if message.role == "customer" else "assistant", "content": message.content}
        for message in conversation.messages
    ]


class Agent:
    """Base class for a specialist. Subclasses supply a category and a prompt."""

    category: ClassVar[Category]
    system_prompt: ClassVar[str]

    def __init__(
        self,
        client: anthropic.Anthropic | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._client = client or anthropic.Anthropic()

    def reply(self, conversation: Conversation) -> AgentReply:
        """Answer the customer, or say this problem belongs to someone else.

        Declining is a normal outcome and comes back as data (handled=False).
        Failing is not, and raises AgentError. The orchestrator decides what
        either means, so that logic lives in one place.
        """
        try:
            response = self._client.messages.parse(
                model=self._settings.agent_model,
                max_tokens=MAX_TOKENS,
                output_config={"effort": EFFORT},
                system=f"{self.system_prompt}\n\nYour specialism is {self.category}.\n\n{SHARED_RULES}",
                messages=to_api_messages(conversation),
                output_format=AgentReply,
            )
        except (anthropic.AnthropicError, ValidationError) as exc:
            raise AgentError(f"{type(self).__name__} could not reply") from exc

        answer = response.parsed_output
        if answer is None:
            raise AgentError(f"{type(self).__name__} returned no answer")
        if answer.handled and not answer.reply.strip():
            raise AgentError(f"{type(self).__name__} handled the conversation with an empty reply")

        return answer
