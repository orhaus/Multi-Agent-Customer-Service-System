"""Talk to the support system from a terminal.

    python -m customer_service

Each line you type is a customer message. Under every reply is one line saying
which specialist answered, whether it was re-routed, and why a conversation was
handed off - so you can watch the routing decisions as they happen.
"""

import logging
import uuid
from collections.abc import Callable

from pydantic import ValidationError

from customer_service.config import get_settings
from customer_service.orchestrator import Orchestrator
from customer_service.schemas import Conversation, Message, Resolution

EXIT_WORDS = {"exit", "quit"}


def new_conversation() -> Conversation:
    return Conversation(id=uuid.uuid4().hex[:8])


def describe(result: Resolution) -> str:
    """The routing detail under each reply - the part worth watching."""
    if result.escalated:
        detail = f"handed to a human: {result.escalation_reason}"
    else:
        detail = f"answered by {result.category}"
    if result.rerouted_from:
        detail += f", rerouted from {result.rerouted_from}"
    return f"[{detail}]"


def chat(
    orchestrator: Orchestrator,
    read: Callable[[str], str] = input,
    write: Callable[[str], None] = print,
) -> None:
    """Run the conversation loop until the user exits.

    `read` and `write` default to the terminal; tests pass their own, which is
    what lets the loop be tested without typing into it.
    """
    conversation = new_conversation()
    write("Customer support. Type a message, or 'exit' to quit.\n")

    while True:
        try:
            text = read("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            write("")
            return

        if not text:
            continue
        if text.lower() in EXIT_WORDS:
            return

        conversation.messages.append(Message(role="customer", content=text))
        result = orchestrator.handle(conversation)

        write(f"support> {result.reply}")
        write(f"         {describe(result)}\n")

        if result.escalated:
            # A person owns this conversation now; the bot shouldn't keep talking in it.
            write("(A person has this conversation now. Starting a new one.)\n")
            conversation = new_conversation()
        else:
            conversation.messages.append(Message(role="assistant", content=result.reply))


def main() -> int:
    try:
        settings = get_settings()
    except ValidationError as exc:
        print("Configuration problem - check your .env against .env.example.\n")
        print(exc)
        return 1

    # Our own logs at the configured level; everyone else's (the HTTP client
    # logs every request at INFO) only when something is wrong.
    logging.basicConfig(level=logging.WARNING, format="  %(levelname)s %(name)s: %(message)s")
    logging.getLogger("customer_service").setLevel(settings.log_level)

    chat(Orchestrator(settings=settings))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
