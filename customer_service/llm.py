"""Everything that knows we are talking to Gemini lives here.

The router and the agents ask for one thing: "a validated instance of this
pydantic model, given this system prompt and these messages". This module turns
that into a Gemini call, and turns every way the call can go wrong into a single
exception. Switching provider means rewriting this file, not the components
that use it.
"""

import copy
from typing import Any, Literal, TypeVar

import httpx
from google import genai
from google.genai import errors, types
from pydantic import BaseModel, ValidationError

from customer_service.config import Settings
from customer_service.schemas import Message

T = TypeVar("T", bound=BaseModel)

ThinkingLevel = Literal["MINIMAL", "LOW", "MEDIUM", "HIGH"]

# Free-tier quotas are per minute, so a short burst of retries on 429s and 5xxs
# often gets through. Kept short, because a customer is waiting.
RETRY = types.HttpRetryOptions(attempts=3, initial_delay=1.0, max_delay=8.0)


class LLMError(Exception):
    """A model call did not produce a valid instance of the requested schema.

    One type for every failure: the API refusing (quota, bad request, server
    error), the network failing, an empty or blocked response, and output that
    doesn't validate. The original is kept as __cause__ for the logs.
    """


def make_client(settings: Settings) -> genai.Client:
    return genai.Client(
        api_key=settings.gemini_api_key.get_secret_value(),
        http_options=types.HttpOptions(retry_options=RETRY),
    )


def json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """The model's JSON Schema with every $ref inlined.

    Gemini documents enum, additionalProperties and minimum/maximum in its JSON
    Schema mode, but not $defs - which pydantic emits for nested types such as
    Category. Inlining them means we only send what the API says it supports.
    Assumes no recursive models, which this project doesn't have.
    """
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                definition = copy.deepcopy(defs[node["$ref"].rsplit("/", 1)[-1]])
                siblings = {key: value for key, value in node.items() if key != "$ref"}
                return resolve(definition | siblings)
            return {key: resolve(value) for key, value in node.items()}
        if isinstance(node, list):
            return [resolve(item) for item in node]
        return node

    return resolve(schema)


def to_contents(messages: list[Message]) -> list[types.Content]:
    """Our roles in Gemini's vocabulary: customer -> user, assistant -> model."""
    return [
        types.Content(
            role="user" if message.role == "customer" else "model",
            parts=[types.Part.from_text(text=message.content)],
        )
        for message in messages
    ]


def generate(
    client: genai.Client,
    *,
    model: str,
    system: str,
    messages: list[Message],
    schema: type[T],
    thinking_level: ThinkingLevel,
) -> T:
    """Ask the model for an instance of `schema`, validated by us.

    The SDK offers response.parsed, but it swallows validation errors and
    returns None - an out-of-range answer would look exactly like an empty one.
    Validating response.text ourselves keeps those failures distinct.
    """
    config = types.GenerateContentConfig(
        system_instruction=system,
        response_mime_type="application/json",
        response_json_schema=json_schema(schema),
        thinking_config=types.ThinkingConfig(thinking_level=thinking_level),
        # We never pass tools. Left enabled, the SDK logs a warning on every call.
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    try:
        response = client.models.generate_content(
            model=model, contents=to_contents(messages), config=config
        )
    except errors.APIError as exc:
        raise LLMError(f"Gemini API error {exc.code}: {exc.message}") from exc
    except httpx.HTTPError as exc:
        # Network failures surface as raw httpx errors, not as APIError.
        raise LLMError(f"Could not reach Gemini: {exc}") from exc

    text = response.text
    if not text:
        reason = response.candidates[0].finish_reason if response.candidates else "no candidates"
        raise LLMError(f"Gemini returned no text (finish reason: {reason})")

    try:
        return schema.model_validate_json(text)
    except ValidationError as exc:
        raise LLMError(f"Gemini output did not match {schema.__name__}") from exc
