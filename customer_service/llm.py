"""Everything that knows we are talking to Gemini lives here.

The router and the agents ask for one thing: "a validated instance of this
pydantic model, given this system prompt and these messages". This module turns
that into a Gemini call, and turns every way the call can go wrong into a single
exception. Switching provider means rewriting this file, not the components
that use it.
"""

import copy
import logging
from collections.abc import Callable, Sequence
from typing import Any, Literal, TypeVar

import httpx
from google import genai
from google.genai import errors, types
from pydantic import BaseModel, ValidationError

from customer_service.config import Settings
from customer_service.schemas import Message

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

ThinkingLevel = Literal["MINIMAL", "LOW", "MEDIUM", "HIGH"]

# How many times the model may call tools before we stop it. A lookup or two
# is normal; a model still asking on the fourth round is looping, not working.
MAX_TOOL_ROUNDS = 3

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


def _declare(client: genai.Client, functions: Sequence[Callable[..., dict]]) -> list[types.Tool]:
    """Describe plain Python functions to Gemini.

    from_callable reads the signature and the docstring, so tools.py stays
    provider-agnostic: plain typed functions, no Gemini types in sight.
    """
    return [
        types.Tool(
            function_declarations=[
                types.FunctionDeclaration.from_callable(client=client, callable=function)
                for function in functions
            ]
        )
    ]


def _run_tool(call: types.FunctionCall, functions: dict[str, Callable[..., dict]]) -> dict:
    """Run one requested tool, turning any failure into data.

    Whatever comes back is sent to the model, so a wrong id or an outright bug
    becomes something it can read and recover from rather than a dead turn.
    """
    function = functions.get(call.name or "")
    if function is None:
        logger.warning("Model asked for unknown tool %s", call.name)
        return {"error": f"No tool named {call.name}."}

    try:
        return function(**dict(call.args or {}))
    except Exception as exc:  # noqa: BLE001 - a tool bug must not kill the conversation
        logger.exception("Tool %s failed", call.name)
        return {"error": f"{call.name} failed: {exc}"}


def _send(
    client: genai.Client,
    model: str,
    contents: list[types.Content],
    config: types.GenerateContentConfig,
) -> types.GenerateContentResponse:
    """One request, with every transport failure translated to LLMError."""
    try:
        return client.models.generate_content(model=model, contents=contents, config=config)
    except errors.APIError as exc:
        raise LLMError(f"Gemini API error {exc.code}: {exc.message}") from exc
    except httpx.HTTPError as exc:
        # Network failures surface as raw httpx errors, not as APIError.
        raise LLMError(f"Could not reach Gemini: {exc}") from exc


def generate(
    client: genai.Client,
    *,
    model: str,
    system: str,
    messages: list[Message],
    schema: type[T],
    thinking_level: ThinkingLevel,
    tools: Sequence[Callable[..., dict]] | None = None,
) -> T:
    """Ask the model for an instance of `schema`, validated by us.

    With `tools`, the model may answer with a function call instead of an
    answer. We run it, hand back the result, and ask again - until it has what
    it needs to reply. Gemini accepts tools and a response schema in the same
    request, so this stays one call path rather than two separate phases.

    The SDK offers response.parsed, but it swallows validation errors and
    returns None - an out-of-range answer would look exactly like an empty one.
    Validating response.text ourselves keeps those failures distinct.
    """
    functions = {function.__name__: function for function in tools or ()}
    config = types.GenerateContentConfig(
        system_instruction=system,
        response_mime_type="application/json",
        response_json_schema=json_schema(schema),
        thinking_config=types.ThinkingConfig(thinking_level=thinking_level),
        # The SDK can run tools itself, but then we could not log or cap them.
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        tools=_declare(client, tools) if functions else None,
    )
    contents = to_contents(messages)

    for _ in range(MAX_TOOL_ROUNDS + 1):
        response = _send(client, model, contents, config)

        # Check for tool calls before touching .text: a turn that calls a tool
        # has no text, and reading it would log an SDK warning every time.
        calls = response.function_calls
        if not calls:
            break

        # Echo the model's own turn back verbatim rather than rebuilding it.
        # On Gemini 3 those parts carry thought signatures, and dropping them
        # loses the reasoning that led to the call.
        contents.append(response.candidates[0].content)
        contents.append(
            types.Content(
                role="user",
                parts=[
                    types.Part.from_function_response(
                        name=call.name, response=_run_tool(call, functions)
                    )
                    for call in calls
                ],
            )
        )
        logger.info("Ran tools: %s", ", ".join(call.name or "?" for call in calls))
    else:
        raise LLMError(f"Still calling tools after {MAX_TOOL_ROUNDS} rounds; giving up")

    text = response.text
    if not text:
        reason = response.candidates[0].finish_reason if response.candidates else "no candidates"
        raise LLMError(f"Gemini returned no text (finish reason: {reason})")

    try:
        return schema.model_validate_json(text)
    except ValidationError as exc:
        raise LLMError(f"Gemini output did not match {schema.__name__}") from exc
