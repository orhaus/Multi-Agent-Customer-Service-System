"""Contract tests for the Gemini layer.

These run the real google-genai SDK against a fake HTTP backend. Requests are
genuinely built and serialised by the SDK; responses are canned JSON that the
SDK genuinely parses. So they check our assumptions about the library - which a
mock of our own code can never do - without a key, a network, or a quota.
"""

import json

import httpx
import pytest
from google import genai
from google.genai import types
from pydantic import ValidationError

from customer_service import llm
from customer_service.schemas import AgentReply, Category, Message, Route

ROUTE_JSON = '{"category": "billing", "confidence": 0.9, "reasoning": "duplicate charge"}'
QUOTA = (429, {"error": {"code": 429, "message": "Resource has been exhausted", "status": "RESOURCE_EXHAUSTED"}})
SERVER_ERROR = (500, {"error": {"code": 500, "message": "Internal error", "status": "INTERNAL"}})
NETWORK_DOWN = "network down"


def ok(text: str, thought: str | None = None) -> tuple[int, dict]:
    parts = ([{"text": thought, "thought": True}] if thought else []) + [{"text": text}]
    return 200, {"candidates": [{"content": {"role": "model", "parts": parts}, "finishReason": "STOP"}]}


class FakeGemini:
    """Stands in for Gemini's HTTP API: records each request, plays back canned responses."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests: list[dict] = []

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        response = self.responses.pop(0)
        if response == NETWORK_DOWN:
            raise httpx.ConnectError("no network", request=request)
        status, body = response
        return httpx.Response(status, json=body)

    def client(self, attempts: int = 1) -> genai.Client:
        return genai.Client(
            api_key="test-key",
            http_options=types.HttpOptions(
                httpx_client=httpx.Client(transport=httpx.MockTransport(self._handle)),
                retry_options=types.HttpRetryOptions(attempts=attempts, initial_delay=0.01, max_delay=0.02),
            ),
        )


def generate(fake: FakeGemini, schema=Route, messages=None, attempts: int = 1):
    return llm.generate(
        fake.client(attempts),
        model="gemini-test",
        system="Route the message.",
        messages=messages or [Message(role="customer", content="I was charged twice")],
        schema=schema,
        thinking_level="MINIMAL",
    )


# --- requests -------------------------------------------------------------


def test_the_request_carries_everything_gemini_needs():
    fake = FakeGemini(ok(ROUTE_JSON))
    generate(
        fake,
        messages=[
            Message(role="customer", content="charged twice"),
            Message(role="assistant", content="which invoice?"),
            Message(role="customer", content="42"),
        ],
    )
    body = fake.requests[0]
    config = body["generationConfig"]

    assert body["systemInstruction"]["parts"][0]["text"] == "Route the message."
    assert [content["role"] for content in body["contents"]] == ["user", "model", "user"]
    assert config["responseMimeType"] == "application/json"
    assert config["responseJsonSchema"]["properties"]["category"]["enum"] == ["billing", "technical", "unknown"]
    assert "MINIMAL" in json.dumps(config["thinkingConfig"])


@pytest.mark.parametrize("model", [Route, AgentReply])
def test_schemas_are_sent_without_defs_or_refs(model):
    """Gemini doesn't document $defs, so we never send it."""
    sent = json.dumps(llm.json_schema(model))
    assert "$defs" not in sent
    assert "$ref" not in sent


def test_inlining_keeps_the_enum_and_the_no_extra_fields_rule():
    schema = llm.json_schema(AgentReply)
    assert schema["properties"]["suggested_category"]["enum"] == ["billing", "technical", "unknown"]
    assert schema["additionalProperties"] is False


def test_our_roles_become_geminis():
    contents = llm.to_contents(
        [Message(role="customer", content="hi"), Message(role="assistant", content="hello")]
    )
    assert [(content.role, content.parts[0].text) for content in contents] == [
        ("user", "hi"),
        ("model", "hello"),
    ]


# --- responses ------------------------------------------------------------


def test_a_valid_response_comes_back_as_a_validated_model():
    result = generate(FakeGemini(ok(ROUTE_JSON)))
    assert result == Route(category=Category.BILLING, confidence=0.9, reasoning="duplicate charge")


def test_agent_replies_round_trip_too():
    reply = '{"handled": false, "suggested_category": "technical", "reply": ""}'
    result = generate(FakeGemini(ok(reply)), schema=AgentReply)
    assert result == AgentReply(handled=False, suggested_category=Category.TECHNICAL, reply="")


def test_thought_parts_are_not_part_of_the_answer():
    result = generate(FakeGemini(ok(ROUTE_JSON, thought="let me think about this")))
    assert result.category is Category.BILLING


def test_out_of_range_values_raise_instead_of_quietly_becoming_none():
    """The SDK's own response.parsed swallows this and returns None. We don't use it."""
    fake = FakeGemini(ok('{"category": "billing", "confidence": 1.4, "reasoning": "overconfident"}'))
    with pytest.raises(llm.LLMError) as info:
        generate(fake)
    assert isinstance(info.value.__cause__, ValidationError)


def test_text_that_is_not_json_raises():
    with pytest.raises(llm.LLMError):
        generate(FakeGemini(ok("Sorry, I can't help with that.")))


def test_an_empty_response_raises():
    with pytest.raises(llm.LLMError, match="no text"):
        generate(FakeGemini((200, {"candidates": []})))


# --- failures -------------------------------------------------------------


def test_a_quota_error_raises_llm_error():
    with pytest.raises(llm.LLMError, match="429"):
        generate(FakeGemini(QUOTA))


def test_a_server_error_raises_llm_error():
    with pytest.raises(llm.LLMError, match="500"):
        generate(FakeGemini(SERVER_ERROR))


def test_a_network_failure_raises_llm_error_not_a_raw_httpx_error():
    """The SDK lets connection errors escape as httpx exceptions, not APIError."""
    with pytest.raises(llm.LLMError, match="Could not reach Gemini"):
        generate(FakeGemini(NETWORK_DOWN))


def test_retries_recover_from_a_brief_quota_error():
    fake = FakeGemini(QUOTA, QUOTA, ok(ROUTE_JSON))
    result = generate(fake, attempts=3)

    assert result.category is Category.BILLING
    assert len(fake.requests) == 3
