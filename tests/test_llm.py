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
from customer_service.schemas import AgentReply, Category, Message, Route, ToolCall

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


def generate(fake: FakeGemini, schema=Route, messages=None, attempts: int = 1, tools=None,
             on_tool_call=None):
    return llm.generate(
        fake.client(attempts),
        model="gemini-test",
        system="Route the message.",
        messages=messages or [Message(role="customer", content="I was charged twice")],
        schema=schema,
        thinking_level="MINIMAL",
        tools=tools,
        on_tool_call=on_tool_call,
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


# --- tool calling ---------------------------------------------------------
#
# Gemini accepts tools and a response schema in one request: the model answers
# with a function call when it needs data, and with the schema once it has it.


def lookup_invoice(invoice_id: str) -> dict:
    """Look up one invoice by its id."""
    return {"invoice_id": invoice_id, "amount": 29.00, "date": "2026-03-03"}


def explode(anything: str) -> dict:
    """A tool whose backend is having a bad day."""
    raise RuntimeError("backend down")


def calls_tool(name: str, **args) -> tuple[int, dict]:
    """A response in which the model asks for a tool instead of answering."""
    return 200, {
        "candidates": [
            {
                "content": {"role": "model", "parts": [{"functionCall": {"name": name, "args": args}}]},
                "finishReason": "STOP",
            }
        ]
    }


def test_a_tool_call_is_run_and_its_result_sent_back():
    fake = FakeGemini(calls_tool("lookup_invoice", invoice_id="inv_1042"), ok(ROUTE_JSON))

    result = generate(fake, tools=[lookup_invoice])

    assert result.category is Category.BILLING  # the model got its answer out
    assert len(fake.requests) == 2

    sent_back = fake.requests[1]["contents"][-1]["parts"][0]["functionResponse"]
    assert sent_back["name"] == "lookup_invoice"
    assert sent_back["response"]["amount"] == 29.00


def test_tools_are_declared_from_the_plain_python_functions():
    """tools.py stays provider-agnostic; the SDK reads signature and docstring."""
    fake = FakeGemini(ok(ROUTE_JSON))
    generate(fake, tools=[lookup_invoice])

    declared = fake.requests[0]["tools"][0]["functionDeclarations"][0]
    assert declared["name"] == "lookup_invoice"
    assert "invoice" in declared["description"].lower()
    assert "invoice_id" in declared["parameters"]["properties"]


def test_without_tools_the_request_declares_none():
    fake = FakeGemini(ok(ROUTE_JSON))
    generate(fake)
    assert "tools" not in fake.requests[0]


def test_the_models_own_turn_is_echoed_back_verbatim():
    """On Gemini 3 those parts carry thought signatures; rebuilding loses them."""
    fake = FakeGemini(calls_tool("lookup_invoice", invoice_id="inv_1042"), ok(ROUTE_JSON))
    generate(fake, tools=[lookup_invoice])

    echoed = fake.requests[1]["contents"][-2]
    assert echoed["role"] == "model"
    assert echoed["parts"][0]["functionCall"]["name"] == "lookup_invoice"


def test_a_tool_the_model_invented_comes_back_as_an_error():
    fake = FakeGemini(calls_tool("get_the_moon", thing="cheese"), ok(ROUTE_JSON))

    result = generate(fake, tools=[lookup_invoice])

    assert result.category is Category.BILLING  # the turn survives
    response = fake.requests[1]["contents"][-1]["parts"][0]["functionResponse"]["response"]
    assert "error" in response


def test_a_tool_that_raises_becomes_an_error_not_a_crash():
    """A bug in our own backend must not kill the customer's turn."""
    fake = FakeGemini(calls_tool("explode", anything="please"), ok(ROUTE_JSON))

    result = generate(fake, tools=[explode])

    assert result.category is Category.BILLING
    response = fake.requests[1]["contents"][-1]["parts"][0]["functionResponse"]["response"]
    assert "backend down" in response["error"]


def test_a_model_that_never_stops_calling_tools_is_cut_off():
    endless = [calls_tool("lookup_invoice", invoice_id="inv_1042")] * (llm.MAX_TOOL_ROUNDS + 1)
    fake = FakeGemini(*endless)

    with pytest.raises(llm.LLMError, match="giving up"):
        generate(fake, tools=[lookup_invoice])

    assert len(fake.requests) == llm.MAX_TOOL_ROUNDS + 1


# --- reporting tool calls to the caller -----------------------------------
#
# Lookups happen deep in here, but the trace that explains a reply is assembled
# further up. The callback is how one reaches the other.


def test_each_tool_call_is_reported_with_its_arguments_and_result():
    fake = FakeGemini(calls_tool("lookup_invoice", invoice_id="inv_1042"), ok(ROUTE_JSON))
    seen: list[ToolCall] = []

    generate(fake, tools=[lookup_invoice], on_tool_call=seen.append)

    assert seen == [
        ToolCall(
            name="lookup_invoice",
            arguments={"invoice_id": "inv_1042"},
            result={"invoice_id": "inv_1042", "amount": 29.00, "date": "2026-03-03"},
        )
    ]


def test_a_failed_lookup_is_reported_too():
    """A tool that errored is a step worth seeing, not one to hide."""
    fake = FakeGemini(calls_tool("explode", anything="please"), ok(ROUTE_JSON))
    seen: list[ToolCall] = []

    generate(fake, tools=[explode], on_tool_call=seen.append)

    assert len(seen) == 1
    assert "backend down" in seen[0].result["error"]


def test_nothing_is_reported_when_the_model_answers_directly():
    seen: list[ToolCall] = []
    generate(FakeGemini(ok(ROUTE_JSON)), tools=[lookup_invoice], on_tool_call=seen.append)
    assert seen == []


def test_reporting_is_optional():
    """Callers that don't want the trace pass nothing and are unaffected."""
    fake = FakeGemini(calls_tool("lookup_invoice", invoice_id="inv_1042"), ok(ROUTE_JSON))
    assert generate(fake, tools=[lookup_invoice]).category is Category.BILLING
