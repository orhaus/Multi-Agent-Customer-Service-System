# Multi-Agent Customer Service System

A customer support system built from several specialist LLM agents rather than one
general one. A cheap classifier routes each conversation to a billing or technical
specialist; the specialist answers from real records using tool calls; anything the
system should not decide on its own goes to a human, with a named reason.

Built on the Gemini API with hand-written orchestration — no agent framework.

---

## What it does

A live session with the billing agent. The invoice numbers are looked up, not invented:

```
you> I think I was charged twice in March. Can you check?

     customer_service.llm: Ran tools: list_invoices

billing> I checked your account and you were indeed billed twice for the Pro plan
         in March. You received two $29.00 charges on March 3, 2026 (invoice
         numbers inv_1042 and inv_1043). Let me get a person to help process a
         refund for the duplicate charge right away.
```

The agent chose to call `list_invoices`, read the customer's records, found two
charges on the same date, and quoted their real ids. Without tools it could only
have asked the customer for details it had no way to check.

---

## Results

The router is measured against a labelled set of 37 support messages
([`evals/router_cases.json`](evals/router_cases.json)), including terse real-world
phrasing, messages that genuinely belong to both specialists, and adversarial ones
whose wording points at the wrong one.

| | Before prompt fix | After |
|---|---|---|
| Accuracy | 35/37 (95%) | **37/37 (100%)** |
| Confidently wrong | 0 | **0** |
| Over-escalated | 3 | **0** |
| Mean confidence when right | 0.90 | 0.96 |
| Mean confidence when wrong | 0.35 | — |

**"Confidently wrong" is the number that matters.** A wrong answer *below* the
confidence threshold escalates to a human and harms nobody. A wrong answer *above*
it reaches the wrong specialist silently. Across both runs that count stayed at
zero — and in the baseline, mean confidence was 0.90 when the router was right
against 0.35 when it was wrong, so the confidence score carries real signal rather
than being decoration.

The eval also caught a bug in the system's own instructions. The router prompt said
`unknown` covered anything that "fits both" categories — written before re-routing
existed, when a two-specialism message really was unroutable. Re-routing made a
wrong pick cheap, but the prompt was never updated, so the router kept sending
those to a human. One paragraph fixed it; the eval proved it, and proved nothing
else regressed.

```bash
python -m evals.run_router --rpm 8 --out evals/runs/today.json
```

---

## How it works

```
customer message
      │
      ▼
 first turn of this conversation?
      │
   yes│                                 no (a specialist already owns it)
      ▼                                        │
 ROUTER  (small model, latest message only)    │
 → category + confidence + reasoning           │
      │                                        │
      ▼                                        ▼
 ESCALATION RULES                        turn limit reached?
 unknown category?    → human                 │ yes → human
 confidence < 0.7?    → human                 │ no
 turn limit reached?  → human                 │
      │ none fire                             │
      └──────────────────┬────────────────────┘
                         ▼
                 SPECIALIST AGENT  (full conversation + its tools)
                 returns: handled? + suggested_category + reply
                         │
       ┌─────────────────┼──────────────────────┐
   handled            declined                 error
       │                 │                       │
       ▼                 ▼                       ▼
  send reply     re-route to the            → human
  and remember   suggested specialist,
  who owns it    once
                         │ declines too
                         ▼
                      → human
```

| Module | Responsibility |
|---|---|
| [`router.py`](customer_service/router.py) | Classifies the opening message. One cheap call, no history. |
| [`escalation.py`](customer_service/escalation.py) | When a human takes over. Pure logic, no model call. |
| [`agents/`](customer_service/agents/) | Billing and technical specialists. A specialist is a prompt, a category and its tools. |
| [`tools.py`](customer_service/tools.py) | In-memory stand-in for the billing backend. |
| [`orchestrator.py`](customer_service/orchestrator.py) | Wires the above together, including the re-route loop. |
| [`llm.py`](customer_service/llm.py) | The only module that knows we use Gemini. |
| [`schemas.py`](customer_service/schemas.py) | The shapes that cross component boundaries. |

---

## Design decisions

**Hand-written orchestration, not LangGraph.** The flow is linear with one branch
and a capped re-route. It has no cycles, no checkpointing and no human-in-the-loop
interrupts — the cases where a graph framework earns its dependency. Worth
revisiting if agents ever need to hand off freely.

**Agents return structured data, not prose.** A specialist answers with
`{handled, suggested_category, reply}`. Because "this isn't mine" is *data* rather
than a sentence buried in text, the orchestrator can act on it and re-route. An
earlier version returned a plain string: the billing agent would tell a customer
"a colleague will pick this up" and no colleague was ever told.

**Failures degrade, they don't crash.** The router returns `unknown` at zero
confidence when a call fails, which escalates. An agent failure escalates. A tool
that raises returns its error *to the model* as data, so a wrong invoice id costs a
round trip instead of the conversation. This was tested by accident: hitting the
free-tier rate limit produced a clean handoff to a human rather than a stack trace.

**The router runs once per conversation, not once per turn.** It only sees the
latest message, so re-running it mid-conversation asked it to classify replies like
"android" in isolation — which correctly came back unclassifiable and threw away a
working conversation. A specialist now keeps the conversation, and its own decline
is what notices if a later message isn't theirs.

**Provider-specific code lives in one file.** The project was migrated from the
Anthropic API to Gemini by rewriting [`llm.py`](customer_service/llm.py) and its
tests. The routing rules, escalation logic, schemas and CLI were untouched.

**`response.parsed` is deliberately unused.** The Gemini SDK's convenience accessor
swallows validation errors and returns `None`, so a confidence of `1.4` would be
indistinguishable from an empty response. `llm.py` validates the raw text against
the pydantic schema itself so those failures stay distinct.

---

## Running it

Requires Python 3.11+ (developed on 3.14) and a free
[Google AI Studio](https://aistudio.google.com/apikey) API key.

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows; use source .venv/bin/activate elsewhere
pip install -r requirements.txt
cp .env.example .env                # then add your GEMINI_API_KEY
python -m customer_service
```

Each reply is annotated with the decision behind it:

```
support> Sorry about that - what error do you see when you sign in?
         [answered by technical, rerouted from billing]
```

Configuration lives in `.env` ([`.env.example`](.env.example) is the contract):
model per role, the confidence threshold, and the agent turn budget.

**Free-tier limits are tight** — 5 requests/minute on `gemini-3.8-flash`. Each
message costs a router call plus one or two agent calls, so a fast conversation can
hit the ceiling. It degrades to a handoff rather than an error.

---

## Tests

```bash
pytest
```

98 tests, and **none of them need an API key, a network connection or quota.** Unit
tests mock at the project's own seams; the Gemini layer is covered by contract tests
that run the real `google-genai` SDK against a fake HTTP transport, so requests are
genuinely built and serialised by the SDK and responses genuinely parsed by it.

That distinction found two real bugs a self-written mock never would: the silent
`response.parsed` failure above, and network errors escaping as raw `httpx`
exceptions rather than the SDK's own error type.

---

## Layout

```
customer_service/      the system
├── llm.py             the only Gemini-aware module
├── router.py          classification
├── escalation.py      handoff rules
├── orchestrator.py    wiring and re-routing
├── agents/            billing and technical specialists
├── tools.py           fake billing backend
├── schemas.py         shared shapes
├── config.py          validated settings
└── __main__.py        terminal chat
evals/                 the router eval set, scorer and saved runs
tests/                 98 offline tests
```

---

## Not built

Deliberate omissions rather than oversights:

- **No HTTP API or web UI.** The system is a library plus a terminal client.
  `Orchestrator.handle()` is the seam an API would sit on — though conversation
  state, which the CLI holds in memory, would need somewhere to live.
- **The technical agent has no tools.** Only billing can look records up.
- **Customer identity is fixed.** Agents are handed a demo customer id rather than
  working out who is speaking.
- **Only the router is evaluated.** Agent reply quality is not measured yet. One
  known issue it would catch: an agent occasionally says a person will follow up
  while still marking the conversation handled, so no handoff actually happens.
