"""Measure how well the router classifies real support messages.

    python -m evals.run_router                 the whole set
    python -m evals.run_router --limit 5       a cheap smoke test
    python -m evals.run_router --rpm 8         slower, for a tighter quota
    python -m evals.run_router --out runs/today.json

Why this calls llm.generate rather than Router.route: Router.route swallows
model failures and returns UNKNOWN at zero confidence. That is right in
production - a customer gets a human instead of an error - but useless here,
because a quota error would be scored as the router answering "unknown".
Going one level down lets a failed call be retried, and reported as a failure
rather than counted as a wrong answer.
"""

import argparse
import json
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from customer_service import llm, router
from customer_service.config import Settings, get_settings
from customer_service.schemas import Message, Route

CASES = Path(__file__).parent / "router_cases.json"


@dataclass
class Result:
    message: str
    expected: list[str]
    note: str
    got: Route | None = None
    error: str | None = None
    seconds: float = 0.0

    @property
    def group(self) -> str:
        """Cases with two acceptable answers are their own bucket."""
        return "/".join(self.expected)

    @property
    def correct(self) -> bool:
        return self.got is not None and self.got.category in self.expected


def classify(client, settings: Settings, message: str, attempts: int = 3) -> tuple[Route | None, str | None]:
    """One routing call, retried through quota windows rather than scored as wrong."""
    for attempt in range(1, attempts + 1):
        try:
            route = llm.generate(
                client,
                model=settings.router_model,
                system=router.SYSTEM_PROMPT,
                messages=[Message(role="customer", content=message)],
                schema=Route,
                thinking_level=router.THINKING,
            )
            return route, None
        except llm.LLMError as exc:
            if attempt == attempts:
                return None, str(exc)
            # Free-tier quotas are per minute, so back off in that order.
            time.sleep(15 * attempt)
    return None, "unreachable"


def run(cases: list[dict], rpm: float) -> list[Result]:
    settings = get_settings()
    client = llm.make_client(settings)
    gap = 60.0 / rpm
    results: list[Result] = []

    for index, case in enumerate(cases, start=1):
        started = time.monotonic()
        route, error = classify(client, settings, case["message"])
        elapsed = time.monotonic() - started

        result = Result(
            message=case["message"],
            expected=case["expected"],
            note=case.get("note", ""),
            got=route,
            error=error,
            seconds=elapsed,
        )
        results.append(result)

        mark = "!" if error else ("." if result.correct else "X")
        print(mark, end="", flush=True)
        if index % 40 == 0:
            print()

        if index < len(cases):
            time.sleep(max(0.0, gap - elapsed))

    print()
    return results


def report(results: list[Result], threshold: float) -> None:
    answered = [r for r in results if r.got is not None]
    failed = [r for r in results if r.got is None]
    correct = [r for r in answered if r.correct]
    wrong = [r for r in answered if not r.correct]

    print(f"\n{'=' * 70}")
    print(f"{len(results)} cases, {len(failed)} could not be scored (call failed)")
    if not answered:
        print("Nothing to score.")
        return

    print(f"Accuracy: {len(correct)}/{len(answered)} = {len(correct) / len(answered):.0%}")
    print(f"Mean latency: {sum(r.seconds for r in answered) / len(answered):.1f}s")

    print(f"\n{'-' * 70}\nBy expected answer")
    groups: dict[str, list[Result]] = {}
    for result in answered:
        groups.setdefault(result.group, []).append(result)
    for group, items in sorted(groups.items()):
        hits = sum(1 for item in items if item.correct)
        print(f"  {group:<22} {hits}/{len(items)}")

    # The distinction that matters for this system: a wrong answer below the
    # threshold escalates to a human, so nobody is harmed. A wrong answer above
    # it reaches the wrong specialist silently.
    confidently_wrong = [r for r in wrong if r.got.confidence >= threshold]
    safely_wrong = [r for r in wrong if r.got.confidence < threshold]

    # The opposite failure: refusing a message it should have routed, which
    # costs a human's time for nothing.
    over_escalated = [
        r for r in answered
        if "unknown" not in r.expected
        and (r.got.category == "unknown" or r.got.confidence < threshold)
    ]

    print(f"\n{'-' * 70}\nFailure shape (threshold {threshold})")
    print(f"  confidently wrong   {len(confidently_wrong):>3}   reaches the wrong specialist silently")
    print(f"  safely wrong        {len(safely_wrong):>3}   wrong, but escalates to a human")
    print(f"  over-escalated      {len(over_escalated):>3}   routable, but sent to a human anyway")

    if correct:
        print(f"\n  mean confidence when right  {sum(r.got.confidence for r in correct) / len(correct):.2f}")
    if wrong:
        print(f"  mean confidence when wrong  {sum(r.got.confidence for r in wrong) / len(wrong):.2f}")
        print("  (a router that knows when it doesn't know scores lower on the second)")

    if confidently_wrong:
        print(f"\n{'-' * 70}\nConfidently wrong - fix these first")
        for result in confidently_wrong:
            print(f"\n  {result.message!r}")
            print(f"    wanted {result.group}, got {result.got.category} at {result.got.confidence:.2f}")
            print(f"    router said: {result.got.reasoning}")
            print(f"    case note:   {result.note}")

    if over_escalated:
        print(f"\n{'-' * 70}\nSent to a human unnecessarily")
        for result in over_escalated:
            print(f"  {result.message!r} -> {result.got.category} at {result.got.confidence:.2f}")

    if failed:
        print(f"\n{'-' * 70}\nCould not be scored")
        for result in failed:
            print(f"  {result.message!r}: {result.error}")

    counts = Counter(str(r.got.category) for r in answered)
    print(f"\n{'-' * 70}\nWhat it answered overall: {dict(counts)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="only run the first N cases")
    parser.add_argument("--rpm", type=float, default=10.0, help="requests per minute (default 10)")
    parser.add_argument("--out", type=Path, help="also write the raw results here as JSON")
    args = parser.parse_args()

    cases = json.loads(CASES.read_text())
    if args.limit:
        cases = cases[: args.limit]

    settings = get_settings()
    print(f"{len(cases)} cases against {settings.router_model} at {args.rpm} rpm")
    print("  . correct    X wrong    ! call failed")

    results = run(cases, args.rpm)
    report(results, settings.router_confidence_threshold)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            [
                {
                    "message": r.message,
                    "expected": r.expected,
                    "note": r.note,
                    "got": r.got.model_dump() if r.got else None,
                    "error": r.error,
                    "seconds": round(r.seconds, 2),
                    "correct": r.correct,
                }
                for r in results
            ],
            indent=2,
        ))
        print(f"\nWrote {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
