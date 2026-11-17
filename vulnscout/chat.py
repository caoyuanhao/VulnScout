"""Interactive chat in the terminal.

    python -m vulnscout.chat
    python -m vulnscout.chat "Our Java service runs Log4j 2.14.1. What should we fix first?"
"""
from __future__ import annotations

import argparse

from .agent import Agent, ClaudeBackend

# USD per million tokens (input, output) for cost estimates in the terminal
PRICES = {"claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-5-5": (0.10, 0.50)}


def _cost(model: str, usage: dict) -> float:
    pin, pout = PRICES.get(model, (0.0, 0.0))
    cached = usage.get("cache_read_input_tokens", 0)
    written = usage.get("cache_creation_input_tokens", 0)
    return (usage.get("input_tokens", 0) * pin + written * pin * 1.25 + cached * pin * 0.1
            + usage.get("output_tokens", 0) * pout) / 1e6


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question", nargs="?", help="ask one question and exit")
    parser.add_argument("--model", default="claude-opus-5-5")
    parser.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    args = parser.parse_args()

    backend = ClaudeBackend(args.model, args.effort)
    agent = Agent(backend, verbose=True)
    questions = [args.question] if args.question else None
    while True:
        q = questions.pop(0) if questions else (None if questions is not None else input("\nyou> ").strip())
        if not q or q in ("exit", "quit"):
            break
        answer = agent.ask(q)
        print(f"\nVulnScout> {answer.text}")
        print(f"\n  ({answer.steps} model calls, {len(answer.tool_calls)} tool calls, "
              f"~${_cost(args.model, backend.usage):.3f} so far)")
        if questions is not None and not questions:
            break


if __name__ == "__main__":
    main()
