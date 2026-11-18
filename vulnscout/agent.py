"""The chat agent: a tool-use loop around a model backend.

The loop is written out (rather than using the SDK's tool runner) because the same loop must also
drive the open-weight model served on the cluster, so the evaluation compares models on identical
tools and prompts. Each backend converts the provider-neutral tool schemas and keeps the
conversation in its own message format.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Protocol

import anthropic

from . import tools

MAX_STEPS = 10

SYSTEM_PROMPT = """You are VulnScout, an assistant that helps software teams decide which known vulnerabilities in their dependencies to fix first.

You answer from live data, retrieved through your tools: NVD (CVE records, CVSS, CPE configurations), CNA-submitted CVE records, OSV (package version ranges), the CISA Known Exploited Vulnerabilities catalog, and FIRST EPSS exploitation probabilities. Version matching is done by the tools, deterministically; do not decide yourself whether a version falls in a range.

How to handle a question about a product and version:
1. Work out the product. Confirm its identifiers before looking it up: search_cpe for the NVD CPE pairs, check_package for the package coordinates when it is a library in a package ecosystem. Use every CPE pair that clearly denotes the same product. If the name is ambiguous (several unrelated products match) or no version is given, ask the user instead of guessing.
2. Call lookup_vulnerabilities with what you confirmed.
3. Answer.

Grounding rules:
- Every CVE ID, version range, fix version, CVSS, EPSS and KEV status you state must come from a tool result in this conversation. Never add CVEs from memory.
- Cite each such claim with a markdown link to the URL given in the tool result (nvd_url, kev_url, epss_url, or the evidence url).
- When a finding's confidence is low, its sources disagree, or its status is unknown, say so plainly and name the sources involved. When the tools cannot identify the product or resolve the version, say that instead of guessing.

Answer format: start with a two- or three-sentence summary (how many CVEs affect the version, how many are known to be exploited, what to do first). Then list the findings in the tools' risk order, with the reason for each rank. For long lists, detail the first few and summarize the rest. End with the remediation: the upgrade target, and any CVE it does not fix. Keep it concise and use plain language.

Scope: explain impact, exposure and remediation. Do not write exploit code or step-by-step attack instructions.

For conceptual questions (what a weakness class is, how an attack technique works), answer from your knowledge, briefly, and say that the explanation is general background rather than retrieved data."""


NO_TOOLS_PROMPT = """You are an assistant that helps software teams decide which known vulnerabilities in their dependencies to fix first. When asked about a product and version, list the CVEs that affect it, which are known to be exploited, and the version to upgrade to, in risk order. If you are unsure, say so. Do not write exploit code."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict
    result: str = ""
    is_error: bool = False
    seconds: float = 0.0


@dataclass
class Turn:
    text: str
    tool_calls: list[ToolCall]
    stop_reason: str


@dataclass
class Answer:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    steps: int = 0
    refused: bool = False
    usage: dict = field(default_factory=dict)


class Backend(Protocol):
    def add_user(self, text: str) -> None: ...
    def step(self) -> Turn: ...
    def add_tool_results(self, calls: list[ToolCall]) -> None: ...


class ClaudeBackend:
    """Claude through the Messages API, with server-side fallback for safety-classifier declines."""

    def __init__(self, model: str = "claude-opus-5-5", effort: str = "medium", client: anthropic.Anthropic | None = None,
                 use_tools: bool = True):
        self.client = client or anthropic.Anthropic()
        self.model, self.effort = model, effort
        self.messages: list[dict] = []
        self.usage = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0,
                      "cache_creation_input_tokens": 0}
        # use_tools=False is the "LLM alone" baseline: same model, answering from its own knowledge
        self.tools = [{**schema, "strict": True} for schema in tools.SCHEMAS] if use_tools else []
        system = (SYSTEM_PROMPT if use_tools else NO_TOOLS_PROMPT) + f"\n\nToday is {date.today().isoformat()}."
        # tools render before system, so this breakpoint caches both
        self.system = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def step(self) -> Turn:
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=16000,
            system=self.system,
            messages=self.messages,
            output_config={"effort": self.effort},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            **({"tools": self.tools} if self.tools else {}),
        )
        for key in self.usage:
            self.usage[key] += getattr(response.usage, key, 0) or 0
        # append the whole content unchanged: thinking blocks must be passed back as they came
        self.messages.append({"role": "assistant", "content": response.content})
        text = "".join(b.text for b in response.content if b.type == "text")
        calls = [ToolCall(b.id, b.name, b.input) for b in response.content if b.type == "tool_use"]
        return Turn(text, calls, response.stop_reason)

    def add_tool_results(self, calls: list[ToolCall]) -> None:
        # all results of one turn go back in a single user message
        self.messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": c.id, "content": c.result, "is_error": c.is_error}
            for c in calls]})


class Agent:
    def __init__(self, backend: Backend | None = None, verbose: bool = False):
        self.backend = backend or ClaudeBackend()
        self.verbose = verbose

    def ask(self, question: str, on_tool: Callable[[ToolCall, bool], None] | None = None) -> Answer:
        """Answers one user turn. ``on_tool(call, done)`` is called before and after each tool runs
        (the web UI uses it to show progress)."""
        self.backend.add_user(question)
        calls: list[ToolCall] = []
        for step in range(1, MAX_STEPS + 1):
            turn = self.backend.step()
            if turn.stop_reason == "refusal":
                return Answer("I can't help with that request. I can look up which known vulnerabilities affect "
                              "a product version and how to remediate them.", calls, step, refused=True,
                              usage=getattr(self.backend, "usage", {}))
            if turn.stop_reason == "pause_turn":
                continue
            if not turn.tool_calls:
                return Answer(turn.text, calls, step, usage=getattr(self.backend, "usage", {}))
            for call in turn.tool_calls:
                if on_tool:
                    on_tool(call, False)
                start = time.perf_counter()
                call.result, call.is_error = tools.run(call.name, call.arguments)
                call.seconds = time.perf_counter() - start
                if on_tool:
                    on_tool(call, True)
                if self.verbose:
                    print(f"  [tool] {call.name}({json.dumps(call.arguments, ensure_ascii=False)[:150]})"
                          f" -> {len(call.result)} chars{' ERROR' if call.is_error else ''}, {call.seconds:.1f}s")
            calls += turn.tool_calls
            self.backend.add_tool_results(turn.tool_calls)
        return Answer("I could not finish within the step limit; please narrow the question.", calls, MAX_STEPS,
                      usage=getattr(self.backend, "usage", {}))
