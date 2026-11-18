"""Web chat interface (Gradio).

    python -m vulnscout.web            # http://127.0.0.1:7860
    python -m vulnscout.web --share    # temporary public link

Each browser session gets its own agent, so follow-up questions keep their context. Tool calls
appear as collapsible steps above the answer, so the user can see what was looked up and check
the raw data behind each claim.
"""
from __future__ import annotations

import argparse
import json
import queue
import threading

import gradio as gr

from .agent import Agent, ClaudeBackend, ToolCall
from .chat import PRICES, _cost

MODELS = list(PRICES)
EXAMPLES = [
    "Our Java service runs Log4j 2.14.1. What should we fix first?",
    "Is jackson-databind 2.9.8 safe to keep using?",
    "We use Django 3.2.10 and requests 2.25.0. Anything urgent?",
    "Is lodash 4.17.15 affected by anything that is actively exploited?",
    "Is spring-core vulnerable?",
    "We pulled in the npm package lodahs 4.17.21. Any known CVEs?",
    "What is insecure deserialization (CWE-502) and how do we prevent it?",
]
LABELS = {
    "search_cpe": "Searching the NVD product dictionary",
    "check_package": "Checking the package registry",
    "lookup_vulnerabilities": "Looking up vulnerabilities",
    "get_cve": "Fetching CVE record",
    "search_security_knowledge": "Searching CWE / ATT&CK",
}
INTRO = """# VulnScout
Ask which known vulnerabilities affect a dependency version and what to fix first.
Answers are built from NVD, CVE.org (CNA), OSV, the CISA KEV catalog and FIRST EPSS, queried live,
and MITRE CWE / ATT&CK for concepts;
each step VulnScout takes is shown above its answer. Not a substitute for a security review."""


def _call_title(call: ToolCall) -> str:
    a = call.arguments
    if call.name == "search_cpe":
        what = a.get("keyword", "")
    elif call.name == "check_package":
        what = " ".join(str(x) for x in (a.get("ecosystem"), a.get("name"), a.get("version")) if x)
    elif call.name == "lookup_vulnerabilities":
        what = f"{a.get('product_name', '')} {a.get('version', '')}"
    elif call.name == "get_cve":
        what = a.get("cve_id", "")
    elif call.name == "search_security_knowledge":
        what = a.get("query", "")
    else:
        what = ""
    return f"🔧 {LABELS.get(call.name, call.name)}: {what}".rstrip(": ")


def _call_summary(call: ToolCall) -> str:
    """One readable line about the result, followed by the raw JSON for anyone who wants to check it."""
    try:
        data = json.loads(call.result)
    except json.JSONDecodeError:
        return call.result[:2000]
    if "error" in data:
        line = f"Error: {data['error']}"
    elif call.name == "lookup_vulnerabilities":
        c = data.get("counts", {})
        line = (f"{c.get('affected', 0)} affecting CVEs ({c.get('in_cisa_kev', 0)} known exploited), "
                f"{c.get('unknown', 0)} undetermined, {c.get('not_affected', 0)} ruled out. "
                f"Upgrade target: {data.get('upgrade_to_fix_all') or 'none found'}.")
    elif call.name == "check_package":
        if not data.get("exists"):
            line = "Package not found in the registry."
        else:
            line = f"Package exists, {data.get('versions_published')} versions, latest {data.get('latest')}."
            if data.get("version_published") is False:
                line += " The requested version was never published."
    elif call.name == "search_security_knowledge":
        hits = data.get("results", [])
        line = "Matches: " + (", ".join(f"{h['id']} {h['name']}" for h in hits[:5]) or "none")
    elif call.name == "search_cpe":
        hits = data.get("results", [])
        line = "Matches: " + (", ".join(f"{h['vendor']}:{h['product']}" for h in hits[:5]) or "none")
    else:
        line = "Done."
    raw = json.dumps(data, indent=2, ensure_ascii=False)
    if len(raw) > 6000:
        raw = raw[:6000] + "\n... (truncated)"
    return f"{line}\n\n```json\n{raw}\n```"


def _new_session(model: str, effort: str) -> dict:
    return {"agent": Agent(ClaudeBackend(model, effort)), "model": model, "effort": effort}


def respond(message: str, history: list, session: dict | None, model: str, effort: str):
    message = (message or "").strip()
    if not message:
        yield history, session, gr.update(), ""
        return
    if session is None or session["model"] != model or session["effort"] != effort:
        session = _new_session(model, effort)       # settings changed: start a fresh conversation
    history = history + [gr.ChatMessage(role="user", content=message)]
    working = gr.ChatMessage(role="assistant", content="", metadata={"title": "⏳ Thinking…", "status": "pending"})
    yield history + [working], session, gr.update(), ""

    events: queue.Queue = queue.Queue()
    result: dict = {}

    def run():
        try:
            result["answer"] = session["agent"].ask(message, on_tool=lambda c, done: events.put((c, done)))
        except Exception as e:  # show API or network failures in the chat instead of hanging
            result["error"] = f"{type(e).__name__}: {e}"
        events.put(None)

    threading.Thread(target=run, daemon=True).start()
    steps: dict[str, gr.ChatMessage] = {}
    while (event := events.get()) is not None:
        call, done = event
        if not done:
            steps[call.id] = gr.ChatMessage(role="assistant", content="Running…",
                                            metadata={"title": _call_title(call), "status": "pending"})
        else:
            steps[call.id] = gr.ChatMessage(
                role="assistant", content=_call_summary(call),
                metadata={"title": _call_title(call) + (" ⚠️" if call.is_error else ""),
                          "status": "done", "duration": round(call.seconds, 1)})
        yield history + list(steps.values()) + [working], session, gr.update(), ""

    history = history + list(steps.values())
    if "error" in result:
        history.append(gr.ChatMessage(role="assistant", content=f"⚠️ Request failed: {result['error']}"))
    else:
        history.append(gr.ChatMessage(role="assistant", content=result["answer"].text))
    backend = session["agent"].backend
    usage = f"Session cost so far: ~${_cost(model, backend.usage):.3f} ({model}, effort {effort})"
    yield history, session, usage, ""


def build() -> gr.Blocks:
    with gr.Blocks(title="VulnScout", fill_height=True) as demo:
        session = gr.State(None)
        gr.Markdown(INTRO)
        with gr.Row(equal_height=False):
            with gr.Column(scale=4):
                chat = gr.Chatbot(height=620, layout="panel", buttons=["copy"], feedback_options=None,
                                  placeholder="Ask about a dependency and its version, e.g. *Log4j 2.14.1*.")
                with gr.Row():
                    box = gr.Textbox(placeholder="e.g. Our Java service runs Log4j 2.14.1. What should we fix first?",
                                     show_label=False, scale=8, autofocus=True)
                    send = gr.Button("Ask", variant="primary", scale=1)
            with gr.Column(scale=1, min_width=240):
                model = gr.Dropdown(MODELS, value=MODELS[0], label="Model")
                effort = gr.Radio(["low", "medium", "high"], value="medium", label="Reasoning effort")
                usage = gr.Markdown("Session cost so far: $0.000")
                clear = gr.Button("New conversation")
                gr.Examples(EXAMPLES, inputs=box, label="Try")

        outputs = [chat, session, usage, box]
        box.submit(respond, [box, chat, session, model, effort], outputs)
        send.click(respond, [box, chat, session, model, effort], outputs)
        clear.click(lambda: ([], None, "Session cost so far: $0.000"), None, [chat, session, usage])
    return demo


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="create a temporary public gradio.live link")
    args = ap.parse_args()
    build().queue(default_concurrency_limit=4).launch(server_port=args.port, share=args.share,
                                                      theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
