"""Evaluation of conceptual questions: CWE / ATT&CK retrieval, and answers with and without it.

    python -m vulnscout.concept_eval                       # retrieval only (offline, free)
    python -m vulnscout.concept_eval --backend claude-kb   # Claude with the retrieval tool
    python -m vulnscout.concept_eval --backend claude      # Claude answering from memory

Each question lists the catalog entries that count as a correct grounding. About a third are
paraphrased so that they do not contain the entry's name, which is the hard case for lexical
retrieval. Answers are scored on whether they cite an expected entry by its official URL and on
whether every CWE / ATT&CK ID they mention is a current catalog entry (an ID that is made up, or was
deprecated or revoked by MITRE, counts as not current).
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from . import knowledge

# (question, acceptable entry IDs, paraphrased?)
QUESTIONS: list[tuple[str, set[str], bool]] = [
    ("What is SQL injection?", {"CWE-89"}, False),
    ("What is cross-site scripting and how do I prevent it?", {"CWE-79"}, False),
    ("Explain CWE-502.", {"CWE-502"}, False),
    ("What does insecure deserialization mean?", {"CWE-502"}, False),
    ("What is a path traversal vulnerability?", {"CWE-22"}, False),
    ("What is server-side request forgery?", {"CWE-918"}, False),
    ("What is CSRF?", {"CWE-352"}, False),
    ("What is XXE and why is it dangerous?", {"CWE-611"}, False),
    ("What is a use-after-free bug?", {"CWE-416"}, False),
    ("What is OS command injection?", {"CWE-78"}, False),
    ("What is prototype pollution in JavaScript?", {"CWE-1321"}, False),
    ("What is ReDoS?", {"CWE-1333"}, False),
    ("What is an out-of-bounds write?", {"CWE-787"}, False),
    ("What does missing authentication for a critical function mean?", {"CWE-306"}, False),
    ("What is an integer overflow?", {"CWE-190"}, False),
    ("What are hard-coded credentials and why are they a problem?", {"CWE-798"}, False),
    ("What is an open redirect?", {"CWE-601"}, False),
    ("What is unrestricted file upload?", {"CWE-434"}, False),
    ("What is ATT&CK technique T1190?", {"T1190"}, False),
    ("What is T1059.001?", {"T1059.001"}, False),
    ("How does spearphishing with attachments work as an attack technique?", {"T1566.001", "T1566"}, False),
    ("What is credential dumping from LSASS memory?", {"T1003.001", "T1003"}, False),
    ("What is the 'exploitation for privilege escalation' technique?", {"T1068"}, False),
    ("What is a supply chain compromise in ATT&CK?", {"T1195", "T1195.001", "T1195.002"}, False),
    # paraphrased: the question avoids the entry's name
    ("An attacker typed ' OR 1=1 -- into our login form and got in. What class of bug is that?", {"CWE-89"}, True),
    ("Someone got our web page to run their JavaScript in other users' browsers. What weakness is that?", {"CWE-79", "CWE-80"}, True),
    ("Our app turns user-supplied Java objects back from bytes, and someone got code execution. What weakness is this?", {"CWE-502"}, True),
    ("A user requested ../../etc/passwd from our file download endpoint and got it. What is this called?", {"CWE-22", "CWE-23"}, True),
    ("Our server fetches any URL the user gives it, and someone used that to reach the cloud metadata endpoint. What weakness is that?", {"CWE-918"}, True),
    ("We passed user input straight into a shell command. What weakness is that?", {"CWE-78", "CWE-77"}, True),
    ("Our XML parser resolved an external entity and leaked a local file. What weakness is that?", {"CWE-611"}, True),
    ("A malicious website made our logged-in users' browsers submit a money transfer form. What is this attack called?", {"CWE-352"}, True),
    ("A crafted regular expression input made our service hang at 100% CPU. What weakness is that?", {"CWE-1333"}, True),
    ("Attackers got in by exploiting a bug in our internet-facing VPN appliance. Which ATT&CK technique is that?", {"T1190"}, True),
    ("Attackers ran encoded PowerShell scripts on our hosts. Which ATT&CK technique is that?", {"T1059.001", "T1059"}, True),
    ("Attackers pushed a backdoored version of a library we depend on to the package registry. Which ATT&CK technique is that?", {"T1195.001", "T1195", "T1195.002"}, True),
]

_IDS = re.compile(r"\b(CWE-\d+|T\d{4}(?:\.\d{3})?)\b")


def retrieval() -> dict:
    rows = []
    for q, expected, para in QUESTIONS:
        ids = [e["id"] for e in knowledge.search(q, limit=5)["results"]]
        rank = next((i + 1 for i, x in enumerate(ids) if x in expected), None)
        rows.append({"question": q, "paraphrased": para, "retrieved": ids, "rank": rank})

    def stats(subset):
        n = len(subset)
        return {"n": n, "hit@1": round(sum(r["rank"] == 1 for r in subset) / n, 3),
                "hit@5": round(sum(r["rank"] is not None for r in subset) / n, 3),
                "mrr": round(sum(1 / r["rank"] for r in subset if r["rank"]) / n, 3)}

    return {"all": stats(rows), "direct": stats([r for r in rows if not r["paraphrased"]]),
            "paraphrased": stats([r for r in rows if r["paraphrased"]]),
            "misses": [(r["question"], r["retrieved"][:3]) for r in rows if r["rank"] != 1]}


def score(answer: str, expected: set[str], tool_calls: list) -> dict:
    _, by_id, _ = knowledge.load()
    mentioned = set(_IDS.findall(answer))
    cited = {e for e in expected if by_id[e]["url"] in answer}
    return {"mentions_expected": bool(mentioned & expected), "cites_expected_url": bool(cited),
            "unknown_ids": sorted(i for i in mentioned if i.upper() not in by_id),
            "used_knowledge_tool": any(c.name == "search_security_knowledge" for c in tool_calls)}


def run_backend(spec: str, out: Path) -> dict:
    from .agent import Agent, ClaudeBackend
    from .chat import _cost
    model = "claude-opus-5-5"
    rows = []
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"concepts_{spec}.jsonl", "w", encoding="utf-8") as f:
        for i, (q, expected, para) in enumerate(QUESTIONS, 1):
            backend = ClaudeBackend(model, knowledge=(spec == "claude-kb"))
            t0 = time.perf_counter()
            answer = Agent(backend).ask(q)
            row = {"question": q, "expected": sorted(expected), "paraphrased": para, "answer": answer.text,
                   "tools": [{"name": c.name, "arguments": c.arguments} for c in answer.tool_calls],
                   "seconds": round(time.perf_counter() - t0, 1), "cost": round(_cost(model, backend.usage), 4),
                   "score": score(answer.text, expected, answer.tool_calls)}
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i}/{len(QUESTIONS)}] {row['seconds']:5.1f}s {row['score']}", flush=True)
    n = len(rows)
    summary = {k: round(sum(bool(r["score"][k]) for r in rows) / n, 3)
               for k in ("mentions_expected", "cites_expected_url", "used_knowledge_tool")}
    summary["answers_with_unknown_ids"] = round(sum(bool(r["score"]["unknown_ids"]) for r in rows) / n, 3)
    summary["cost_total"] = round(sum(r["cost"] for r in rows), 2)
    (out / f"concepts_{spec}.summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["claude-kb", "claude"])
    ap.add_argument("--out", type=Path, default=Path("outputs/eval"))
    args = ap.parse_args()
    if args.backend:
        print(json.dumps(run_backend(args.backend, args.out), indent=2))
    else:
        result = retrieval()
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "concepts_retrieval.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
