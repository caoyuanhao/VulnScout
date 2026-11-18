"""Runs a backend over held-out test questions and scores the answers automatically.

    python -m vulnscout.evaluate --backend claude --limit 20
    python -m vulnscout.evaluate --backend claude-no-tools
    python -m vulnscout.evaluate --backend hf:Qwen/Qwen3-4B-Instruct-2507
    python -m vulnscout.evaluate --backend hf:Qwen/Qwen3-4B-Instruct-2507+outputs/lora-qwen3-4b

The reference for each question is the generated conversation in ``data/sft/test.jsonl``: its
tool results are the ground truth (which CVEs affect the version, the upgrade target), and its kind
says what a correct reply must do. Scores per conversation are written to a JSONL file and
averaged into a summary.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from .agent import Agent, Answer, ClaudeBackend, ToolCall

# vulnerability IDs: CVEs and, for OSV-only findings, GitHub advisories. The lookahead skips group
# shorthands such as "CVE-2026-1064xx" (several IDs sharing a prefix), which are not IDs themselves.
_VULN_ID = r"(?:CVE-\d{4}-\d{4,}|GHSA(?:-[0-9a-z]{4}){3})(?![0-9A-Za-z])"
CVE = re.compile(_VULN_ID)
GHSA = re.compile(r"GHSA(?:-[0-9a-z]{4}){3}")
LINKED_CVE = re.compile(r"\[(" + _VULN_ID + r")\]\(https?://[^)]+\)")


def reference(record: dict) -> dict:
    """Ground truth from a generated conversation."""
    msgs = record["messages"]
    question = next(m["content"] for m in msgs if m["role"] == "user")
    ref = {"id": record["id"], "kind": record["kind"], "question": question}
    for m in msgs:
        if m["role"] == "tool" and m["name"] == "lookup_vulnerabilities":
            r = json.loads(m["content"])
            ref["affected"] = [f["cve"] for f in r["affected"]] + r.get("more_affected", [])
            # GitHub advisory IDs that the tool output ties to a CVE (e.g. in an OSV evidence URL), so an
            # answer citing the advisory instead of the CVE is credited with that CVE
            ref["aliases"] = {g: f["cve"] for f in r["affected"] + r.get("unknown", []) if f["cve"].startswith("CVE-")
                              for g in GHSA.findall(json.dumps(f))}
            ref["unknown"] = [u["cve"] for u in r.get("unknown", [])]
            ref["kev"] = [f["cve"] for f in r["affected"] if f.get("cisa_kev")]
            ref["top"] = r["affected"][0]["cve"] if r["affected"] else None
            ref["upgrade"] = r.get("upgrade_to_fix_all")
            ref["package"] = r["product"]
        if m["role"] == "assistant" and m.get("tool_calls"):
            for c in m["tool_calls"]:
                if c["function"]["name"] == "lookup_vulnerabilities":
                    ref["lookup_args"] = c["function"]["arguments"]
    return ref

# the answer must actually tell the user the package or version was not found (an empty answer is wrong)
NOT_FOUND = re.compile(r"(does ?n[o']t|did ?n[o']t) exist|is ?n[o']t (a )?(known|published|listed)|not (a )?known|"
                       r"(could ?n[o']t|can ?n[o']t|could not|cannot|did ?n[o']t) find|never (been )?(published|released)|"
                       r"not (been )?(published|released)|no (\S+ ){0,3}(release|version|match|package|product)|not found|unknown|"
                       r"(could ?n[o']t|could not|can ?n[o']t|cannot) (identify|locate|confirm)|found nothing|"
                       r"(is ?n[o']t|not) a real", re.I)


def score(ref: dict, answer) -> dict:
    text = answer.text
    alias = ref.get("aliases", {})
    mentioned = {alias.get(i, i) for i in CVE.findall(text)}
    linked = {alias.get(i, i) for i in LINKED_CVE.findall(text)}
    supported = set(ref.get("affected", [])) | set(ref.get("unknown", []))   # ground truth for this question
    for c in answer.tool_calls:
        supported |= set(CVE.findall(c.result))
    s: dict = {"steps": answer.steps, "tool_calls": len(answer.tool_calls),
               "tool_errors": sum(c.is_error for c in answer.tool_calls),
               # IDs that neither the ground truth nor any tool result in this run contains
               "hallucinated_cves": sorted(mentioned - supported),
               "refused": answer.refused}
    s["citation_rate"] = len(linked & mentioned) / len(mentioned) if mentioned else None

    if ref["kind"] == "lookup":
        lookups = [c for c in answer.tool_calls if c.name == "lookup_vulnerabilities" and not c.is_error]
        s["called_lookup"] = bool(lookups)
        want = {(c["vendor"], c["product"]) for c in ref["lookup_args"]["cpes"]}
        if lookups:
            got = {(c["vendor"], c["product"]) for c in lookups[-1].arguments.get("cpes", [])}
            s["cpes_exact"] = got == want
            s["package_correct"] = lookups[-1].arguments.get("package") == ref["lookup_args"]["package"]
        affected = set(ref["affected"])
        s["n_affected"] = len(affected)
        if affected:
            s["cve_recall"] = len(mentioned & affected) / len(affected)
            s["cve_precision"] = len(mentioned & affected) / len(mentioned) if mentioned else 0.0
            s["top_cve_mentioned"] = ref["top"] in mentioned
            s["kev_recall"] = (len(mentioned & set(ref["kev"])) / len(ref["kev"])) if ref["kev"] else None
        else:
            s["clean_correct"] = not mentioned          # nothing to report, so no CVE should be named
        if ref.get("upgrade"):
            s["upgrade_correct"] = ref["upgrade"] in text
    elif ref["kind"] == "need_version":
        s["asked_for_version"] = "version" in text.lower()
        s["correct"] = s["asked_for_version"] and not mentioned
    elif ref["kind"] in ("unknown_package", "unpublished_version"):
        s["correct"] = bool(NOT_FOUND.search(text)) and not mentioned and not any(c.name == "lookup_vulnerabilities" and not c.is_error
                                                 and json.loads(c.result).get("affected") for c in answer.tool_calls)
    return s


def make_backend(spec: str):
    # the lookup evaluation leaves out the CWE/ATT&CK tool, as in the fine-tuning data
    if spec == "claude":
        return ClaudeBackend(knowledge=False)
    if spec == "claude-no-tools":
        return ClaudeBackend(use_tools=False)
    if spec.startswith("claude:"):
        return ClaudeBackend(model=spec.split(":", 1)[1], knowledge=False)
    if spec.startswith("hf:"):
        from .hf_backend import HFBackend
        model, _, adapter = spec[3:].partition("+")
        return HFBackend(model, adapter or None)
    raise ValueError(f"unknown backend {spec!r}")


def summarize(rows: list[dict]) -> dict:
    def mean(key, kind=None):
        vals = [r["score"][key] for r in rows if key in r["score"] and r["score"][key] is not None
                and (kind is None or r["kind"] == kind)]
        if not vals:
            return None
        return round(sum(float(v) for v in vals) / len(vals), 3)

    lookups = [r for r in rows if r["kind"] == "lookup"]
    return {
        "n": len(rows),
        "lookup": {k: mean(k, "lookup") for k in ("called_lookup", "cpes_exact", "package_correct", "cve_recall",
                                                   "cve_precision", "top_cve_mentioned", "kev_recall",
                                                   "upgrade_correct", "clean_correct", "citation_rate")},
        "answers_with_hallucinated_cves": round(sum(bool(r["score"]["hallucinated_cves"]) for r in rows) / len(rows), 3),
        "hallucinated_cves_total": sum(len(r["score"]["hallucinated_cves"]) for r in rows),
        "special_cases_correct": mean("correct"),
        "mean_tool_calls": mean("tool_calls"),
        "tool_errors_total": sum(r["score"]["tool_errors"] for r in rows),
        "n_lookup": len(lookups),
    }


def rescore(path: Path, data: Path) -> None:
    """Re-scores saved answers with the current scoring rules, without calling any model.

    Runs saved before tool results were stored have no ``result`` fields. For those, everything that
    depends only on the answer text, the tool arguments and the reference is recomputed; the
    hallucination check, which needs the tool results, keeps its earlier verdicts (minus IDs the
    current pattern no longer extracts).
    """
    refs = {r["id"]: r for r in (reference(json.loads(line)) for line in data.read_text(encoding="utf-8").splitlines())}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    changed = 0
    for row in rows:
        have_results = all("result" in t for t in row["tools"])
        calls = [ToolCall(f"c{k}", t["name"], t["arguments"], t.get("result", "{}"), t["error"])
                 for k, t in enumerate(row["tools"])]
        new = score(refs[row["id"]], Answer(text=row["answer"], tool_calls=calls, steps=row["score"].get("steps", 0),
                                             refused=row["score"].get("refused", False)))
        if not have_results:
            # without the tool results an ID cannot be newly flagged as unsupported; keep the earlier
            # verdicts, minus any the current ID pattern no longer extracts from the answer
            new["hallucinated_cves"] = sorted(set(row["score"]["hallucinated_cves"]) & set(new["hallucinated_cves"]))
        changed += new != row["score"]
        row["score"] = new
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    summary = summarize(rows)
    path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"{changed} of {len(rows)} rows changed")
    print(json.dumps(summary, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend")
    ap.add_argument("--data", type=Path, default=Path("data/sft/test.jsonl"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=Path("outputs/eval"))
    ap.add_argument("--rescore", type=Path, metavar="JSONL", help="re-score a saved run instead of running a backend")
    args = ap.parse_args()
    if args.rescore:
        rescore(args.rescore, args.data)
        return
    if not args.backend:
        ap.error("--backend is required unless --rescore is given")

    records = [json.loads(line) for line in args.data.read_text(encoding="utf-8").splitlines()][:args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    tag = re.sub(r"[^\w.+-]", "_", args.backend)
    rows, backend = [], None
    with open(args.out / f"{tag}.jsonl", "w", encoding="utf-8") as f:
        for i, rec in enumerate(records, 1):
            ref = reference(rec)
            backend = make_backend(args.backend) if not args.backend.startswith("hf:") or backend is None else backend
            if args.backend.startswith("hf:"):       # reuse the loaded model; reset the conversation
                backend.messages = backend.messages[:1]
            t0 = time.perf_counter()
            try:
                answer = Agent(backend).ask(ref["question"])
            except Exception as e:  # a GPU out-of-memory on one long conversation should not end the run
                if type(e).__name__ != "OutOfMemoryError":
                    raise
                import torch
                torch.cuda.empty_cache()
                print(f"[{i}/{len(records)}] {ref['id']}: out of memory, scored as a failed answer", flush=True)
                answer = Answer(text="")
            row = {"id": ref["id"], "kind": ref["kind"], "question": ref["question"], "answer": answer.text,
                   "tools": [{"name": c.name, "arguments": c.arguments, "error": c.is_error, "result": c.result}
                             for c in answer.tool_calls],
                   "seconds": round(time.perf_counter() - t0, 1), "usage": dict(getattr(backend, "usage", {})),
                   "score": score(ref, answer)}
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i}/{len(records)}] {ref['kind']:20} {ref['id']:40} {row['seconds']:6.1f}s "
                  f"halluc={len(row['score']['hallucinated_cves'])} recall={row['score'].get('cve_recall')}", flush=True)
    summary = summarize(rows)
    (args.out / f"{tag}.summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
