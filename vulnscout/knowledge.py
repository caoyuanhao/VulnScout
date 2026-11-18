"""Retrieval over MITRE CWE (weakness classes) and ATT&CK Enterprise (attack techniques).

Grounds answers to conceptual questions ("what is CWE-502?", "how does SQL injection work?",
"what is T1190?") in the official entries, with links, instead of the model's memory.

    python -m vulnscout.knowledge --build          # download both catalogs and build the index
    python -m vulnscout.knowledge "sql injection"  # search from the command line

Retrieval is BM25 over each entry's name, alternate terms and description (the name and terms are
weighted higher), plus exact lookup of IDs such as ``CWE-79`` or ``T1059.001``. The catalogs are
small (about 1,000 weaknesses and 700 techniques), so a lexical index is fast and needs no extra
dependencies; it also makes the retrieval step transparent and easy to evaluate.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import re
import zipfile
from collections import Counter
from functools import lru_cache
from pathlib import Path
from xml.etree import ElementTree

import requests

DATA = Path(__file__).resolve().parent.parent / "data" / "knowledge"
INDEX = DATA / "index.json"
CWE_URL = "https://cwe.mitre.org/data/xml/cwec_latest.xml.zip"
ATTACK_URL = "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/enterprise-attack/enterprise-attack.json"

_ID = re.compile(r"\b(CWE-\d+|T\d{4}(?:\.\d{3})?)\b", re.I)
_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = set("a an and are as at be by can do does for from how i in into is it its of on or that the this to "
            "what when which who why with you your my we our explain tell me about".split())


def _text(el) -> str:
    return " ".join(" ".join(el.itertext()).split()) if el is not None else ""


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _children(el, name: str) -> list:
    return [c for c in el if _local(c.tag) == name] if el is not None else []


def _child(el, name: str):
    found = _children(el, name)
    return found[0] if found else None


def parse_cwe(xml: bytes) -> list[dict]:
    """Weakness entries from the CWE XML catalog (any schema version; deprecated entries skipped)."""
    root = ElementTree.fromstring(xml)
    weaknesses = _child(root, "Weaknesses")
    out = []
    for w in _children(weaknesses, "Weakness"):
        if w.get("Status") == "Deprecated":
            continue
        cid = f"CWE-{w.get('ID')}"
        terms = [_text(_child(t, "Term")) for t in _children(_child(w, "Alternate_Terms"), "Alternate_Term")]
        consequences = []
        for c in _children(_child(w, "Common_Consequences"), "Consequence"):
            scope = ", ".join(_text(s) for s in _children(c, "Scope"))
            impact = ", ".join(_text(s) for s in _children(c, "Impact"))
            consequences.append(f"{scope}: {impact}".strip(": "))
        mitigations = []
        for m in _children(_child(w, "Potential_Mitigations"), "Mitigation"):
            phase = ", ".join(_text(p) for p in _children(m, "Phase"))
            desc = _text(_child(m, "Description"))
            if desc:
                mitigations.append(f"[{phase}] {desc}" if phase else desc)
        examples = [{"cve": _text(_child(e, "Reference")), "description": _text(_child(e, "Description"))}
                    for e in _children(_child(w, "Observed_Examples"), "Observed_Example")]
        parents = [f"CWE-{r.get('CWE_ID')}" for r in _children(_child(w, "Related_Weaknesses"), "Related_Weakness")
                   if r.get("Nature") == "ChildOf" and r.get("View_ID") == "1000"]
        out.append({
            "id": cid, "source": "CWE", "name": w.get("Name"), "abstraction": w.get("Abstraction"),
            "terms": [t for t in terms if t],
            "description": _text(_child(w, "Description")),
            "extended": _text(_child(w, "Extended_Description"))[:1500],
            "likelihood": _text(_child(w, "Likelihood_Of_Exploit")),
            "consequences": consequences[:6], "mitigations": [m[:500] for m in mitigations[:4]],
            "examples": [e for e in examples if e["cve"].startswith("CVE-")][:4], "parents": parents,
            "url": f"https://cwe.mitre.org/data/definitions/{w.get('ID')}.html",
        })
    return out


_CITATION = re.compile(r"\(Citation:[^)]*\)")
_MDLINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")


def _clean(text: str) -> str:
    return " ".join(_MDLINK.sub(r"\1", _CITATION.sub("", text or "")).split())


def parse_attack(bundle: dict) -> list[dict]:
    """Techniques and sub-techniques from the ATT&CK Enterprise STIX bundle, with their mitigations."""
    objects = bundle["objects"]
    by_id = {o["id"]: o for o in objects if "id" in o}

    def ext_id(o):
        return next((r.get("external_id") for r in o.get("external_references", [])
                     if r.get("source_name") == "mitre-attack"), None)

    mitigations: dict[str, list[str]] = {}
    for r in objects:
        if r["type"] == "relationship" and r.get("relationship_type") == "mitigates" and not r.get("revoked"):
            src = by_id.get(r["source_ref"])
            if src and src["type"] == "course-of-action" and not src.get("x_mitre_deprecated"):
                mitigations.setdefault(r["target_ref"], []).append(f"{ext_id(src)} {src['name']}")
    out = []
    for o in objects:
        if o["type"] != "attack-pattern" or o.get("revoked") or o.get("x_mitre_deprecated"):
            continue
        tid = ext_id(o)
        if not tid:
            continue
        out.append({
            "id": tid, "source": "ATT&CK", "name": o["name"], "terms": [],
            "description": _clean(o.get("description", ""))[:2500],
            "tactics": [p["phase_name"] for p in o.get("kill_chain_phases", [])],
            "platforms": o.get("x_mitre_platforms", []),
            "mitigations": sorted(set(mitigations.get(o["id"], [])))[:8],
            "url": next(r["url"] for r in o["external_references"] if r.get("source_name") == "mitre-attack"),
        })
    return out


def build(cwe_xml: bytes | None = None, attack_bundle: dict | None = None) -> list[dict]:
    if cwe_xml is None:
        z = zipfile.ZipFile(io.BytesIO(requests.get(CWE_URL, timeout=120).content))
        cwe_xml = z.read(next(n for n in z.namelist() if n.endswith(".xml")))
    if attack_bundle is None:
        attack_bundle = requests.get(ATTACK_URL, timeout=300).json()
    entries = parse_cwe(cwe_xml) + parse_attack(attack_bundle)
    DATA.mkdir(parents=True, exist_ok=True)
    INDEX.write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")
    load.cache_clear()
    return entries


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class Bm25:
    def __init__(self, docs: list[list[str]], k1: float = 1.2, b: float = 0.75):
        self.docs, self.k1, self.b = [Counter(d) for d in docs], k1, b
        self.lens = [len(d) for d in docs]
        self.avg = sum(self.lens) / max(1, len(docs))
        df = Counter(t for d in self.docs for t in d)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> list[float]:
        out = []
        for d, length in zip(self.docs, self.lens):
            s = 0.0
            for t in query:
                f = d.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * length / self.avg))
            out.append(s)
        return out


@lru_cache(maxsize=1)
def load() -> tuple[list[dict], dict[str, dict], Bm25]:
    if not INDEX.exists():
        raise FileNotFoundError(f"{INDEX} is missing; run `python -m vulnscout.knowledge --build`")
    entries = json.loads(INDEX.read_text(encoding="utf-8"))
    # name and alternate terms count three times, so "SQL injection" ranks CWE-89 above entries that
    # merely mention SQL injection in passing
    docs = [_tokens(" ".join([e["name"], *e["terms"]] * 3 + [e["description"]])) for e in entries]
    return entries, {e["id"].upper(): e for e in entries}, Bm25(docs)


def _brief(e: dict) -> dict:
    keep = {"id": e["id"], "source": e["source"], "name": e["name"], "url": e["url"],
            "description": e["description"][:600]}
    if e["source"] == "ATT&CK":
        keep["tactics"] = e["tactics"]
    return keep


def search(query: str, source: str = "any", limit: int = 5) -> dict:
    """Entries matching ``query``: IDs named in the query first (in full), then BM25 matches (brief)."""
    entries, by_id, bm25 = load()
    wanted = {"cwe": "CWE", "attack": "ATT&CK"}.get(source)
    exact = [by_id[m.upper()] for m in _ID.findall(query) if m.upper() in by_id]
    seen = {e["id"] for e in exact}
    ranked = []
    q = _tokens(_ID.sub(" ", query))
    if q:
        scores = bm25.scores(q)
        order = sorted(range(len(entries)), key=lambda i: -scores[i])
        for i in order:
            if scores[i] <= 0 or len(ranked) >= limit:
                break
            e = entries[i]
            if e["id"] not in seen and (wanted is None or e["source"] == wanted):
                ranked.append(e)
                seen.add(e["id"])
    results = exact + ranked
    if not results:
        return {"query": query, "results": [], "note": "no matching CWE or ATT&CK entry"}
    # the best match in full (consequences, mitigations, examples); the rest briefly
    return {"query": query, "results": [results[0]] + [_brief(e) for e in results[1:limit]]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("query", nargs="?")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--source", default="any", choices=["any", "cwe", "attack"])
    args = ap.parse_args()
    if args.build:
        entries = build()
        print(f"built {INDEX}: {Counter(e['source'] for e in entries)}")
    if args.query:
        for e in search(args.query, args.source)["results"]:
            print(f"{e['id']:12} {e['name'][:90]}")


if __name__ == "__main__":
    main()
