"""The tools the chat model can call, as JSON schemas plus their implementations.

Tool outputs are compact JSON: every claim the model may repeat carries the URL of the record it
comes from, so the answer can cite it. The schemas are provider-neutral; the agent wraps them in
whatever format its model backend expects.
"""
from __future__ import annotations

import inspect
import json
from typing import Any, Callable

from . import catalog, ranker, versions
from .models import Finding, Product
from .pipeline import lookup
from .sources import epss, kev, nvd

MAX_DETAILED = 15   # affected findings returned in full; the rest are listed by ID


def _obj(properties: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}


SCHEMAS: list[dict] = [
    {
        "name": "search_cpe",
        "description": (
            "Search the NVD CPE dictionary for the (vendor, product) identifiers NVD uses for a piece of software. "
            "Use it to confirm or discover CPE identifiers before calling lookup_vulnerabilities. A product can "
            "appear under several vendors after acquisitions (e.g. Spring Framework under vmware, "
            "pivotal_software and springsource); include all that clearly denote the same product."),
        "input_schema": _obj({"keyword": {"type": "string", "description": "Product name, e.g. 'log4j'"}},
                             ["keyword"]),
    },
    {
        "name": "check_package",
        "description": (
            "Check that a package exists in an ecosystem's registry (via deps.dev) and, optionally, that a version "
            "was published. Use it to confirm package coordinates such as Maven "
            "'org.apache.logging.log4j:log4j-core' or PyPI 'requests' before calling lookup_vulnerabilities."),
        "input_schema": _obj({
            "ecosystem": {"type": "string", "enum": sorted(catalog.ECOSYSTEMS)},
            "name": {"type": "string", "description": "Package name; Maven uses 'groupId:artifactId'"},
            "version": {"type": ["string", "null"], "description": "Version to check, or null"},
        }, ["ecosystem", "name", "version"]),
    },
    {
        "name": "lookup_vulnerabilities",
        "description": (
            "Find the CVEs that affect one version of one product, decided deterministically from NVD CPE "
            "configurations, CNA records and OSV ranges, enriched with CISA KEV and FIRST EPSS, and ranked by "
            "exploitation risk. Give the CPE pairs and/or the package coordinates you have confirmed."),
        "input_schema": _obj({
            "product_name": {"type": "string", "description": "Human-readable product name"},
            "version": {"type": "string"},
            "cpes": {"type": "array", "description": "Confirmed NVD CPE identifiers (may be empty)",
                     "items": _obj({"vendor": {"type": "string"}, "product": {"type": "string"}},
                                   ["vendor", "product"])},
            "ecosystem": {"type": ["string", "null"],
                          "description": f"One of {', '.join(sorted(catalog.ECOSYSTEMS))}, or null"},
            "package": {"type": ["string", "null"], "description": "Confirmed package name, or null"},
            "cna_names": {"type": "array", "items": {"type": "string"},
                          "description": "Names CVE records may use for the product, e.g. ['log4j', 'log4j2']"},
        }, ["product_name", "version", "cpes", "ecosystem", "package", "cna_names"]),
    },
    {
        "name": "get_cve",
        "description": "Details of one CVE: description, CVSS, CWE, NVD analysis status, KEV and EPSS.",
        "input_schema": _obj({"cve_id": {"type": "string", "description": "e.g. CVE-2021-44228"}}, ["cve_id"]),
    },
]


def _finding_json(f: Finding) -> dict:
    out: dict[str, Any] = {
        "cve": f.cve.id, "rank": f.rank, "priority": ranker.tier(f) if f.status == "affected" else None,
        "status": f.status, "decided_by": f.source, "confidence": f.confidence, "fixed_in": f.fixed,
        "cvss": f.cve.cvss, "severity": f.cve.severity, "epss": f.epss, "epss_percentile": f.epss_percentile,
        "cisa_kev": f.in_kev, "kev_date_added": f.kev_date_added, "ransomware": f.kev_ransomware,
        "cwe": f.cve.cwes, "published": f.cve.published, "summary": f.cve.description[:300],
        "evidence": [{"source": e.source, "says": e.status, "detail": e.detail, "url": e.record}
                     for e in f.evidence],
        "nvd_url": nvd.record_url(f.cve.id),
    }
    if f.in_kev:
        out["kev_url"] = kev.record_url(f.cve.id)
    if f.epss is not None:
        out["epss_url"] = epss.record_url(f.cve.id)
    return {k: v for k, v in out.items() if v not in (None, [], "")}


def _upgrade_target(product: Product, affected: list[Finding], rounds: int = 5) -> tuple[str | None, list[str]]:
    """The lowest version that fixes every affected CVE, found by iterating to a fixed point.

    Moving to the highest fix version can land in the range of a newer CVE that did not affect the
    original version, so the candidate is looked up again and raised until nothing with a known fix
    remains (the CVE lists are cached, so each round is cheap). Returns the target and the CVEs that
    still affect it.
    """
    key = versions.sort_key(product.ecosystem)
    fixes = [f.fixed for f in affected if f.fixed]
    if not fixes:
        return None, []
    target = max(fixes, key=key)
    for _ in range(rounds):
        remaining = lookup(product, target).ranked
        newer = [f.fixed for f in remaining if f.fixed and key(f.fixed) > key(target)]
        if not newer:
            return target, [f.cve.id for f in remaining]
        target = max(newer, key=key)
    return target, [f.cve.id for f in lookup(product, target).ranked]


def lookup_vulnerabilities(product_name: str, version: str, cpes: list[dict], ecosystem: str | None,
                           package: str | None, cna_names: list[str]) -> dict:
    if not cpes and not (ecosystem and package):
        return {"error": "give at least one confirmed CPE pair or an ecosystem and package"}
    product = Product(product_name, tuple((c["vendor"], c["product"]) for c in cpes),
                      ecosystem if package else None, package, tuple(cna_names) or (product_name,))
    result = lookup(product, version)
    affected, unknown = result.ranked, result.by_status("unknown")
    upgrade, still = _upgrade_target(product, affected)
    return {
        "product": product_name, "version": version, "data_as_of": result.as_of,
        "counts": {"affected": len(affected), "unknown": len(unknown),
                   "not_affected": len(result.by_status("not_affected")),
                   "in_cisa_kev": sum(f.in_kev for f in affected)},
        "upgrade_to_fix_all": upgrade,
        "upgrade_target_still_affected_by": still,
        "affected_without_known_fix": [f.cve.id for f in affected if not f.fixed],
        "affected": [_finding_json(f) for f in affected[:MAX_DETAILED]],
        "more_affected": [f.cve.id for f in affected[MAX_DETAILED:]],
        "unknown": [{"cve": f.cve.id, "why": "; ".join(e.detail for e in f.evidence) or "no version data",
                     "nvd_url": nvd.record_url(f.cve.id)} for f in unknown],
    }


def get_cve(cve_id: str) -> dict:
    cve = nvd.get(cve_id.strip().upper())
    if cve is None:
        return {"error": f"{cve_id} not found in NVD"}
    entry = kev.lookup(cve.id)
    score = epss.scores([cve.id]).get(cve.id)
    return {k: v for k, v in {
        "cve": cve.id, "published": cve.published, "nvd_status": cve.nvd_status, "description": cve.description,
        "cvss": cve.cvss, "severity": cve.severity, "cvss_vector": cve.cvss_vector, "cwe": cve.cwes,
        "cisa_kev": entry is not None, "kev_date_added": entry and entry.get("dateAdded"),
        "kev_required_action": entry and entry.get("requiredAction"),
        "ransomware": bool(entry) and entry.get("knownRansomwareCampaignUse", "").lower() == "known",
        "epss": score and score[0], "epss_percentile": score and score[1],
        "nvd_url": nvd.record_url(cve.id), "kev_url": entry and kev.record_url(cve.id),
    }.items() if v not in (None, [], "")}


IMPLEMENTATIONS: dict[str, Callable[..., Any]] = {
    "search_cpe": lambda keyword: {"results": catalog.search_cpe(keyword)},
    "check_package": catalog.check_package,
    "lookup_vulnerabilities": lookup_vulnerabilities,
    "get_cve": get_cve,
}


def run(name: str, arguments: dict) -> tuple[str, bool]:
    """Executes a tool; returns (JSON text, is_error)."""
    if name not in IMPLEMENTATIONS:
        return json.dumps({"error": f"unknown tool {name}"}), True
    try:
        inspect.signature(IMPLEMENTATIONS[name]).bind(**arguments)
    except TypeError as e:
        return json.dumps({"error": f"bad arguments for {name}: {e}"}), True
    try:
        return json.dumps(IMPLEMENTATIONS[name](**arguments), ensure_ascii=False), False
    except Exception as e:          # network errors and the like: let the model report them
        return json.dumps({"error": f"{type(e).__name__}: {e}"}), True
