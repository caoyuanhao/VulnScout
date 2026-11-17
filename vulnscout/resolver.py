"""Multi-source, provenance-tracked version resolution (technical contribution 1).

For one CVE and one product version, each source is asked separately and returns
:class:`~vulnscout.models.Evidence`. The verdict comes from the most reliable source that gives a
definite answer, in the order NVD CPE configuration > CNA record > OSV range. Disagreement between
sources, or a CNA entry that had to be partly guessed, lowers the confidence.
"""
from __future__ import annotations

import re

from . import versions
from .models import Cve, Evidence, Finding, Product
from .sources import cna, nvd, osv

PRIORITY = ["NVD CPE", "CNA", "OSV"]
_CPE_SPLIT = re.compile(r"(?<!\\):")


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _cpe_version(parts: list[str]) -> str | None:
    """The CPE version, with a pre-release ``update`` field appended (2.0 + beta9 -> 2.0-beta9)."""
    version, update = parts[5], parts[6] if len(parts) > 6 else "*"
    if version in ("*", "-"):
        return None
    if update not in ("*", "-", ""):
        return f"{version}-{update}"
    return version.replace("\\", "")


def nvd_cpe_evidence(cve: Cve, product: Product, version: str) -> Evidence | None:
    if not (product.cpe_vendor and product.cpe_product):
        return None
    mentioned, matched, platform = False, [], False
    for config in cve.configurations:
        for node in config.get("nodes", []):
            for m in node.get("cpeMatch", []):
                parts = _CPE_SPLIT.split(m["criteria"])
                if len(parts) < 7 or parts[3] != product.cpe_vendor or parts[4] != product.cpe_product:
                    continue
                if not m.get("vulnerable", True):
                    continue
                mentioned = True
                exact = _cpe_version(parts)
                if exact is not None:
                    if versions.compare(version, exact, product.ecosystem) == 0:
                        matched.append(f"exact {exact}")
                else:
                    rng = versions.cpe_range(m)
                    if rng.contains(version, product.ecosystem):
                        matched.append(str(rng))
                if config.get("operator") == "AND":
                    platform = True
    if not mentioned:
        return None
    detail = (f"matches {', '.join(matched)}" if matched else "no vulnerable CPE range contains this version")
    if matched and platform:
        detail += " (only in combination with a platform listed in the configuration)"
    return Evidence("NVD CPE", "affected" if matched else "not_affected", detail, nvd.record_url(cve.id))


def _cna_entries(affected: list[dict], product: Product) -> tuple[list[dict], bool]:
    """Selects the CNA ``affected`` entries describing ``product``; the flag says if the match is certain."""
    names = [_norm(n) for n in (product.cna_names or (product.name,)) if n]
    chosen = []
    for a in affected:
        label = _norm(a.get("product", "")) + "|" + _norm(a.get("packageName", ""))
        if any(n and n in label for n in names):
            chosen.append(a)
    if chosen:
        return chosen, True
    if len(affected) == 1:
        return affected, False
    return [], False


def cna_evidence(cve: Cve, product: Product, version: str) -> Evidence | None:
    affected = cve.cna_affected or cna.affected(cve.id)
    entries, certain = _cna_entries(affected, product)
    if not entries:
        return None
    verdicts = [versions.cna_status(version, e.get("versions", []), e.get("defaultStatus"), product.ecosystem)
                for e in entries]
    decided = [v for v in verdicts if v.status != "unknown"]
    if not decided:
        return Evidence("CNA", "unknown", verdicts[0].detail, cna.record_url(cve.id), clean=False)
    v = next((d for d in decided if d.status == "affected"), decided[0])
    status = "affected" if v.status == "affected" else "not_affected"
    detail = v.detail if certain else f"{v.detail}; product name did not match, used the only entry"
    return Evidence("CNA", status, detail, cna.record_url(cve.id), clean=v.clean and certain, fixed=v.fixed)


def osv_evidence(advisories: list[dict], product: Product, version: str) -> Evidence | None:
    if not (product.osv_ecosystem and product.osv_name):
        return None
    best: Evidence | None = None
    for adv in advisories:
        for entry in osv.package_entries(adv, product.osv_ecosystem, product.osv_name):
            url = osv.record_url(adv["id"])
            listed = entry.get("versions", [])
            fixed = None
            hit = version in listed
            for r in entry.get("ranges", []):
                if r.get("type") not in ("ECOSYSTEM", "SEMVER"):
                    continue
                affected, f = versions.osv_range_status(version, r.get("events", []), product.ecosystem)
                hit = hit or affected
                fixed = fixed or f
            if hit:
                how = "listed version" if version in listed else "inside range"
                return Evidence("OSV", "affected", f"{how} ({adv['id']})", url, fixed=fixed)
            best = best or Evidence("OSV", "not_affected", f"outside all ranges ({adv['id']})", url)
    return best


def resolve(cve: Cve, product: Product, version: str, advisories: list[dict]) -> Finding:
    evidence = [e for e in (nvd_cpe_evidence(cve, product, version),
                            cna_evidence(cve, product, version),
                            osv_evidence(advisories, product, version)) if e is not None]
    definite = sorted((e for e in evidence if e.status != "unknown"), key=lambda e: PRIORITY.index(e.source))
    if not definite:
        return Finding(cve, "unknown", None, "low", evidence)

    top = definite[0]
    agree = [e for e in definite[1:] if e.status == top.status]
    disagree = [e for e in definite[1:] if e.status != top.status]
    if top.source == "NVD CPE":
        confidence = "medium" if disagree else "high"
    elif not top.clean or disagree:
        confidence = "low"
    else:
        confidence = "high" if agree else "medium"
    fixed = next((e.fixed for e in definite if e.fixed and e.status == "affected"), None)
    return Finding(cve, top.status, top.source, confidence, evidence,
                   fixed=fixed if top.status == "affected" else None)
