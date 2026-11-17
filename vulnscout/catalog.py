"""Product identification: checks the identifiers the LLM proposes against real indexes.

The LLM is good at recalling that "log4j" means Maven ``org.apache.logging.log4j:log4j-core``
and CPE ``apache:log4j``, but it can misremember. These functions confirm a proposed identifier
exists (deps.dev for packages, the NVD CPE dictionary for CPEs) before any CVE lookup uses it.
"""
from __future__ import annotations

import re
from collections import Counter
from urllib.parse import quote

import requests

from . import versions
from .http import fetch_json

DEPS_DEV = "https://api.deps.dev/v3/systems"
NVD_CPES = "https://services.nvd.nist.gov/rest/json/cpes/2.0"
# OSV ecosystem name -> deps.dev system name
ECOSYSTEMS = {"Maven": "MAVEN", "PyPI": "PYPI", "npm": "NPM", "Go": "GO", "crates.io": "CARGO", "NuGet": "NUGET"}


def normalize_ecosystem(ecosystem: str) -> str | None:
    for name in ECOSYSTEMS:
        if name.lower() == ecosystem.strip().lower():
            return name
    return {"pip": "PyPI", "python": "PyPI", "node": "npm", "javascript": "npm", "java": "Maven",
            "cargo": "crates.io", "rust": "crates.io", "golang": "Go", ".net": "NuGet"}.get(ecosystem.strip().lower())


def check_package(ecosystem: str, name: str, version: str | None = None) -> dict:
    """Whether ``name`` is a published package in ``ecosystem`` (and ``version`` a published release)."""
    eco = normalize_ecosystem(ecosystem)
    if eco is None:
        return {"exists": False, "error": f"unsupported ecosystem {ecosystem!r}; use one of {sorted(ECOSYSTEMS)}"}
    if eco == "PyPI":
        name = re.sub(r"[-_.]+", "-", name).lower()   # PEP 503 normalization
    try:
        data = fetch_json(f"{DEPS_DEV}/{ECOSYSTEMS[eco]}/packages/{quote(name, safe='')}")
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            return {"exists": False, "ecosystem": eco, "name": name}
        raise
    published = [v["versionKey"]["version"] for v in data.get("versions", [])]
    default = next((v["versionKey"]["version"] for v in data.get("versions", []) if v.get("isDefault")), None)
    out = {"exists": True, "ecosystem": eco, "name": data["packageKey"]["name"],
           "versions_published": len(published), "latest": default}
    if default and default.endswith("-security") and len(published) == 1:
        out["warning"] = ("npm security placeholder: the original package was removed as malicious; "
                          "the name is probably a typo of a popular package")
    if version is not None:
        out["version_published"] = version in published
        if not out["version_published"]:
            # the published versions closest to the requested one
            parsable = [v for v in published if versions.is_version(v)]
            if versions.is_version(version) and parsable:
                key = versions.sort_key(eco)
                ordered = sorted(parsable + [version], key=key)
                i = ordered.index(version)
                out["nearest_versions"] = ordered[max(0, i - 4):i] + ordered[i + 1:i + 5]
    return out


def search_cpe(keyword: str, limit: int = 8) -> list[dict]:
    """CPE (vendor, product) pairs whose dictionary entries match ``keyword``, most entries first."""
    data = fetch_json(NVD_CPES, {"keywordSearch": keyword, "resultsPerPage": 1000})
    counts: Counter[tuple[str, str, str]] = Counter()
    titles: dict[tuple[str, str, str], str] = {}
    for p in data.get("products", []):
        cpe = p["cpe"]
        part, vendor, product = cpe["cpeName"].split(":")[2:5]
        key = (part, vendor, product)
        counts[key] += 1
        if key not in titles:
            titles[key] = next((t["title"] for t in cpe.get("titles", []) if t.get("lang") == "en"), "")
    return [{"vendor": v, "product": pr, "type": {"a": "application", "o": "os", "h": "hardware"}.get(pa, pa),
             "dictionary_entries": n, "example_title": titles[(pa, v, pr)]}
            for (pa, v, pr), n in counts.most_common(limit)]
