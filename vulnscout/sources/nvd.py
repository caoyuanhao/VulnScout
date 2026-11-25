"""NVD CVE API 2.0: CVE records, CVSS, CWE and CPE applicability configurations."""
from __future__ import annotations

from .. import versions
from ..http import fetch_json
from ..models import Cve

API = "https://services.nvd.nist.gov/rest/json/cves/2.0"
PAGE = 2000


def record_url(cve_id: str) -> str:
    """The public page of a vulnerability ID. Findings known only from OSV carry an advisory ID
    (e.g. GHSA-...) that NVD has no page for, so those link to OSV instead."""
    if not cve_id.startswith("CVE-"):
        return f"https://osv.dev/vulnerability/{cve_id}"
    return f"https://nvd.nist.gov/vuln/detail/{cve_id}"


def parse(item: dict) -> Cve:
    c = item["cve"]
    desc = next((d["value"] for d in c.get("descriptions", []) if d.get("lang") == "en"), "")
    metrics = c.get("metrics", {})
    cvss = None
    # CVSS 3.1 first: it is the version most CVEs have, so scores stay comparable across findings
    for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV40", "cvssMetricV2"):
        if metrics.get(key):
            # prefer NVD's own (Primary) score, fall back to the CNA's
            entries = sorted(metrics[key], key=lambda m: m.get("type") != "Primary")
            cvss = entries[0]["cvssData"] | {"baseSeverity": entries[0]["cvssData"].get(
                "baseSeverity", entries[0].get("baseSeverity"))}
            break
    cwes = sorted({d["value"] for w in c.get("weaknesses", []) for d in w.get("description", [])
                   if d["value"].startswith("CWE-")})
    cna_affected = [a for block in c.get("affected", []) for a in block.get("affectedData", [])]
    return Cve(
        id=c["id"],
        published=c.get("published", "")[:10],
        description=desc,
        cvss=cvss.get("baseScore") if cvss else None,
        severity=cvss.get("baseSeverity") if cvss else None,
        cvss_vector=cvss.get("vectorString") if cvss else None,
        cwes=cwes,
        nvd_status=c.get("vulnStatus", ""),
        configurations=c.get("configurations", []),
        cna_affected=cna_affected,
    )


def get(cve_id: str) -> Cve | None:
    data = fetch_json(API, {"cveId": cve_id})
    items = data.get("vulnerabilities", [])
    return parse(items[0]) if items else None


def search_product(vendor: str, product: str, version: str | None = None) -> list[Cve]:
    """All CVEs whose CPE configurations mention ``vendor:product``.

    Without ``version``, NVD returns every CVE for the product and VulnScout decides version
    applicability itself. With ``version``, NVD does the matching server-side; we use that only
    as a cross-check in the tests.
    """
    if version:
        # CPE keeps pre-release qualifiers in the "update" field: 2.0-beta9 -> 2.0:beta9
        base = versions.release_base(version)
        if base is not None:
            version = f"{base}:{version[len(base):].lstrip('.-_')}"
    cpe = f"cpe:2.3:*:{vendor}:{product}" + (f":{version}" if version else "")
    out, start = [], 0
    while True:
        data = fetch_json(API, {"virtualMatchString": cpe, "resultsPerPage": PAGE, "startIndex": start})
        out += [parse(v) for v in data.get("vulnerabilities", [])]
        start += data.get("resultsPerPage", PAGE)
        if start >= data.get("totalResults", 0):
            return out
