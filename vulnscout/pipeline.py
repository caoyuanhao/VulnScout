"""End-to-end lookup: product + version -> resolved, enriched and ranked findings.

This is the deterministic core that the agent will call as a tool. It gathers candidate CVEs from
NVD (by CPE product) and OSV (by package), resolves each one with :mod:`vulnscout.resolver`,
attaches KEV and EPSS signals, and ranks the affected ones with :mod:`vulnscout.ranker`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from . import ranker, resolver
from .models import Cve, Finding, Product
from .sources import epss, kev, nvd, osv


@dataclass
class LookupResult:
    product: Product
    version: str
    findings: list[Finding]
    as_of: str = field(default_factory=lambda: date.today().isoformat())

    def by_status(self, status: str) -> list[Finding]:
        return [f for f in self.findings if f.status == status]

    @property
    def ranked(self) -> list[Finding]:
        return sorted(self.by_status("affected"), key=lambda f: f.rank or 0)


def lookup(product: Product, version: str) -> LookupResult:
    cves: dict[str, Cve] = {}
    for vendor, name in product.cpes:
        for c in nvd.search_product(vendor, name):
            cves[c.id] = c

    advisories_by_cve: dict[str, list[dict]] = {}
    if product.osv_ecosystem and product.osv_name:
        for adv in osv.query(product.osv_ecosystem, product.osv_name):
            for cid in osv.cve_ids(adv) or [adv["id"]]:
                advisories_by_cve.setdefault(cid, []).append(adv)

    for cid, advs in advisories_by_cve.items():
        if cid in cves:
            continue
        record = nvd.get(cid) if cid.startswith("CVE-") else None
        cves[cid] = record or Cve(id=cid, published=advs[0].get("published", "")[:10],
                                  description=advs[0].get("summary") or advs[0].get("details", ""))

    findings = [resolver.resolve(c, product, version, advisories_by_cve.get(c.id, [])) for c in cves.values()]

    relevant = [f for f in findings if f.status != "not_affected" and f.cve.id.startswith("CVE-")]
    scores = epss.scores([f.cve.id for f in relevant])
    for f in relevant:
        entry = kev.lookup(f.cve.id)
        if entry:
            f.in_kev = True
            f.kev_date_added = entry.get("dateAdded")
            f.kev_ransomware = entry.get("knownRansomwareCampaignUse", "").lower() == "known"
        if f.cve.id in scores:
            f.epss, f.epss_percentile = scores[f.cve.id]

    ranker.rank(findings)
    findings.sort(key=lambda f: (f.rank is None, f.rank or 0, f.cve.id))
    return LookupResult(product, version, findings)
