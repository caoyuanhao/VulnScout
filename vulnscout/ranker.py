"""Exploitation-aware prioritization (technical contribution 2).

Affected CVEs are ordered by confirmed exploitation (CISA KEV, ransomware use first), then by the
probability of exploitation (EPSS), then by severity (CVSS). Each finding gets a priority tier and
a one-line reason the answer generator can quote.
"""
from __future__ import annotations

from .models import Finding

EPSS_HIGH = 0.1      # roughly the top 5% of all CVEs


def tier(f: Finding) -> str:
    if f.in_kev:
        return "patch now"
    if (f.epss or 0) >= EPSS_HIGH:
        return "high"
    if (f.cve.cvss or 0) >= 7.0:
        return "medium"
    return "low"


def _reason(f: Finding) -> str:
    epss = f"EPSS {f.epss:.3f} (percentile {f.epss_percentile:.0%})" if f.epss is not None else "no EPSS score"
    cvss = f"CVSS {f.cve.cvss} {f.cve.severity or ''}".strip() if f.cve.cvss is not None else "no CVSS score"
    if f.in_kev:
        parts = [f"exploited in the wild (CISA KEV since {f.kev_date_added})"]
        if f.kev_ransomware:
            parts.append("used in ransomware campaigns")
        return "; ".join(parts + [epss, cvss])
    if (f.epss or 0) >= EPSS_HIGH:
        return f"not in KEV, but likely to be exploited: {epss}; {cvss}"
    return f"no known exploitation; {epss}; {cvss}"


def rank(findings: list[Finding]) -> list[Finding]:
    """Ranks affected findings in place of their order; others keep ``rank = None``."""
    affected = [f for f in findings if f.status == "affected"]
    affected.sort(key=lambda f: (not f.in_kev, not f.kev_ransomware, -(f.epss or 0.0), -(f.cve.cvss or 0.0)))
    for i, f in enumerate(affected, 1):
        f.rank = i
        f.reason = f"{tier(f)}: {_reason(f)}"
    return affected


def cvss_order(findings: list[Finding]) -> list[Finding]:
    """The naive baseline: affected findings sorted by CVSS only."""
    return sorted((f for f in findings if f.status == "affected"), key=lambda f: -(f.cve.cvss or 0.0))
