"""FIRST EPSS: the probability that a CVE is exploited in the next 30 days (updated daily)."""
from __future__ import annotations

from ..http import DAY, fetch_json

API = "https://api.first.org/data/v1/epss"
BATCH = 50


def record_url(cve_id: str) -> str:
    return f"{API}?cve={cve_id}"


def scores(cve_ids: list[str]) -> dict[str, tuple[float, float]]:
    """Maps each CVE to (epss, percentile); CVEs without a score are left out."""
    out: dict[str, tuple[float, float]] = {}
    ids = sorted(set(cve_ids))
    for i in range(0, len(ids), BATCH):
        data = fetch_json(API, {"cve": ",".join(ids[i:i + BATCH])}, ttl=DAY)
        for row in data.get("data", []):
            out[row["cve"]] = (float(row["epss"]), float(row["percentile"]))
    return out
