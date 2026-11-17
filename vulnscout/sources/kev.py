"""CISA Known Exploited Vulnerabilities catalog, re-downloaded at most once a day."""
from __future__ import annotations

from functools import lru_cache

from ..http import DAY, fetch_json

FEED = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
CATALOG_URL = "https://www.cisa.gov/known-exploited-vulnerabilities-catalog"


@lru_cache(maxsize=1)
def catalog() -> dict[str, dict]:
    data = fetch_json(FEED, ttl=DAY)
    return {v["cveID"]: v for v in data.get("vulnerabilities", [])}


def lookup(cve_id: str) -> dict | None:
    return catalog().get(cve_id)


def record_url(cve_id: str) -> str:
    return f"{CATALOG_URL}?search_api_fulltext={cve_id}"
