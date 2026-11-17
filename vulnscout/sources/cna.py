"""CVE Services (cveawg.mitre.org): the CNA-submitted CVE JSON 5 record.

NVD copies the CNA's ``affected`` block into its own record, so this is only needed when NVD has
not ingested a CVE yet or omits the block.
"""
from __future__ import annotations

import requests

from ..http import fetch_json

API = "https://cveawg.mitre.org/api/cve/"


def record_url(cve_id: str) -> str:
    return f"https://www.cve.org/CVERecord?id={cve_id}"


def affected(cve_id: str) -> list[dict]:
    try:
        data = fetch_json(API + cve_id)
    except requests.HTTPError:
        return []
    return data.get("containers", {}).get("cna", {}).get("affected", [])
