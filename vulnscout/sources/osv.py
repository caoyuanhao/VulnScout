"""OSV (osv.dev): package-level affected version ranges, keyed by ecosystem and package name."""
from __future__ import annotations

from ..http import fetch_json

API = "https://api.osv.dev/v1"


def record_url(osv_id: str) -> str:
    return f"https://osv.dev/vulnerability/{osv_id}"


def query(ecosystem: str, name: str, version: str | None = None) -> list[dict]:
    """OSV advisories for a package (only those affecting ``version`` when it is given)."""
    body: dict = {"package": {"ecosystem": ecosystem, "name": name}}
    if version:
        body["version"] = version
    vulns, token = [], None
    while True:
        data = fetch_json(f"{API}/query", body=body | ({"page_token": token} if token else {}))
        vulns += data.get("vulns", [])
        token = data.get("next_page_token")
        if not token:
            return vulns


def cve_ids(vuln: dict) -> list[str]:
    ids = [vuln["id"]] + vuln.get("aliases", [])
    return sorted({i for i in ids if i.startswith("CVE-")})


def package_entries(vuln: dict, ecosystem: str, name: str) -> list[dict]:
    return [a for a in vuln.get("affected", [])
            if a.get("package", {}).get("ecosystem", "").lower() == ecosystem.lower()
            and a.get("package", {}).get("name", "").lower() == name.lower()]
