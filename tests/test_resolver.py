"""Offline tests on real API responses saved in tests/fixtures (Log4Shell, CVE-2021-44228)."""
import json
from pathlib import Path

import pytest

from vulnscout import products, ranker, resolver, tools
from vulnscout.models import Cve, Finding
from vulnscout.sources import nvd

FIX = Path(__file__).parent / "fixtures"
LOG4J = products.find("log4j")


@pytest.fixture(scope="module")
def log4shell() -> Cve:
    return nvd.parse(json.loads((FIX / "nvd_44228.json").read_text(encoding="utf-8"))["vulnerabilities"][0])


@pytest.fixture(scope="module")
def osv_advisories() -> list[dict]:
    data = json.loads((FIX / "osv_query.json").read_text(encoding="utf-8"))
    return [a for a in data["vulns"] if "CVE-2021-44228" in a.get("aliases", [])]


def test_parse_nvd(log4shell):
    assert log4shell.cvss == 10.0
    assert log4shell.severity == "CRITICAL"
    assert "CWE-917" in log4shell.cwes
    assert log4shell.cna_affected, "NVD embeds the CNA's affected block"


@pytest.mark.parametrize("version, status", [
    ("2.14.1", "affected"),
    ("2.0-beta9", "affected"),        # exact CPE with update field: log4j:2.0:beta9
    ("2.12.1", "affected"),
    ("2.12.2", "not_affected"),       # gap between the 2.4-2.12.2 and 2.13-2.15 windows
    ("2.15.0", "not_affected"),
    ("2.17.1", "not_affected"),
])
def test_nvd_cpe(log4shell, version, status):
    ev = resolver.nvd_cpe_evidence(log4shell, LOG4J, version)
    assert ev.status == status
    assert ev.record == "https://nvd.nist.gov/vuln/detail/CVE-2021-44228"


def test_nvd_cpe_fix_is_exclusive_upper_bound(log4shell):
    assert resolver.nvd_cpe_evidence(log4shell, LOG4J, "2.14.1").fixed == "2.15.0"
    assert resolver.nvd_cpe_evidence(log4shell, LOG4J, "2.5").fixed == "2.12.2"


def test_upgrade_target_iterates_past_newer_cves(monkeypatch):
    # 1.0 is hit by A (fixed 4.17.21); 4.17.21 is hit by a newer B (fixed 4.18.1); 4.18.1 is clean
    affected_by = {"4.17.21": [_finding("B", 9.8, 0.1, fixed="4.18.1")], "4.18.1": []}

    class Result:
        def __init__(self, version):
            self.ranked = affected_by[version]

    monkeypatch.setattr(tools, "lookup", lambda product, version: Result(version))
    target, still = tools._upgrade_target(LOG4J, [_finding("A", 7.2, 0.2, fixed="4.17.21")])
    assert (target, still) == ("4.18.1", [])


def test_nvd_cpe_ignores_other_products(log4shell):
    other = products.find("requests")
    assert resolver.nvd_cpe_evidence(log4shell, other, "2.14.1") is None


def test_cna(log4shell):
    ev = resolver.cna_evidence(log4shell, LOG4J, "2.14.1")
    assert (ev.status, ev.fixed) == ("affected", "2.15.0")
    assert not ev.clean                                   # "lessThan": "log4j-core*"
    assert resolver.cna_evidence(log4shell, LOG4J, "2.3.1").status == "not_affected"


def test_osv(osv_advisories):
    assert osv_advisories, "fixture contains the Log4Shell advisory"
    ev = resolver.osv_evidence(osv_advisories, LOG4J, "2.14.1")
    assert ev.status == "affected"
    assert ev.record.startswith("https://osv.dev/vulnerability/GHSA-")


def test_resolve_agreement(log4shell, osv_advisories):
    f = resolver.resolve(log4shell, LOG4J, "2.14.1", osv_advisories)
    assert (f.status, f.source, f.confidence, f.fixed) == ("affected", "NVD CPE", "high", "2.15.0")
    assert {e.source for e in f.evidence} == {"NVD CPE", "CNA", "OSV"}


def test_resolve_falls_back_to_cna_when_nvd_has_no_cpe(log4shell, osv_advisories):
    unenriched = Cve(id=log4shell.id, cna_affected=log4shell.cna_affected)   # like 77% of new CVEs
    f = resolver.resolve(unenriched, LOG4J, "2.14.1", osv_advisories)
    assert (f.status, f.source) == ("affected", "CNA")
    assert f.confidence == "low"         # the CNA entry needed guessing, even though OSV agrees


def test_resolve_unknown_without_sources():
    f = resolver.resolve(Cve(id="CVE-2099-0001"), LOG4J, "2.14.1", [])
    assert (f.status, f.confidence) == ("unknown", "low")


def _finding(cid, cvss, epss, kev=False, ransomware=False, fixed=None):
    return Finding(Cve(id=cid, cvss=cvss), "affected", "NVD CPE", "high", [], fixed=fixed,
                   in_kev=kev, kev_ransomware=ransomware, kev_date_added="2021-12-10" if kev else None,
                   epss=epss, epss_percentile=0.5)


def test_ranker_prefers_exploitation_over_severity():
    fs = [_finding("A", 9.8, 0.01), _finding("B", 5.9, 0.97), _finding("C", 7.5, 0.30, kev=True),
          _finding("D", 6.0, 0.20, kev=True, ransomware=True)]
    ordered = [f.cve.id for f in ranker.rank(fs)]
    assert ordered == ["D", "C", "B", "A"]
    assert [f.cve.id for f in ranker.cvss_order(fs)] == ["A", "C", "D", "B"]
    assert fs[3].reason.startswith("patch now")
    assert "ransomware" in fs[3].reason


def test_record_url_links_advisories_to_osv():
    from vulnscout.sources import nvd
    assert nvd.record_url("CVE-2021-44228") == "https://nvd.nist.gov/vuln/detail/CVE-2021-44228"
    assert nvd.record_url("GHSA-442j-39wm-28r2") == "https://osv.dev/vulnerability/GHSA-442j-39wm-28r2"
