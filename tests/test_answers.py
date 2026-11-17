import random

from vulnscout import answers

RESULT = {
    "product": "Log4j", "version": "2.14.1", "data_as_of": "2026-11-17",
    "counts": {"affected": 3, "unknown": 1, "not_affected": 10, "in_cisa_kev": 1},
    "upgrade_to_fix_all": "2.17.1", "upgrade_target_still_affected_by": [], "affected_without_known_fix": [],
    "affected": [
        {"cve": "CVE-2021-44228", "nvd_url": "https://nvd.nist.gov/vuln/detail/CVE-2021-44228", "cvss": 10.0,
         "severity": "CRITICAL", "cisa_kev": True, "kev_date_added": "2021-12-10", "ransomware": True,
         "kev_url": "https://kev/CVE-2021-44228", "epss": 0.99999, "epss_url": "https://epss/CVE-2021-44228",
         "fixed_in": "2.15.0", "confidence": "high"},
        {"cve": "CVE-2021-45105", "nvd_url": "https://nvd.nist.gov/vuln/detail/CVE-2021-45105", "cvss": 5.9,
         "epss": 0.97, "epss_url": "https://epss/CVE-2021-45105", "fixed_in": "2.17.0", "confidence": "medium",
         "evidence": [{"source": "NVD CPE", "says": "affected"}, {"source": "OSV", "says": "not_affected"}]},
        {"cve": "CVE-2021-44832", "nvd_url": "https://nvd.nist.gov/vuln/detail/CVE-2021-44832", "cvss": 6.6,
         "fixed_in": "2.17.1", "confidence": "high"},
    ],
    "more_affected": [],
    "unknown": [{"cve": "CVE-2099-0001", "nvd_url": "https://nvd.nist.gov/vuln/detail/CVE-2099-0001"}],
}


def test_answer_reports_tool_facts_with_citations():
    text = answers.vulnerabilities(RESULT, random.Random(0))
    assert "3 known CVEs affect Log4j 2.14.1" in text
    assert "1 of them is known to be exploited" in text and "ransomware" in text
    assert "[CVE-2021-44228](https://nvd.nist.gov/vuln/detail/CVE-2021-44228)" in text
    assert "[in the CISA KEV catalog](https://kev/CVE-2021-44228)" in text
    assert "Confidence is medium: NVD CPE says affected, OSV says not affected" in text
    assert "CVE-2099-0001" in text                                # unresolved findings are disclosed
    assert "Upgrade to 2.17.1." in text
    # nothing that is not in the tool result
    assert "CVE-2021-45046" not in text


def test_answer_for_clean_version():
    clean = {**RESULT, "counts": {"affected": 0, "unknown": 0, "not_affected": 20, "in_cisa_kev": 0},
             "affected": [], "unknown": [], "upgrade_to_fix_all": None}
    text = answers.vulnerabilities(clean, random.Random(1))
    assert "Log4j 2.14.1" in text and "CVE-" not in text
