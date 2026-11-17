"""Cross-checks against NVD's own server-side version matching (needs network).

Run with:  VULNSCOUT_LIVE=1 pytest tests/test_live.py
For every CVE that NVD has CPE data for, our NVD-CPE verdict should equal NVD's own
``virtualMatchString`` matching for the same product version.
"""
import os

import pytest

from vulnscout import products
from vulnscout.pipeline import lookup
from vulnscout.sources import nvd

pytestmark = pytest.mark.skipif(not os.environ.get("VULNSCOUT_LIVE"), reason="set VULNSCOUT_LIVE=1")

CASES = [("log4j-core", "2.14.1"), ("log4j-core", "2.17.0"), ("jackson-databind", "2.9.8"),
         ("requests", "2.19.0"), ("django", "3.2.0"), ("lodash", "4.17.15")]


def _ours_and_nvd(name: str, version: str) -> tuple[set[str], set[str]]:
    product = products.find(name)
    result = lookup(product, version)
    ours = {f.cve.id for f in result.findings
            if any(e.source == "NVD CPE" and e.status == "affected" for e in f.evidence)}
    theirs = {c.id for vendor, name in product.cpes for c in nvd.search_product(vendor, name, version)}
    return ours, theirs


@pytest.mark.parametrize("name, version", CASES)
def test_matches_nvd_server_side(name, version):
    ours, theirs = _ours_and_nvd(name, version)
    assert ours == theirs, f"only ours: {sorted(ours - theirs)}, only NVD: {sorted(theirs - ours)}"


def test_pre_release_finds_more_than_nvd():
    """NVD does not apply version ranges to CPEs with an update field (2.0:beta9), so it misses
    CVEs whose descriptions clearly cover 2.0-beta9, e.g. "2.0-alpha1 through 2.16.0"."""
    ours, theirs = _ours_and_nvd("log4j-core", "2.0-beta9")
    assert theirs <= ours
    assert {"CVE-2017-5645", "CVE-2021-45105"} <= ours - theirs
    assert not {"CVE-2020-9493", "CVE-2022-23307"} & ours      # log4j 1.x CVEs, range [1.2, 2.0)
