import pytest

from vulnscout import versions as v


@pytest.mark.parametrize("a, b, expected", [
    ("2.14.1", "2.15.0", -1),
    ("2.0", "2.0.0", 0),
    ("2.10.0", "2.9.9", 1),                 # numeric, not lexicographic
    ("2.0-beta9", "2.0", -1),               # pre-release before release
    ("2.0-alpha1", "2.0-beta1", -1),
    ("2.0-rc2", "2.0-beta9", 1),
    ("2.0-rc1", "2.0.1", -1),
    ("v1.2.3", "1.2.3", 0),                 # Go-style prefix
    ("1.2.3+build.5", "1.2.3", 0),          # build metadata ignored
    ("2.3.2", "2.3.10", -1),
    ("1.0.sp1", "1.0", 1),                  # unknown qualifier after the release...
    ("1.0.sp1", "1.0.1", -1),               # ...but before the next number
])
def test_compare_generic(a, b, expected):
    assert v.compare(a, b) == expected


def test_compare_pypi_uses_pep440():
    assert v.compare("2.0.0rc1", "2.0.0", "PyPI") == -1
    assert v.compare("1.0.post1", "1.0", "PyPI") == 1


def test_not_a_version():
    with pytest.raises(v.VersionError):
        v.compare("log4j-core*", "2.0")
    assert not v.is_version("*")
    assert not v.is_version("n/a")
    assert v.is_version("2.0-beta9")


def test_cpe_range():
    rng = v.cpe_range({"versionStartIncluding": "2.13.0", "versionEndExcluding": "2.15.0"})
    assert rng.contains("2.14.1")
    assert rng.contains("2.13.0")
    assert not rng.contains("2.15.0")
    assert not rng.contains("2.12.4")
    assert v.cpe_range({"versionEndIncluding": "1.0"}).contains("1.0")


def test_cpe_range_treats_bounds_as_release_lines():
    # NVD writes [1.2, 2.0) for "log4j 1.x" and [2.0, 2.8.2) for "2.x before 2.8.2"
    assert not v.cpe_range({"versionStartIncluding": "1.2", "versionEndExcluding": "2.0"}).contains("2.0-beta9")
    assert v.cpe_range({"versionStartIncluding": "2.0", "versionEndExcluding": "2.8.2"}).contains("2.0-beta9")
    # a bound that is itself a pre-release keeps the strict order
    assert not v.cpe_range({"versionStartIncluding": "2.0-rc1"}).contains("2.0-beta9")
    # OSV keeps the ecosystem's strict order
    assert v.osv_range_status("2.0-beta9", [{"introduced": "1.2"}, {"fixed": "2.0"}])[0] is True


def test_osv_range():
    events = [{"introduced": "0"}, {"fixed": "2.15.0"}]
    assert v.osv_range_status("2.14.1", events) == (True, "2.15.0")
    assert v.osv_range_status("2.15.0", events) == (False, None)
    # several affected windows in one range, events out of order
    events = [{"introduced": "2.13.0"}, {"fixed": "2.12.2"}, {"introduced": "2.0-beta9"}, {"fixed": "2.15.0"}]
    assert v.osv_range_status("2.12.2", events)[0] is False
    assert v.osv_range_status("2.14.0", events) == (True, "2.15.0")
    assert v.osv_range_status("1.5", [{"introduced": "0"}, {"last_affected": "1.5"}])[0] is True
    assert v.osv_range_status("1.6", [{"introduced": "0"}, {"last_affected": "1.5"}])[0] is False


# The real Log4Shell CNA entry: "lessThan" is not a version, the range lives in "changes".
LOG4J_CNA = [{"version": "2.0-beta9", "status": "affected", "lessThan": "log4j-core*", "versionType": "custom",
              "changes": [{"at": "2.3.1", "status": "unaffected"}, {"at": "2.4", "status": "affected"},
                          {"at": "2.12.2", "status": "unaffected"}, {"at": "2.13.0", "status": "affected"},
                          {"at": "2.15.0", "status": "unaffected"}]}]


@pytest.mark.parametrize("version, status, fixed", [
    ("2.14.1", "affected", "2.15.0"),
    ("2.3.1", "unaffected", None),
    ("2.5", "affected", "2.12.2"),
    ("2.12.2", "unaffected", None),
    ("2.15.0", "unaffected", None),
])
def test_cna_changes(version, status, fixed):
    verdict = v.cna_status(version, LOG4J_CNA)
    assert verdict.status == status
    assert verdict.fixed == fixed
    assert not verdict.clean            # the malformed bound is reported


def test_cna_before_first_version_is_unknown():
    assert v.cna_status("1.2.17", LOG4J_CNA).status == "unknown"


def test_cna_less_than():
    entries = [{"version": "0", "lessThan": "2.17.1", "status": "affected", "versionType": "semver"}]
    verdict = v.cna_status("2.17.0", entries)
    assert (verdict.status, verdict.fixed, verdict.clean) == ("affected", "2.17.1", True)
    assert v.cna_status("2.17.1", entries).status == "unknown"
    assert v.cna_status("2.17.1", entries, default_status="unaffected").status == "unaffected"


def test_cna_exact_and_default():
    entries = [{"version": "1.4.2", "status": "affected"}]
    assert v.cna_status("1.4.2", entries).status == "affected"
    assert v.cna_status("1.4.3", entries, default_status="unaffected").status == "unaffected"


@pytest.mark.parametrize("text, inside, outside", [
    ("< 2.17.1", "2.17.0", "2.17.1"),
    ("before 3.0", "2.9", "3.0"),
    ("2.0 through 2.14.1", "2.14.1", "2.15.0"),
    ("2.14.1 and earlier", "2.0", "2.14.2"),
    ("versions 1.0 to 1.5", "1.5", "1.6"),
    ("<= 4.2", "4.2", "4.3"),
])
def test_text_ranges(text, inside, outside):
    rng = v.parse_text_range(text)
    assert rng is not None
    assert rng.contains(inside)
    assert not rng.contains(outside)


def test_text_range_rejects_prose():
    assert v.parse_text_range("see advisory") is None
    assert v.cna_status("1.0", [{"version": "see advisory", "status": "affected"}]).status == "unknown"
