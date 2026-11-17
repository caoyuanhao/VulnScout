"""Deterministic version parsing and range checks.

Every "is version X affected?" decision in VulnScout goes through this module, never through the LLM.
Three range formats are supported:

* NVD CPE match criteria (``versionStartIncluding`` ... ``versionEndExcluding``),
* OSV ranges (``introduced`` / ``fixed`` / ``last_affected`` events, plus explicit version lists),
* CVE JSON 5 ``affected[].versions`` entries written by CNAs (``version`` + ``lessThan`` /
  ``lessThanOrEqual`` + ``changes``), including common free-text forms such as "< 2.17.1" or
  "2.0 through 2.14.1".
"""
from __future__ import annotations

import functools
import re
from dataclasses import dataclass

from packaging.version import InvalidVersion, Version

# Pre-release qualifiers sort before the release; anything else unknown sorts after it.
_PRE_RELEASE = {
    "dev": -6, "snapshot": -5, "alpha": -4, "a": -4, "beta": -3, "b": -3,
    "milestone": -2, "m": -2, "rc": -1, "cr": -1, "pre": -1, "preview": -1,
}
_RELEASE = {"", "ga", "final", "release", "r"}
_PLACEHOLDERS = {"", "*", "-", "n/a", "na", "unspecified", "unknown", "all", "none", "0"}
_TOKEN = re.compile(r"\d+|[a-z]+")


class VersionError(ValueError):
    """Raised when a string cannot be read as a version."""


def _tokens(version: str) -> list[int | str]:
    v = version.strip().lower()
    if v.startswith("v") and len(v) > 1 and v[1].isdigit():
        v = v[1:]
    v = v.split("+", 1)[0]  # build metadata never affects ordering
    tokens: list[int | str] = []
    for t in _TOKEN.findall(v):
        tokens.append(int(t) if t.isdigit() else t)
    if not tokens or not isinstance(tokens[0], int):
        raise VersionError(f"not a version: {version!r}")
    return tokens


def _rank(token: int | str | None) -> tuple[int, int, str]:
    """Orders tokens so that pre-release < release (missing) < unknown qualifier < number."""
    if token is None or token in _RELEASE:
        return (1, 0, "")
    if isinstance(token, int):
        return (2, token, "")
    if token in _PRE_RELEASE:
        return (0, _PRE_RELEASE[token], "")
    return (1, 1, token)  # unknown qualifier such as "sp" or "jre": just after the plain release


def compare(a: str, b: str, ecosystem: str | None = None) -> int:
    """Returns -1, 0 or 1. PyPI versions use PEP 440; everything else a Maven-like generic order."""
    if ecosystem and ecosystem.lower() == "pypi":
        try:
            va, vb = Version(a), Version(b)
            return (va > vb) - (va < vb)
        except InvalidVersion:
            pass
    ta, tb = _tokens(a), _tokens(b)
    for i in range(max(len(ta), len(tb))):
        x = ta[i] if i < len(ta) else None
        y = tb[i] if i < len(tb) else None
        # a missing numeric component counts as 0, so 2.0 == 2.0.0
        if x is None and isinstance(y, int):
            x = 0
        if y is None and isinstance(x, int):
            y = 0
        rx, ry = _rank(x), _rank(y)
        if rx != ry:
            return -1 if rx < ry else 1
    return 0


def sort_key(ecosystem: str | None = None):
    return functools.cmp_to_key(lambda a, b: compare(a, b, ecosystem))


def is_version(s: str | None) -> bool:
    if s is None or s.strip().lower() in _PLACEHOLDERS - {"0"}:
        return False
    try:
        _tokens(s)
        return True
    except VersionError:
        return False


# --------------------------------------------------------------------------- ranges

def release_base(version: str) -> str | None:
    """For a pre-release such as ``2.0-beta9`` returns ``2.0``; for a plain release returns None."""
    tokens = _tokens(version)
    for i, t in enumerate(tokens):
        if isinstance(t, str) and t in _PRE_RELEASE:
            return ".".join(str(x) for x in tokens[:i])
    return None


@dataclass(frozen=True)
class Range:
    """A half-open or closed interval; ``None`` bounds are unbounded.

    With ``release_line_bounds`` (used for NVD CPE ranges), a pre-release compared with a bound
    equal to its own release counts as that release: ``2.0-beta9`` is outside ``[1.2, 2.0)``
    and inside ``[2.0, 2.8.2)``. NVD writes such bounds to mean a release line ("1.x", "2.x"),
    whereas OSV ranges follow the ecosystem's strict order, where ``2.0-beta9 < 2.0``.
    """

    start: str | None = None
    start_inclusive: bool = True
    end: str | None = None
    end_inclusive: bool = False
    release_line_bounds: bool = False

    def _cmp(self, version: str, bound: str, ecosystem: str | None) -> int:
        c = compare(version, bound, ecosystem)
        if c < 0 and self.release_line_bounds and release_base(bound) is None:
            base = release_base(version)
            if base is not None and compare(base, bound, ecosystem) == 0:
                return 0
        return c

    def contains(self, version: str, ecosystem: str | None = None) -> bool:
        if self.start is not None:
            c = self._cmp(version, self.start, ecosystem)
            if c < 0 or (c == 0 and not self.start_inclusive):
                return False
        if self.end is not None:
            c = self._cmp(version, self.end, ecosystem)
            if c > 0 or (c == 0 and not self.end_inclusive):
                return False
        return True

    def __str__(self) -> str:
        lo = f"{'[' if self.start_inclusive else '('}{self.start or ''}"
        hi = f"{self.end or ''}{']' if self.end_inclusive else ')'}"
        return f"{lo}, {hi}"


def cpe_range(match: dict) -> Range:
    """Builds the range of an NVD cpeMatch entry whose CPE version field is ``*``."""
    if "versionStartIncluding" in match:
        start, s_inc = match["versionStartIncluding"], True
    elif "versionStartExcluding" in match:
        start, s_inc = match["versionStartExcluding"], False
    else:
        start, s_inc = None, True
    if "versionEndIncluding" in match:
        end, e_inc = match["versionEndIncluding"], True
    elif "versionEndExcluding" in match:
        end, e_inc = match["versionEndExcluding"], False
    else:
        end, e_inc = None, False
    return Range(start, s_inc, end, e_inc, release_line_bounds=True)


def osv_range_status(version: str, events: list[dict], ecosystem: str | None = None) -> tuple[bool, str | None]:
    """Evaluates one OSV ``ECOSYSTEM``/``SEMVER`` range.

    Returns (affected, fixed_version), where fixed_version is the first fix after ``version``.
    """
    def key(ev: dict) -> str:
        v = next(iter(ev.values()))
        return "0" if v in ("0", "") else v

    ordered = sorted(events, key=lambda ev: sort_key(ecosystem)(key(ev)))
    affected = False
    for ev in ordered:
        if "introduced" in ev:
            if ev["introduced"] in ("0", "") or compare(version, ev["introduced"], ecosystem) >= 0:
                affected = True
        elif "fixed" in ev:
            if compare(version, ev["fixed"], ecosystem) >= 0:
                affected = False
        elif "last_affected" in ev:
            if compare(version, ev["last_affected"], ecosystem) > 0:
                affected = False
    fixed = None
    if affected:
        later = [ev["fixed"] for ev in ordered if "fixed" in ev and compare(ev["fixed"], version, ecosystem) > 0]
        fixed = later[0] if later else None
    return affected, fixed


# Free-text forms CNAs put in the "version" field.
_TEXT_PATTERNS = [
    (re.compile(r"^(?:<|before|prior to|earlier than|below)\s*(?P<end>\S+)$"), "lt"),
    (re.compile(r"^(?:<=|=<|up to(?: and including)?|through|until)\s*(?P<end>\S+)$"), "le"),
    (re.compile(r"^(?P<end>\S+)\s+(?:and|or)\s+(?:earlier|before|prior|below|lower|older)$"), "le"),
    (re.compile(r"^(?P<start>\S+)\s*(?:through|thru|to|-|–)\s*(?P<end>\S+)$"), "range_le"),
    (re.compile(r"^(?P<start>\S+)\s*(?:before|<)\s*(?P<end>\S+)$"), "range_lt"),
    (re.compile(r"^(?:>=|from)\s*(?P<start>\S+)(?:\s*(?:,|and)?\s*(?:<|before)\s*(?P<end>\S+))?$"), "ge"),
]


def parse_text_range(text: str) -> Range | None:
    """Parses CNA free text like "< 2.17.1", "2.0 through 2.14.1" or "2.14.1 and earlier"."""
    t = re.sub(r"\s+", " ", text.strip().lower())
    t = re.sub(r"^(?:versions?|all versions?|version)\s+", "", t)
    for pattern, kind in _TEXT_PATTERNS:
        m = pattern.match(t)
        if not m:
            continue
        g = {k: v for k, v in m.groupdict().items() if v}
        if any(not is_version(v) for v in g.values()):
            continue
        if kind == "lt":
            return Range(end=g["end"], end_inclusive=False)
        if kind == "le":
            return Range(end=g["end"], end_inclusive=True)
        if kind == "range_le":
            return Range(g["start"], True, g["end"], True)
        if kind == "range_lt":
            return Range(g["start"], True, g["end"], False)
        if kind == "ge":
            return Range(g["start"], True, g.get("end"), False)
    return None


@dataclass(frozen=True)
class CnaVerdict:
    status: str          # "affected", "unaffected" or "unknown"
    clean: bool          # False when part of the entry had to be ignored or guessed
    detail: str
    fixed: str | None = None


def cna_status(version: str, entries: list[dict], default_status: str | None = None,
               ecosystem: str | None = None) -> CnaVerdict:
    """Evaluates CVE JSON 5 ``versions`` entries (one product) for ``version``.

    Implements the schema's semantics: an entry with ``lessThan``/``lessThanOrEqual`` covers the
    range from ``version`` to that bound, with ``changes`` switching the status at given versions;
    an entry without a bound matches one exact version. Free text in ``version`` is parsed with
    :func:`parse_text_range`. Malformed bounds are ignored and the verdict marked unclean.
    """
    clean = True
    notes: list[str] = []
    for e in entries:
        status = e.get("status", "affected")
        lo = e.get("version", "")
        lt, lte = e.get("lessThan"), e.get("lessThanOrEqual")
        bound = lt if lt is not None else lte

        if bound is not None:
            start = lo if is_version(lo) else None
            end = bound if is_version(bound) else None
            if end is None and bound.strip() not in ("*", ""):
                clean = False
                notes.append(f"ignored malformed bound {bound!r}")
            if lo and start is None and lo.strip().lower() not in _PLACEHOLDERS:
                clean = False
                notes.append(f"ignored malformed start {lo!r}")
            rng = Range(start, True, end, lte is not None and end is not None)
            if not rng.contains(version, ecosystem):
                continue
            current, fixed = status, None
            changes = sorted(e.get("changes", []), key=lambda c: sort_key(ecosystem)(c["at"])
                             if is_version(c.get("at")) else sort_key(ecosystem)("0"))
            for ch in changes:
                if not is_version(ch.get("at")):
                    clean = False
                    continue
                if compare(version, ch["at"], ecosystem) >= 0:
                    current = ch["status"]
                elif fixed is None and current == "affected" and ch["status"] == "unaffected":
                    fixed = ch["at"]
            if fixed is None and current == "affected" and end is not None and not lte:
                fixed = end
            return CnaVerdict(current, clean, f"in {rng}" + (f"; {'; '.join(notes)}" if notes else ""), fixed)

        if is_version(lo):
            if compare(version, lo, ecosystem) == 0:
                return CnaVerdict(status, clean, f"exact match {lo}")
            continue
        rng = parse_text_range(lo)
        if rng is not None:
            if rng.contains(version, ecosystem):
                fixed = rng.end if status == "affected" and rng.end and not rng.end_inclusive else None
                return CnaVerdict(status, False, f"parsed text {lo!r} as {rng}", fixed)
            continue
        if lo.strip().lower() not in _PLACEHOLDERS:
            clean = False
            notes.append(f"could not parse {lo!r}")

    if default_status in ("affected", "unaffected"):
        return CnaVerdict(default_status, clean, "defaultStatus" + (f"; {'; '.join(notes)}" if notes else ""))
    return CnaVerdict("unknown", clean, "; ".join(notes) or "no entry covers this version")
