"""Template answers rendered from tool results.

Used to build the fine-tuning data: every sentence is filled in from a tool result, so a model
trained on these answers learns to report what the tools returned, with the tools' citations,
and nothing else. A few phrasings are sampled so the model does not memorize one wording.
"""
from __future__ import annotations

import random

DETAIL = 5   # findings described one by one; the rest are listed


def _link(cve: str, url: str) -> str:
    return f"[{cve}]({url})"


def _finding_line(i: int, f: dict) -> str:
    parts = [f"{i}. **{_link(f['cve'], f['nvd_url'])}**"]
    if "cvss" in f:
        parts.append(f": CVSS {f['cvss']}" + (f" ({f['severity'].lower()})" if "severity" in f else ""))
    line = "".join(parts) + "."
    why = []
    if f.get("cisa_kev"):
        kev = f"[in the CISA KEV catalog]({f['kev_url']}) since {f['kev_date_added']}"
        why.append(kev + (", used in ransomware campaigns" if f.get("ransomware") else ""))
    if "epss" in f:
        why.append(f"[EPSS {f['epss']:.3f}]({f['epss_url']})")
    if why:
        line += " Why: " + "; ".join(why) + "."
    else:
        line += " No known exploitation and no EPSS score."
    line += f" Fixed in {f['fixed_in']}." if "fixed_in" in f else " No fixed version is recorded."
    if f.get("confidence") in ("low", "medium"):
        says = ", ".join(f"{e['source']} says {e['says'].replace('_', ' ')}" for e in f.get("evidence", []))
        line += f" Confidence is {f['confidence']}: {says}."
    return line


def vulnerabilities(result: dict, rng: random.Random) -> str:
    name, version = result["product"], result["version"]
    c = result["counts"]
    affected = result["affected"]
    as_of = result["data_as_of"]

    if c["affected"] == 0:
        text = rng.choice([
            f"No published CVEs affect {name} {version} in the sources I checked (NVD, CVE records and OSV, as of {as_of}).",
            f"I found no known CVEs for {name} {version} (NVD, CVE records and OSV, checked {as_of}).",
        ])
        if result.get("unknown"):
            ids = ", ".join(_link(u["cve"], u["nvd_url"]) for u in result["unknown"])
            text += f" For {ids} the version data could not be resolved, so check those manually."
        return text + " New vulnerabilities are published every day, so re-check before each release."

    kev = [f for f in affected if f.get("cisa_kev")]
    top = affected[0]
    first = (f"{c['affected']} known CVE{'s' if c['affected'] != 1 else ''} affect {name} {version}"
             if c["affected"] != 1 else f"1 known CVE affects {name} {version}")
    if c["in_cisa_kev"]:
        exploited = (f"{c['in_cisa_kev']} of them {'is' if c['in_cisa_kev'] == 1 else 'are'} known to be exploited in the wild"
                     + (" and used in ransomware campaigns" if any(f.get("ransomware") for f in kev) else ""))
        summary = f"**{first}, and {exploited}.** Fix {_link(top['cve'], top['nvd_url'])} first."
    else:
        summary = (f"**{first}; none of them is in the CISA KEV catalog.** "
                   f"The highest risk is {_link(top['cve'], top['nvd_url'])}, which has the highest exploitation probability.")
    if result.get("upgrade_to_fix_all"):
        summary += f" Upgrading to {result['upgrade_to_fix_all']} fixes {'them all' if not result.get('upgrade_target_still_affected_by') else 'most of them'}."

    lines = [summary, "", rng.choice(["Findings, highest risk first:", "In order of risk:", "Ranked by exploitation risk:"])]
    for i, f in enumerate(affected[:DETAIL], 1):
        lines.append(_finding_line(i, f))
    rest = affected[DETAIL:]
    more = result.get("more_affected", [])
    if rest or more:
        listed = ", ".join(_link(f["cve"], f["nvd_url"]) for f in rest)
        extra = f" and {len(more)} more ({', '.join(more)})" if more else ""
        lines.append(f"Lower priority (no known exploitation, low EPSS): {listed}{extra}.")
    if result.get("unknown"):
        ids = ", ".join(_link(u["cve"], u["nvd_url"]) for u in result["unknown"])
        lines.append(f"I could not resolve whether {version} is affected by {ids}; the version data is missing or unclear.")

    lines += ["", "**Remediation:**"]
    target = result.get("upgrade_to_fix_all")
    if target:
        lines.append(f"Upgrade to {target}.")
        if result.get("upgrade_target_still_affected_by"):
            lines.append("It still does not fix " + ", ".join(result["upgrade_target_still_affected_by"]) + ".")
    if result.get("affected_without_known_fix"):
        lines.append("No fixed version is recorded for " + ", ".join(result["affected_without_known_fix"])
                     + "; check the advisories for workarounds.")
    lines.append(f"Data as of {as_of}.")
    return "\n".join(lines)


def need_version(product: str, rng: random.Random) -> str:
    return rng.choice([
        f"Which version of {product} do you run? Vulnerabilities depend on the exact version, so I need it "
        "to look anything up. Your lockfile or build file (e.g. `pom.xml`, `requirements.txt`, `package-lock.json`) shows it.",
        f"I need the exact {product} version to check this, because CVEs apply to specific version ranges. "
        "Could you tell me the version from your dependency file?",
    ])


def unknown_package(name: str, ecosystem: str, rng: random.Random) -> str:
    return rng.choice([
        f"I could not find a package called `{name}` in the {ecosystem} registry or a matching product in the NVD "
        "CPE dictionary, so I can't look up its vulnerabilities. Could you check the exact package name?",
        f"`{name}` doesn't exist in {ecosystem} as far as I can tell (deps.dev and the NVD CPE dictionary have no "
        "match). Please confirm the name, for example from your dependency file.",
    ])


def unpublished_version(name: str, version: str, nearest: list[str], rng: random.Random) -> str:
    near = ", ".join(nearest) if nearest else "none found"
    return rng.choice([
        f"{name} {version} was never published, so I can't check it. Nearby releases: {near}. Which one do you run?",
        f"There is no {name} release {version} in the registry. The closest published versions are {near}. "
        "Could you confirm the version?",
    ])
