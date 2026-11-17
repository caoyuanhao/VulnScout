"""Generates tool-use conversations for fine-tuning and evaluation.

Every conversation is built by code: a question is sampled from templates, the real tools are
called in the right order, and the final answer is rendered from their results
(:mod:`vulnscout.answers`). No LLM output is used, so the data is correct by construction and
free of third-party terms. Products are split into train and test sets, so test questions are
about products the model never saw in training.

    python -m vulnscout.datagen --out data/sft --per-product 12

Writes ``train.jsonl`` and ``test.jsonl``; each line is ``{"id", "split", "kind", "messages"}``
in the common chat format with ``tool_calls`` (accepted by Hugging Face chat templates).
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from . import answers, catalog, tools, versions
from .agent import SYSTEM_PROMPT


@dataclass(frozen=True)
class PoolEntry:
    ecosystem: str
    package: str
    cpes: tuple[tuple[str, str], ...]
    aliases: tuple[str, ...]          # how users refer to it; the first is the display name
    cna_names: tuple[str, ...]
    language: str                     # used in question templates


def _p(eco, pkg, cpes, aliases, cna=None):
    lang = {"Maven": "Java", "PyPI": "Python", "npm": "Node.js"}[eco]
    return PoolEntry(eco, pkg, tuple(cpes), tuple(aliases), tuple(cna or aliases), lang)


# Identifiers verified against deps.dev (package exists) and NVD (each CPE pair has CVEs), 2026-11-17.
POOL: list[PoolEntry] = [
    _p("Maven", "org.apache.logging.log4j:log4j-core", [("apache", "log4j")], ["Log4j", "log4j-core", "Log4j 2"],
       ["log4j", "log4j2", "log4j-core"]),
    _p("Maven", "com.fasterxml.jackson.core:jackson-databind", [("fasterxml", "jackson-databind")],
       ["jackson-databind", "Jackson Databind"]),
    _p("Maven", "org.springframework:spring-core", [("vmware", "spring_framework"), ("pivotal_software", "spring_framework")],
       ["Spring Framework", "spring-core"], ["spring framework", "spring-core"]),
    _p("Maven", "org.apache.commons:commons-text", [("apache", "commons_text")], ["Apache Commons Text", "commons-text"]),
    _p("Maven", "commons-collections:commons-collections", [("apache", "commons_collections")],
       ["Apache Commons Collections", "commons-collections"]),
    _p("Maven", "org.yaml:snakeyaml", [("snakeyaml_project", "snakeyaml")], ["SnakeYAML", "snakeyaml"]),
    _p("Maven", "org.apache.tomcat.embed:tomcat-embed-core", [("apache", "tomcat")],
       ["Apache Tomcat", "tomcat-embed-core", "embedded Tomcat"], ["tomcat", "apache tomcat"]),
    _p("Maven", "com.h2database:h2", [("h2database", "h2")], ["H2 Database", "H2"], ["h2", "h2 database"]),
    _p("Maven", "com.thoughtworks.xstream:xstream", [("xstream", "xstream")], ["XStream", "xstream"]),
    _p("Maven", "io.netty:netty-codec-http", [("netty", "netty")], ["Netty", "netty-codec-http"], ["netty"]),
    _p("Maven", "com.google.guava:guava", [("google", "guava")], ["Guava", "Google Guava"], ["guava"]),
    _p("Maven", "org.apache.struts:struts2-core", [("apache", "struts")], ["Apache Struts", "Struts 2", "struts2-core"],
       ["struts", "struts 2"]),
    _p("Maven", "org.apache.shiro:shiro-core", [("apache", "shiro")], ["Apache Shiro", "shiro-core"], ["shiro"]),
    _p("Maven", "com.alibaba:fastjson", [("alibaba", "fastjson")], ["Fastjson", "fastjson"]),
    _p("Maven", "ch.qos.logback:logback-classic", [("qos", "logback")], ["Logback", "logback-classic"], ["logback"]),
    _p("PyPI", "requests", [("python", "requests")], ["requests", "Requests"]),
    _p("PyPI", "django", [("djangoproject", "django")], ["Django", "django"]),
    _p("PyPI", "flask", [("palletsprojects", "flask")], ["Flask", "flask"]),
    _p("PyPI", "jinja2", [("palletsprojects", "jinja")], ["Jinja2", "jinja2", "Jinja"], ["jinja", "jinja2"]),
    _p("PyPI", "pyyaml", [("pyyaml", "pyyaml")], ["PyYAML", "pyyaml"]),
    _p("PyPI", "pillow", [("python", "pillow")], ["Pillow", "pillow", "PIL"], ["pillow"]),
    _p("PyPI", "urllib3", [("python", "urllib3")], ["urllib3"]),
    _p("PyPI", "cryptography", [("cryptography.io", "cryptography")], ["cryptography", "pyca/cryptography"],
       ["cryptography"]),
    _p("PyPI", "werkzeug", [("palletsprojects", "werkzeug")], ["Werkzeug", "werkzeug"]),
    _p("PyPI", "numpy", [("numpy", "numpy")], ["NumPy", "numpy"]),
    _p("PyPI", "lxml", [("lxml", "lxml")], ["lxml"]),
    _p("PyPI", "paramiko", [("paramiko", "paramiko")], ["Paramiko", "paramiko"]),
    _p("PyPI", "aiohttp", [("aiohttp", "aiohttp")], ["aiohttp"]),
    _p("npm", "lodash", [("lodash", "lodash")], ["lodash", "Lodash"]),
    _p("npm", "express", [("openjsf", "express")], ["Express", "express"]),
    _p("npm", "axios", [("axios", "axios")], ["axios", "Axios"]),
    _p("npm", "minimist", [("substack", "minimist")], ["minimist"]),
    _p("npm", "moment", [("momentjs", "moment")], ["Moment.js", "moment"], ["moment"]),
    _p("npm", "jquery", [("jquery", "jquery")], ["jQuery", "jquery"]),
    _p("npm", "handlebars", [("handlebarsjs", "handlebars")], ["Handlebars", "handlebars"]),
    _p("npm", "node-fetch", [("node-fetch_project", "node-fetch")], ["node-fetch"]),
    _p("npm", "ws", [("ws_project", "ws")], ["ws", "the ws WebSocket library"], ["ws"]),
    _p("npm", "jsonwebtoken", [("auth0", "jsonwebtoken")], ["jsonwebtoken"]),
]

# Products held out for evaluation: the model never sees them in training.
TEST_PACKAGES = {"org.yaml:snakeyaml", "org.apache.shiro:shiro-core", "com.h2database:h2",
                 "werkzeug", "paramiko", "lxml", "handlebars", "ws"}

QUESTIONS = [
    "Our {lang} service runs {alias} {version}. What should we fix first?",
    "Is {alias} {version} affected by any known vulnerabilities?",
    "Which CVEs affect {alias} version {version}, and what should we upgrade to?",
    "We have {alias} {version} in production. Anything exploited in the wild?",
    "Security review: we depend on {alias} {version}. How bad is it?",
    "Do we need to patch {alias} {version}?",
    "What vulnerabilities does {alias} {version} have, ranked by risk?",
    "Our dependency scan flagged {alias} {version}. What matters most?",
]
PIN = {"PyPI": "We pinned {package}=={version} in requirements.txt. Any known vulnerabilities?",
       "npm": "package-lock.json has {package}@{version}. Should we be worried?",
       "Maven": "Our pom.xml declares {package}:{version}. Which CVEs apply?"}
NO_VERSION = ["Is {alias} safe to use?", "Does {alias} have any serious vulnerabilities?",
              "We use {alias} in our {lang} app. Are we vulnerable?"]


def _assistant_calls(calls: list[tuple[str, dict]], counter: list[int]) -> tuple[dict, list[dict]]:
    """An assistant turn with tool calls, plus the tool messages answering them (real tool output)."""
    tool_calls, results = [], []
    for name, args in calls:
        counter[0] += 1
        call_id = f"call_{counter[0]}"
        tool_calls.append({"id": call_id, "type": "function", "function": {"name": name, "arguments": args}})
        output, _ = tools.run(name, args)
        results.append({"role": "tool", "tool_call_id": call_id, "name": name, "content": output})
    return {"role": "assistant", "content": "", "tool_calls": tool_calls}, results


def _lookup_args(p: PoolEntry, version: str) -> dict:
    return {"product_name": p.aliases[0], "version": version,
            "cpes": [{"vendor": v, "product": pr} for v, pr in p.cpes],
            "ecosystem": p.ecosystem, "package": p.package, "cna_names": list(p.cna_names)}


def lookup_conversation(p: PoolEntry, version: str, rng: random.Random, pinned: bool = False) -> dict:
    alias = rng.choice(p.aliases)
    template = PIN[p.ecosystem] if pinned else rng.choice(QUESTIONS)
    question = template.format(lang=p.language, alias=alias, version=version, package=p.package)
    counter = [0]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]
    turn, results = _assistant_calls([("search_cpe", {"keyword": alias}),
                                      ("check_package", {"ecosystem": p.ecosystem, "name": p.package, "version": version})],
                                     counter)
    messages += [turn, *results]
    turn, results = _assistant_calls([("lookup_vulnerabilities", _lookup_args(p, version))], counter)
    messages += [turn, *results]
    result = json.loads(results[0]["content"])
    messages.append({"role": "assistant", "content": answers.vulnerabilities(result, rng)})
    return {"kind": "lookup", "messages": messages}


def no_version_conversation(p: PoolEntry, rng: random.Random) -> dict:
    alias = rng.choice(p.aliases)
    question = rng.choice(NO_VERSION).format(alias=alias, lang=p.language)
    return {"kind": "need_version", "messages": [
        {"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question},
        {"role": "assistant", "content": answers.need_version(alias, rng)}]}


def unpublished_version_conversation(p: PoolEntry, rng: random.Random) -> dict | None:
    info = catalog.check_package(p.ecosystem, p.package)
    latest = info.get("latest")
    if not latest or not versions.is_version(latest):
        return None
    major = str(int(versions._tokens(latest)[0]) + rng.randint(3, 9))   # a major version that does not exist
    bogus = f"{major}.{rng.randint(0, 9)}.{rng.randint(0, 9)}"
    alias = rng.choice(p.aliases)
    counter = [0]
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": rng.choice(QUESTIONS).format(lang=p.language, alias=alias, version=bogus,
                                                                          package=p.package)}]
    turn, results = _assistant_calls([("check_package", {"ecosystem": p.ecosystem, "name": p.package, "version": bogus})],
                                     counter)
    messages += [turn, *results]
    nearest = json.loads(results[0]["content"]).get("nearest_versions", [])
    messages.append({"role": "assistant", "content": answers.unpublished_version(alias, bogus, nearest, rng)})
    return {"kind": "unpublished_version", "messages": messages}


def unknown_package_conversation(p: PoolEntry, rng: random.Random) -> dict | None:
    # a misspelling that is not a published package
    name = p.package.split(":")[-1]
    i = rng.randrange(0, max(1, len(name) - 1))          # swap two neighbouring letters, then add a suffix
    swapped = name[:i] + name[i + 1:i + 2] + name[i] + name[i + 2:] if len(name) > 1 else name
    typo = swapped + rng.choice(["x", "z", "q", "-js", "2k"])
    if catalog.check_package(p.ecosystem, typo).get("exists"):
        return None
    counter = [0]
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Is {typo} {rng.choice(['1.0.0', '2.3.1', '0.9.4'])} vulnerable? It's a {p.ecosystem} package."}]
    turn, results = _assistant_calls([("check_package", {"ecosystem": p.ecosystem, "name": typo, "version": None}),
                                      ("search_cpe", {"keyword": typo})], counter)
    messages += [turn, *results]
    messages.append({"role": "assistant", "content": answers.unknown_package(typo, p.ecosystem, rng)})
    return {"kind": "unknown_package", "messages": messages}


def sample_versions(p: PoolEntry, n: int, rng: random.Random) -> list[str]:
    """Published versions, spread over the release history (older ones are more often vulnerable)."""
    info = catalog.check_package(p.ecosystem, p.package)
    data = catalog.fetch_json(f"{catalog.DEPS_DEV}/{catalog.ECOSYSTEMS[p.ecosystem]}/packages/"
                              f"{catalog.quote(p.package, safe='')}")
    published = [v["versionKey"]["version"] for v in data.get("versions", [])]
    stable = sorted({v for v in published if versions.is_version(v) and versions.release_base(v) is None},
                    key=versions.sort_key(p.ecosystem))
    if not stable:
        return []
    picks = {stable[-1]} if info.get("latest") else set()   # include the newest release
    while len(picks) < min(n, len(stable)):
        picks.add(stable[int(len(stable) * rng.random() ** 0.7)])   # skewed towards recent releases
    return sorted(picks, key=versions.sort_key(p.ecosystem))


def generate(per_product: int, seed: int, out: Path) -> None:
    rng = random.Random(seed)
    out.mkdir(parents=True, exist_ok=True)
    files = {split: open(out / f"{split}.jsonl", "w", encoding="utf-8") for split in ("train", "test")}
    counts: dict[tuple[str, str], int] = {}
    for p in POOL:
        split = "test" if p.package in TEST_PACKAGES else "train"
        convs = []
        for i, version in enumerate(sample_versions(p, per_product, rng)):
            convs.append(lookup_conversation(p, version, rng, pinned=(i % 4 == 3)))
        convs.append(no_version_conversation(p, rng))
        for extra in (unpublished_version_conversation(p, rng), unknown_package_conversation(p, rng)):
            if extra:
                convs.append(extra)
        for j, conv in enumerate(convs):
            record = {"id": f"{p.package}#{j}", "split": split, **conv}
            files[split].write(json.dumps(record, ensure_ascii=False) + "\n")
            counts[(split, conv["kind"])] = counts.get((split, conv["kind"]), 0) + 1
        print(f"{split:5} {p.package:45} {len(convs)} conversations")
    for f in files.values():
        f.close()
    # the data date of the generated references; evaluation pins tool results to it (VULNSCOUT_AS_OF)
    (out.parent / "AS_OF").write_text(date.today().isoformat(), encoding="utf-8")
    print({f"{s}/{k}": n for (s, k), n in sorted(counts.items())})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("data/sft"))
    parser.add_argument("--per-product", type=int, default=12, help="versions sampled per product")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    generate(args.per_product, args.seed, args.out)


if __name__ == "__main__":
    main()
