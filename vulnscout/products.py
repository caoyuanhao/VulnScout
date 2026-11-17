"""A seed table of products with their identifiers in each source.

The agent resolves arbitrary product names itself (see :mod:`vulnscout.catalog`); these entries
are used by the command-line check and the tests.
"""
from __future__ import annotations

from .models import Product

KNOWN: dict[str, Product] = {p.name: p for p in [
    Product("log4j-core", (("apache", "log4j"),), "Maven", "org.apache.logging.log4j:log4j-core",
            ("log4j", "log4j2", "log4j-core")),
    Product("jackson-databind", (("fasterxml", "jackson-databind"),), "Maven",
            "com.fasterxml.jackson.core:jackson-databind", ("jackson-databind",)),
    Product("spring-core", (("vmware", "spring_framework"), ("pivotal_software", "spring_framework"),
                            ("springsource", "spring_framework")),
            "Maven", "org.springframework:spring-core", ("spring framework", "spring-core")),
    Product("requests", (("python", "requests"),), "PyPI", "requests", ("requests",)),
    Product("django", (("djangoproject", "django"),), "PyPI", "django", ("django",)),
    Product("lodash", (("lodash", "lodash"),), "npm", "lodash", ("lodash",)),
]}

ALIASES = {"log4j": "log4j-core", "log4j2": "log4j-core", "jackson": "jackson-databind",
           "spring": "spring-core", "spring framework": "spring-core"}


def find(name: str) -> Product | None:
    key = name.strip().lower()
    return KNOWN.get(ALIASES.get(key, key))
