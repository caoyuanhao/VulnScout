"""Data types shared by the sources, the resolver and the ranker."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Product:
    """One software product, with its identifiers in each data source.

    ``cpe_vendor``/``cpe_product`` select NVD records; ``osv_ecosystem``/``osv_name`` select OSV
    records; ``cna_names`` are the names CNAs use for the product in CVE records.
    """

    name: str
    cpe_vendor: str | None = None
    cpe_product: str | None = None
    osv_ecosystem: str | None = None
    osv_name: str | None = None
    cna_names: tuple[str, ...] = ()

    @property
    def ecosystem(self) -> str | None:
        return self.osv_ecosystem


@dataclass
class Evidence:
    """What one source says about one CVE and one version."""

    source: str            # "NVD CPE", "CNA", "OSV"
    status: str            # "affected", "not_affected" or "unknown"
    detail: str
    record: str            # citation: URL of the record the claim comes from
    clean: bool = True
    fixed: str | None = None


@dataclass
class Cve:
    id: str
    published: str = ""
    description: str = ""
    cvss: float | None = None
    severity: str | None = None
    cvss_vector: str | None = None
    cwes: list[str] = field(default_factory=list)
    nvd_status: str = ""                       # e.g. "Analyzed", "Awaiting Analysis"
    configurations: list[dict] = field(default_factory=list)
    cna_affected: list[dict] = field(default_factory=list)


@dataclass
class Finding:
    """The resolved verdict for one CVE, with provenance, plus the risk signals used for ranking."""

    cve: Cve
    status: str                                 # "affected", "not_affected" or "unknown"
    source: str | None
    confidence: str                             # "high", "medium" or "low"
    evidence: list[Evidence]
    fixed: str | None = None
    in_kev: bool = False
    kev_ransomware: bool = False
    kev_date_added: str | None = None
    epss: float | None = None
    epss_percentile: float | None = None
    rank: int | None = None
    reason: str = ""
