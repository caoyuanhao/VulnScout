"""Command-line check of the deterministic core.

    python -m vulnscout.cli log4j-core 2.14.1
    python -m vulnscout.cli requests 2.19.0 --all
"""
from __future__ import annotations

import argparse
import sys

from . import products
from .pipeline import lookup


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("product", help=f"one of: {', '.join(products.KNOWN)}")
    parser.add_argument("version")
    parser.add_argument("--all", action="store_true", help="also list not-affected and unknown CVEs")
    args = parser.parse_args()

    product = products.find(args.product)
    if product is None:
        sys.exit(f"unknown product {args.product!r}; known: {', '.join(products.KNOWN)}")
    result = lookup(product, args.version)

    affected, unknown = result.ranked, result.by_status("unknown")
    print(f"{product.name} {args.version}: {len(affected)} affected, {len(unknown)} unknown, "
          f"{len(result.by_status('not_affected'))} not affected (data as of {result.as_of})\n")
    print(f"{'#':>2}  {'CVE':<16} {'CVSS':>4}  {'EPSS':>7}  {'KEV':<4} {'fixed in':<10} {'source':<8} {'conf':<6}")
    for f in affected:
        print(f"{f.rank:>2}  {f.cve.id:<16} {f.cve.cvss or '-':>4}  "
              f"{f'{f.epss:.5f}' if f.epss is not None else '-':>7}  {'yes' if f.in_kev else '-':<4} "
              f"{f.fixed or '-':<10} {f.source:<8} {f.confidence:<6}")
        print(f"    {f.reason}")
    if unknown:
        print("\nCould not resolve:", ", ".join(f.cve.id for f in unknown))
    if args.all:
        print("\nNot affected:", ", ".join(f.cve.id for f in result.by_status("not_affected")))


if __name__ == "__main__":
    main()
