"""Renders the cover figure from the real Log4j 2.14.1 query (NVD + CISA KEV + FIRST EPSS).

Run from hw5/phase1 with the query outputs in DATA:  python make_cover.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

DATA = Path(r"C:\Users\86151\AppData\Local\Temp\claude\e--Northeastern-ENCP7375"
            r"\14859652-00b6-4a47-b836-d2922cbc5366\scratchpad")
EPSS = {  # FIRST EPSS API, 2026-10-27
    "CVE-2021-44228": 0.99999, "CVE-2021-45046": 0.99977, "CVE-2021-45105": 0.99999,
    "CVE-2021-44832": 0.97906, "CVE-2026-34480": 0.01046, "CVE-2025-68161": 0.00767,
    "CVE-2026-34481": 0.00691, "CVE-2026-49844": 0.00595, "CVE-2026-34477": 0.00560,
    "CVE-2026-34479": 0.00535,
}

nvd = json.load(open(DATA / "log4j_2141.json", encoding="utf-8"))
kev = {v["cveID"]: v for v in json.load(open(DATA / "kev.json", encoding="utf-8"))["vulnerabilities"]}
rows = []
for item in nvd["vulnerabilities"]:
    c = item["cve"]
    m = c["metrics"]
    cv = (m.get("cvssMetricV31") or m.get("cvssMetricV30") or [{}])[0].get("cvssData", {})
    rows.append((c["id"], c["published"][:7], cv.get("baseScore"), c["id"] in kev, EPSS[c["id"]]))
rows.sort(key=lambda r: (not r[3], -r[4]))

INK, MUTED, LINE, BG = "#1f2328", "#57606a", "#d0d7de", "#ffffff"
RED, ORANGE, YELLOW = "#cf222e", "#bc4c00", "#9a6700"

fig = plt.figure(figsize=(12, 4.6), dpi=200)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 12)
ax.set_ylim(0, 4.6)
ax.axis("off")
fig.patch.set_facecolor(BG)

# left: the user's question and the answer summary
ax.add_patch(FancyBboxPatch((0.25, 3.35), 3.6, 0.95, boxstyle="round,pad=0.03,rounding_size=0.12",
                            fc="#ddf4ff", ec="#54aeff", lw=1))
ax.text(0.4, 4.08, "User", fontsize=9, color=MUTED, weight="bold")
ax.text(0.4, 3.62, "Our Java service runs Log4j 2.14.1.\nWhat should we fix first?",
        fontsize=10.5, color=INK, va="center")

ax.add_patch(FancyBboxPatch((0.25, 0.3), 3.6, 2.85, boxstyle="round,pad=0.03,rounding_size=0.12",
                            fc="#f6f8fa", ec=LINE, lw=1))
ax.text(0.4, 2.93, "Assistant", fontsize=9, color=MUTED, weight="bold")
summary = [
    (INK, "10 published CVEs affect 2.14.1", 2.55),
    (MUTED, "(NVD, CISA KEV, FIRST EPSS; 2026-10-27)", 2.27),
    (RED, "2 are being exploited in the wild", 1.88),
    (MUTED, "used in ransomware campaigns: patch now", 1.6),
    (ORANGE, "2 more have EPSS above 0.97", 1.22),
    (YELLOW, "6 recent ones (2025-2026) have EPSS below 0.011,", 0.84),
    (YELLOW, "although three of them score CVSS 7.5", 0.56),
]
for color, text, y in summary:
    ax.text(0.4, y, text, fontsize=9.6, color=color, va="center",
            weight="bold" if color in (RED, ORANGE, INK) else "normal")

# right: the risk-ranked table
x0, ytop = 4.25, 4.25
cols = [("Rank", 0.0), ("CVE", 0.55), ("Published", 2.25), ("CVSS", 3.45), ("EPSS", 4.3), ("CISA KEV", 5.35)]
for name, dx in cols:
    ax.text(x0 + dx, ytop, name, fontsize=9.5, color=MUTED, weight="bold", va="center")
ax.plot([x0, 11.75], [ytop - 0.2, ytop - 0.2], color=LINE, lw=1)
row_h = 0.37
for i, (cid, pub, score, in_kev, epss) in enumerate(rows):
    y = ytop - 0.5 - i * row_h
    color = RED if in_kev else (ORANGE if epss > 0.5 else YELLOW)
    ax.add_patch(FancyBboxPatch((x0 - 0.02, y - 0.13), 0.32, 0.26, boxstyle="round,pad=0,rounding_size=0.06",
                                fc=color, ec="none"))
    ax.text(x0 + 0.14, y, str(i + 1), fontsize=9, color="white", ha="center", va="center", weight="bold")
    ax.text(x0 + 0.55, y, cid, fontsize=9.5, color=INK, va="center", family="monospace")
    ax.text(x0 + 2.25, y, pub, fontsize=9.5, color=INK, va="center")
    ax.text(x0 + 3.45, y, f"{score:.1f}", fontsize=9.5, color=INK, va="center")
    ax.text(x0 + 4.3, y, f"{epss:.5f}", fontsize=9.5, color=INK, va="center")
    ax.text(x0 + 5.35, y, "exploited" if in_kev else "-", fontsize=9.5,
            color=RED if in_kev else MUTED, va="center", weight="bold" if in_kev else "normal")

Path("fig").mkdir(exist_ok=True)
fig.savefig("fig/cover.png", facecolor=BG)
print("wrote fig/cover.png with", len(rows), "rows")
