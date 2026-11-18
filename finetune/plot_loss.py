"""Plots the LoRA training and held-out loss from the training log.

    python finetune/plot_loss.py outputs/logs/train_lora.log phase2/fig/loss_curve.pdf

The training loss in the log is that of the last conversation in an optimizer step (printed every
10 steps), so it is noisy; the held-out loss is the token-weighted mean over all 119 test
conversations, measured before training and after each epoch.
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

# reference categorical palette, slots 1 and 2 (validated: CVD dE 24.7, normal-vision dE 33.6)
TRAIN, TEST = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

STEP = re.compile(r"epoch (\d+) step (\d+)/(\d+) loss ([\d.]+)")
EPOCH_TEST = re.compile(r"epoch (\d+): test loss ([\d.]+)")
BEFORE = re.compile(r"test loss before training: ([\d.]+)")


def parse(text: str):
    train = [(int(m[2]), float(m[4])) for m in STEP.finditer(text)]
    total = int(STEP.search(text)[3])
    epochs = [(int(m[1]), float(m[2])) for m in EPOCH_TEST.finditer(text)]
    n_epochs = max(e for e, _ in epochs)
    per_epoch = math.ceil(total / n_epochs)
    test = [(0, float(BEFORE.search(text)[1]))] + [(min(e * per_epoch, total), v) for e, v in epochs]
    return train, test, total, per_epoch


def main(log: Path, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    train, test, total, per_epoch = parse(log.read_text(encoding="utf-8", errors="replace"))
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.3, 2.05))
    ax.set_yscale("log")
    ax.grid(True, which="major", axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    ax.axvline(per_epoch, color=MUTED, linewidth=0.6, linestyle=(0, (3, 3)))
    ax.text(per_epoch / 2, 1.0, "epoch 1", ha="center", va="top", color=MUTED, transform=ax.get_xaxis_transform())
    ax.text((per_epoch + total) / 2, 1.0, "epoch 2", ha="center", va="top", color=MUTED,
            transform=ax.get_xaxis_transform())

    xs, ys = zip(*train)
    ax.plot(xs, ys, color=TRAIN, linewidth=1.2, marker="o", markersize=3.2,
            markeredgecolor="white", markeredgewidth=0.6, label="training (one dialogue)")
    tx, ty = zip(*test)
    ax.plot(tx, ty, color=TEST, linewidth=1.2, marker="s", markersize=4.2,
            markeredgecolor="white", markeredgewidth=0.6, label="held-out (119 dialogues)")
    # direct labels on the held-out points: the three numbers the text refers to
    for x, y in test:
        ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points", xytext=(4, 5), color=INK, fontsize=7)

    ax.set_xlim(-3, total + 8)
    ax.set_xlabel("optimizer step (8 dialogues each)")
    ax.set_ylabel("cross-entropy, assistant tokens")
    ax.legend(loc="lower left", frameon=False, handlelength=1.6, borderaxespad=0.2)
    fig.tight_layout(pad=0.3)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    print(f"wrote {out} ({len(train)} training points, held-out {[round(v, 4) for _, v in test]})")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
