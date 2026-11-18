"""Uploads a finished LoRA run's logged losses to Weights & Biases, after the fact.

    python finetune/log_to_wandb.py logs/train_11959691.log --project vulnscout --name lora-qwen3-4b

train_lora.py prints the losses instead of logging them live, so this replays the printed points
(the training loss every 10 optimizer steps and the held-out loss before training and after each
epoch) into one W&B run, against the optimizer step. Credentials come from ``wandb login``.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import wandb

from plot_loss import parse

LR = re.compile(r"step (\d+)/\d+ loss [\d.]+ lr ([\d.e+-]+)")
PARAMS = re.compile(r"trainable params: ([\d,]+)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", type=Path)
    ap.add_argument("--project", default="vulnscout")
    ap.add_argument("--name", default="lora-qwen3-4b")
    args = ap.parse_args()

    text = args.log.read_text(encoding="utf-8", errors="replace")
    train, test, total, per_epoch = parse(text)
    lrs = {int(m[1]): float(m[2]) for m in LR.finditer(text)}
    config = {"model": "Qwen/Qwen3-4B-Instruct-2507", "method": "LoRA", "rank": 16, "alpha": 32, "dropout": 0.05,
              "target_modules": "all-linear", "lr": 2e-4, "schedule": "cosine, 5% warmup", "epochs": 2,
              "grad_accum": 8, "optimizer_steps": total, "train_dialogues": 450, "test_dialogues": 119,
              "trainable_params": int(PARAMS.search(text)[1].replace(",", "")), "gpu": "V100-SXM2-32GB",
              "precision": "fp16 weights, fp32 adapters", "source_log": args.log.name}
    run = wandb.init(project=args.project, name=args.name, config=config, job_type="train")
    run.define_metric("optimizer_step")
    for key in ("train/loss_one_dialogue", "train/lr", "heldout/loss"):
        run.define_metric(key, step_metric="optimizer_step")

    points: dict[int, dict] = {}
    for step, loss in train:
        points.setdefault(step, {})["train/loss_one_dialogue"] = loss
        if step in lrs:
            points[step]["train/lr"] = lrs[step]
    for step, loss in test:
        points.setdefault(step, {})["heldout/loss"] = loss
    for step in sorted(points):
        run.log({"optimizer_step": step, **points[step]})
    run.summary.update({"heldout/loss_before": test[0][1], "heldout/loss_final": test[-1][1]})
    print(f"logged {len(points)} steps to {run.url}")
    run.finish()


if __name__ == "__main__":
    main()
