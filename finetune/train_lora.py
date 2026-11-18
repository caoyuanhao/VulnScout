"""LoRA fine-tuning of an open-weight instruct model on the generated tool-use conversations.

The loss covers only the assistant turns (tool calls and final answers); system, user and tool
messages are context. Conversations are rendered with the model's own chat template, which
formats ``tool_calls`` and tool results the way the model expects at inference time.

    python finetune/train_lora.py --model Qwen/Qwen3-4B-Instruct-2507 --data data/sft --out outputs/lora-qwen3-4b

Defaults suit a 32 GB V100: fp16 weights, LoRA adapters in fp32, gradient checkpointing,
one conversation per step with gradient accumulation.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vulnscout.hf_backend import disable_sdpa_gqa  # noqa: E402
from vulnscout.tools import function_schemas  # noqa: E402

TOOLS = function_schemas()
disable_sdpa_gqa()   # otherwise SDPA uses the O(L^2)-memory math kernel on V100s (see the docstring)


def encode(tokenizer, messages: list[dict], max_len: int) -> tuple[list[int], list[int]] | None:
    """Token ids and labels (-100 outside assistant turns) for one conversation."""
    ids = tokenizer.apply_chat_template(messages, tools=TOOLS, tokenize=True)
    labels = [-100] * len(ids)
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        prefix = tokenizer.apply_chat_template(messages[:i], tools=TOOLS, tokenize=True, add_generation_prompt=True)
        end = len(tokenizer.apply_chat_template(messages[:i + 1], tools=TOOLS, tokenize=True))
        if ids[:len(prefix)] != prefix:
            raise ValueError("chat template is not prefix-stable; cannot locate assistant turns")
        labels[len(prefix):end] = ids[len(prefix):end]
    if len(ids) > max_len:
        return None
    return ids, labels


def load(path: Path, tokenizer, max_len: int) -> list[tuple[list[int], list[int]]]:
    out, skipped = [], 0
    for line in path.read_text(encoding="utf-8").splitlines():
        enc = encode(tokenizer, json.loads(line)["messages"], max_len)
        if enc is None:
            skipped += 1
        else:
            out.append(enc)
    print(f"{path.name}: {len(out)} conversations, {skipped} longer than {max_len} tokens skipped, "
          f"max {max(len(i) for i, _ in out)} tokens")
    return out


def lm_loss(model, ids: list[int], labels: list[int], device: str) -> torch.Tensor:
    """Mean next-token loss over the labelled positions only.

    The output layer is applied only where a label follows (about 800 of ~4,000 tokens), instead
    of to every position: with a 152k-token vocabulary, full logits for a 13k-token conversation
    would need over 8 GB on their own.
    """
    inner = model.get_base_model()                      # the causal LM with LoRA layers injected
    hidden = inner.model(input_ids=torch.tensor([ids], device=device)).last_hidden_state[0]
    target = torch.tensor(labels[1:], device=device)
    keep = (target != -100).nonzero(as_tuple=True)[0]
    logits = inner.lm_head(hidden[keep]).float()
    return torch.nn.functional.cross_entropy(logits, target[keep])


@torch.no_grad()
def evaluate(model, data, device) -> float:
    model.eval()
    total, n = 0.0, 0
    for ids, labels in data:
        with torch.autocast("cuda", dtype=next(model.parameters()).dtype):
            loss = lm_loss(model, ids, labels, device)
        tokens = sum(l != -100 for l in labels[1:])
        total += loss.item() * tokens
        n += tokens
    model.train()
    return total / n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--data", type=Path, default=Path("data/sft"))
    ap.add_argument("--out", type=Path, default=Path("outputs/lora"))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=14000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda"
    bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
    dtype = torch.bfloat16 if bf16 else torch.float16
    print(f"GPU {torch.cuda.get_device_name()}, weights in {dtype}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    train = load(args.data / "train.jsonl", tokenizer, args.max_len)
    test = load(args.data / "test.jsonl", tokenizer, args.max_len)

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, attn_implementation="sdpa",
                                                 device_map={"": device})
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
                                             target_modules="all-linear", task_type="CAUSAL_LM"))
    for p in model.parameters():          # train the adapters in fp32 for stable fp16 training
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    steps = math.ceil(len(train) * args.epochs / args.grad_accum)
    sched = get_cosine_schedule_with_warmup(opt, max(1, steps // 20), steps)
    scaler = torch.amp.GradScaler(enabled=not bf16)

    print(f"test loss before training: {evaluate(model, test, device):.4f}")
    step, t0 = 0, time.perf_counter()
    for epoch in range(args.epochs):
        random.shuffle(train)
        for i, (ids, labels) in enumerate(train):
            with torch.autocast("cuda", dtype=dtype):
                loss = lm_loss(model, ids, labels, device) / args.grad_accum
            scaler.scale(loss).backward()
            if (i + 1) % args.grad_accum == 0 or i == len(train) - 1:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
                if step % 10 == 0:
                    print(f"epoch {epoch + 1} step {step}/{steps} loss {loss.item() * args.grad_accum:.4f} "
                          f"lr {sched.get_last_lr()[0]:.2e} {time.perf_counter() - t0:.0f}s", flush=True)
        print(f"epoch {epoch + 1}: test loss {evaluate(model, test, device):.4f}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"saved adapter to {args.out}")


if __name__ == "__main__":
    main()
