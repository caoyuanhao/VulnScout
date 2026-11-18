"""An open-weight model (optionally with a LoRA adapter) as an agent backend, via Hugging Face.

Implements the same interface as :class:`vulnscout.agent.ClaudeBackend`, so the base model, the
fine-tuned model and Claude run the identical agent loop, prompt and tools. Tool calls are parsed
from the ``<tool_call>{json}</tool_call>`` blocks that Qwen-family chat templates use.

Needs ``torch``, ``transformers`` and (for adapters) ``peft``; meant to run on the GPU cluster.
"""
from __future__ import annotations

import json
import re
from datetime import date

from . import tools
from .agent import SYSTEM_PROMPT, ToolCall, Turn

_TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def parse_tool_calls(text: str, step: int) -> tuple[str, list[ToolCall]]:
    """Splits generated text into plain content and tool calls. Malformed calls are reported back
    to the model as errors instead of being silently dropped."""
    calls = []
    for k, m in enumerate(_TOOL_CALL.finditer(text)):
        try:
            obj = json.loads(m.group(1))
            name, args = obj["name"], obj.get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            name, args = "__invalid__", {"raw": m.group(1)[:500], "error": str(e)}
        calls.append(ToolCall(f"call_{step}_{k}", name, args))
    content = _TOOL_CALL.sub("", text).strip()
    return content, calls


def disable_sdpa_gqa() -> None:
    """Makes transformers repeat the key/value heads instead of passing ``enable_gqa`` to SDPA.

    With ``enable_gqa`` PyTorch 2.6 can only use the FlashAttention or math kernels. GPUs without
    FlashAttention (e.g. V100) then fall back to the math kernel, which materialises the full
    heads x L x L attention matrix (about 10 GB for a 13k-token conversation) and runs out of memory.
    Repeating the 8 KV heads lets SDPA use the memory-efficient kernel, whose memory grows linearly.
    """
    try:
        from transformers.integrations import sdpa_attention
    except ImportError:
        return
    if hasattr(sdpa_attention, "use_gqa_in_sdpa"):
        sdpa_attention.use_gqa_in_sdpa = lambda *args, **kwargs: False


class HFBackend:
    def __init__(self, model_name: str, adapter: str | None = None, max_new_tokens: int = 1500,
                 device: str = "cuda"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        disable_sdpa_gqa()
        self.torch = torch
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
        self.tokenizer = AutoTokenizer.from_pretrained(adapter or model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, attn_implementation="sdpa",
                                                     device_map={"": device})
        if adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
        self.model = model.eval()
        self.device, self.max_new_tokens = device, max_new_tokens
        self.tools = tools.function_schemas()
        self.messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT + f"\n\nToday is {date.today().isoformat()}."}]
        self.steps = 0
        self.usage = {"input_tokens": 0, "output_tokens": 0}

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def step(self) -> Turn:
        self.steps += 1
        ids = self.tokenizer.apply_chat_template(self.messages, tools=self.tools, add_generation_prompt=True,
                                                 return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(ids, max_new_tokens=self.max_new_tokens, do_sample=False,
                                      pad_token_id=self.tokenizer.eos_token_id)
        new = out[0, ids.shape[1]:]
        self.usage["input_tokens"] += ids.shape[1]
        self.usage["output_tokens"] += len(new)
        text = self.tokenizer.decode(new, skip_special_tokens=True)
        content, calls = parse_tool_calls(text, self.steps)
        message: dict = {"role": "assistant", "content": content}
        if calls:
            message["tool_calls"] = [{"id": c.id, "type": "function",
                                      "function": {"name": c.name, "arguments": c.arguments}} for c in calls]
        self.messages.append(message)
        stop = "tool_use" if calls else ("max_tokens" if len(new) >= self.max_new_tokens else "end_turn")
        return Turn(content, calls, stop)

    def add_tool_results(self, calls: list[ToolCall]) -> None:
        for c in calls:
            self.messages.append({"role": "tool", "tool_call_id": c.id, "name": c.name, "content": c.result})
