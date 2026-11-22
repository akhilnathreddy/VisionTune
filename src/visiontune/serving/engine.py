"""One model in memory, several adapters, switchable per request.

The base model is loaded once per worker (blueprint §8: never reload per request).
Every adapter is attached to that single base with PEFT's multi-adapter support;
"base" runs with adapters disabled. A lock serialises generation, so one GPU
worker handles one request at a time. Scale out with more workers, not threads.
"""

from __future__ import annotations

import re
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from visiontune.data.records import LETTERS, normalize_text
from visiontune.evaluation.extraction import extract_answer, extract_explanation
from visiontune.prompts import build_messages, build_open_messages

BASE = "base"


class VQAEngine:
    def __init__(self, cfg: dict[str, Any], adapters: dict[str, str | Path] | None = None,
                 quantized: bool = True):
        from visiontune.models.loading import load_model, load_processor

        self.cfg = cfg
        self.quantized = quantized
        self.processor = load_processor(cfg, padding_side="left")
        model = load_model(cfg, quantized=quantized)
        self.adapter_names: list[str] = []
        for name, path in (adapters or {}).items():
            if not self.adapter_names:
                from peft import PeftModel

                model = PeftModel.from_pretrained(model, str(path), adapter_name=name)
            else:
                model.load_adapter(str(path), adapter_name=name)
            self.adapter_names.append(name)
        model.eval()
        model.config.use_cache = True
        self.model = model
        self.lock = threading.Lock()
        prompt = cfg.get("prompt", {})
        self.system = prompt.get("system")
        gen = cfg.get("generation", {})
        self.max_new_tokens = int(gen.get("max_new_tokens", 160))

    @property
    def variants(self) -> list[str]:
        return [BASE, *self.adapter_names]

    def info(self) -> dict[str, Any]:
        m = self.cfg["model"]
        return {"base_model": m["name_or_path"], "revision": m.get("revision"),
                "quantized_base": self.quantized, "variants": self.variants,
                "device": str(self.model.device), "max_new_tokens": self.max_new_tokens,
                "max_pixels": self.cfg.get("processor", {}).get("max_pixels")}

    def _variant_context(self, variant: str):
        if variant not in self.variants:
            raise ValueError(f"unknown model variant {variant!r}; available: {self.variants}")
        if not self.adapter_names:
            return nullcontext()
        if variant == BASE:
            return self.model.disable_adapter()
        self.model.set_adapter(variant)
        return nullcontext()

    @torch.inference_mode()
    def _generate(self, messages: list[dict[str, Any]], image: Image.Image) -> tuple[str, int]:
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = self.processor(text=[text], images=[image], return_tensors="pt").to(self.model.device)
        out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                                  pad_token_id=self.processor.tokenizer.pad_token_id)
        new = out[0, enc["input_ids"].shape[1]:]
        n_tokens = int((new != self.processor.tokenizer.pad_token_id).sum())
        return self.processor.decode(new, skip_special_tokens=True), n_tokens

    def predict(self, image: Image.Image, question: str, choices: list[str] | None = None,
                context: str | None = None, variant: str = BASE) -> dict[str, Any]:
        image = image.convert("RGB")
        choices = [normalize_text(c) for c in (choices or []) if normalize_text(c)]
        if choices:
            record = {"question": normalize_text(question), "choices": choices,
                      "context": normalize_text(context), "has_image": True}
            messages = build_messages(record, self.system, include_explanation=True)
        else:
            messages = build_open_messages(normalize_text(question), normalize_text(context), self.system)
        with self.lock:
            t0 = time.perf_counter()
            with self._variant_context(variant):
                raw, n_tokens = self._generate(messages, image)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latency = time.perf_counter() - t0
        result: dict[str, Any] = {"model": variant, "raw_output": raw, "latency_s": round(latency, 3),
                                  "new_tokens": n_tokens, "explanation": extract_explanation(raw)}
        if choices:
            ext = extract_answer(raw, choices)
            result.update({"answer_index": ext.index, "extraction": ext.method,
                           "answer": (f"{LETTERS[ext.index]}. {choices[ext.index]}" if ext.index is not None else None)})
        else:
            m = re.search(r"answer\s*[:：]\s*(.+)", raw, flags=re.IGNORECASE)
            result.update({"answer_index": None, "extraction": "open",
                           "answer": normalize_text(m.group(1) if m else raw.split("\n", 1)[0])})
        return result
