"""LoRA target-module presets for the ablations in blueprint 5.3.

Qwen2.5-VL reuses the names `gate_proj`, `up_proj` and `down_proj` in both the
vision tower (model.visual.blocks.N.mlp.*) and the language model
(model.language_model.layers.N.mlp.*). Passing bare names such as
["gate_proj", "up_proj", "down_proj"] to PEFT would therefore silently adapt the
vision encoder too. We resolve presets to explicit full module paths instead,
so each ablation trains exactly the components it claims to.

Module paths differ slightly across transformers versions (`model.layers.N` vs
`model.language_model.layers.N`; `visual.` vs `model.visual.`), so rules match
on path *segments*, not on fixed prefixes.
"""

from __future__ import annotations

import re
from collections import Counter

import torch.nn as nn

LM_ATTN = re.compile(r"(^|\.)self_attn\.(q_proj|k_proj|v_proj|o_proj)$")
LM_MLP = re.compile(r"(^|\.)mlp\.(gate_proj|up_proj|down_proj)$")
MERGER = re.compile(r"(^|\.)visual\.merger\.")
VISION_ATTN = re.compile(r"(^|\.)visual\.blocks\.\d+\.attn\.(qkv|proj)$")

PRESETS: dict[str, tuple[str, ...]] = {
    "lm_attn": ("lm_attn",),                                                      # A1
    "lm_attn_mlp": ("lm_attn", "lm_mlp"),                                         # A2
    "lm_attn_mlp_merger": ("lm_attn", "lm_mlp", "merger"),                        # A3
    "lm_attn_mlp_merger_vision_attn": ("lm_attn", "lm_mlp", "merger", "vision_attn"),  # A4
}


def _is_vision(name: str) -> bool:
    return re.search(r"(^|\.)visual\.", name) is not None


def classify(name: str) -> str | None:
    """Which component group a Linear module belongs to (None = never a LoRA target)."""
    if name.endswith("lm_head"):
        return None
    if _is_vision(name):
        if MERGER.search(name):
            return "merger"
        if VISION_ATTN.search(name):
            return "vision_attn"
        return "vision_other"
    if LM_ATTN.search(name):
        return "lm_attn"
    if LM_MLP.search(name):
        return "lm_mlp"
    return None


def resolve_target_modules(model: nn.Module, preset: str) -> list[str]:
    if preset not in PRESETS:
        raise ValueError(f"Unknown LoRA target preset {preset!r}; choose from {sorted(PRESETS)}")
    wanted = set(PRESETS[preset])
    names = [name for name, module in model.named_modules()
             if isinstance(module, nn.Linear) and classify(name) in wanted]
    if not names:
        raise RuntimeError(f"Preset {preset!r} matched no Linear modules - inspect model.named_modules()")
    return names


def describe_targets(names: list[str]) -> dict[str, int]:
    """Count resolved targets per component group, for logs and run summaries."""
    return dict(Counter(classify(n) for n in names))


def lora_wrapped_modules(model: nn.Module) -> list[str]:
    """Names of modules that actually received LoRA weights.

    PEFT compresses long explicit target lists (>= 20 entries) into a minimal set of
    name suffixes before matching, so the saved `target_modules` can read e.g.
    ['q_proj', 'k_proj', ...]. This inspects the model itself instead."""
    names = []
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and len(getattr(module, "lora_A")) > 0:
            names.append(name.replace("base_model.model.", "", 1))
    return names

