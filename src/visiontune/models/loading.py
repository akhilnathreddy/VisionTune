"""Load the Qwen2.5-VL processor and model for LoRA, QLoRA or plain inference."""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from visiontune.models.targets import describe_targets, resolve_target_modules

log = logging.getLogger(__name__)


def native_bf16() -> bool:
    """True only for GPUs with hardware bf16 (compute capability >= 8.0, Ampere+).

    torch.cuda.is_bf16_supported() also returns True on a T4 (7.5), where bf16 is
    emulated and slow; observed on Kaggle with torch 2.11, so we check capability."""
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8


def resolve_dtype(name: str | None) -> torch.dtype:
    """auto -> bf16 on Ampere+ GPUs, fp16 on older GPUs (T4, P100), fp32 on CPU."""
    name = (name or "auto").lower()
    if name == "auto":
        if torch.cuda.is_available():
            return torch.bfloat16 if native_bf16() else torch.float16
        return torch.float32
    return {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16, "fp16": torch.float16,
            "float16": torch.float16, "fp32": torch.float32, "float32": torch.float32}[name]


def load_processor(cfg: dict[str, Any], padding_side: str = "right"):
    from transformers import AutoProcessor

    m, p = cfg["model"], cfg.get("processor", {})
    processor = AutoProcessor.from_pretrained(
        m["name_or_path"], revision=m.get("revision", "main"),
        min_pixels=p.get("min_pixels"), max_pixels=p.get("max_pixels"),
    )
    processor.tokenizer.padding_side = padding_side
    if p.get("max_pixels"):
        check_visual_token_cap(processor, p["max_pixels"])
    return processor


def check_visual_token_cap(processor, max_pixels: int) -> int:
    """Fail loudly if the pixel cap is not applied. Some transformers versions silently
    ignore unrecognized resize options, and an uncapped 2000x1500 figure becomes
    ~3,800 visual tokens instead of <=512, which is an easy way to OOM a T4."""
    from PIL import Image

    grid = processor.image_processor(images=[Image.new("RGB", (2000, 1500))], return_tensors="pt")["image_grid_thw"]
    merge = getattr(processor.image_processor, "merge_size", 2)
    tokens = int(grid.prod()) // (merge * merge)
    cap = max_pixels // (28 * 28)
    if tokens > cap:
        raise RuntimeError(f"Processor produced {tokens} visual tokens for a large image; expected <= {cap}. "
                           "The installed transformers version ignores min_pixels/max_pixels.")
    return tokens


def _quant_config(cfg: dict[str, Any], dtype: torch.dtype):
    from transformers import BitsAndBytesConfig

    q = cfg["quantization"]
    compute = dtype if q.get("compute_dtype", "auto") == "auto" else resolve_dtype(q["compute_dtype"])
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=q.get("quant_type", "nf4"),
        bnb_4bit_use_double_quant=q.get("double_quant", True),
        bnb_4bit_compute_dtype=compute,
        llm_int8_skip_modules=list(q.get("skip_modules", ["model.visual", "visual", "lm_head"])),
    )


def load_model(cfg: dict[str, Any], quantized: bool | None = None):
    """Load the base model. `quantized` defaults to cfg['method'] == 'qlora'."""
    from transformers import Qwen2_5_VLForConditionalGeneration

    m = cfg["model"]
    dtype = resolve_dtype(m.get("dtype"))
    quantized = cfg.get("method") == "qlora" if quantized is None else quantized
    kwargs: dict[str, Any] = {
        "revision": m.get("revision", "main"),
        "dtype": dtype,
        "attn_implementation": m.get("attn_implementation", "sdpa"),
    }
    if torch.cuda.is_available():
        kwargs["device_map"] = {"": torch.cuda.current_device()}
    if quantized:
        kwargs["quantization_config"] = _quant_config(cfg, dtype)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(m["name_or_path"], **kwargs)
    if m.get("vision_fp32"):
        visual = getattr(model, "visual", None) or model.model.visual
        visual.float()
    log.info("Loaded %s (dtype=%s, quantized=%s)", m["name_or_path"], dtype, quantized)
    return model


def linear_layer_summary(model: torch.nn.Module) -> dict[str, dict[str, int]]:
    """Count Linear layer classes in the vision tower vs the rest, to confirm what
    was actually quantized (e.g. Linear4bit only outside `visual`)."""
    counts: dict[str, Counter] = {"vision": Counter(), "language": Counter()}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            part = "vision" if ".visual." in f".{name}." else "language"
            counts[part][type(module).__name__] += 1
    return {k: dict(v) for k, v in counts.items()}


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    total = trainable = 0
    for p in model.parameters():
        n = p.numel()
        # bitsandbytes packs two 4-bit values per byte; report logical parameters.
        if p.__class__.__name__ == "Params4bit":
            n *= 2
        total += n
        if p.requires_grad:
            trainable += n
    return {"total": total, "trainable": trainable}


def attach_lora(model, cfg: dict[str, Any], gradient_checkpointing: bool = True):
    """Freeze the base model and add LoRA adapters to the configured targets."""
    from peft import LoraConfig, get_peft_model

    for p in model.parameters():
        p.requires_grad_(False)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.config.use_cache = False

    lcfg = cfg["lora"]
    targets = resolve_target_modules(model, lcfg.get("targets", "lm_attn"))
    log.info("LoRA targets (%d modules): %s", len(targets), describe_targets(targets))
    peft_config = LoraConfig(
        r=lcfg.get("r", 16),
        lora_alpha=lcfg.get("alpha", 32),
        lora_dropout=lcfg.get("dropout", 0.05),
        target_modules=targets,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    params = count_parameters(model)
    if params["trainable"] == 0:
        raise RuntimeError("No trainable parameters after attaching LoRA")
    log.info("Parameters: %s (trainable %.3f%%)", params, 100 * params["trainable"] / params["total"])
    return model


def load_for_inference(cfg: dict[str, Any], adapter_path: str | Path | None = None,
                       quantized: bool | None = None):
    """Base model (optionally 4-bit) plus an optional saved adapter, in eval mode."""
    model = load_model(cfg, quantized=quantized)
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()
    model.config.use_cache = True
    return model
