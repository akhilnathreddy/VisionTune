"""LoRA / QLoRA supervised fine-tuning.

    python -m visiontune.training.train --config configs/lora.yaml
    python -m visiontune.training.train --config configs/qlora.yaml train.max_train_samples=2000
    python -m visiontune.training.train --config configs/lora.yaml --resume   # after a Kaggle restart

Everything needed to reproduce a run is written to <output_root>/<run_name>/:
    config.yaml, sample_ids.json, environment.json, run_summary.json,
    checkpoint-*/ (for resuming), adapter/ (final LoRA weights + processor)
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import torch

from visiontune.config import load_config, save_config
from visiontune.data.loaders import load_image_source, load_records
from visiontune.data.sampling import select_subset
from visiontune.env import environment_report
from visiontune.models.loading import (attach_lora, count_parameters, linear_layer_summary,
                                       load_model, load_processor, resolve_dtype)
from visiontune.models.targets import describe_targets
from visiontune.training.collator import SFTCollator, VQADataset

log = logging.getLogger(__name__)


def build_training_args(cfg: dict[str, Any], output_dir: Path, has_eval: bool):
    import inspect

    from transformers import TrainingArguments

    t = cfg["train"]
    dtype = resolve_dtype(cfg["model"].get("dtype"))
    on_gpu = torch.cuda.is_available()
    # transformers v5 folded warmup_ratio into warmup_steps (a float < 1 is a ratio).
    accepted = inspect.signature(TrainingArguments.__init__).parameters
    warmup = ({"warmup_ratio": t.get("warmup_ratio", 0.03)} if "warmup_ratio" in accepted
              else {"warmup_steps": float(t.get("warmup_ratio", 0.03))})
    return TrainingArguments(
        **warmup,
        output_dir=str(output_dir),
        run_name=cfg["run_name"],
        seed=cfg.get("seed", 42),
        num_train_epochs=t.get("epochs", 2),
        learning_rate=float(t.get("learning_rate", 1e-4)),
        lr_scheduler_type=t.get("lr_scheduler", "cosine"),
        weight_decay=t.get("weight_decay", 0.0),
        per_device_train_batch_size=t.get("per_device_batch_size", 1),
        per_device_eval_batch_size=t.get("per_device_batch_size", 1),
        gradient_accumulation_steps=t.get("gradient_accumulation_steps", 8),
        max_grad_norm=t.get("max_grad_norm", 1.0),
        optim=t.get("optim", "adamw_torch") if on_gpu else "adamw_torch",
        bf16=on_gpu and dtype == torch.bfloat16,
        fp16=on_gpu and dtype == torch.float16,
        logging_steps=t.get("logging_steps", 10),
        eval_strategy="steps" if has_eval else "no",
        eval_steps=t.get("eval_steps", 100),
        save_strategy="steps",
        save_steps=t.get("save_steps", 100),
        save_total_limit=t.get("save_total_limit", 3),
        report_to=t.get("report_to", "none"),
        remove_unused_columns=False,   # the collator needs the raw record + image
        dataloader_num_workers=t.get("dataloader_num_workers", 2 if on_gpu else 0),
        gradient_checkpointing=False,  # enabled on the model in attach_lora()
        use_cpu=not on_gpu,
    )


def last_checkpoint(output_dir: Path) -> str | None:
    ckpts = sorted(output_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    return str(ckpts[-1]) if ckpts else None


def directory_size_mb(path: Path) -> float:
    return round(sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1024**2, 2)


def run(cfg: dict[str, Any], resume: bool = False) -> dict[str, Any]:
    from transformers import Trainer, set_seed

    set_seed(cfg.get("seed", 42))
    t = cfg["train"]
    out_dir = Path(t.get("output_root", "outputs")) / cfg["run_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / "config.yaml")
    env = environment_report()
    (out_dir / "environment.json").write_text(json.dumps(env, indent=2))

    # Data: nested deterministic subsets of the cleaned official splits.
    train_records = select_subset(load_records(cfg, "scienceqa", "train"), t.get("max_train_samples"), cfg["seed"])
    eval_records = select_subset(load_records(cfg, "scienceqa", "validation"), t.get("max_eval_samples"), cfg["seed"])
    (out_dir / "sample_ids.json").write_text(json.dumps(
        {"train": [r["id"] for r in train_records], "validation": [r["id"] for r in eval_records]}, indent=1))
    log.info("Training on %d examples, evaluating loss on %d", len(train_records), len(eval_records))
    train_ds = VQADataset(train_records, load_image_source(cfg, "scienceqa", "train"))
    eval_ds = VQADataset(eval_records, load_image_source(cfg, "scienceqa", "validation")) if eval_records else None

    processor = load_processor(cfg)
    model = load_model(cfg)
    model = attach_lora(model, cfg, gradient_checkpointing=t.get("gradient_checkpointing", True))
    collator = SFTCollator(processor, cfg["prompt"].get("system"), cfg["prompt"].get("include_explanation", True))

    trainer = Trainer(
        model=model,
        args=build_training_args(cfg, out_dir, has_eval=eval_ds is not None),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        processing_class=processor,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    resume_from = last_checkpoint(out_dir) if resume else None
    if resume and not resume_from:
        log.warning("--resume given but no checkpoint found in %s; starting fresh", out_dir)
    start = time.perf_counter()
    result = trainer.train(resume_from_checkpoint=resume_from)
    train_seconds = time.perf_counter() - start

    adapter_dir = out_dir / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)
    final_eval = trainer.evaluate() if eval_ds is not None else {}

    peft_cfg = trainer.model.peft_config["default"]
    summary = {
        "run_name": cfg["run_name"],
        "method": cfg.get("method"),
        "train_examples": len(train_records),
        "eval_examples": len(eval_records),
        "train_seconds_this_session": round(train_seconds, 1),
        "resumed_from": resume_from,
        "train_metrics": result.metrics,
        "final_eval": final_eval,
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2) if torch.cuda.is_available() else None,
        "parameters": count_parameters(trainer.model),
        "lora_targets": describe_targets(sorted(peft_cfg.target_modules)),
        "linear_layers": linear_layer_summary(trainer.model),
        "adapter_size_mb": directory_size_mb(adapter_dir),
        "log_history": trainer.state.log_history,
        "gpu": env["gpus"],
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log.info("Done: %s", {k: summary[k] for k in ("train_seconds_this_session", "peak_vram_gb", "adapter_size_mb")})
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true", help="continue from the latest checkpoint-*")
    parser.add_argument("overrides", nargs="*", help="key.path=value overrides")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run(load_config(args.config, args.overrides), resume=args.resume)


if __name__ == "__main__":
    main()
