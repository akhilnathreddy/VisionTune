"""Generate answers with the base model or a LoRA/QLoRA adapter and score them.

    # M0 baseline on 200 validation questions
    python -m visiontune.inference.predict --config configs/base.yaml --split validation --limit 200

    # M1 LoRA on the full ScienceQA test split (final evaluation only)
    python -m visiontune.inference.predict --config configs/lora.yaml \
        --adapter outputs/scienceqa_qwen25vl3b_lora_r16_seed42/adapter --split test

    # Cross-dataset: AI2D
    python -m visiontune.inference.predict --config configs/lora.yaml --adapter ... --dataset ai2d --split test

Predictions are appended to a JSONL file as they are produced, so an interrupted
Kaggle session resumes where it stopped. A metrics JSON is written at the end.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Sequence

import torch

from visiontune.config import load_config
from visiontune.data.io import iter_jsonl
from visiontune.data.loaders import load_image_source, load_records
from visiontune.data.sampling import select_subset
from visiontune.evaluation.extraction import extract_answer, extract_explanation
from visiontune.env import pin_single_gpu
from visiontune.evaluation.metrics import group_summary, summarize
from visiontune.prompts import build_messages

log = logging.getLogger(__name__)


class Predictor:
    def __init__(self, model, processor, cfg: dict[str, Any]):
        self.model = model
        self.processor = processor
        self.system = cfg["prompt"].get("system")
        self.include_explanation = cfg["prompt"].get("include_explanation", True)
        g = cfg.get("generation", {})
        self.gen_kwargs = {"max_new_tokens": g.get("max_new_tokens", 160), "do_sample": g.get("do_sample", False)}

    @torch.inference_mode()
    def generate(self, records: Sequence[dict[str, Any]], images: Sequence[Any]) -> list[str]:
        return [text for text, _ in self.generate_with_counts(records, images)]

    @torch.inference_mode()
    def generate_with_counts(self, records: Sequence[dict[str, Any]], images: Sequence[Any]) -> list[tuple[str, int]]:
        """Decoded answers plus the number of generated tokens (up to and including EOS)."""
        texts = [self.processor.apply_chat_template(
                    build_messages(r, self.system, self.include_explanation),
                    tokenize=False, add_generation_prompt=True) for r in records]
        imgs = [im for im in images if im is not None]
        self.processor.tokenizer.padding_side = "left"   # generation needs left padding
        enc = self.processor(text=texts, images=imgs or None, padding=True, return_tensors="pt")
        enc = enc.to(self.model.device)
        out = self.model.generate(**enc, **self.gen_kwargs, pad_token_id=self.processor.tokenizer.pad_token_id)
        new_tokens = out[:, enc["input_ids"].shape[1]:]
        pad = self.processor.tokenizer.pad_token_id
        counts = (new_tokens != pad).sum(dim=1).tolist()
        texts = self.processor.batch_decode(new_tokens, skip_special_tokens=True)
        return list(zip(texts, counts))


def score(record: dict[str, Any], output: str) -> dict[str, Any]:
    ext = extract_answer(output, record["choices"])
    return {
        "pred_index": ext.index,
        "extraction": ext.method,
        "correct": ext.index == record["answer_index"],
        "pred_explanation": extract_explanation(output),
    }


def compute_metrics(records: list[dict[str, Any]], preds: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {p["id"]: p for p in preds}
    paired = [(r, by_id[r["id"]]) for r in records if r["id"] in by_id]
    recs = [r for r, _ in paired]
    correct = [p["correct"] for _, p in paired]
    methods = [p["extraction"] for _, p in paired]
    n = len(paired)
    metrics = {
        "overall": summarize(correct),
        "format": {m: methods.count(m) / n for m in ("strict", "lenient", "text", "none")} if n else {},
        "by_subject": group_summary(recs, correct, "subject"),
        "by_topic": group_summary(recs, correct, "topic"),
        "by_grade": group_summary(recs, correct, "grade"),
        "by_near_dup_in_train": group_summary(recs, correct, "near_dup_in_train"),
    }
    lat = [p["latency_s"] for _, p in paired if p.get("latency_s") is not None]
    if lat:
        metrics["latency_s_per_example_batched"] = sum(lat) / len(lat)
    toks = [p["new_tokens"] for _, p in paired if p.get("new_tokens") is not None]
    if toks:
        metrics["mean_new_tokens"] = sum(toks) / len(toks)
        rates = [p["batch_tokens_per_s"] for _, p in paired if p.get("batch_tokens_per_s")]
        metrics["generated_tokens_per_s_batched"] = sum(rates) / len(rates)
    return metrics


def run(cfg: dict[str, Any], dataset: str, split: str, adapter: str | None, limit: int | None,
        out_path: Path, quantized: bool | None = None) -> dict[str, Any]:
    from visiontune.models.loading import load_for_inference, load_processor

    records = load_records(cfg, dataset, split)
    if limit:
        records = select_subset(records, limit, cfg.get("seed", 42))
    done: dict[str, dict] = {}
    if out_path.exists():
        done = {p["id"]: p for p in iter_jsonl(out_path)}
        log.info("Resuming: %d predictions already in %s", len(done), out_path)
    todo = [r for r in records if r["id"] not in done]

    if todo:
        images = load_image_source(cfg, dataset, split)
        processor = load_processor(cfg, padding_side="left")
        model = load_for_inference(cfg, adapter, quantized=quantized)
        predictor = Predictor(model, processor, cfg)
        bs = cfg.get("generation", {}).get("batch_size", 8)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("a") as f:
            for start in range(0, len(todo), bs):
                batch = todo[start:start + bs]
                imgs = [images[r["hf_index"]]["image"].convert("RGB") if r["has_image"] else None for r in batch]
                t0 = time.perf_counter()
                outputs = predictor.generate_with_counts(batch, imgs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                batch_seconds = time.perf_counter() - t0
                per_example = batch_seconds / len(batch)
                batch_tokens = max(1, sum(n for _, n in outputs))
                for rec, (output, n_tokens) in zip(batch, outputs):
                    row = {"id": rec["id"], "answer_index": rec["answer_index"], "output": output,
                           "latency_s": per_example, "new_tokens": n_tokens,
                           "batch_tokens_per_s": batch_tokens / batch_seconds, **score(rec, output)}
                    done[rec["id"]] = row
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                log.info("%d / %d", min(start + bs, len(todo)), len(todo))

    metrics = compute_metrics(records, list(done.values()))
    metrics.update({
        "dataset": dataset, "split": split, "adapter": adapter, "model": cfg["model"]["name_or_path"],
        "quantized": cfg.get("method") == "qlora" if quantized is None else quantized,
        "generation": cfg.get("generation", {}),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2)
        if torch.cuda.is_available() and todo else None,
    })
    metrics_path = out_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2))
    o = metrics["overall"]
    log.info("Accuracy %.4f (95%% CI %.4f-%.4f, n=%d) -> %s",
             o["accuracy"], o["ci95_low"], o["ci95_high"], o["n"], metrics_path)
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", default="scienceqa", choices=["scienceqa", "ai2d"])
    parser.add_argument("--split", default="validation")
    parser.add_argument("--adapter", default=None, help="path to a saved LoRA/QLoRA adapter")
    parser.add_argument("--limit", type=int, default=None, help="evaluate a deterministic subset")
    parser.add_argument("--quantized", choices=["auto", "yes", "no"], default="auto",
                        help="load the base in 4-bit (default: only for method=qlora)")
    parser.add_argument("--out", default=None)
    parser.add_argument("overrides", nargs="*", help="key.path=value overrides")
    args = parser.parse_args(argv)
    pin_single_gpu()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.config, args.overrides)
    quantized = {"auto": None, "yes": True, "no": False}[args.quantized]
    tag = Path(args.adapter).parent.name if args.adapter else "base"
    suffix = f"_n{args.limit}" if args.limit else ""
    out = Path(args.out) if args.out else Path("results/predictions") / f"{tag}_{args.dataset}_{args.split}{suffix}.jsonl"
    run(cfg, args.dataset, args.split, args.adapter, args.limit, out, quantized)


if __name__ == "__main__":
    main()
