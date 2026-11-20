"""Build validated, de-duplicated record files from ScienceQA and AI2D.

    python -m visiontune.data.prepare --config configs/base.yaml

Writes to `data.processed_dir`:
    scienceqa_train.jsonl, scienceqa_validation.jsonl, scienceqa_test.jsonl,
    ai2d_test.jsonl, prepare_report.json (counts and every exclusion reason)

Rules (blueprint 4.5):
  * official splits are preserved; nothing moves between splits
  * malformed rows and rows whose image cannot be decoded are excluded
  * exact duplicates inside a split are dropped (first occurrence kept)
  * a train/validation record that exactly duplicates a test record is dropped;
    a train record that exactly duplicates a validation record is dropped
  * near-duplicates (same question and choices, image dHash within 6 bits) are
    *flagged* on validation/test records (`near_dup_in_train`) rather than
    removed: ScienceQA's templated figures (magnets, particle diagrams) often
    differ by a detail that changes the answer, which a perceptual hash cannot
    see. Set data.near_dup_policy=exclude to drop them from train instead.
  * every exclusion is counted by reason in the report
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from visiontune.config import load_config
from visiontune.data.io import write_jsonl
from visiontune.data.loaders import load_hf_split
from visiontune.data.records import ai2d_record, hamming, scienceqa_record, validate_record

log = logging.getLogger(__name__)

SCIENCEQA_SPLITS = ("train", "validation", "test")


def _decode(row: dict[str, Any]):
    """Return (PIL image or None, error string or None)."""
    image = row.get("image")
    if image is None:
        return None, None
    try:
        image.load()
        return image, None
    except Exception as exc:  # corrupt bytes, truncated file, ...
        return None, f"{type(exc).__name__}: {exc}"


def build_split(ds, split: str, make_record: Callable, require_image: bool,
                progress_every: int = 2000) -> tuple[list[dict], Counter]:
    records: list[dict] = []
    reasons: Counter = Counter()
    seen_hashes: set[str] = set()
    for idx in range(len(ds)):
        row = ds[idx]
        image, err = _decode(row)
        if err is not None:
            reasons["image_decode_error"] += 1
            continue
        try:
            rec = make_record(row, split, idx, image)
        except (KeyError, ValueError, TypeError):
            reasons["malformed_row"] += 1
            continue
        reason = validate_record(rec, require_image)
        if reason:
            reasons[reason] += 1
            continue
        if rec["content_hash"] in seen_hashes:
            reasons["duplicate_in_split"] += 1
            continue
        seen_hashes.add(rec["content_hash"])
        records.append(rec)
        if progress_every and (idx + 1) % progress_every == 0:
            log.info("%s: processed %d / %d", split, idx + 1, len(ds))
    return records, reasons


NEAR_DUP_MAX_HAMMING = 6  # of 64 dHash bits


class NearDupIndex:
    """Near-duplicate lookup: identical question and choice set, and image dHash within
    `max_hamming` bits (catches re-encoded or resized copies of the same figure)."""

    def __init__(self, records: list[dict], max_hamming: int = NEAR_DUP_MAX_HAMMING):
        self.max_hamming = max_hamming
        self.by_text: dict[str, list[dict]] = {}
        for r in records:
            self.by_text.setdefault(r["text_key"], []).append(r)

    def match(self, record: dict) -> dict | None:
        mine = record.get("dhash")
        for other in self.by_text.get(record["text_key"], []):
            theirs = other.get("dhash")
            if theirs == mine or (theirs and mine and hamming(theirs, mine) <= self.max_hamming):
                return other
        return None


def remove_overlap(records: list[dict], protected: list[dict], label: str, reasons: Counter,
                   exclude_near: bool = False) -> list[dict]:
    """Drop records that exactly duplicate (question, choices, context and decoded
    pixels) a record in a protected split; optionally drop near-duplicates too."""
    exact = {r["content_hash"] for r in protected}
    near = NearDupIndex(protected) if exclude_near else None
    kept = []
    for r in records:
        if r["content_hash"] in exact:
            reasons[f"exact_dup_of_{label}"] += 1
        elif near is not None and near.match(r) is not None:
            reasons[f"near_dup_of_{label}"] += 1
        else:
            kept.append(r)
    return kept


def flag_near_duplicates(eval_records: list[dict], train_records: list[dict]) -> dict[str, int]:
    """Mark eval records whose template + figure also appears in train, so results can
    be reported separately on seen-template and novel questions."""
    index = NearDupIndex(train_records)
    stats = Counter()
    for r in eval_records:
        other = index.match(r)
        r["near_dup_in_train"] = other is not None
        if other is None:
            continue
        same = other["choices"][other["answer_index"]] == r["choices"][r["answer_index"]]
        stats["flagged"] += 1
        stats["same_answer" if same else "different_answer"] += 1
    return dict(stats)


def prepare_scienceqa(cfg: dict, load_split=load_hf_split) -> tuple[dict[str, list[dict]], dict]:
    data_cfg = cfg["data"]
    revision = data_cfg.get("scienceqa_revision")
    max_chars = data_cfg.get("max_explanation_chars", 600)

    def make(row, split, idx, image):
        return scienceqa_record(row, split, idx, image, max_chars)

    out: dict[str, list[dict]] = {}
    report: dict[str, Any] = {"dataset": data_cfg["scienceqa_id"], "revision": revision, "splits": {}}
    reasons_by_split: dict[str, Counter] = {}
    for split in SCIENCEQA_SPLITS:
        ds = load_split(data_cfg["scienceqa_id"], split, revision)
        records, reasons = build_split(ds, split, make, data_cfg.get("require_image", True))
        out[split] = records
        reasons_by_split[split] = reasons
        report["splits"][split] = {"raw": len(ds)}

    # Leakage guards: test is protected from train and validation; validation from train.
    exclude_near = data_cfg.get("near_dup_policy", "flag") == "exclude"
    out["validation"] = remove_overlap(out["validation"], out["test"], "test",
                                       reasons_by_split["validation"], exclude_near)
    out["train"] = remove_overlap(out["train"], out["test"], "test", reasons_by_split["train"], exclude_near)
    out["train"] = remove_overlap(out["train"], out["validation"], "validation",
                                  reasons_by_split["train"], exclude_near)
    for split in ("validation", "test"):
        report["splits"][split]["near_dup_in_train"] = flag_near_duplicates(out[split], out["train"])

    for split in SCIENCEQA_SPLITS:
        report["splits"][split].update({
            "kept": len(out[split]),
            "excluded": dict(sorted(reasons_by_split[split].items())),
            "by_subject": dict(Counter(r["subject"] for r in out[split]).most_common()),
        })
    # Final assertion: no shared content between splits.
    for a, b in (("train", "test"), ("train", "validation"), ("validation", "test")):
        shared = {r["content_hash"] for r in out[a]} & {r["content_hash"] for r in out[b]}
        assert not shared, f"{len(shared)} records shared between {a} and {b}"
    return out, report


def prepare_ai2d(cfg: dict, load_split=load_hf_split) -> tuple[list[dict], dict]:
    dataset_id, revision = cfg["data"]["ai2d_id"], cfg["data"].get("ai2d_revision")
    ds = load_split(dataset_id, "test", revision)
    records, reasons = build_split(ds, "test", ai2d_record, require_image=True)
    return records, {"dataset": dataset_id, "revision": revision, "raw": len(ds), "kept": len(records),
                     "excluded": dict(sorted(reasons.items()))}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--skip-ai2d", action="store_true")
    parser.add_argument("overrides", nargs="*", help="key.path=value overrides")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.config, args.overrides)
    out_dir = Path(cfg["data"]["processed_dir"])
    splits, report = prepare_scienceqa(cfg)
    for split, records in splits.items():
        write_jsonl(records, out_dir / f"scienceqa_{split}.jsonl")
    full_report = {"scienceqa": report, "config": cfg["data"]}
    if not args.skip_ai2d:
        ai2d, ai2d_report = prepare_ai2d(cfg)
        write_jsonl(ai2d, out_dir / "ai2d_test.jsonl")
        full_report["ai2d"] = ai2d_report
    (out_dir / "prepare_report.json").write_text(json.dumps(full_report, indent=2))
    print(json.dumps(full_report, indent=2))


if __name__ == "__main__":
    main()
