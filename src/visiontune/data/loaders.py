"""Locate processed record files and the HF splits that hold their images."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from visiontune.data.io import read_jsonl

DATASET_KEYS = {"scienceqa": "scienceqa", "ai2d": "ai2d"}


def processed_path(cfg: dict[str, Any], source: str, split: str) -> Path:
    return Path(cfg["data"]["processed_dir"]) / f"{source}_{split}.jsonl"


def load_records(cfg: dict[str, Any], source: str, split: str) -> list[dict[str, Any]]:
    path = processed_path(cfg, source, split)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python -m visiontune.data.prepare --config <config>")
    return read_jsonl(path)


def load_hf_split(dataset_id: str, split: str, revision: str | None = None):
    from datasets import load_dataset

    return load_dataset(dataset_id, split=split, revision=revision)


def load_image_source(cfg: dict[str, Any], source: str, split: str):
    key = DATASET_KEYS[source]
    return load_hf_split(cfg["data"][f"{key}_id"], split, cfg["data"].get(f"{key}_revision"))
