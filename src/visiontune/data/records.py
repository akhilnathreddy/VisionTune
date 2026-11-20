"""Normalize raw dataset rows into one record schema shared by every pipeline stage.

A record is a plain JSON-serializable dict. Images are *not* stored in records;
`hf_index` points back into the Hugging Face split so images are decoded lazily.

    {
      "id": "sqa-train-00012", "source": "scienceqa", "split": "train", "hf_index": 12,
      "question": "...", "choices": ["...", "..."], "answer_index": 1,
      "context": "...", "explanation": "...",
      "subject": "natural science", "topic": "biology", "category": "...",
      "skill": "...", "grade": "grade4",
      "has_image": true, "image_size": [w, h],
      "image_sha1": "...", "content_hash": "...", "text_key": "...", "dhash": "...",
      "near_dup_in_train": false   # validation/test only, set by data.prepare
    }
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

from PIL import Image

LETTERS = "ABCDEFGHIJ"

_SPLIT_PREFIX = {"train": "train", "validation": "val", "val": "val", "test": "test"}


def normalize_text(text: str | None) -> str:
    """Collapse whitespace; used for hashing and answer matching."""
    return re.sub(r"\s+", " ", (text or "")).strip()


def trim_explanation(text: str | None, max_chars: int) -> str:
    """Trim a long explanation to <= max_chars, cutting at a sentence boundary when possible."""
    text = normalize_text(text)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    window = text[:max_chars]
    cut = max(window.rfind(". "), window.rfind("? "), window.rfind("! "))
    if cut >= max_chars // 3:
        return window[: cut + 1]
    return window.rstrip() + "..."


def image_digest(image: Image.Image) -> str:
    """Exact hash of decoded pixels (format-independent)."""
    rgb = image.convert("RGB")
    h = hashlib.sha1()
    h.update(f"{rgb.size}".encode())
    h.update(rgb.tobytes())
    return h.hexdigest()


def image_dhash(image: Image.Image, size: int = 8) -> str:
    """64-bit difference hash: robust to re-encoding and small resizes (near-duplicate check)."""
    gray = image.convert("L").resize((size + 1, size), Image.Resampling.BOX)
    px = gray.tobytes()  # one byte per pixel in mode L
    bits = 0
    for row in range(size):
        for col in range(size):
            left = px[row * (size + 1) + col]
            right = px[row * (size + 1) + col + 1]
            bits = (bits << 1) | int(left > right)
    return f"{bits:016x}"


def content_hash(question: str, choices: list[str], context: str, img_digest: str | None) -> str:
    payload = "\x1f".join([normalize_text(question), *map(normalize_text, choices),
                           normalize_text(context), img_digest or ""])
    return hashlib.sha1(payload.encode()).hexdigest()


def hamming(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def text_key(question: str, choices: list[str]) -> str:
    """Same question text and same set of choices (any order). Combined with a small
    dHash distance this defines a near-duplicate (see prepare.remove_overlap)."""
    q = normalize_text(question).lower()
    c = "|".join(sorted(normalize_text(x).lower() for x in choices))
    return hashlib.sha1(f"{q}\x1f{c}".encode()).hexdigest()


def make_id(source: str, split: str, index: int) -> str:
    prefix = {"scienceqa": "sqa", "ai2d": "ai2d"}[source]
    return f"{prefix}-{_SPLIT_PREFIX.get(split, split)}-{index:05d}"


def scienceqa_record(row: dict[str, Any], split: str, index: int, image: Image.Image | None,
                     max_explanation_chars: int = 600) -> dict[str, Any]:
    choices = [normalize_text(c) for c in row["choices"]]
    digest = image_digest(image) if image is not None else None
    context = normalize_text(row.get("hint"))
    return {
        "id": make_id("scienceqa", split, index),
        "source": "scienceqa",
        "split": split,
        "hf_index": index,
        "question": normalize_text(row["question"]),
        "choices": choices,
        "answer_index": int(row["answer"]),
        "context": context,
        "explanation": trim_explanation(row.get("solution"), max_explanation_chars),
        "subject": row.get("subject") or "",
        "topic": row.get("topic") or "",
        "category": row.get("category") or "",
        "skill": row.get("skill") or "",
        "grade": row.get("grade") or "",
        "has_image": image is not None,
        "image_size": list(image.size) if image is not None else None,
        "image_sha1": digest,
        "content_hash": content_hash(row["question"], choices, context, digest),
        "text_key": text_key(row["question"], choices),
        "dhash": image_dhash(image) if image is not None else None,
    }


def ai2d_record(row: dict[str, Any], split: str, index: int,
                image: Image.Image | None) -> dict[str, Any]:
    choices = [normalize_text(c) for c in row["options"]]
    digest = image_digest(image) if image is not None else None
    return {
        "id": make_id("ai2d", split, index),
        "source": "ai2d",
        "split": split,
        "hf_index": index,
        "question": normalize_text(row["question"]),
        "choices": choices,
        "answer_index": int(row["answer"]),
        "context": "",
        "explanation": "",
        "subject": "diagram",
        "topic": "",
        "category": "",
        "skill": "",
        "grade": "",
        "has_image": image is not None,
        "image_size": list(image.size) if image is not None else None,
        "image_sha1": digest,
        "content_hash": content_hash(row["question"], choices, "", digest),
        "text_key": text_key(row["question"], choices),
        "dhash": image_dhash(image) if image is not None else None,
    }


def validate_record(record: dict[str, Any], require_image: bool) -> str | None:
    """Return an exclusion reason, or None if the record is usable."""
    if not record["question"]:
        return "empty_question"
    n = len(record["choices"])
    if n < 2:
        return "too_few_choices"
    if n > len(LETTERS):
        return "too_many_choices"
    if any(not c for c in record["choices"]):
        return "empty_choice"
    if not 0 <= record["answer_index"] < n:
        return "answer_out_of_range"
    if require_image and not record["has_image"]:
        return "no_image"
    return None
