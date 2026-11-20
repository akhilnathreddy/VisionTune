"""Dataset wrapper and the SFT collator with assistant-only loss masking (blueprint 5.4)."""

from __future__ import annotations

from typing import Any, Sequence

import torch
from torch.utils.data import Dataset

from visiontune.prompts import build_messages

IGNORE_INDEX = -100


class VQADataset(Dataset):
    """Pairs normalized records with images decoded lazily from the HF split."""

    def __init__(self, records: Sequence[dict[str, Any]], image_source):
        self.records = list(records)
        self.image_source = image_source  # HF Dataset (or any indexable of {"image": PIL})

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> dict[str, Any]:
        rec = self.records[i]
        image = self.image_source[rec["hf_index"]]["image"] if rec.get("has_image") else None
        if image is not None:
            image = image.convert("RGB")
        return {"record": rec, "image": image}


def _find_subsequence(seq: list[int], pattern: list[int], start: int = 0) -> int:
    n = len(pattern)
    for i in range(start, len(seq) - n + 1):
        if seq[i:i + n] == pattern:
            return i
    return -1


class SFTCollator:
    """Tokenizes full conversations and computes loss only on the assistant reply.

    Label layout for one example (`x` = ignored, `y` = trained):

        <|im_start|>system ... <|im_end|>          x x x
        <|im_start|>user <vision tokens> ... <|im_end|>  x x x
        <|im_start|>assistant\\n                   x x x
        Answer: B. ... Explanation: ...<|im_end|>   y y y   (end token is trained so the model learns to stop)
        \\n  <pad> <pad>                            x x
    """

    def __init__(self, processor, system: str | None, include_explanation: bool = True):
        self.processor = processor
        self.system = system
        self.include_explanation = include_explanation
        tok = processor.tokenizer
        self.header_ids = tok.encode("<|im_start|>assistant\n", add_special_tokens=False)
        self.end_id = tok.convert_tokens_to_ids("<|im_end|>")
        self.pad_id = tok.pad_token_id

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        texts, images = [], []
        for item in batch:
            msgs = build_messages(item["record"], self.system, self.include_explanation, with_answer=True)
            texts.append(self.processor.apply_chat_template(msgs, tokenize=False))
            if item["image"] is not None:
                images.append(item["image"])
        self.processor.tokenizer.padding_side = "right"
        enc = self.processor(text=texts, images=images or None, padding=True, return_tensors="pt")
        enc["labels"] = self.make_labels(enc["input_ids"], enc["attention_mask"])
        return dict(enc)

    def make_labels(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        labels = torch.full_like(input_ids, IGNORE_INDEX)
        for row, ids in enumerate(input_ids.tolist()):
            start = _find_subsequence(ids, self.header_ids)
            if start < 0:
                raise ValueError("assistant header not found; check the chat template")
            begin = start + len(self.header_ids)
            try:
                end = ids.index(self.end_id, begin) + 1  # include <|im_end|>
            except ValueError:
                end = len(ids)
            labels[row, begin:end] = input_ids[row, begin:end]
        labels[attention_mask == 0] = IGNORE_INDEX
        return labels
