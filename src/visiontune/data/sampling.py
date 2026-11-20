"""Deterministic, nested training subsets for the data-scaling study (blueprint 7.3).

Each record gets a rank from sha1(seed, id). Taking the first n by rank gives
subsets that are nested: the 500-example set is contained in the 2,000-example
set, which is contained in the 5,000-example set. That keeps the scaling curve
about data quantity rather than about which examples happened to be drawn.
"""

from __future__ import annotations

import hashlib
from typing import Any, Sequence


def _rank(seed: int, record_id: str) -> str:
    return hashlib.sha1(f"{seed}:{record_id}".encode()).hexdigest()


def select_subset(records: Sequence[dict[str, Any]], n: int | None, seed: int) -> list[dict[str, Any]]:
    """Return min(n, len(records)) records; n=None or n<=0 returns all of them (rank-ordered)."""
    ranked = sorted(records, key=lambda r: _rank(seed, r["id"]))
    if n is None or n <= 0:
        return ranked
    return ranked[:n]
