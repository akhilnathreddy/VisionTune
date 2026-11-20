"""Accuracy with uncertainty, paired model comparison and per-group breakdowns
(blueprint 7.7 / 7.8). Pure numpy so it runs anywhere."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Sequence

import numpy as np


def accuracy(correct: Sequence[bool]) -> float:
    return float(np.mean(correct)) if len(correct) else float("nan")


def wilson_ci(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (well-behaved near 0 and 1)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_ci(correct: Sequence[bool], n_boot: int = 10_000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float]:
    x = np.asarray(correct, dtype=float)
    if x.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, x.size, size=(n_boot, x.size))].mean(axis=1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def paired_bootstrap(a: Sequence[bool], b: Sequence[bool], n_boot: int = 10_000,
                     alpha: float = 0.05, seed: int = 0) -> dict[str, float]:
    """Accuracy difference (b - a) on the *same* examples, with a bootstrap CI and
    a two-sided p-value for the null hypothesis of no difference."""
    xa, xb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if xa.shape != xb.shape:
        raise ValueError("paired comparison needs predictions for the same examples")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, xa.size, size=(n_boot, xa.size))
    diffs = xb[idx].mean(axis=1) - xa[idx].mean(axis=1)
    observed = float(xb.mean() - xa.mean())
    # Shift the bootstrap distribution to the null and count more extreme values.
    p = float(np.mean(np.abs(diffs - observed) >= abs(observed)))
    return {
        "diff": observed,
        "ci_low": float(np.quantile(diffs, alpha / 2)),
        "ci_high": float(np.quantile(diffs, 1 - alpha / 2)),
        "p_value": p,
        "n": int(xa.size),
    }


def mcnemar_exact(a: Sequence[bool], b: Sequence[bool]) -> dict[str, float]:
    """Exact McNemar test on discordant pairs. a_only = a right & b wrong."""
    xa, xb = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    a_only = int(np.sum(xa & ~xb))
    b_only = int(np.sum(~xa & xb))
    n = a_only + b_only
    if n == 0:
        return {"a_only": 0, "b_only": 0, "p_value": 1.0}
    k = min(a_only, b_only)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return {"a_only": a_only, "b_only": b_only, "p_value": min(1.0, 2 * tail)}


def summarize(correct: Sequence[bool], seed: int = 0) -> dict[str, Any]:
    n, k = len(correct), int(np.sum(correct))
    lo, hi = wilson_ci(k, n)
    return {"n": n, "correct": k, "accuracy": accuracy(correct), "ci95_low": lo, "ci95_high": hi}


def group_summary(records: Sequence[dict[str, Any]], correct: Sequence[bool],
                  key: str) -> dict[str, dict[str, Any]]:
    """Accuracy per value of `key` (e.g. subject, topic, grade), with counts and CIs."""
    groups: dict[str, list[bool]] = defaultdict(list)
    for rec, ok in zip(records, correct):
        value = rec.get(key)
        groups["unknown" if value is None or value == "" else str(value)].append(bool(ok))
    return {g: summarize(v) for g, v in sorted(groups.items(), key=lambda kv: -len(kv[1]))}
