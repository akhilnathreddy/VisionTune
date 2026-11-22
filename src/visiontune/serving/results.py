"""Compact, read-only summary of the committed evaluation results for the /metrics endpoint."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPORTS = Path(__file__).resolve().parents[3] / "results" / "reports"


def _acc(m: dict) -> dict[str, float]:
    o = m["overall"]
    return {"accuracy": o["accuracy"], "ci95_low": o["ci95_low"], "ci95_high": o["ci95_high"], "n": o["n"]}


def summary(reports: Path = REPORTS) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, rel in (("scienceqa_test", "test/comparison_test.json"), ("ai2d_test", "ai2d/comparison_ai2d.json")):
        path = reports / rel
        if path.exists():
            data = json.loads(path.read_text())
            out[key] = {"n": data["n"],
                        "models": {k: {**_acc(v), "strict_format": v["format"].get("strict")}
                                   for k, v in data["models"].items()},
                        "pairs": {k: {"diff": v["bootstrap"]["diff"], "ci_low": v["bootstrap"]["ci_low"],
                                      "ci_high": v["bootstrap"]["ci_high"], "p_value": v["bootstrap"]["p_value"]}
                                  for k, v in data["pairs"].items()}}
    scaling = []
    for result in sorted((reports / "scaling").glob("*/result.json")):
        r = json.loads(result.read_text())
        scaling.append({"method": r["method"], "n_train": r["n_train"], **_acc(r["validation"]),
                        "train_minutes": round(r["train"]["train_seconds_this_session"] / 60, 1),
                        "peak_train_vram_gb": r["train"]["peak_vram_gb"]})
    if scaling:
        out["scienceqa_validation_scaling"] = sorted(scaling, key=lambda x: (x["method"], x["n_train"]))
    return out
