"""Plot the data-scaling curve from results/reports/scaling/*.

    python scripts/plot_scaling.py   # writes results/plots/scaling_curve.png

One chart, one y-axis (validation accuracy, %). The x-axis is training examples (0 = base model M0).
Three QLoRA series, direct-labelled (overall, novel, train-template near-duplicates), plus the
full-data LoRA run as a separate marker. 95% Wilson CIs are shown as bands on the overall line.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCALING = ROOT / "results/reports/scaling"
BASE = SCALING / "scienceqa_qwen25vl3b_qlora_r16_n2000_seed42/predictions/base_validation_full.metrics.json"

# Reference palette (dataviz skill), validated all-pairs for 3 series on the light surface.
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
OVERALL, NOVEL, NEAR = "#2a78d6", "#eb6834", "#1baf7a"


def point(metrics: dict) -> dict:
    nd = metrics["by_near_dup_in_train"]
    o = metrics["overall"]
    return {"acc": 100 * o["accuracy"], "lo": 100 * o["ci95_low"], "hi": 100 * o["ci95_high"],
            "novel": 100 * nd["False"]["accuracy"], "near": 100 * nd["True"]["accuracy"]}


def load() -> tuple[dict[int, dict], dict | None]:
    qlora = {0: point(json.loads(BASE.read_text()))}
    lora = None
    for result in sorted(SCALING.glob("*/result.json")):
        r = json.loads(result.read_text())
        p = point(r["validation"])
        if r["method"] == "qlora":
            qlora[r["n_train"]] = p
        elif r["method"] == "lora" and r["n_train"] == 6218:
            lora = p
    return dict(sorted(qlora.items())), lora


def main() -> None:
    qlora, lora = load()
    xs = list(qlora)
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ax.fill_between(xs, [qlora[x]["lo"] for x in xs], [qlora[x]["hi"] for x in xs],
                    color=OVERALL, alpha=0.15, linewidth=0)
    series = [("novel", "Novel questions", NOVEL), ("acc", "All questions (QLoRA)", OVERALL),
              ("near", "Train-template near-duplicates", NEAR)]
    for key, label, color in series:
        ys = [qlora[x][key] for x in xs]
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=label, zorder=3)
        ax.annotate(f"{label}  {ys[-1]:.1f}%", (xs[-1], ys[-1]), xytext=(26, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK)
    if lora:
        ax.plot([6218 + 160], [lora["acc"]], linestyle="none", marker="D", markersize=7, markerfacecolor=SURFACE,
                markeredgecolor=OVERALL, markeredgewidth=2, label="All questions (LoRA, 6,218)", zorder=4)
        ax.annotate(f"LoRA {lora['acc']:.1f}%", (6218 + 160, lora["acc"]), xytext=(0, -16), textcoords="offset points",
                    ha="center", fontsize=8, color=INK_2)
    for x in xs:
        ax.annotate(f"{qlora[x]['acc']:.1f}", (x, qlora[x]["acc"]), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=8, color=INK_2)

    ax.set_xlim(-200, 8600)
    ax.set_ylim(50, 100)
    ax.set_xticks(xs)
    ax.set_xticklabels(["0\n(base)" if x == 0 else f"{x:,}" for x in xs], fontsize=8.5, color=INK_2)
    ax.tick_params(axis="y", labelsize=8.5, colors=INK_2)
    ax.set_xlabel("Training examples (nested subsets, 2 epochs)", fontsize=9, color=INK_2)
    ax.set_ylabel("Validation accuracy (%)", fontsize=9, color=INK_2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.set_title("ScienceQA validation accuracy vs training data (Qwen2.5-VL-3B, n = 2,097)",
                 fontsize=10.5, color=INK, loc="left")
    ax.legend(loc="lower right", fontsize=8, frameon=False, labelcolor=INK)
    fig.text(0.01, 0.01, "Band: 95% Wilson CI for all questions (QLoRA). Near-duplicate = same question, choices and "
             "near-identical figure as a training item.", fontsize=7, color=INK_2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out = ROOT / "results/plots/scaling_curve.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    print(f"wrote {out} with QLoRA points {xs}" + (" + LoRA" if lora else ""))


if __name__ == "__main__":
    main()
