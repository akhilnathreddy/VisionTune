# Data-scaling study (blueprint §7.3): QLoRA, A1 targets, full validation split

Every model is scored on the **full ScienceQA validation split: 2,097 image questions**, greedy decoding, batch 8.
Training subsets are nested (seed 42), and all other settings are the pilot's: r=16, α=32, 2 epochs, LR 1e-4,
batch 1 × 8, fp16 on one Kaggle T4. The test split is untouched.

## Results so far

| Model | Train examples | Accuracy (95% CI) | Δ vs M0 (paired bootstrap 95% CI) | McNemar (M0-only / model-only) | Novel (n=1,583) | Near-dup (n=514) | Train time | Peak train VRAM | Val loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M0 base | n/a | 78.7% (76.9–80.4) | n/a | n/a | 85.9% | 56.4% | n/a | n/a | n/a |
| QLoRA D2 | 2,000 | **83.8%** (82.2–85.3) | **+5.2 pp (+3.7, +6.5)**, p < 0.001 | 60 / 168, p = 5e-13 | 89.8% (+3.9) | 65.4% (+8.9) | 93 min | 4.96 GB | 0.179 |
| QLoRA D3 | 5,000 | running | | | | | | | |
| QLoRA D4 | 6,218 | running | | | | | | | |
| LoRA D4 | 6,218 | queued | | | | | | | |
| QLoRA D1 | 500 | queued (rescore on full validation) | | | | | | | |

D2 by subject: natural science 75.9 → 78.9% (+3.0, p < 0.001); social science 82.5 → 92.0% (+9.4, p < 0.001);
language science 91.7 → 87.5% (n = 48, not significant).

## Notes

- **The gain is not only template memorisation.** Near-duplicate items gain more (+8.9 pp), but novel items also improve
  significantly (+3.9 pp, CI +2.5 to +5.3).
- **Batch-composition effect:** M0's outputs on the 500 pilot questions differ in explanation *wording* on 19/500 items when
  they are batched with different neighbours (fp16 + left padding). The predicted answer letter is identical on all 500,
  so accuracy is unaffected. Within one evaluation run, results are deterministic.
- **Kaggle inputs:** attaching an earlier notebook's output through the API did not mount anything (`/kaggle/input` was empty).
  Adapters for the final test-set evaluation will be published as a Kaggle dataset instead.
