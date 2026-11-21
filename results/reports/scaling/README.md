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
| QLoRA D4 | 6,218 (all) | **89.5%** (88.1–90.7) | **+10.8 pp (+9.1, +12.5)**, p < 0.001 | 62 / 289, p = 4e-36 | 93.6% (+7.6) | 77.0% (+20.6) | 282 min | 4.95 GB | 0.104 |
| LoRA D4 | 6,218 | queued | | | | | | | |
| QLoRA D1 | 500 | queued (rescore on full validation) | | | | | | | |

**D4 vs D2:** +5.7 pp (+4.2, +7.1), McNemar 62 / 181, p = 1e-14. More data keeps helping between 2,000 and 6,218 examples.

D4 by subject: natural science 85.2%, social science 96.5%, language science 93.8% (n = 48). D4's validation loss
(300 examples) fell 0.199 → 0.147 → 0.125 → 0.113 → 0.105 → 0.104 at steps 250–1,500, flattening only in the last
quarter. Generation: 59.3 tokens per answer, 32.7 tokens/s (batch 8), peak inference VRAM 3.9 GB.

D2 by subject: natural science 75.9 → 78.9% (+3.0, p < 0.001); social science 82.5 → 92.0% (+9.4, p < 0.001);
language science 91.7 → 87.5% (n = 48, not significant).

## Notes

- **The gain is not only template memorisation.** Near-duplicate items gain more (D2 +8.9 pp, D4 +20.6 pp), but novel
  items also improve significantly (D2 +3.9 pp, CI +2.5 to +5.3; D4 +7.6 pp, CI +6.1 to +9.2). The gap between the two groups
  grows with data, so part of the D4 gain on templated questions is likely familiarity with those templates. Report both
  numbers rather than only the headline.
- **Batch-composition effect:** M0's outputs on the 500 pilot questions differ in explanation *wording* on 19/500 items when
  they are batched with different neighbours (fp16 + left padding). The predicted answer letter is identical on all 500,
  so accuracy is unaffected. Within one evaluation run, results are deterministic.
- **Kaggle inputs:** attaching an earlier notebook's output through the API did not mount anything (`/kaggle/input` was empty).
  Adapters for the final test-set evaluation will be published as a Kaggle dataset instead.
