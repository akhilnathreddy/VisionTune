# Held-out evaluation: ScienceQA test split (n = 2,017 image questions)

This is the first and only use of the test split. The models, hyperparameters, prompt and decoding were all fixed on the
validation split beforehand (`results/reports/scaling/`). Nothing was tuned on these results.
Kaggle notebooks `visiontune-05-test-base-qlora` and `visiontune-05-test-lora` (2026-11-22), one Tesla T4, fp16, greedy
decoding, max 160 new tokens, batch 8. Adapters: the full-data runs `scienceqa_qwen25vl3b_{qlora,lora}_r16_n6218_seed42`.

## Results

| Metric | M0 base | M1 LoRA (6,218) | M2 QLoRA (6,218) |
|---|---:|---:|---:|
| **Test accuracy (95% Wilson CI)** | 81.1% (79.3–82.7) | **90.8%** (89.4–92.0) | **91.5%** (90.2–92.7) |
| Δ vs M0 (paired bootstrap 95% CI) | n/a | **+9.7 pp (+8.1, +11.4)** | **+10.5 pp (+8.8, +12.2)** |
| McNemar vs M0 (M0-only / model-only) | n/a | 59 / 255, p = 3e-30 | 55 / 266, p = 2e-34 |
| Strict format compliance | 3.6% | 100% | 100% |
| Novel items (n = 1,535) | 87.4% | 94.5% (+7.1) | 94.9% (+7.5) |
| Train-template near-duplicates (n = 482) | 60.8% | 78.8% (+18.0) | 80.7% (+19.9) |
| Natural / social / language science | 80.2 / 81.7 / 95.5% | 86.9 / 96.9 / 90.9% | 87.8 / 97.1 / 95.5% |
| Mean generated tokens | 29.7 | 58.3 | 58.5 |
| Generated tokens/s (batch 8) | 51.7 | 42.3 | 33.3 |
| Time per example (batch 8) | 0.58 s | 1.37 s | 1.75 s |
| Peak inference VRAM | 7.69 GB | 7.72 GB | **3.89 GB** |

**QLoRA − LoRA:** +0.7 pp (95% CI 0.0 to +1.5), p = 0.07; McNemar 24 / 39, p = 0.08. They give the same answer on 96.7%
of questions. The language-science row has only 22 test items.

## Validation vs test

| Model | Validation (n = 2,097) | Test (n = 2,017) |
|---|---:|---:|
| M0 base | 78.7% | 81.1% |
| LoRA 6,218 | 89.3% | 90.8% |
| QLoRA 6,218 | 89.5% | 91.5% |
| Gain (QLoRA − M0) | +10.8 pp | +10.5 pp |

The test split is slightly easier for every model (+1.5 to +2.4 pp), but the **gain over the base model transfers
almost unchanged** (+10.8 → +10.5 pp). Nothing suggests the validation-based choices overfit the validation split.

## Hypotheses on the held-out test split

- **H1 (fine-tuning improves STEM VQA): supported.** About +10 pp for both methods, p < 1e-29.
- **H2 (LoRA ≈ QLoRA): supported.** +0.7 pp in QLoRA's favour, not significant. The CI excludes LoRA being better by
  any margin and QLoRA being better by more than 1.5 pp.
- **H3 (QLoRA reduces memory): supported.** 4.95 vs 8.82 GB to train (validation runs); 3.89 vs 7.72 GB to serve.
- **Caveat, again:** about two-thirds of the per-item gain on near-duplicates (+20 pp) vs novel items (+7.5 pp) suggests some
  benefit comes from familiarity with ScienceQA's question templates. Novel items still improve by 7 pp.

Files: `base_qlora/` and `lora/` hold the per-question predictions and metrics. `comparison_test.json` holds the paired statistics.
