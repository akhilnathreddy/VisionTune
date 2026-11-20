# Run 03: QLoRA pilot (M2), and M0 / M1 / M2 compared

Kaggle notebook `akhilnathreddy/visiontune-03-qlora-pilot`, version 1 (2026-11-20), one Tesla T4, fp16 compute.
Code commit `899ccb1`. The data, hyperparameters, seed and 500 validation questions are identical to run 02. The only
change is the method:
the language model is loaded in 4-bit NF4 with double quantization (252 `Linear4bit` layers), the vision tower stays in
16-bit (162 `Linear`), and the optimizer is paged AdamW 8-bit. **Validation data only.**

## Results (n = 500, same questions for every model)

| Metric | M0 base | M1 LoRA pilot | M2 QLoRA pilot |
|---|---:|---:|---:|
| Accuracy (95% Wilson CI) | 79.0% (75.2–82.3) | 79.2% (75.4–82.5) | 79.0% (75.2–82.3) |
| Paired diff vs M0 (bootstrap 95% CI, p) | n/a | +0.2 pp (−1.8, +2.2), p = 0.91 | 0.0 pp (−2.4, +2.2), p = 1.0 |
| McNemar discordant pairs vs M0 | n/a | 11 / 12 | 18 / 18 |
| Strict format compliance | 4% | 100% | 100% |
| Novel items (n = 374) | 87.2% | 88.8% | 88.5% |
| Train-template near-duplicates (n = 126) | 54.8% | 50.8% | 50.8% |
| Mean generated tokens | 31.8 | 48.9 ¹ | 48.5 |
| Generated tokens/s (batch 8) | 53.4 | 38.7 ¹ | 31.0 |
| Time per example (batch 8) | 0.60 s | 1.28 s | 1.60 s |
| **Peak inference VRAM** | 7.68 GB | 7.71 GB | **3.88 GB** |
| **Peak training VRAM** | n/a | 8.83 GB | **4.96 GB** |
| Training time (1,000 example-passes) | n/a | 25.6 min | 26.5 min |
| Final validation loss (300 examples) | n/a | 0.304 | 0.320 |
| Trainable parameters / adapter size | n/a | 7.37 M / 28.2 MB | 7.37 M / 28.2 MB |

M2 − M1: −0.2 pp (−2.0, +1.6), p = 0.90, discordant 11 / 10. The two adapters give the same answer on 94.2% of questions.

¹ M1 was not re-scored in this run, because notebook 02's output did not mount as an input (`LORA_ADAPTER: None`). Its token counts
are computed offline by re-tokenizing run 02's saved outputs. The same method on M0 and M2 reproduces the measured values
(31.8 vs 31.78 tokens; 53.3 vs 53.4 tok/s), so the estimate is reliable.

Determinism: M0 produced byte-identical outputs for all 500 questions in all three sessions (run 02 v1, v2 and run 03).

## Interpretation

- **H3 (QLoRA reduces training memory): supported.** Peak training VRAM fell 44% (8.83 → 4.96 GB), and inference VRAM
  halved (7.7 → 3.9 GB), with the same wall-clock training time on a T4.
- **H2 (LoRA ≈ QLoRA in task performance): consistent with the data at this budget.** The difference is −0.2 pp with 94%
  answer agreement. Neither adapter beats the base model yet, so this is "equal at no gain", not yet "equal at a gain".
- **Cost of 4-bit at inference:** generation is about 20% slower than LoRA (31 vs 39 tok/s), from dequantization on a T4.
  Both adapters are also slower than the base per token, because the LoRA layers are not merged.
- **H1 (fine-tuning improves accuracy): not shown at 500 examples.** Both adapters reach 100% format compliance and copy
  ScienceQA's explanation style (+50% tokens), but answer accuracy is unchanged.

## Recommendation for the scaling study

Use **QLoRA** for the D2–D4 runs (2,000 / 5,000 / 6,218 examples). It matches LoRA on every quality measure here, at about half the
memory and the same training speed. This leaves room to raise the image token cap or batch size later. Keep one LoRA run at
the largest budget, for the final M1-vs-M2 comparison the blueprint asks for.
