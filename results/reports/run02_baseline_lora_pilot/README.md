# Run 02: M0 baseline vs LoRA pilot (D1, 500 examples)

Kaggle notebook `akhilnathreddy/visiontune-02-baseline-lora-pilot`, version 2 (2026-11-20), one Tesla T4, fp16.
Code commit `4944011`. **Validation data only**: 500 fixed ScienceQA validation image questions (seed 42 subset).
The test split was not used.

| Setting | Value |
|---|---|
| Model | Qwen2.5-VL-3B-Instruct @ `66285546` |
| LoRA | A1 (`q/k/v/o_proj`, 36 language layers = 144 modules), r=16, α=32, dropout 0.05 |
| Training | 500 train examples × 2 epochs, LR 1e-4 cosine, batch 1 × grad-accum 8 (126 optimizer steps), grad checkpointing |
| Decoding | greedy, max 160 new tokens, batch 8 |

## Results

| Metric | M0 base | M1 LoRA pilot |
|---|---:|---:|
| Accuracy (95% Wilson CI) | 79.0% (75.2–82.3) | 79.2% (75.4–82.5) |
| Paired difference, LoRA − base (bootstrap 95% CI) | n/a | **+0.2 pp (−1.8 to +2.2), p = 0.91** |
| McNemar discordant pairs (base-only / LoRA-only correct) | n/a | 11 / 12, p = 1.0 |
| Strict format compliance (`Answer: <letter>`) | 4% | **100%** |
| Accuracy on novel items (n = 374) | 87.2% | 88.8% |
| Accuracy on train-template near-duplicates (n = 126) | 54.8% | 50.8% |
| Mean output length (characters) | 141 | 206 |
| Time per example, batched (batch 8) | 0.63 s | 1.28 s |
| Peak training VRAM | n/a | 8.83 GB |
| Training time (1,000 example-passes) | n/a | 25.6 min (1.54 s per example) |
| Trainable parameters | n/a | 7.37 M (0.196%) |
| Adapter size (weights) | n/a | 28.2 MB |
| Validation loss (300 examples), step 100 → 126 | n/a | 0.307 → 0.304 |

Smoke test (8 examples, 15 epochs): training loss 0.73 → 0.000. The adapter, loss masking and fp16 training all work.

Determinism: re-running M0 in a new session reproduced all 500 outputs exactly.

## Interpretation

- **No measurable accuracy change at 500 examples.** The +0.2 pp is far inside the noise: 23 discordant questions split
  11/12. H1 is *not* supported at this budget. This is a pilot, so this does not reject H1.
- **The adapter did learn the output format and style.** Strict format compliance went from 4% to 100%, and the
  model now writes ScienceQA-style explanations. That is why outputs are longer and batched latency doubled: more tokens
  generated, plus the unmerged LoRA layers. Token counts are now recorded per prediction, so later runs can separate the two.
- **The loss is mostly explanation tokens.** The answer letter is about 2–3 tokens of a ~60-token target. Validation loss
  flattened by step 100, which suggests the pilot mostly learned to imitate explanations rather than pick better answers.
- **Near-duplicate split:** novel +1.6 pp, near-dup −4.0 pp. Both are within noise (n = 374 and 126). No sign yet of template
  memorisation helping.

## Next

1. QLoRA pilot with identical data and settings (M2 vs M1 vs M0), recording memory, time and tokens/s.
2. Data scaling D2–D4 (2,000 / 5,000 / 6,218). At ~1.5 s per example-pass on a T4, 5,000 × 2 epochs ≈ 4.3 h per run.
3. If gains stay flat: an answer-focused ablation (shorter or no explanation target), and A2 targets (attention + MLP).

Files: `comparison_base_vs_lora_pilot.json`, per-example predictions and metrics in `predictions/`, run configs,
sample IDs and summaries in `runs/`, and v1's baseline-only output in `v1_baseline_only/` (identical to v2).
Adapter weights are kept on Kaggle (notebook output), not in git.
