# Cross-dataset evaluation: AI2D test (n = 3,088), hypothesis H5

AI2D (`lmms-lab/ai2d`, pinned revision) is a separate benchmark of grade-school science diagrams, all four-option multiple choice.
None of it was used for training, model selection or prompt design. The models are the same full-data adapters as the
ScienceQA test evaluation, with identical prompt and decoding (greedy, max 160 new tokens, batch 8).
Kaggle notebooks `visiontune-05-ai2d-base-qlora` and `visiontune-05-ai2d-lora` (2026-11-22), one Tesla T4, fp16.

## Results

| Metric | M0 base | M1 LoRA (6,218) | M2 QLoRA (6,218) |
|---|---:|---:|---:|
| **AI2D accuracy (95% Wilson CI)** | **79.3%** (77.9–80.7) | 77.9% (76.4–79.3) | 78.5% (77.0–79.9) |
| Δ vs M0 (paired bootstrap 95% CI) | n/a | **−1.5 pp (−2.4, −0.5), p = 0.003** | −0.9 pp (−1.9, +0.2), p = 0.12 |
| McNemar (M0-only / model-only correct) | n/a | 139 / 94, p = 0.004 | 152 / 125, p = 0.12 |
| Strict format compliance | 2.8% | 100% | 100% |
| Unparseable answers | 1 | 0 | 0 |
| Mean generated tokens | 31.8 | 64.4 | 62.6 |
| Generated tokens/s (batch 8) | 53.5 | 47.4 | 37.9 |
| Peak inference VRAM | 7.77 GB | 7.80 GB | 3.97 GB |

QLoRA − LoRA on AI2D: +0.6 pp (−0.4, +1.5), p = 0.22.

## Same models, two benchmarks

| | ScienceQA test (in-domain) | AI2D test (out-of-domain) |
|---|---:|---:|
| M0 base | 81.1% | 79.3% |
| QLoRA 6,218 | 91.5% (**+10.5 pp**) | 78.5% (−0.9 pp, n.s.) |
| LoRA 6,218 | 90.8% (**+9.7 pp**) | 77.9% (**−1.5 pp**, p = 0.003) |

## Interpretation

- **H5 (in-domain gains may not transfer): supported, strongly.** The ~10 pp ScienceQA gain does not carry over to AI2D at
  all. Accuracy is flat for QLoRA and slightly *worse* for LoRA. Fine-tuning taught ScienceQA-specific skills and conventions,
  not general diagram understanding.
- **Style transferred, competence did not.** Both adapters reproduce ScienceQA's output format (100% strict) and its explanation
  boilerplate on AI2D (for example "Look at the pictures. Think about what each word means…"), even when it does not fit the
  diagram. The base model writes shorter, diagram-specific explanations. A plausible cause of the small LoRA loss is answers
  being steered toward ScienceQA's templates.
- **This matches the near-duplicate analysis.** On ScienceQA, the largest gains were on questions whose template also
  appears in training (+20 pp vs +7.5 pp on novel ones). AI2D has none of those templates, and the gain disappears.
- **QLoRA vs LoRA remain indistinguishable** on AI2D too (+0.6 pp, n.s.). QLoRA's slightly smaller drop is not significant.
- **Implication:** the adapters are ScienceQA specialists. For general science-diagram VQA, the base model is as good or better.
  Mixing AI2D-style data into training, or training on answer-only targets, would be the next experiment for transfer.

Files: per-question predictions and metrics in `base_qlora/` and `lora/`; paired statistics in `comparison_ai2d.json`.
