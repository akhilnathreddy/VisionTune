# VisionTune AI
## Complete VLM Fine-Tuning Project Blueprint

**Project:** Parameter-Efficient Fine-Tuning of an Open-Source Vision-Language Model for Scientific Image Understanding  
**Application:** STEM visual question answering (VQA)  
**Priority:** Balanced — research quality + engineering portfolio  
**Primary compute environment:** Kaggle Notebooks  
**Primary model:** Qwen2.5-VL-3B-Instruct  
**Primary dataset:** ScienceQA  
**Secondary evaluation benchmark:** AI2D
**Blueprint version:** 1.1 (2026-11-20). See the revision log below.

---

## Revision log (v1.0 → v1.1)

These changes came from inspecting the real model, datasets and libraries while building Phase 1.
Each one is reflected in the section it affects.

| # | Section | Change | Why |
|---|---|---|---|
| 1 | 4.1, 4.4, 7.3 | ScienceQA is restricted to questions **with an image**: 6,218 train / 2,097 validation / 2,017 test. The scaling budgets become 500 / 2,000 / 5,000 / **all 6,218**. | 51% of ScienceQA is text-only, which is irrelevant to a VLM study. The 8,000-example budget is not reachable. |
| 2 | 4.5 | Exact duplicates are excluded. Near-duplicates are **flagged, not removed**. | 1,093 train items share a question, choice set and near-identical figure with a test item, but **261 of those have a different correct answer** (templated magnet and particle diagrams). Dropping them would discard legitimate training examples. Instead, 482 test items carry `near_dup_in_train`, and results are reported for both groups. |
| 3 | 4.6 | Record IDs are `sqa-{split}-{index}`. | The Hugging Face mirror has no official problem IDs. |
| 4 | 4.7 | Target is `Answer: <letter>. <option text>` plus `Explanation: <ScienceQA solution>` (≤600 chars). | Letter answers make extraction unambiguous. 540 train items have no solution, so they are trained on the answer only. |
| 5 | 7.2, 7.7 | For multiple-choice, "exact match" is identical to accuracy. It is replaced by **format compliance** (strict / lenient / text / unparseable). | This avoids reporting the same number twice, and separates "can't follow the format" from "wrong answer". |
| 6 | 7.5 | ScienceQA `subject` is *natural / social / language science*. Physics, chemistry and biology are `topic` values. Results are broken down by subject, topic and grade. | Matches the real schema. |
| 7 | 5.3 | LoRA targets are resolved to **explicit module paths**, not bare names. | Qwen2.5-VL reuses `gate_proj/up_proj/down_proj` in the vision tower, so the A2 preset would silently adapt the vision encoder too. |
| 8 | 5.2 | QLoRA keeps the vision tower and `lm_head` in 16-bit. Skip patterns are `[model.visual, visual, lm_head]`. | transformers v5 matches skip patterns as path *prefixes*, so a bare `visual` would quantize the vision tower. A unit test guards this. |
| 9 | 5, 16 | Training uses the Transformers `Trainer` with a custom collator. TRL is optional. | Loss masking is explicit and unit-tested, and there are fewer moving APIs. TRL's VLM interface changes often. |
| 10 | 6.3 | Kaggle T4/P100 have no bf16, so `dtype: auto` → fp16 there. Attention is `sdpa` (no flash-attn on T4). Images are capped at 128–512 visual tokens, enforced by a startup check. `vision_fp32` is a fallback if fp16 overflows. | Hardware facts. One transformers version silently ignored a resize option, which left a 2000×1500 image at 3,834 tokens. |
| 11 | 3 | Qwen2.5-VL-3B-Instruct is under the **Qwen Research License** (non-commercial). The 7B model is Apache-2.0. | Affects publishing adapters and the demo. |
| 12 | 4.2 | AI2D uses `lmms-lab/ai2d` (test, 3,088 four-option questions). Model and dataset revisions are pinned in `configs/base.yaml`. | Reproducibility. |
| 13 | 11 | Code lives in an installable package, `src/visiontune/`. | Notebooks stay thin, and the same code runs on Kaggle, in tests and in the backend. |
| 14 | 7.4 | Latency is recorded both amortized over a batch and as single-request latency. | These measure different things. |

---

## 1. Project overview

VisionTune AI is an end-to-end project to adapt an open-source vision-language model (VLM) for scientific image understanding using parameter-efficient fine-tuning.

A user provides a scientific image or diagram and asks a question. The system produces an answer and a concise explanation. The project compares the original pretrained model with LoRA- and QLoRA-fine-tuned variants.

The project is not intended to train a VLM from scratch. It studies whether targeted adaptation of a pretrained model can improve scientific visual question answering while reducing training costs compared with full fine-tuning.

### Core research question

Can LoRA and QLoRA fine-tuning improve the scientific visual question-answering performance of an open-source VLM while maintaining reasonable computational efficiency?

### Objectives

1. Understand the selected VLM's vision encoder, multimodal projector and language model.
2. Prepare a reproducible scientific image-question-answer dataset.
3. Evaluate the pretrained model before fine-tuning.
4. Fine-tune using LoRA and QLoRA.
5. Compare task performance, memory use, training time and inference latency.
6. Evaluate generalization to a separate scientific diagram benchmark.
7. Build and deploy an interactive application.
8. Publish reproducible code, adapter weights, model cards and results.

### Research hypotheses

- **H1:** Fine-tuning improves STEM VQA performance over the pretrained baseline.
- **H2:** LoRA and QLoRA can achieve similar task performance, subject to experimental verification.
- **H3:** QLoRA reduces training memory requirements.
- **H4:** Increasing training data improves performance up to a point, after which gains may diminish.
- **H5:** Improvements on the training domain may not transfer equally to unseen scientific diagram domains.

These are hypotheses to test, not results to assume.

---

## 2. Scope and boundaries

### Included in the initial project

- Scientific image understanding.
- Visual question answering.
- Scientific diagram interpretation.
- LoRA fine-tuning.
- QLoRA fine-tuning.
- Baseline and fine-tuned model comparison.
- Cross-dataset evaluation.
- Interactive web demonstration.
- Public GitHub and Hugging Face artifacts.

### Out of scope for the initial version

- Training a VLM from scratch.
- Image generation.
- Video generation or video understanding.
- Pixel-level image segmentation.
- A dedicated object-detection system.
- Medical diagnosis or clinical decision support.

These are distinct tasks that require different annotations, model outputs and evaluation protocols. They should not be added to the first version.

---

## 3. Model selection

### Primary model

**Qwen2.5-VL-3B-Instruct**

Model repository: https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct
Pinned revision: `66285546d2b821cf421d4f5eb2576359d3770cd3`

**License:** the 3B model is released under the *Qwen Research License* (research / non-commercial use).
Adapters trained on it inherit those terms. If commercial use ever matters, Qwen2.5-VL-7B-Instruct is Apache-2.0.

Why this model:

- Supports image and text inputs.
- Can perform visual question answering and image understanding.
- Is smaller than many larger VLM alternatives.
- Provides a practical starting point for parameter-efficient fine-tuning.
- Has an established Hugging Face model implementation.

### Model components to understand

1. **Vision encoder:** Extracts visual representations from the input image.
2. **Multimodal projector / connector:** Connects visual representations to the language model's representation space.
3. **Language model:** Processes the question and visual context to generate an answer.
4. **Processor:** Handles image preprocessing and text tokenization.

### Fine-tuning policy

Keep the pretrained architecture intact. Train parameter-efficient adapters rather than updating every model parameter in the initial experiments.

Begin with selected language-model attention projections. Then run separate ablations to investigate whether adapting the multimodal projector or selected vision components improves performance.

Do not assume every module name or target layer is identical across model implementations. Inspect the actual model modules and verify that gradients reach the intended trainable parameters.

**Verified module layout (transformers ≥ 4.57):**

| Component | Module path | Linear layers |
|---|---|---|
| Vision encoder (32 blocks) | `model.visual.blocks.N` | `attn.qkv`, `attn.proj`, `mlp.gate_proj/up_proj/down_proj` |
| Projector ("merger") | `model.visual.merger.mlp.{0,2}` | 2-layer MLP, 1280·4 → 2048 |
| Language model (36 layers) | `model.language_model.layers.N` | `self_attn.q/k/v/o_proj`, `mlp.gate_proj/up_proj/down_proj` |
| Output head | `lm_head` | never adapted |

---

## 4. Dataset strategy

Use one primary training dataset and one separate evaluation dataset.

### 4.1 Primary dataset: ScienceQA

Official website: https://scienceqa.github.io/  
GitHub: https://github.com/lupantech/ScienceQA

ScienceQA contains science questions across subjects such as physics, chemistry and biology. It includes multiple-choice questions, explanations and multimodal examples, including image context for a subset of questions.

Use the official train and validation splits for model development. Use the official test split only for final evaluation.

**Measured (HF mirror `derek-thomas/ScienceQA`, pinned revision):**

| Split | Raw | With image (kept) | By subject (natural / social / language) |
|---|---:|---:|---|
| train | 12,726 | 6,218 | 3,836 / 2,246 / 136 |
| validation | 4,241 | 2,097 | 1,287 / 762 / 48 |
| test | 4,241 | 2,017 | 1,209 / 764 / 44 |

Answer options: 2–5 per question. 63% of image questions include a text context (`hint`). Full counts and exclusion reasons: `results/reports/data/prepare_report.json`.

**Important:** ScienceQA is distributed under a non-commercial Creative Commons license (CC BY-NC-SA). The Hugging Face mirror's card says CC BY-SA 4.0. Follow the more restrictive official license. Review the dataset's current license and attribution requirements before redistributing data or using the project commercially.

### 4.2 Secondary benchmark: AI2D

Dataset information: https://allenai.org/data/diagrams

AI2D contains educational science diagrams and associated annotations/questions. Use it as an external evaluation benchmark to test whether the model generalizes to scientific diagrams not used for fine-tuning.

Do not train on AI2D in the primary experiments. Confirm the license and exact split details for the version downloaded before use.

**Version used:** `lmms-lab/ai2d` test split: 3,088 questions, all with 4 options and an image. The dataset card does not state a license. The original AI2D release from AI2 should be checked before redistributing anything derived from it.

### 4.3 Dataset roles

| Dataset split | Purpose |
|---|---|
| ScienceQA train | Fine-tuning |
| ScienceQA validation | Hyperparameter and checkpoint selection |
| ScienceQA test | Final in-domain evaluation |
| AI2D evaluation split | Cross-dataset generalization |

### 4.4 Initial sample budgets

| Experiment | Training examples |
|---|---:|
| Pipeline pilot | 500 |
| Small experiment | 2,000 |
| Main experiment | 5,000 |
| Extended experiment | all eligible (6,218) |

Subsets are **nested** and deterministic (`visiontune.data.sampling`): the 500-example set is inside the 2,000-example set, and so on. The scaling curve then reflects quantity, not which examples were drawn. Each run writes its exact sample IDs to `sample_ids.json`.

### 4.5 Preprocessing pipeline

1. Download the dataset using its official instructions.
2. Inspect the dataset schema and image references.
3. Validate that referenced images exist and can be decoded.
4. Remove malformed examples and document exclusions.
5. Detect exact duplicates and, where feasible, near-duplicates.
   - *Exact* = same question, choices, context and decoded pixels → excluded (none exist across splits after filtering).
   - *Near* = same question and choice set, and image dHash within 6/64 bits → **flagged** on validation/test (`near_dup_in_train`), not removed. 482 of 2,017 test items (24%) are flagged. 119 of these have a different answer from their train counterpart. Report accuracy on flagged and novel items separately (`by_near_dup_in_train` in every metrics file). Use `data.near_dup_policy=exclude` for a stricter run.
6. Preserve the official train/validation/test split.
7. Normalize answer labels and choice formatting.
8. Convert examples to the model's supported multimodal conversational format.
9. Preserve subject, topic, question ID and source metadata.
10. Verify that no test examples or derived variants enter training.

Do not resize images arbitrarily before the model processor. Use the model's supported image preprocessing and resolution controls.

### 4.6 Example training record

A logical record may look like this:

```json
{
  "question_id": "sample_001",
  "image": "images/sample_001.png",
  "question": "What is the function of mitochondria?",
  "choices": [
    "Protein synthesis",
    "Energy production",
    "Photosynthesis",
    "Cell division"
  ],
  "answer": "Energy production",
  "explanation": "Mitochondria produce ATP through cellular respiration.",
  "subject": "biology"
}
```

The actual serialized training example must follow the selected Qwen2.5-VL processor and chat-template requirements. The JSON above is an illustrative intermediate schema, not necessarily the final trainer input format.

### 4.7 Target response format

Train the model to return a concise answer and explanation, for example:

```text
Answer: B. Energy production
Explanation: Mitochondria generate most of the cell's usable ATP through cellular respiration.
```

The prompt lists options as `A. …`, `B. …` and asks for exactly this format. Extraction (`visiontune.evaluation.extraction`) tries, in order: strict `Answer: <letter>` → a lenient letter pattern → a unique option-text match. It records which method succeeded, so format compliance is reported separately from accuracy.

For multiple-choice tasks, preserve the choices in the prompt and use a consistent answer-extraction procedure during evaluation.

---

## 5. Training strategy

Run two principal experiments using the same base model, training examples, validation examples and evaluation protocol.

### 5.1 Experiment A: LoRA

LoRA adds trainable low-rank matrices to selected pretrained layers while keeping the original weights frozen.

Conceptually:

\[
W' = W + \frac{\alpha}{r}BA
\]

Where:

- \(W\): original pretrained weight matrix.
- \(A\) and \(B\): trainable low-rank matrices.
- \(r\): rank.
- \(\alpha\): scaling factor.

Initial configuration:

| Parameter | Initial value |
|---|---:|
| Rank | 16 |
| Alpha | 32 |
| Dropout | 0.05 |
| Learning rate | 1e-4 |
| Epochs | 2–3 |
| Optimizer | AdamW |
| Precision | BF16 where supported; otherwise supported FP16 |
| Per-device batch size | 1 initially |
| Gradient accumulation | 8 initially |

These values are starting points. Tune them using validation performance and available GPU memory.

### 5.2 Experiment B: QLoRA

QLoRA loads the frozen base model in 4-bit quantized form and trains LoRA adapters.

Initial configuration:

| Parameter | Initial value |
|---|---:|
| Quantization | 4-bit NF4 |
| Double quantization | Enabled |
| Compute dtype | BF16 where supported |
| Rank | 16 |
| Alpha | 32 |
| Dropout | 0.05 |
| Learning rate | 1e-4 |
| Epochs | 2–3 |
| Optimizer | Compatible 8-bit/paged optimizer, if supported |
| Per-device batch size | 1 initially |
| Gradient accumulation | 8 initially |

**Vision tower precision:** only the language model is quantized. The vision encoder and `lm_head` stay in 16-bit (`quantization.skip_modules`). The notebook confirms this with `linear_layer_summary()`, and `tests/test_quant_config.py` guards the skip patterns against transformers' matching rules.

**Compatibility warning:** Quantized multimodal training support depends on the exact Transformers, PEFT, TRL and BitsAndBytes versions and the model implementation. Verify that the selected Qwen2.5-VL modules can be loaded, quantized and trained correctly before committing to a full run. If a component is unsupported, document the limitation and adjust the experiment rather than silently changing the method.

### 5.3 Target modules and ablations

Start with the language model's attention projections where supported:

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`

Inspect the model to confirm actual module names.

Run controlled ablations. Each preset in `src/visiontune/models/targets.py` resolves to explicit module paths:

| Experiment | Preset (`lora.targets`) | Trainable components |
|---|---|---|
| A1 | `lm_attn` | Language-model attention projections (q/k/v/o) |
| A2 | `lm_attn_mlp` | A1 + language-model MLP (gate/up/down), **not** the vision MLPs |
| A3 | `lm_attn_mlp_merger` | A2 + multimodal projector (merger) |
| A4 | `lm_attn_mlp_merger_vision_attn` | A3 + vision-encoder attention (qkv/proj) |

Keep A1 as the initial baseline fine-tuning configuration. Do not unfreeze the entire vision encoder at the beginning.

### 5.4 Supervised fine-tuning objective

Use supervised fine-tuning (SFT). The model receives an image and question, optionally including answer choices, and learns to generate the expected answer and explanation.

\[
\mathcal{L}_{SFT}
=
-\sum_{t=1}^{T}\log P(y_t \mid y_{<t}, x, I)
\]

Where:

- \(I\) is the image.
- \(x\) is the question and optional choices.
- \(y\) is the target assistant response.

Ensure that the training collator and chat template mask user/system tokens appropriately so the loss is applied to the intended assistant response tokens.

Implemented in `visiontune.training.collator.SFTCollator`. Labels cover only the assistant reply and its closing `<|im_end|>`, so the model learns to stop. System text, user text, vision tokens and padding are set to −100. `tests/test_model_pipeline.py` checks this by decoding the kept labels.

### 5.5 Training pipeline

```text
ScienceQA
   |
   v
Validate images and annotations
   |
   v
Preserve official data splits
   |
   v
Format multimodal conversations
   |
   v
Load Qwen2.5-VL processor and base model
   |
   +-------------------------+
   |                         |
   v                         v
Attach LoRA adapters     4-bit quantize + LoRA
   |                         |
   v                         v
LoRA SFT training        QLoRA SFT training
   |                         |
   v                         v
Save LoRA adapter        Save QLoRA adapter
   |                         |
   +------------+------------+
                |
                v
       Evaluate all variants
                |
                v
       Compare results and costs
```

### 5.6 Training safeguards

- Run a tiny end-to-end training test before a full run.
- Confirm that trainable parameter counts are non-zero and expected.
- Confirm that loss decreases on a small overfit test.
- Monitor validation loss and task metrics.
- Save checkpoints to persistent Kaggle output or external storage.
- Save tokenizer/processor and adapter configuration as required.
- Record random seeds, package versions and hardware details.
- Do not use the test set for checkpoint selection.
- Keep a copy of the training configuration for every run.

---

## 6. Kaggle-first compute strategy

**Primary environment: Kaggle Notebooks**

Kaggle is the preferred environment for this project. Use it for dataset exploration, baseline inference, pilot training and, subject to available accelerator quotas and session limits, the main fine-tuning experiments.

Kaggle: https://www.kaggle.com/code

### 6.1 Kaggle workflow

1. Create a Kaggle account and enable notebook access.
2. Create a private working notebook for initial setup.
3. Enable a GPU accelerator in notebook settings when available.
4. Attach the ScienceQA dataset or download it using the official repository instructions.
5. Install and pin compatible Python packages.
6. Verify GPU visibility and CUDA/PyTorch compatibility.
7. Load the processor and pretrained model.
8. Run a single-image inference test.
9. Run baseline evaluation on a small subset.
10. Run a 500-example LoRA pilot.
11. Test QLoRA compatibility and run a small pilot.
12. Scale up only after memory use and training behavior are understood.
13. Save outputs and checkpoints to Kaggle's persistent output mechanism or external storage before a session ends.

### 6.2 Suggested Kaggle notebook organization

| Notebook | Responsibility |
|---|---|
| `01_environment_setup.ipynb` | Verify accelerator, install packages and load model |
| `02_dataset_exploration.ipynb` | Inspect ScienceQA and image examples |
| `03_preprocessing.ipynb` | Validate, clean and format data |
| `04_baseline_evaluation.ipynb` | Evaluate original model |
| `05_lora_training.ipynb` | Train LoRA adapter |
| `06_qlora_training.ipynb` | Train QLoRA adapter |
| `07_benchmarking.ipynb` | Compare model variants |
| `08_results_analysis.ipynb` | Generate tables and plots |
| `09_demo_validation.ipynb` | Test exported adapters and inference |

### 6.3 GPU planning

Use the accelerator actually available in the Kaggle session. Availability, VRAM, quotas and session limits can change.

Kaggle facts that shape the configs:

- **T4 (16 GB, compute 7.5) and P100 have no bf16.** `model.dtype: auto` picks fp16 there and bf16 on Ampere+. If fp16 gives non-finite logits (notebook 01 checks), set `model.vision_fp32: true`.
- **No FlashAttention-2 on T4.** Use `attn_implementation: sdpa`.
- **"T4 x2" means two 16 GB GPUs, not one 32 GB GPU.** The pipeline uses one GPU. Multi-GPU (DDP) is a later optimisation.
- **Visual tokens are the main memory lever.** Qwen2.5-VL makes one token per 28×28 pixel block. `processor.min_pixels/max_pixels` caps every image at 128–512 tokens, and `load_processor()` refuses to continue if the cap is not applied.

A 16–24 GB GPU is a useful planning target for initial constrained experiments, but a particular model configuration is not guaranteed to fit. Actual memory use depends on image resolution, visual token count, sequence length, optimizer state, batch size, precision and implementation.

If the selected Kaggle GPU cannot support a configuration:

- Reduce per-device batch size to 1.
- Increase gradient accumulation to preserve effective batch size.
- Enable gradient checkpointing if supported.
- Reduce image pixel limits using the model's supported processor options.
- Shorten sequence length where appropriate.
- Freeze the vision encoder initially.
- Use QLoRA only after validating compatibility.
- Reduce the pilot dataset size to debug the pipeline.

### 6.4 Kaggle session and storage management

Kaggle notebook sessions are not permanent training machines.

- Save checkpoints periodically.
- Save adapters, configs, logs and metrics separately.
- Keep a manifest of completed steps and last saved checkpoint.
- Use Kaggle notebook outputs or attached datasets for reproducible artifacts.
- Use an external persistent location for important checkpoints if needed.
- Never assume files in a temporary working directory will survive a session reset.
- Avoid storing access tokens or secrets in notebook output or GitHub.

### 6.5 Fallback compute

If Kaggle's accelerator quota or session duration becomes a bottleneck, use a rented cloud GPU only for the specific long-running training job. Keep Kaggle as the primary development and experiment environment.

---

## 7. Evaluation and benchmarking

Evaluation is central to the project. The goal is to establish whether fine-tuning helps, where it helps and what computational cost it introduces.

### 7.1 Main model comparison

| ID | Model |
|---|---|
| M0 | Original Qwen2.5-VL-3B-Instruct |
| M1 | Qwen2.5-VL-3B + LoRA |
| M2 | Qwen2.5-VL-3B + QLoRA |

Use identical test examples, prompts, answer extraction and decoding settings.

### 7.2 Experiment 1: Baseline versus fine-tuned

Compare M0, M1 and M2 on the ScienceQA test split.

Measure:

- Multiple-choice accuracy (with 95% Wilson CI).
- Format compliance (share of answers parsed strictly / leniently / by option text / unparseable).
- Accuracy on test items flagged `near_dup_in_train` vs novel items.
- Explanation quality.
- Inference latency.
- Peak inference GPU memory.

### 7.3 Experiment 2: Training data scaling

Compare training budgets:

| Run | Training examples |
|---|---:|
| D1 | 500 |
| D2 | 2,000 |
| D3 | 5,000 |
| D4 | 8,000 or all eligible examples |

Use the same validation and test sets. Investigate performance gains, saturation and overfitting.

### 7.4 Experiment 3: LoRA versus QLoRA

Compare:

- Test accuracy.
- Exact match.
- Training time.
- Peak training VRAM.
- Trainable parameter count.
- Adapter size.
- Inference latency.

Distinguish training memory savings from inference memory savings. QLoRA's 4-bit training setup does not automatically mean every deployment will use the same quantization configuration.

### 7.5 Experiment 4: Subject-wise performance

Break down results by available ScienceQA metadata. In the real schema, `subject` has three values (natural / social / language science). Physics, chemistry, biology, geography and earth science are `topic` values. Report:

- by `subject` (3 groups),
- by `topic` (e.g. geography 600, physics 425, biology 403, chemistry 94 test items),
- by `grade` (grade 1–12).

Report sample counts alongside scores. Do not overinterpret categories with very few examples.

### 7.6 Experiment 5: Cross-dataset generalization

Evaluate the final selected variants on AI2D without training on AI2D.

Investigate:

- Diagram interpretation.
- Reading labels.
- Understanding arrows and connections.
- Identifying relationships between diagram components.
- Answering questions from unseen diagrams.

Check that the AI2D evaluation format and answer annotations are compatible with the chosen evaluation procedure.

### 7.7 Metrics

| Metric | Purpose |
|---|---|
| Accuracy | Correct answers divided by total evaluated answers |
| Format compliance | How the answer was extracted (strict / lenient / text / none). For MC this replaces exact match, which equals accuracy. |
| F1 | Token-level overlap for suitable text answers |
| ROUGE-L | Text overlap based on longest common subsequence |
| BERTScore | Semantic similarity to reference text |
| Training time | Time required to complete training |
| Peak VRAM | Maximum GPU memory consumed |
| Inference latency | Single-request latency (batch 1) and batch-amortized time per example, both recorded |

For multiple-choice questions, accuracy and carefully normalized answer extraction should be the primary metrics. For free-text explanations, combine automatic metrics with human review of factual correctness.

ROUGE-L and BERTScore do not establish scientific correctness by themselves.

### 7.8 Statistical reliability

- Use the same test examples for every model.
- Report the number of evaluated examples.
- Report confidence intervals for accuracy.
- Use paired bootstrap comparisons or an appropriate paired statistical test.
- Fix prompts and decoding settings.
- Repeat training with multiple random seeds where compute permits.
- Preserve per-example predictions for error analysis.

### 7.9 Results template

Fill this table only with measured results. Measured on **ScienceQA validation** (n = 2,097 image questions), one Kaggle T4,
fp16. LoRA and QLoRA are the full-data runs (6,218 examples, A1 targets, r=16, 2 epochs). The **test** row was measured once,
after every choice had been fixed on validation (`results/reports/test/README.md`).
Details: `results/reports/scaling/README.md`.

| Metric | Base | LoRA | QLoRA |
|---|---:|---:|---:|
| ScienceQA **validation** accuracy (95% CI) | 78.7% (76.9–80.4) | 89.3% (87.9–90.5) | 89.5% (88.1–90.7) |
| ScienceQA **test** accuracy (95% CI, n = 2,017) | 81.1% (79.3–82.7) | 90.8% (89.4–92.0) | 91.5% (90.2–92.7) |
| Format compliance (strict) | 4.2% | 100% | 100% |
| Accuracy: novel / near-dup items | 85.9% / 56.4% | 93.1% / 77.4% | 93.6% / 77.0% |
| Explanation evaluation | TBD | TBD | TBD |
| Training time | N/A | 288 min | 282 min |
| Peak training VRAM | N/A | 8.82 GB | 4.95 GB |
| Peak inference VRAM | 7.69 GB | 7.72 GB | 3.90 GB |
| Inference: s/example (batch 8) · tokens/s | 0.60 s · 52.1 | 1.41 s · 41.6 | 1.82 s · 32.7 |
| Adapter size | N/A | 28.2 MB | 28.2 MB |

On the held-out test split: H1 +10.5 / +9.7 pp (p < 1e-29), H2 QLoRA − LoRA = +0.7 pp (p = 0.07, n.s.).
Hypotheses on validation: **H1 supported** (+10.6 / +10.8 pp, p < 0.001). **H2 supported** (QLoRA − LoRA = +0.2 pp,
95% CI −0.6 to +1.1). **H3 supported** (−44% training VRAM). **H4 supported** (+4.9 pp from 2,000 → 5,000 examples, +0.8 pp
n.s. from 5,000 → 6,218). **H5** is pending the AI2D evaluation.

---

## 8. System architecture

The application has four layers:

1. Data engineering.
2. Model training and evaluation.
3. Inference API.
4. Frontend.

```text
             SCIENCEQA + AI2D
                    |
                    v
          DATA VALIDATION / CLEANING
                    |
                    v
          MULTIMODAL PREPROCESSING
                    |
                    v
          QWEN2.5-VL-3B BASE MODEL
                    |
              +-----+-----+
              |           |
              v           v
            LoRA        QLoRA
              |           |
              v           v
         LORA ADAPTER  QLORA ADAPTER
              |           |
              +-----+-----+
                    |
                    v
          BENCHMARKING / EVALUATION
                    |
                    v
             MODEL ARTIFACTS
                    |
                    v
              FASTAPI BACKEND
                    |
                    v
           STREAMLIT FRONTEND
                    |
                    v
            DEPLOYED APPLICATION
```

### Inference flow

1. User uploads a scientific image.
2. User enters a question.
3. Frontend sends the image, question and selected model variant to the backend.
4. Backend preprocesses inputs using the model's processor.
5. The selected base model and adapter generate an answer.
6. Backend returns the answer, explanation, model identifier and timing information.
7. Frontend displays the response.

Load the base model once per worker. Do not reload the full model for every request. Use supported adapter switching or separate workers depending on the deployment hardware and library support.

---

## 9. Frontend application

**Recommended first frontend: Streamlit**

Streamlit allows the project to focus on ML experimentation while still providing an interactive demonstration.

### Application pages

| Page | Functionality |
|---|---|
| Home | Project overview and model information |
| Playground | Upload an image and ask a STEM question |
| Model comparison | Compare base, LoRA and QLoRA outputs |
| Evaluation | Display accuracy, F1 and latency |
| Training dashboard | Show loss curves and training configuration |
| About | Explain datasets, architecture and limitations |

### Core features

- Image upload (PNG/JPG).
- Question input.
- Base/LoRA/QLoRA model selection.
- Answer and explanation display.
- Side-by-side response comparison.
- Inference time and model information.
- Sample scientific images.
- Session conversation history.
- Downloadable response.
- User feedback for incorrect answers.

### Example interface flow

```text
VisionTune AI
|
+-- Upload scientific image
|
+-- Enter question
|
+-- Choose model
|     +-- Base
|     +-- LoRA
|     +-- QLoRA
|
+-- Generate answer
|
+-- Display answer and explanation
|
+-- Compare model responses
|
+-- Display inference metrics
```

Clearly label the application as an experimental research demonstration. Do not present generated answers as scientifically verified facts.

---

## 10. Backend and deployment

### Backend: FastAPI

Suggested endpoints:

| Endpoint | Method | Function |
|---|---|---|
| `/health` | GET | Service health |
| `/models` | GET | Available model variants |
| `/predict` | POST | Single image question answering |
| `/compare` | POST | Compare base and adapted models |
| `/metrics` | GET | Evaluation results |
| `/model-info` | GET | Model and adapter metadata |

### Recommended deployment arrangement

| Component | Recommended service |
|---|---|
| Frontend | Hugging Face Spaces |
| Inference backend | GPU-backed endpoint |
| Model adapters | Hugging Face Hub |
| Source code | GitHub |
| Training | Kaggle Notebooks |
| Experiment tracking | Weights & Biases |

Potential GPU inference providers:

- RunPod: https://www.runpod.io/
- Modal: https://modal.com/

Hugging Face Spaces: https://huggingface.co/spaces

Keep the frontend separate from the GPU inference service if hosting the full VLM in the frontend environment is too expensive or resource-intensive.

### Deployment requirements

- Load model weights once when a worker starts.
- Add request-size and image-dimension limits.
- Validate uploaded file types.
- Add request timeouts and error handling.
- Avoid exposing API keys.
- Restrict access or add rate limits where appropriate.
- Monitor GPU memory and request latency.
- Document the model's expected hardware requirements.

---

## 11. GitHub repository structure

Repository: **VisionTune**. The layout below is the original plan. The implemented layout keeps the same
responsibilities but puts library code in an installable package:

```text
src/visiontune/
  config.py            YAML configs with `extends:` + dotted CLI overrides
  env.py               environment / hardware report
  prompts.py           the single prompt + target formatter
  data/                records.py, prepare.py (validate, dedup, leakage), sampling.py, loaders.py
  models/              loading.py (LoRA / QLoRA / inference), targets.py (ablation presets)
  training/            collator.py (loss masking), train.py (CLI)
  evaluation/          extraction.py, metrics.py (CI, paired bootstrap, McNemar)
  inference/           predict.py (resumable batched evaluation CLI)
```

Original plan:

```text
VisionTune-AI/
|
+-- configs/
|   +-- model.yaml
|   +-- lora.yaml
|   +-- qlora.yaml
|   +-- training.yaml
|
+-- data/
|   +-- README.md
|   +-- processed/
|
+-- notebooks/
|   +-- 01_environment_setup.ipynb
|   +-- 02_dataset_exploration.ipynb
|   +-- 03_preprocessing.ipynb
|   +-- 04_baseline_evaluation.ipynb
|   +-- 05_lora_training.ipynb
|   +-- 06_qlora_training.ipynb
|   +-- 07_benchmarking.ipynb
|   +-- 08_results_analysis.ipynb
|   +-- 09_demo_validation.ipynb
|
+-- src/
|   +-- data/
|   |   +-- download.py
|   |   +-- preprocess.py
|   |   +-- validate.py
|   |
|   +-- models/
|   |   +-- load_model.py
|   |   +-- adapters.py
|   |
|   +-- training/
|   |   +-- trainer.py
|   |   +-- train_lora.py
|   |   +-- train_qlora.py
|   |
|   +-- evaluation/
|   |   +-- accuracy.py
|   |   +-- answer_metrics.py
|   |   +-- benchmark.py
|   |   +-- statistical_tests.py
|   |
|   +-- inference/
|       +-- predict.py
|       +-- compare.py
|
+-- backend/
|   +-- main.py
|   +-- routes/
|   +-- schemas/
|
+-- frontend/
|   +-- app.py
|   +-- components/
|
+-- tests/
|
+-- results/
|   +-- metrics/
|   +-- plots/
|   +-- reports/
|
+-- docs/
|   +-- architecture.md
|   +-- methodology.md
|   +-- experiments.md
|
+-- requirements.txt
+-- .gitignore
+-- LICENSE
+-- README.md
```

### What belongs in GitHub

- Source code.
- Kaggle notebook exports.
- Configuration files.
- Tests.
- Documentation.
- Small sample assets that can legally be redistributed.
- Evaluation plots and results.
- Model cards and demo screenshots.

Do not commit large raw datasets, model checkpoints, Kaggle credentials, Hugging Face tokens or other secrets.

### Hugging Face repositories

Publish separate adapter repositories, for example:

```text
your-username/
|
+-- visiontune-qwen25vl-lora
|
+-- visiontune-qwen25vl-qlora
|
+-- visiontune-stem-demo
```

Each adapter model card should include:

- Base model reference.
- Adapter configuration.
- Training dataset and license.
- Training method.
- Training configuration.
- Evaluation results.
- Intended use and limitations.
- Reproduction instructions.

Publish adapter weights rather than duplicating the entire pretrained base model unless there is a clear reason to merge or export it.

---

## 12. Experiment tracking and reproducibility

Use Weights & Biases to record experiments.

Website: https://wandb.ai/

Track at least:

- Experiment name.
- Model and model revision.
- Dataset version and sample IDs.
- Random seed.
- LoRA rank and alpha.
- Target modules.
- Learning rate.
- Batch size and gradient accumulation.
- Epoch and global step.
- Training and validation loss.
- Validation accuracy.
- Training duration.
- Peak GPU memory.
- Package versions.
- Hardware type.

Use a unique run name, for example:

```text
scienceqa_qwen25vl3b_lora_r16_seed42
scienceqa_qwen25vl3b_qlora_r16_seed42
```

Save a machine-readable configuration with every experiment. Record the Kaggle accelerator type and the date of the run.

---

## 13. Milestones and completion criteria

| Milestone | Work | Completion criteria |
|---|---|---|
| M1: Setup | Kaggle environment and model | Model loads and answers a sample image question |
| M2: Dataset | ScienceQA processing | Validated, reproducible dataset pipeline |
| M3: Baseline | Original model evaluation | Baseline predictions and metrics saved |
| M4: LoRA | Fine-tuning | Adapter trained and reloadable |
| M5: QLoRA | Quantized fine-tuning | Compatible QLoRA run completed |
| M6: Benchmarking | Compare experiments | Evaluation tables and error analysis complete |
| M7: Application | API and frontend | Working image-Q&A interface |
| M8: Deployment | Public hosting | Accessible demo URL |
| M9: Documentation | GitHub and model cards | Reproduction-ready public repository |

### Recommended implementation order

1. Set up Kaggle and verify the GPU.
2. Load the base model and processor.
3. Run a single-image inference test.
4. Download and inspect ScienceQA.
5. Build preprocessing and validation.
6. Run baseline evaluation.
7. Train a small LoRA pilot.
8. Verify QLoRA compatibility and run a pilot.
9. Scale the training experiments.
10. Evaluate and analyze results.
11. Build the frontend and backend.
12. Deploy and document the project.

Do not begin frontend development before the model inference pipeline works.

---

## 14. Risks and mitigation

| Risk | Mitigation |
|---|---|
| GPU out of memory | Batch size 1, gradient accumulation, checkpointing and supported image pixel limits |
| Kaggle session ends | Save checkpoints and logs to persistent outputs regularly |
| QLoRA incompatibility | Test package and model compatibility with a small pilot |
| Dataset leakage | Preserve official splits and deduplicate before training |
| Overfitting | Monitor validation metrics and use early stopping/checkpoint selection |
| Poor explanation quality | Add human review and factual correctness checks |
| Weak cross-domain generalization | Evaluate on AI2D without training on it |
| Unfair comparison | Fix prompts, test samples and decoding settings |
| Deployment cost | Separate frontend from on-demand GPU inference |
| Licensing restrictions | Review dataset and model licenses before redistribution or commercial use |

### Research limitations to disclose

- Foundation models may have encountered related benchmark content during pretraining.
- Performance on ScienceQA does not prove general scientific reasoning.
- Automatic text-similarity metrics do not guarantee scientific correctness.
- AI2D and ScienceQA may differ in image styles, question formats and annotation conventions.
- Results depend on the selected model revision, processor settings and evaluation protocol.

---

## 15. Suggested effort allocation

| Workstream | Suggested effort |
|---|---:|
| Model training and experimentation | 30% |
| Evaluation and research | 25% |
| Dataset engineering | 15% |
| Frontend and deployment | 20% |
| Documentation | 10% |

Prioritize credible experimental results over frontend polish.

---

## 16. Final project specification

| Component | Decision |
|---|---|
| Project | VisionTune AI |
| Application | STEM visual question answering |
| Foundation model | Qwen2.5-VL-3B-Instruct (Qwen Research License) |
| Primary dataset | ScienceQA |
| Secondary benchmark | AI2D |
| Fine-tuning method 1 | LoRA |
| Fine-tuning method 2 | QLoRA |
| Framework | PyTorch, Transformers, PEFT (TRL optional) |
| Primary compute | Kaggle Notebooks |
| Experiment tracking | Weights & Biases |
| Backend | FastAPI |
| Frontend | Streamlit |
| Demo hosting | Hugging Face Spaces |
| GPU inference | RunPod or Modal, if required |
| Model hosting | Hugging Face Hub |
| Source control | GitHub |
| Priority | Balanced research and engineering |

### Final objective

Demonstrate measurable changes in scientific image understanding through parameter-efficient fine-tuning, while comparing LoRA and QLoRA in predictive performance and computational efficiency.

---

## 17. Immediate next step: Phase 1

Begin with a Kaggle notebook named:

`01_environment_setup.ipynb`

Complete these tasks before starting training:

- [ ] Enable and verify a Kaggle GPU accelerator.
- [ ] Record GPU model and available VRAM.
- [ ] Check Python, PyTorch, CUDA and package versions.
- [ ] Install compatible Transformers, PEFT, TRL and supporting libraries.
- [ ] Load Qwen2.5-VL-3B-Instruct and its processor.
- [ ] Run inference on one sample image.
- [ ] Save the environment details and successful test output.

Then proceed to `02_dataset_exploration.ipynb`.

**Do not start full LoRA or QLoRA training until model loading, image inference, dataset preprocessing and a small end-to-end training test have all succeeded.**

---

## 18. Key references

### Model
- Qwen2.5-VL-3B-Instruct: https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct
- Qwen2.5-VL documentation: https://huggingface.co/docs/transformers/en/model_doc/qwen2_5_vl

### Datasets
- ScienceQA: https://scienceqa.github.io/
- ScienceQA GitHub: https://github.com/lupantech/ScienceQA
- AI2D: https://allenai.org/data/diagrams

### Fine-tuning
- PEFT documentation: https://huggingface.co/docs/peft/main/en/index
- TRL SFTTrainer: https://huggingface.co/docs/trl/en/sft_trainer
- BitsAndBytes: https://huggingface.co/docs/bitsandbytes/main/en/index

### Compute and deployment
- Kaggle Notebooks: https://www.kaggle.com/code
- Hugging Face Spaces: https://huggingface.co/spaces
- RunPod: https://www.runpod.io/
- Modal: https://modal.com/

### Experiment tracking
- Weights & Biases: https://wandb.ai/

---

**Project status:** Blueprint v1.1. M1–M5 done; validation scaling study and held-out test evaluation complete.  
**Compute preference:** Kaggle-first.  
**Next action:** AI2D cross-dataset evaluation (H5), then the demo application (M7).
