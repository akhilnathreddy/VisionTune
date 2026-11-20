# Phase 1: Kaggle environment check

Kaggle notebook `akhilnathreddy/visiontune-01-environment-setup`, Tesla T4 (2× 15 GB, compute 7.5, one GPU used),
torch 2.11.0+cu130, transformers 5.23.0, peft 0.22.0, bitsandbytes 0.50.2, Python 3.13.

## run_v2_bf16_emulated (2026-11-20)

`dtype: auto` resolved to **bf16**, because `torch.cuda.is_bf16_supported()` returns True on a T4 even though bf16 is only
emulated there. Fixed afterwards: `auto` now checks compute capability, so a T4 gets fp16.

| Check | Result |
|---|---|
| Model load (fp weights, from HF) | 44 s, 6.99 GB VRAM, 3.75 B parameters |
| Visual-token cap (2000×1500 image) | 494 tokens (cap 512) ✓ |
| LoRA presets A1 / A2 / A3 / A4 | 144 / 252 / 254 / 318 modules; no vision MLPs in A2 ✓ |
| Logits finite | ✓ |
| Inference, batch 1, 8 validation questions | 2.4–4.7 s each, peak 7.51 GB, 6/8 correct (smoke test, not a benchmark) |
| Answer format | 8/8 parsed "lenient": the base model answers `B` / `A. weather` without the `Answer:` prefix |
| QLoRA load | vision 162 × Linear (16-bit), language 252 × Linear4bit ✓, 3.27 GB VRAM |
| QLoRA forward + backward (1 example) | loss 1.057, 288 LoRA tensors got gradients, 7.37 M trainable (0.196%), peak 7.14 GB ✓ |
