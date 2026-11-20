"""End-to-end checks on a tiny, randomly initialized Qwen2.5-VL (CPU, ~1 minute).

Verifies the parts that are easy to get silently wrong (blueprint 3 / 5.6):
  * LoRA presets resolve to the intended modules only (vision tower untouched for A1/A2)
  * the loss mask covers exactly the assistant reply
  * gradients reach the LoRA weights and the training loss falls
    (a random 64-dim model cannot overfit quickly; the real overfit check is
    configs/smoke.yaml on the 3B model)
  * a saved adapter reloads and the prediction CLI produces scored output

Needs torch, transformers, peft and network access to download the real
Qwen2.5-VL processor (tokenizer + image processor, no model weights).
"""

import json

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")

from conftest import make_image, sqa_row  # noqa: E402
from visiontune.data.io import write_jsonl  # noqa: E402
from visiontune.data.records import scienceqa_record  # noqa: E402
from visiontune.models.targets import describe_targets, resolve_target_modules  # noqa: E402
from visiontune.training.collator import IGNORE_INDEX, SFTCollator, VQADataset  # noqa: E402

BASE = "Qwen/Qwen2.5-VL-3B-Instruct"


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory):
    from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration

    try:
        processor = AutoProcessor.from_pretrained(BASE, min_pixels=16 * 28 * 28, max_pixels=32 * 28 * 28)
        cfg = AutoConfig.from_pretrained(BASE)
    except Exception as exc:  # offline
        pytest.skip(f"cannot download processor/config: {exc}")
    cfg.text_config.update(dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                                num_attention_heads=4, num_key_value_heads=2,
                                layer_types=["full_attention"] * 2, max_window_layers=2))
    cfg.text_config.rope_parameters["mrope_section"] = [2, 3, 3]
    cfg.vision_config.update(dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=2,
                                  out_hidden_size=64, fullatt_block_indexes=[1]))
    torch.manual_seed(0)
    model = Qwen2_5_VLForConditionalGeneration(cfg)
    path = tmp_path_factory.mktemp("tiny_qwen25vl")
    model.save_pretrained(path)
    processor.save_pretrained(path)
    return path


@pytest.fixture
def cfg(base_cfg, tiny_model_dir, tmp_path):
    base_cfg.update({
        "run_name": "tiny", "method": "lora",
        "model": {"name_or_path": str(tiny_model_dir), "dtype": "fp32", "attn_implementation": "sdpa"},
        "processor": {"min_pixels": 16 * 28 * 28, "max_pixels": 32 * 28 * 28},
        "lora": {"r": 8, "alpha": 16, "dropout": 0.0, "targets": "lm_attn"},
        "train": {"max_train_samples": 4, "max_eval_samples": 2, "epochs": 30, "learning_rate": 5e-3,
                  "per_device_batch_size": 2, "gradient_accumulation_steps": 1, "logging_steps": 5,
                  "eval_steps": 1000, "save_steps": 1000, "warmup_ratio": 0.0,
                  "gradient_checkpointing": True, "output_root": str(tmp_path / "outputs")},
    })
    return base_cfg


def records_and_images(n, split, offset=0):
    rows = [sqa_row(question=f"Question {i}?", choices=(f"yes {i}", f"no {i}", "maybe"),
                    answer=i % 3, image_seed=i + offset) for i in range(n)]
    recs = [scienceqa_record(r, split, i, r["image"]) for i, r in enumerate(rows)]
    return recs, rows


def test_target_presets_respect_vision_boundary(tiny_model_dir):
    from transformers import Qwen2_5_VLForConditionalGeneration

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(tiny_model_dir)
    a1 = resolve_target_modules(model, "lm_attn")
    a2 = resolve_target_modules(model, "lm_attn_mlp")
    a3 = resolve_target_modules(model, "lm_attn_mlp_merger")
    a4 = resolve_target_modules(model, "lm_attn_mlp_merger_vision_attn")
    assert describe_targets(a1) == {"lm_attn": 8}
    assert describe_targets(a2) == {"lm_attn": 8, "lm_mlp": 6}          # NOT the vision MLPs
    assert not any("visual" in n for n in a1 + a2)
    assert describe_targets(a3) == {"lm_attn": 8, "lm_mlp": 6, "merger": 2}
    assert describe_targets(a4)["vision_attn"] == 4
    assert not any(n.endswith("lm_head") for n in a4)


def test_collator_masks_everything_but_the_answer(cfg):
    from visiontune.models.loading import load_processor

    processor = load_processor(cfg)
    recs, rows = records_and_images(2, "train")
    recs[1]["explanation"] = "A much longer explanation so the two rows pad differently."
    collator = SFTCollator(processor, cfg["prompt"]["system"])
    ds = VQADataset(recs, rows)
    batch = collator([ds[0], ds[1]])
    tok = processor.tokenizer
    for row, rec in enumerate(recs):
        kept = batch["labels"][row][batch["labels"][row] != IGNORE_INDEX]
        decoded = tok.decode(kept)
        expected = f"Answer: {'ABC'[rec['answer_index']]}. {rec['choices'][rec['answer_index']]}"
        assert decoded.startswith(expected)
        assert decoded.endswith("<|im_end|>")
        assert "Question" not in decoded and "<|image_pad|>" not in decoded
    assert (batch["labels"][batch["attention_mask"] == 0] == IGNORE_INDEX).all()
    assert "pixel_values" in batch and "image_grid_thw" in batch


def test_train_overfit_save_and_predict(cfg, tmp_path, monkeypatch):
    from visiontune.inference import predict
    from visiontune.training import train

    train_recs, train_rows = records_and_images(4, "train")
    val_recs, val_rows = records_and_images(2, "validation", offset=100)
    write_jsonl(train_recs, tmp_path / "processed/scienceqa_train.jsonl")
    write_jsonl(val_recs, tmp_path / "processed/scienceqa_validation.jsonl")
    sources = {"train": train_rows, "validation": val_rows}
    monkeypatch.setattr(train, "load_image_source", lambda _c, _d, split: sources[split])
    monkeypatch.setattr(predict, "load_image_source", lambda _c, _d, split: sources[split])

    summary = train.run(cfg)
    losses = [h["loss"] for h in summary["log_history"] if "loss" in h]
    assert losses[-1] < losses[0] - 0.25, losses
    assert 0 < summary["parameters"]["trainable"] < summary["parameters"]["total"]
    assert summary["lora_targets"] == {"lm_attn": 8}
    assert 0 < summary["adapter_size_mb"] < 1 < summary["adapter_dir_size_mb"]   # tiny adapter, ~11 MB tokenizer
    run_dir = tmp_path / "outputs/tiny"
    assert (run_dir / "adapter/adapter_config.json").exists()
    from safetensors.torch import load_file
    weights = load_file(run_dir / "adapter/adapter_model.safetensors")
    lora_b = [v for k, v in weights.items() if "lora_B" in k]
    assert lora_b and all(v.abs().sum() > 0 for v in lora_b)   # B starts at zero; non-zero means it trained
    assert not any("visual" in k for k in weights)
    ids = json.loads((run_dir / "sample_ids.json").read_text())
    assert sorted(ids["train"]) == sorted(r["id"] for r in train_recs)

    out = tmp_path / "preds.jsonl"
    metrics = predict.run(cfg, "scienceqa", "train", str(run_dir / "adapter"), None, out)
    assert metrics["overall"]["n"] == 4
    preds = [json.loads(line) for line in out.read_text().splitlines()]
    assert {p["id"] for p in preds} == {r["id"] for r in train_recs}
    # Resuming with every prediction present must not regenerate anything.
    assert predict.run(cfg, "scienceqa", "train", str(run_dir / "adapter"), None, out)["overall"]["n"] == 4
    assert len(out.read_text().splitlines()) == 4
