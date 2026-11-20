from pathlib import Path

from visiontune.config import apply_overrides, deep_merge, load_config

ROOT = Path(__file__).resolve().parents[1]


def test_deep_merge_keeps_unrelated_keys():
    assert deep_merge({"a": {"x": 1, "y": 2}}, {"a": {"y": 3}}) == {"a": {"x": 1, "y": 3}}


def test_overrides_parse_yaml_values():
    cfg = apply_overrides({"train": {"lr": 1}}, ["train.lr=2e-4", "train.flag=true", "new.key=[a, b]"])
    assert cfg["train"]["lr"] == 2e-4 and cfg["train"]["flag"] is True and cfg["new"]["key"] == ["a", "b"]


def test_repo_configs_resolve_inheritance():
    qlora = load_config(ROOT / "configs/qlora.yaml")
    assert qlora["method"] == "qlora"
    assert qlora["lora"]["r"] == 16                      # from lora.yaml
    assert qlora["model"]["name_or_path"].startswith("Qwen/Qwen2.5-VL")  # from base.yaml
    assert qlora["train"]["optim"] == "paged_adamw_8bit"
    assert qlora["train"]["gradient_accumulation_steps"] == 8
    smoke = load_config(ROOT / "configs/smoke.yaml")
    assert smoke["train"]["max_train_samples"] == 8 and smoke["method"] == "lora"
