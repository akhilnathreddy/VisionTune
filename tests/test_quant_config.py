"""The QLoRA skip list must keep the vision tower and lm_head out of 4-bit on the
installed transformers version (matching rules changed between v4 and v5)."""

from pathlib import Path

import pytest

from visiontune.config import load_config

pytest.importorskip("transformers")

ROOT = Path(__file__).resolve().parents[1]
VISION = ["model.visual.blocks.0.attn.qkv", "model.visual.blocks.31.mlp.gate_proj", "model.visual.merger.mlp.0"]
LANGUAGE = ["model.language_model.layers.0.self_attn.q_proj", "model.language_model.layers.35.mlp.down_proj"]


def test_skip_modules_match_vision_tower_only():
    try:
        from transformers.quantizers.quantizers_utils import should_convert_module
    except ImportError:
        pytest.skip("this transformers version uses substring matching (covered by the 'visual' entry)")
    skip = load_config(ROOT / "configs/qlora.yaml")["quantization"]["skip_modules"]
    assert not any(should_convert_module(n, skip) for n in VISION + ["lm_head"])
    assert all(should_convert_module(n, skip) for n in LANGUAGE)
    # Document the trap: a bare "visual" pattern does not protect the vision tower here.
    assert should_convert_module(VISION[0], ["visual", "lm_head"])
