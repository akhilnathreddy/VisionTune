import random

import pytest
from PIL import Image


def make_image(seed: int, size=(64, 48)) -> Image.Image:
    rng = random.Random(seed)
    img = Image.new("RGB", size)
    img.putdata([(rng.randrange(256), rng.randrange(256), rng.randrange(256))
                 for _ in range(size[0] * size[1])])
    return img


def sqa_row(question="Which is a mammal?", choices=("whale", "shark"), answer=0, image_seed=0,
            hint="", solution="Whales breathe air. Sharks have gills."):
    return {
        "image": make_image(image_seed) if image_seed is not None else None,
        "question": question, "choices": list(choices), "answer": answer, "hint": hint,
        "task": "closed choice", "grade": "grade4", "subject": "natural science", "topic": "biology",
        "category": "Classification", "skill": "Identify mammals", "lecture": "", "solution": solution,
    }


@pytest.fixture
def base_cfg(tmp_path):
    return {
        "seed": 42,
        "data": {"scienceqa_id": "fake/sqa", "ai2d_id": "fake/ai2d", "require_image": True,
                 "max_explanation_chars": 600, "processed_dir": str(tmp_path / "processed")},
        "prompt": {"system": "You are a careful science tutor.", "include_explanation": True},
        "generation": {"max_new_tokens": 8, "do_sample": False, "batch_size": 2},
    }


BASE = "Qwen/Qwen2.5-VL-3B-Instruct"


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
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
