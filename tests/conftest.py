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
