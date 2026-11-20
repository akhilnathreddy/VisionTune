import os

from visiontune.env import pin_single_gpu


def test_pin_single_gpu_respects_existing_choice(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    pin_single_gpu()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    pin_single_gpu()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"
