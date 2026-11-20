import pytest

torch = pytest.importorskip("torch")

from visiontune.models import loading  # noqa: E402


@pytest.mark.parametrize("capability,expected", [((7, 5), "float16"), ((6, 0), "float16"),
                                                 ((8, 0), "bfloat16"), ((9, 0), "bfloat16")])
def test_auto_dtype_follows_hardware_bf16(monkeypatch, capability, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: capability)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *a, **k: True)  # T4 reports True too
    assert loading.resolve_dtype("auto") == getattr(torch, expected)


def test_explicit_and_cpu_dtypes(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert loading.resolve_dtype("auto") == torch.float32
    assert loading.resolve_dtype("fp16") == torch.float16
    assert loading.resolve_dtype("bf16") == torch.bfloat16
