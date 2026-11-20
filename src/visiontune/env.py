"""Record the software and hardware environment of a run (blueprint 5.6 / 12)."""

from __future__ import annotations

import importlib
import json
import platform
import sys
from datetime import datetime, timezone

PACKAGES = [
    "torch", "torchvision", "transformers", "peft", "trl", "accelerate",
    "datasets", "bitsandbytes", "qwen_vl_utils", "flash_attn", "wandb", "numpy", "PIL",
]


def pin_single_gpu(index: int = 0) -> None:
    """Expose only one GPU to this process, unless the caller already chose.

    On multi-GPU machines (Kaggle "T4 x2") transformers' Trainer otherwise wraps the
    model in nn.DataParallel. DataParallel replicas have no registered parameters,
    so Qwen2.5-VL's `self.visual.dtype` raises StopIteration in the forward pass
    (observed on Kaggle, transformers 5.23). Must run before CUDA is initialized."""
    import os

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(index))


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            module = importlib.import_module(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception:  # not installed, or broken install
            versions[name] = None
    return versions


def gpu_info() -> list[dict]:
    try:
        import torch
    except ImportError:
        return []
    if not torch.cuda.is_available():
        return []
    gpus = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        gpus.append({
            "index": i,
            "name": props.name,
            "total_memory_gb": round(props.total_memory / 1024**3, 2),
            "compute_capability": f"{props.major}.{props.minor}",
            "bf16_supported": props.major >= 8,
        })
    return gpus


def environment_report() -> dict:
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": package_versions(),
        "gpus": gpu_info(),
    }
    try:
        import torch
        report["cuda_runtime"] = torch.version.cuda
        report["cudnn"] = torch.backends.cudnn.version() if torch.cuda.is_available() else None
    except ImportError:
        pass
    return report


if __name__ == "__main__":
    print(json.dumps(environment_report(), indent=2))
