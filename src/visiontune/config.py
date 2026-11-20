"""YAML experiment configs with `extends:` inheritance and dotted CLI overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable

import yaml


def deep_merge(base: dict, override: dict) -> dict:
    """Return a new dict where `override` is merged recursively into `base`."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def apply_overrides(cfg: dict, overrides: Iterable[str]) -> dict:
    """Apply `a.b.c=value` overrides. Values are parsed as YAML (so 1e-4, true, [a, b] work)."""
    cfg = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must look like key.path=value, got {item!r}")
        dotted, raw = item.split("=", 1)
        keys = dotted.strip().split(".")
        node = cfg
        for key in keys[:-1]:
            node = node.setdefault(key, {})
            if not isinstance(node, dict):
                raise ValueError(f"Cannot set {dotted}: {key} is not a mapping")
        node[keys[-1]] = _parse_value(raw)
    return cfg


def _parse_value(raw: str) -> Any:
    value = yaml.safe_load(raw)
    if isinstance(value, str):
        # PyYAML follows YAML 1.1, which reads "2e-4" (no dot) as a string.
        try:
            return float(value)
        except ValueError:
            pass
    return value


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> dict[str, Any]:
    """Load a config file, resolving `extends:` chains relative to each file."""
    path = Path(path)
    with path.open() as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("extends", None)
    if parent:
        cfg = deep_merge(load_config(path.parent / parent), cfg)
    return apply_overrides(cfg, overrides)


def save_config(cfg: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
