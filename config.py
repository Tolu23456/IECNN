"""Config loading with inheritance.

A config may contain `base: other.yaml` (path relative to its own folder).
Overlays (for example tiny.yaml) are merged last.
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml


def _deep_merge(a: dict, b: dict) -> dict:
    """Return a new dict: b merged over a, recursively."""
    out = copy.deepcopy(a)
    for key, value in b.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path, *overlays) -> dict:
    """Load a YAML config, following `base:` links, then apply overlays."""
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("base", None)
    if base:
        cfg = _deep_merge(load_config(path.parent / base), cfg)
    for overlay in overlays:
        overlay_path = Path(overlay)
        if not overlay_path.exists():
            overlay_path = path.parent / overlay
        cfg = _deep_merge(cfg, load_config(overlay_path))
    return cfg
