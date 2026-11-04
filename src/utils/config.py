"""YAML config loading: ``base.yaml`` defaults, the condition file on top, then CLI overrides."""

import copy
from pathlib import Path

import yaml

BASE_CONFIG = "base.yaml"


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into a copy of ``base`` (override wins)."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def apply_overrides(config: dict, overrides: list[str]) -> dict:
    """Apply ``section.key=value`` overrides; values are parsed as YAML (``1e-3``, ``true``...)."""
    config = copy.deepcopy(config)
    for item in overrides:
        dotted, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"override must look like section.key=value, got {item!r}")
        *parents, leaf = dotted.split(".")
        node = config
        for key in parents:
            node = node.setdefault(key, {})
        node[leaf] = _parse_value(raw)
    return config


def _parse_value(raw: str) -> object:
    """YAML-parse a value, also reading ``1e-3``-style numbers (YAML 1.1 leaves them strings)."""
    value = yaml.safe_load(raw)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            pass
    return value


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    """Load a condition config merged over ``base.yaml`` from the same directory.

    Args:
        path: Condition YAML, e.g. ``configs/lpl.yaml``.
        overrides: Optional ``section.key=value`` strings applied last.

    Returns:
        The merged config, with ``config["name"]`` set to the condition file's stem.
    """
    path = Path(path)
    with open(path.parent / BASE_CONFIG) as f:
        config = yaml.safe_load(f) or {}
    if path.name != BASE_CONFIG:
        with open(path) as f:
            config = deep_merge(config, yaml.safe_load(f) or {})
    config = apply_overrides(config, overrides or [])
    config.setdefault("name", path.stem)
    return config
