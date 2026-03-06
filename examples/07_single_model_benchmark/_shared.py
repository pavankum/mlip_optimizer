"""Shared config helpers for the single-model benchmark workflow.

Provides trivial JSON config loading and path resolution used by both
``optimize_single_model.py`` (Phase 1) and ``compare_models.py``
(Phase 2).
"""

from __future__ import annotations

import json
from pathlib import Path


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def load_json_config(
    path: str | Path,
    required_keys: tuple[str, ...],
) -> dict:
    """Read a JSON config file and validate that *required_keys* are present."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as fh:
        config = json.load(fh)
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: '{key}'")
    return config


def resolve_path(raw: str, config_path: str | Path) -> Path:
    """Resolve *raw* path relative to the config file's directory."""
    p = Path(raw)
    if p.is_absolute():
        return p
    return (Path(config_path).resolve().parent / p).resolve()
