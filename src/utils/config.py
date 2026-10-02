from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_get(cfg: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = cfg
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def resolve_path(base_dir: str | Path, path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else Path(base_dir) / p


def clone_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(cfg)
