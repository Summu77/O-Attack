"""Attack-stage path helpers."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"
ATTACK_CONFIG = CONFIG_DIR / "attack" / "default.yaml"
ATTACK_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "attack"
DATA_ROOT = PROJECT_ROOT / "data"


def resolve_path(path) -> Path:
    p = Path(os.path.expanduser(str(path)))
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()
